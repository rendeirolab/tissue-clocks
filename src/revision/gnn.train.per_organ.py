"""
GNN-based prediction of biological age from WSI graphs.
"""

from pathlib import Path

from tqdm import tqdm
import h5py
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
import torch
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from torch.nn import Linear, ReLU, Dropout, BatchNorm1d
from torch_geometric.nn import Sequential, GCNConv, JumpingKnowledge
from torch_geometric.nn.aggr import AttentionalAggregation
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn.functional import cross_entropy, l1_loss, mse_loss
from torcheval.metrics.functional import (
    multiclass_accuracy,
    multiclass_auroc,
    multiclass_auprc,
    r2_score,
)
import matplotlib.pyplot as plt
import seaborn as sns

from src.utils import get_restricted_info


data_dir = Path("data") / "gtex" / "svs"
output_dir = Path("results") / "gtex" / "gnn.train.age.per_organ"
output_dir.mkdir(exist_ok=True, parents=True)
meta = pd.read_csv(data_dir.parent / "GTEx Portal.csv")
meta["Organ"] = meta["Tissue"].str.replace(" - .*", "", regex=True)
indi, _ = get_restricted_info()
indi = indi.loc[:, ~indi.columns.isin(meta.columns)]
meta = meta.merge(indi, how="left", left_on="Subject ID", right_index=True).set_index(
    "Tissue Sample ID"
)
figkws = dict(dpi=300, bbox_inches="tight")


def main():
    feat_name = "cemm-convnext_base_fine_tune_63.224px"
    slides = sorted(data_dir.glob("*.h5"))
    all_slides = [f.stem for f in slides if f.with_suffix(f".{feat_name}.npy").exists()]
    # 25420

    new_target_variable = "Age"
    target_type = (
        "continuous"
        if meta[new_target_variable].dtype not in ["O", "categorical"]
        else "class"
    )

    for organ in sorted(meta.loc[all_slides, "Organ"].unique()):
        slide_names = meta.loc[all_slides].query("Organ == @organ").index.tolist()
        train_slides = (
            meta.loc[all_slides]
            .query("Organ == @organ")
            .sample(frac=0.8, random_state=42)
            .index.tolist()
        )

        # In memory (works well and feasible for a few thousand slides)
        # train_data = [get_data(s) for s in tqdm(train_slides)]
        # Lazy (works well and doesn't seem to be slower than in memory)
        train_data = WSIGraphDataset(slide_names=train_slides, target_variable="Age")
        train_dl = DataLoader(train_data, batch_size=64, shuffle=True, num_workers=12)
        train_targets = meta.loc[train_slides, new_target_variable]
        if target_type == "continuous":
            train_targets = train_targets.astype("float32").to_frame()
        else:
            train_targets = pd.get_dummies(train_targets).astype("float32")

        num_features = train_data[0].num_features
        num_classes = train_targets.shape[1]

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        epochs = 20
        for num_layers in [2, 4, 8]:
            for hidden_dim in [128, 256]:
                for dropout in [0.1, 0.25]:
                    arch_name, arch = make_gcn_attn(
                        num_features,
                        num_classes,
                        num_layers=num_layers,
                        hidden_dim=hidden_dim,
                        dropout=dropout,
                    )
                    tqdm.write(arch_name)

                    checkpoints = sorted(
                        output_dir.glob(
                            f"GCN-{num_layers}-{hidden_dim}-{dropout}-attn.{feat_name}.{organ}.Age.{num_classes}.steps=*.perf.pt"
                        )
                    )
                    if not checkpoints:
                        train(
                            model_name=f"{arch_name}.{feat_name}.{organ}",
                            target_variable=new_target_variable,
                            arch=arch,
                            feat_name=feat_name,
                            device=device,
                            epochs=epochs,
                            dl=train_dl,
                            target_type=target_type,
                            target_values=train_targets,
                            num_classes=num_classes,
                            output_dir=output_dir,
                        )

                    checkpoints = sorted(
                        output_dir.glob(
                            f"GCN-{num_layers}-{hidden_dim}-{dropout}-attn.{feat_name}.{organ}.Age.{num_classes}.steps=*.perf.pt"
                        )
                    )
                    checkpoint = checkpoints[-1]
                    perf_output = checkpoint.with_suffix(".csv")
                    if perf_output.exists():
                        continue
                    checkpoint = sorted(
                        output_dir.glob(
                            f"GCN-{num_layers}-{hidden_dim}-{dropout}-attn.{feat_name}.{organ}.Age.{num_classes}.steps=*.model_state_dict.pt"
                        )
                    )[-1]
                    arch.load_state_dict(
                        torch.load(checkpoint, map_location=device), strict=False
                    )

                    preds = inference(
                        arch,
                        slide_names=slide_names,
                        device=device,
                        new_target_variable=new_target_variable,
                    )
                    preds["split"] = (
                        preds.index.to_series()
                        .isin(train_slides)
                        .replace({True: "train", False: "val"})
                    )
                    preds.to_csv(perf_output)


def collect():
    from scipy.stats import pearsonr

    result_files = sorted(output_dir.glob("*steps*.perf.csv"))
    df = pd.concat(
        [
            pd.read_csv(f, index_col=0).assign(
                steps=f.name.split("steps=")[1].split(".")[0]
            )
            for f in result_files
        ]
    )

    r = (
        df.groupby(["Organ", "split", "steps"])
        .mean()
        .loc[:, "val", :]
        .sort_values("Error_abs")
    )  # .drop(['Fallopian Tube', 'Cervix', 'Bladder'])

    r = r.join(
        df.groupby(["Organ", "split", "steps"])
        .apply(
            lambda x: pearsonr(
                x["pred:Age"].values,
                x["Age"].values,
            ).statistic
        )
        .rename("pearson_r")
    )

    r = r.join(
        df.groupby(["Organ", "split", "steps"])
        .apply(
            lambda x: r2_score(
                torch.tensor(x["pred:Age"].values),
                torch.tensor(x["Age"].values),
            ).item()
        )
        .rename("r2_score")
    )
    r.to_csv(output_dir / "all.organs.perf.csv")

    p = (
        r.loc[:, :, "val", :][["Error_abs", "pearson_r", "r2_score"]]
        .drop(["Bladder", "Fallopian Tube", "Cervix"])
        .reset_index(level=1, drop=True)
    )

    mapping = {
        "Error_abs": {"cmap": "Reds_r"},
        "pearson_r": {"cmap": "vlag", "center": 0, "vmin": -1, "vmax": 1},
        "r2_score": {"cmap": "vlag", "center": 0, "vmin": -1, "vmax": 1},
    }

    fig, axes = plt.subplots(1, 3, figsize=(6, 6), sharey=True)
    for metric, ax in zip(p.columns, axes):
        sns.heatmap(p[[metric]], annot=True, fmt=".2f", ax=ax, **mapping[metric])
    fig.savefig(output_dir / "all.organs.perf.heatmap.svg", **figkws)


def inference(arch, slide_names: list[str], device, new_target_variable):
    arch = arch.to(device)
    vali_data = WSIGraphDataset(slide_names=slide_names, target_variable="Age")
    vali_dl = DataLoader(vali_data, batch_size=64, shuffle=False, num_workers=12)

    arch = arch.eval()
    _preds = list()
    for d in tqdm(vali_dl):
        d = d.to(device)
        with torch.no_grad():
            pred = arch(d.x, d.edge_index, d.batch)
        _preds.append(
            pd.DataFrame(
                pred.cpu().numpy(),
                index=d.attrs["name"],
                columns=["pred:" + new_target_variable],
            )
        )
    preds = pd.concat(_preds).join(meta[["Organ", new_target_variable]])
    preds["Error"] = preds["pred:" + new_target_variable] - preds[new_target_variable]
    preds["Error_abs"] = preds["Error"].abs()
    return preds


class WSIGraphDataset(Dataset):
    """
    A lazy dataset for WSI graphs.
    https://pytorch-geometric.readthedocs.io/en/latest/tutorial/create_dataset.html#creating-larger-datasets
    """

    def __init__(
        self,
        *,
        slide_names: list[str],
        root: Path = Path("data") / "gtex" / "svs",
        feat_name: str = "cemm-convnext_base_fine_tune_63.224px",
        edge_radius: int = 895,
        target_variable: str | None = None,
        transform=None,
        pre_transform=None,
        pre_filter=None,
    ):
        self.slide_names = slide_names
        self.root = root
        self.individual_ids = [s[:-5] for s in self.slide_names]
        self.edge_radius = edge_radius
        self.target_variable = target_variable
        if self.target_variable is not None:
            self.targets = meta.loc[self.slide_names, self.target_variable].tolist()
            # TODO: if target_variable is categorical, convert to one-hot
            if meta[self.target_variable].dtype in ["O", "category"]:
                self.targets = torch.tensor(
                    pd.get_dummies(self.targets).astype(float).values
                )
        else:
            self.targets = [None] * len(self.slide_names)
        self.coords_files = [(root / s).with_suffix(".h5") for s in self.slide_names]
        self.feats_files = [
            (root / s).with_suffix(f".{feat_name}.npy") for s in self.slide_names
        ]
        self.processed_files = [
            self.root / f"{s}.{feat_name}.graph.pt" for s in self.slide_names
        ]
        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self):
        return tuple(self.feats_files)

    @property
    def processed_file_names(self):
        return tuple(self.processed_files)

    def process(self):
        for slide_name, coords_file, feats_file, processed_file, target in tqdm(
            zip(
                self.slide_names,
                self.coords_files,
                self.feats_files,
                self.processed_files,
                self.targets,
            ),
            total=len(self.slide_names),
        ):
            if processed_file.exists():
                # data = torch.load(processed_file, weights_only=False)
                # data.y = target
                # torch.save(data, processed_file)
                continue
            coords = h5py.File(coords_file)["coords"][:]
            edge_index = self._get_edges(coords, self.edge_radius)
            x = torch.tensor(np.load(feats_file))
            data = Data(x=x, edge_index=edge_index, y=target)
            if self.pre_filter is not None and not self.pre_filter(data):
                return
            if self.pre_transform is not None:
                data = self.pre_transform(data)
            data.attrs["name"] = self.slide_name
            torch.save(data, processed_file)

    @staticmethod
    def _get_edges(coords, radius) -> torch.Tensor:
        tree = cKDTree(coords)
        pairs = tree.query_pairs(radius)
        edge_index = torch.tensor(np.array(list(pairs)).T)
        return edge_index

    def len(self):
        return len(self.processed_files)

    def get(self, idx):
        data = torch.load(self.processed_files[idx], weights_only=False)
        if not hasattr(data, "attrs"):
            data.attrs["name"] = self.slide_names[idx]
        return data


def get_data(
    slide_name: str,
    target_variable: str = "Age",
    feat_name: str = "cemm-convnext_base_fine_tune_63.224px",
) -> Data:
    processed_file = data_dir / f"{slide_name}.{feat_name}.graph.pt"
    if processed_file.exists():
        return torch.load(processed_file, weights_only=False)
    coords_file = data_dir / (slide_name + ".h5")
    feats_file = data_dir / (slide_name + f".{feat_name}.npy")
    coords = h5py.File(coords_file)["coords"][:]
    edge_index = get_edges(coords, radius=897)
    x = torch.tensor(np.load(feats_file))
    y = torch.tensor(
        meta.loc[slide_name, target_variable],
    )

    d = Data(x=x, edge_index=edge_index, y=y)
    return d


def get_edges(coords: np.ndarray, radius: int) -> torch.Tensor:
    tree = cKDTree(coords)
    pairs = tree.query_pairs(radius)
    edge_index = torch.tensor(np.array(list(pairs)).T)
    return edge_index


def make_gcn_attn(
    num_features: int,
    num_classes: int,
    num_layers: int,
    hidden_dim: int = 64,
    dropout: float = 0.5,
):
    """
    Build a GCN-n-attn model with Jumping Knowledge and Attentional Aggregation.

    Args:
        num_features (int): Number of input features.
        num_classes (int): Number of output classes.
        num_layers (int): Number of GCNConv layers.
        hidden_dim (int): Hidden dimension (default: 64).
        dropout (float): Dropout probability (default: 0.5).

    Returns:
        arch_name (str): Architecture name, e.g., "GCN-4-attn".
        arch (torch.nn.Module): The constructed model as a Sequential module.
    """
    layers = []
    # Dropout first
    layers.append((Dropout(p=dropout), "x -> x"))

    # Build GCNConv + BN + ReLU blocks
    for i in range(num_layers):
        in_dim = num_features if i == 0 else hidden_dim
        out_name = f"x{i+1}"
        prev_name = "x" if i == 0 else f"x{i}"
        layers.extend(
            [
                (GCNConv(in_dim, hidden_dim), f"{prev_name}, edge_index -> {out_name}"),
                (BatchNorm1d(hidden_dim), f"{out_name} -> {out_name}"),
                ReLU(inplace=True),
            ]
        )

    # Jumping Knowledge across all layers
    input_names = ", ".join([f"x{i+1}" for i in range(num_layers)])
    layers.append((lambda *args: list(args), f"{input_names} -> xs"))
    layers.append(
        (JumpingKnowledge("cat", hidden_dim, num_layers=num_layers), "xs -> x")
    )

    # Attentional Pooling
    layers.append(
        (
            AttentionalAggregation(
                torch.nn.Sequential(
                    Linear(num_layers * hidden_dim, 1), torch.nn.Sigmoid()
                )
            ),
            "x, batch -> x",
        )
    )

    # Final classifier
    layers.append(Linear(num_layers * hidden_dim, num_classes))

    arch_name = f"GCN-{num_layers}-{hidden_dim}-{dropout}-attn"
    arch = Sequential("x, edge_index, batch", layers)

    return arch_name, arch


def record(
    model_name: str,
    new_target_variable: str,
    target_type: str,
    num_classes: int,
    output_dir: Path,
    arch,
    optim,
    perf,
):
    steps = sum([len(e) for e in perf["loss"].values()])
    f = output_dir / f"{model_name}.{new_target_variable}.{num_classes}.{steps=}.X"
    torch.save(arch.state_dict(), f.with_suffix(".model_state_dict.pt"))
    torch.save(optim.state_dict(), f.with_suffix(".optim_state_dict.pt"))
    torch.save(perf, f.with_suffix(".perf.pt"))
    plot_performance(perf, target_type, prefix=f)


def plot_performance(perf, target_type, prefix: Path):
    rows = len(perf["metrics"]) + 1
    fig, axes = plt.subplots(rows, 1, figsize=(rows * 6, 3), sharex=True)
    axes[0].plot([y for x in perf["loss"].values() for y in x], label="Loss (GCN)")
    ll = "MAE" if target_type == "continuous" else "Cross-entropy"
    axes[0].set(xlabel="global step", ylabel=f"Loss ({ll})", yscale="log")
    for ax, metric, color in zip(axes[1:], perf["metrics"], sns.color_palette()[1:]):
        ax.plot(
            [y for x in perf["metrics"][metric].values() for y in x],
            label=f"{metric} (GCN)",
            color=color,
        )
        ax.set(
            xlabel="global step",
            ylabel=metric,
            yscale="log" if metric != "r2" else "linear",
            ylim=(0, None) if metric != "r2" else (-1, 1),
        )
    # for ax in axes:
    #     i = 0
    #     for epoch in perf["loss"]:
    #         i += len(perf["loss"][epoch])
    #         ax.axvline(i, color="grey", linestyle="--")
    fig.savefig(prefix.with_suffix(".perf.svg"), dpi=300, bbox_inches="tight")
    for ax in axes:
        ax.set_xscale("log")
    fig.savefig(prefix.with_suffix(".perf.log.svg"), dpi=300, bbox_inches="tight")


def train(
    model_name: str,
    target_variable: str,
    arch: torch.nn,
    feat_name: str,
    device: torch.device,
    epochs: int,
    dl: DataLoader,
    target_type: str,
    target_values: pd.Series,
    num_classes: int,
    output_dir: Path,
):
    arch = arch.to(device)
    optim = AdamW(
        arch.parameters(), lr=1e-3, weight_decay=1e-4, amsgrad=True, fused=True
    )
    scheduler = CosineAnnealingLR(optim, T_max=epochs)

    perf: dict = dict(
        loss=dict(),
        lr=dict(),
        metrics=(
            dict(mae=dict(), r2=dict())
            if target_type == "continuous"
            else dict(
                multiclass_accuracy=dict(),
                multiclass_auroc=dict(),
                multiclass_auprc=dict(),
            )
        ),
    )
    epoch = 0
    t0 = tqdm(total=epochs, position=0, leave=True)
    for epoch in range(epoch, epoch + epochs):
        perf["loss"][epoch] = list()
        for metric in perf["metrics"]:
            perf["metrics"][metric][epoch] = list()
        t1 = tqdm(total=len(dl), position=1, leave=False)
        for d in dl:
            # # To change the target on the fly
            d.y = torch.tensor(target_values.loc[d.attrs["name"]].values)
            if len(d.y.shape) == 1:
                d.y = d.y.reshape(-1, 1)

            # Training
            optim.zero_grad()
            d = d.to(device)
            pred = arch(d.x, d.edge_index, d.batch)
            if target_type == "continuous":
                loss = mse_loss(pred, d.y)
            else:
                loss = cross_entropy(pred, d.y)
            loss.backward()
            optim.step()

            # Metrics
            perf["loss"][epoch].append(loss.item())
            if target_type == "continuous":
                perf["metrics"]["mae"][epoch].append(l1_loss(pred, d.y).item())
                if d.y.shape[0] > 1:
                    perf["metrics"]["r2"][epoch].append(r2_score(pred, d.y).item())
            else:
                perf["metrics"]["multiclass_accuracy"][epoch].append(
                    multiclass_accuracy(pred, d.y).item()
                )
                perf["metrics"]["multiclass_auroc"][epoch].append(
                    multiclass_auroc(pred, d.y, num_classes=num_classes).item()
                )
                perf["metrics"]["multiclass_auprc"][epoch].append(
                    multiclass_auprc(pred, d.y, num_classes=num_classes).item()
                )

            t1.set_postfix_str(f"loss={loss:.3f}")
            t1.update()
        mean_loss = np.mean(perf["loss"][epoch])
        # perf["lr"][epoch] = scheduler.get_last_lr()[0]
        t0.set_postfix_str(f"loss={mean_loss:.3f}; lr={scheduler.get_last_lr()[0]:.2e}")
        t0.update()

        scheduler.step()

        if (epoch % 5 == 0) and (epoch > 0):
            record(
                model_name,
                target_variable,
                target_type,
                num_classes,
                output_dir,
                arch,
                optim,
                perf,
            )
            # validate()


if __name__ == "__main__":
    main()
