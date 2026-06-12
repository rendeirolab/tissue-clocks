# SPDX-FileCopyrightText: Copyright (c) 2026 Rendeiro Lab, CeMM - Research Center for Molecular Medicine, Austrian Academy of Sciences
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#
# Licensed under the PolyForm Noncommercial License 1.0.0 (see LICENSE).

"""
GNN-based prediction of biological age from WSI graphs.

NOTE: This script contains hardcoded machine-specific paths for remote
data retrieval via scp. It is included for transparency and is not
expected to run outside the original machine.
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
)
import matplotlib.pyplot as plt
import seaborn as sns

from src.utils import get_restricted_info


data_dir = Path("data") / "gtex" / "svs"
output_dir = Path("results") / "gtex" / "gnn.train.age.balanced_organ"
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
    slide_names = [
        f.stem for f in slides if f.with_suffix(f".{feat_name}.npy").exists()
    ]
    # 25420

    new_target_variable = "Age"
    target_type = (
        "continuous"
        if meta[new_target_variable].dtype not in ["O", "categorical"]
        else "class"
    )
    # n = meta.loc[slide_names, new_target_variable].value_counts().min()
    n = meta.loc[slide_names, "Organ"].value_counts().min()
    train_slides = (
        meta.loc[slide_names]
        .groupby("Organ")  # .groupby(new_target_variable)
        .sample(n=n, random_state=42)
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

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    epochs = 40
    steps = 720
    for num_layers in [2, 4, 8]:
        for hidden_dim in [32, 64, 128, 256]:
            for dropout in [0.1, 0.25, 0.5]:
                o = f"GCN-{num_layers}-{hidden_dim}-{dropout}-attn.{feat_name}.Age.{num_classes}.{steps=}.perf.pt"
                arch_name, arch = make_gcn_attn(
                    num_features,
                    num_classes,
                    num_layers=num_layers,
                    hidden_dim=hidden_dim,
                    dropout=dropout,
                )
                tqdm.write(arch_name)

                if not (output_dir / o).exists():
                    train(
                        model_name=f"{arch_name}.{feat_name}",
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

                o = f"{arch_name}.{feat_name}.{new_target_variable}.pred.csv"
                if (output_dir / o).exists():
                    continue
                f = (
                    output_dir
                    / f"{arch_name}.{feat_name}.{new_target_variable}.{num_classes}.{steps=}.X"
                )
                arch.load_state_dict(
                    torch.load(f.with_suffix(".model_state_dict.pt")), strict=False
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
                preds.to_csv(output_dir / o)

    # Train
    # # Collect perfs from all models at same iter
    # perf_files = sorted(output_dir.glob("*steps=900.perf.pt"))
    # _res = dict()
    # for f in perf_files:
    #     _res[f.stem.split("-attn")[0]] = np.asarray(
    #         torch.load(f, weights_only=False)["metrics"]["mae"][35]
    #     ).mean()
    # res = pd.Series(_res).to_frame("mae").sort_index()
    # res = pd.concat(
    #     [
    #         res,
    #         res.index.to_series()
    #         .str.extract(
    #             r"GCN-(?P<layers>\d+)-(?P<hidden_dim>\d+)-(?P<dropout>[\d\.]+)"
    #         )
    #         .astype(float),
    #     ],
    #     axis=1,
    # )
    # res = res.sort_values(res.columns[1:].tolist())
    # res.to_csv(output_dir / "gcn-attn.hyperparams_effect_on_performance.csv")

    # # # quantify
    # import statsmodels.api as sm

    # p = res.query("(mae < 40) & (dropout < 0.9)")
    # x = p[["layers", "hidden_dim", "dropout"]]
    # x = (x - x.mean()) / x.std()
    # y = p["mae"]
    # y = (y - y.mean()) / y.std()
    # fit_res = sm.OLS(y, sm.add_constant(x)).fit().summary2().tables[1]
    # fit_res.to_csv(output_dir / "gcn-attn.hyperparams_effect_on_performance.stats.csv")

    # fig, ax = plt.subplots(figsize=(3, 3))
    # ax.axvline(0, linestyle="--", color="grey", alpha=0.5, label="No effect")
    # ax.scatter(
    #     fit_res["Coef."],
    #     -np.log10(fit_res["P>|t|"]),
    #     # s=fit_res["Std.Err."] * 300,
    #     alpha=0.85,
    #     color="black",
    # )
    # for t in fit_res.index:
    #     ax.text(
    #         fit_res["Coef."][t],
    #         -np.log10(fit_res["P>|t|"][t]),
    #         s=t,
    #         ha="center",
    #         va="center",
    #         fontsize=8,
    #     )
    # ax.set(
    #     xlabel="Coefficient",
    #     ylabel="-log10(p-value)",
    #     title="GCN-attn Hyperparameters Effect on Performance",
    # )
    # fig.savefig(
    #     output_dir / "gcn-attn.hyperparams_effect_on_performance.model_fit.train.svg",
    #     **figkws,
    # )

    # # # Plot comparing hyperparameters
    # p = p.pivot_table(index="layers", columns=["hidden_dim", "dropout"], values="mae").T

    # fig, ax = plt.subplots(figsize=(5, 5))
    # sns.heatmap(
    #     p,
    #     annot=True,
    #     fmt=".3f",
    #     cmap="inferno_r",
    #     cbar_kws=dict(label="MAE (normalized)"),
    #     ax=ax,
    #     vmin=0.0,
    #     vmax=20,
    # )
    # ax.set(
    #     xlabel="Layers",
    #     ylabel="Hidden dim - Dropout",
    #     title="GCN-attn Hyperparameters Effect on Performance",
    # )
    # fig.savefig(
    #     output_dir / "gcn-attn.hyperparams_effect_on_performance.heatmap.train.svg",
    #     **figkws,
    # )

    # Validation
    # Collect inference from all models at same iter
    res_files = sorted(output_dir.glob("*Age.pred.csv"))
    _res = list()
    for f in res_files:
        model = f.stem.split("-attn")[0]
        _res.append(pd.read_csv(f, index_col=0).assign(model=model))
    res = pd.concat(_res)
    res = pd.concat(
        [
            res,
            res["model"]
            .str.extract(
                r"GCN-(?P<layers>\d+)-(?P<hidden_dim>\d+)-(?P<dropout>[\d\.]+)"
            )
            .astype(float),
        ],
        axis=1,
    )

    p = (
        res.query("(split == 'val') & (dropout < 0.5)")  # & (Error_abs < 40)
        .groupby(["Organ", "split", "model"])["Error_abs"]
        .mean()
        .loc[:, "val", :]
    )
    pp = p.unstack("model").T
    i = (
        pp.index.to_series()
        .str.extract(r"GCN-(?P<layers>\d+)-(?P<hidden_dim>\d+)-(?P<dropout>[\d\.]+)")
        .astype(float)
    )
    i.sort_values(i.columns.tolist(), inplace=True)
    fig, ax = plt.subplots(figsize=(15, 10))
    sns.heatmap(
        pp.reindex(i.index),
        cmap="Reds_r",
        cbar_kws=dict(label="MAE (years)"),
        ax=ax,
        vmin=0,
        vmax=25,
        annot=True,
        fmt=".2f",
    )
    ax.set(
        xlabel="Organ",
        ylabel="Model (layers-hidden_dim-dropout)",
        title="GCN-attn Performance per Organ",
    )
    fig.savefig(
        output_dir / "gcn-attn.performance_per_organ.heatmap.valid.abs.svg", **figkws
    )
    fig, ax = plt.subplots(figsize=(15, 10))
    sns.heatmap(
        ((pp - pp.mean()) / pp.std()).reindex(i.index),
        cmap="coolwarm",
        cbar_kws=dict(label="MAE (years)"),
        ax=ax,
        vmin=-3,
        vmax=3,
    )
    ax.set(
        xlabel="Organ",
        ylabel="Model (layers-hidden_dim-dropout)",
        title="GCN-attn Performance per Organ",
    )
    fig.savefig(
        output_dir / "gcn-attn.performance_per_organ.heatmap.valid.z.svg", **figkws
    )

    # More condensed

    pp = (
        res.query(
            "(split == 'val') & (Error_abs < 40) & (dropout < 0.9) & Organ == 'Artery'"
        )
        .groupby(["Organ", "split", "model"])
        .mean()
        .loc[:, "val", :]
        .pivot_table(
            index=["hidden_dim", "dropout"], columns="layers", values="Error_abs"
        )
    )

    # # quantify
    import statsmodels.api as sm

    p = (
        res.query("(split == 'val') & (Error_abs < 40) & (dropout < 0.9)")
        .groupby(["model", "Organ", "split"])
        .mean()
        .reset_index()
    )
    x = p[["layers", "hidden_dim", "dropout"]]
    x = (x - x.mean()) / x.std()
    y = p["Error_abs"]
    y = (y - y.mean()) / y.std()
    fit_res = sm.OLS(y, sm.add_constant(x)).fit().summary2().tables[1]
    fit_res.to_csv(
        output_dir / "gcn-attn.hyperparams_effect_on_performance.stats.valid.csv"
    )
    fit_res["P>|t|"] = fit_res["P>|t|"].replace(0, 1e-300)

    fig, ax = plt.subplots(figsize=(2, 2))
    ax.axvline(0, linestyle="--", color="grey", alpha=0.5, label="No effect")
    ax.scatter(
        fit_res["Coef."],
        -np.log10(fit_res["P>|t|"]),
        # s=fit_res["Std.Err."] * 300,
        alpha=0.85,
        color="black",
    )
    for t in fit_res.index:
        ax.text(
            fit_res["Coef."][t],
            -np.log10(fit_res["P>|t|"][t]),
            s=t,
            ha="center",
            va="center",
            fontsize=8,
        )
    ax.set(
        xlabel="Coefficient",
        ylabel="-log10(p-value)",
        title="GCN-attn Hyperparameters Effect on Performance",
    )
    fig.savefig(
        output_dir / "gcn-attn.hyperparams_effect_on_performance.model_fit.valid.svg",
        **figkws,
    )

    fig, axes = plt.subplots(1, 3, figsize=(3 * 3, 1 * 3))
    for ax, var in zip(axes, ["layers", "hidden_dim", "dropout"]):
        sns.boxplot(x=p[var].astype(np.float32), y=p["Error_abs"], ax=ax)
        ax.set(ylabel="MAE (years)", title=var)
    fig.savefig(
        output_dir / "gcn-attn.hyperparams_effect_on_performance.boxplots.valid.svg",
        **figkws,
    )

    fig, axes = plt.subplots(3, 10, figsize=(10 * 3, 3 * 3), sharex=True, sharey=False)
    for organ, ax in zip(sorted(res["Organ"].unique()), axes.flatten()):
        p = res.query("(Organ == @organ) & (split == 'val')")
        if p.empty:
            continue
        model = p.groupby("model")["Error_abs"].mean().idxmin()
        p = p.query("model == @model")
        if p.empty:
            continue
        ax.scatter(
            p[new_target_variable],
            p["pred:" + new_target_variable],
            c=p["Error"],
            cmap="coolwarm",
            vmin=-10,
            vmax=10,
            s=0.5,
            alpha=0.5,
            rasterized=True,
        )
        r = np.corrcoef(p[new_target_variable], p["pred:" + new_target_variable])[0, 1]
        from sklearn.metrics import r2_score

        r2 = r2_score(
            p[new_target_variable].astype(float).values[:, None],
            p["pred:" + new_target_variable].astype(float).values[:, None],
        )
        ax.text(
            20, 55, s=f"MAE = {p['Error_abs'].mean():.2f}\nr = {r:.2f}\nR² = {r2:.2f}"
        )
        ax.plot((20, 70), (20, 70), linestyle="--", color="grey")
        ax.set(title=organ, xlabel="", ylabel="")
    for ax in axes[-1, :]:
        ax.set(xlabel=new_target_variable)
    for ax in axes[:, 0]:
        ax.set(ylabel="Predicted")
    for ax in axes.flatten():
        # ax.label_outer()
        if not ax.has_data():
            ax.axis("off")
    fig.savefig(
        output_dir
        / f"gcn-attn.all_models.{new_target_variable}.pred.per_organ.valid.svg",
        **figkws,
    )

    # Plot attention
    num_features = 1024
    num_layers = 8
    hidden_dim = 128
    dropout = 0.1
    num_classes = 1
    steps = 900
    arch_name, arch = make_gcn_attn(
        num_features,
        num_classes,
        num_layers=num_layers,
        hidden_dim=hidden_dim,
        dropout=dropout,
    )
    o = f"GCN-{num_layers}-{hidden_dim}-{dropout}-attn.{feat_name}.Age.{num_classes}.{steps=}.perf.pt"
    f = (
        output_dir
        / f"{arch_name}.{feat_name}.{new_target_variable}.{num_classes}.{steps=}.X"
    )
    arch.load_state_dict(
        torch.load(f.with_suffix(".model_state_dict.pt")), strict=False
    )

    viz_slides = (
        meta.query("`Age Bracket`.isin(['20-29', '60-69'])")
        .groupby(["Organ", "Age Bracket"])
        .sample(n=2, replace=False)
        .index
    )
    viz_slides = [s for s in viz_slides if (data_dir / f"{s}.{feat_name}.npy").exists()]

    viz_data = WSIGraphDataset(slide_names=viz_slides, target_variable="Age")
    viz_dl = DataLoader(viz_data, batch_size=64, shuffle=False, num_workers=12)

    for batch in viz_dl:
        plot_graph_attention(arch, batch)


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


def get_attention_outputs(arch, data):
    """
    Forward through a PyG Sequential GCN-n-attn model up to the attention layer.
    Extract node-level attention weights for each node in the batch.

    Args:
        arch: torch_geometric.nn.Sequential model (GCN-n-attn)
        data: PyG Data object with attributes x, edge_index, batch

    Returns:
        attn_weights (Tensor): Normalized per-node attention weights [num_nodes]
        attn_scores (Tensor): Raw gate outputs per node
        pooled (Tensor): Graph-level pooled embeddings [num_graphs, hidden_dim * num_layers]
        x_jk (Tensor): Node embeddings after JumpingKnowledge
    """
    from torch_scatter import scatter_softmax

    x, edge_index, batch = data.x, data.edge_index, data.batch

    # Dropout
    x = arch[0](x)

    # Collect outputs from each GCN block
    conv_outputs = []
    i = 1
    while (
        isinstance(arch[i], type(arch[1]))
        or isinstance(arch[i], torch.nn.BatchNorm1d)
        or isinstance(arch[i], torch.nn.ReLU)
    ):
        # We expect blocks of 3: (GCNConv, BatchNorm, ReLU)
        x = arch[i](x, edge_index)  # GCNConv
        x = arch[i + 1](x)  # BatchNorm
        x = arch[i + 2](x)  # ReLU
        conv_outputs.append(x)
        i += 3

    # Lambda collects them
    xs = arch[i](*conv_outputs)
    i += 1

    # JumpingKnowledge
    x_jk = arch[i](xs)
    i += 1

    # AttentionalAggregation
    attn = arch[i]
    attn_scores = attn.gate_nn(x_jk).squeeze()  # raw gate outputs per node
    attn_weights = scatter_softmax(attn_scores, batch)
    pooled = attn(x_jk, batch)  # pooled graph-level representation

    return attn_weights, attn_scores, pooled


def smooth_attention(G, attn_weights, alpha=0.5, n_iter=5):
    """
    Smooth attention values over the graph using neighbor averaging.
    alpha: smoothing factor (0=no smoothing, 1=full neighbors)
    n_iter: number of smoothing iterations
    """
    attn = attn_weights.copy()
    for _ in range(n_iter):
        new_attn = attn.copy()
        for i in range(len(attn)):
            neighbors = list(G.neighbors(i))
            if neighbors:
                new_attn[i] = (1 - alpha) * attn[i] + alpha * np.mean(attn[neighbors])
        attn = new_attn
    return attn


def plot_graph_attention(arch, d, overwrite: bool = False):
    """
    Plot graphs coloring nodes by attention weight.

    Args:
        d: PyG Data object
        x_jk: Node embeddings after JumpingKnowledge (Tensor [num_nodes, feat])
        attn_weights: Node attention weights (Tensor [num_nodes])
    """
    import os
    import networkx as nx
    import matplotlib.pyplot as plt

    (output_dir / "attention_maps").mkdir(exist_ok=True, parents=True)

    with torch.no_grad():
        attn_weights, attn_scores, pooled = get_attention_outputs(arch, d)

    for slide_idx, slide_name in enumerate(d.attrs["name"]):
        output_file = output_dir / "attention_maps" / f"{slide_name}.smoothed.svg"
        if output_file.exists() and not overwrite:
            continue
        mask = d.batch == slide_idx
        nodes_idx = mask.nonzero(as_tuple=True)[0]

        # Subset edges: only edges where both nodes are in first graph
        edge_mask = mask[d.edge_index[0]] & mask[d.edge_index[1]]
        edge_index_sub = d.edge_index[:, edge_mask]

        # Remap node indices to consecutive numbers for plotting
        node_mapping = {old.item(): i for i, old in enumerate(nodes_idx)}
        edges = [
            (node_mapping[u.item()], node_mapping[v.item()])
            for u, v in edge_index_sub.t()
        ]

        # Create NetworkX graph
        G = nx.Graph()
        G.add_nodes_from(range(len(nodes_idx)))
        G.add_edges_from(edges)

        # Node colors by attention weight
        node_colors = attn_weights[mask].cpu().numpy()
        smoothed_colors = smooth_attention(G, node_colors, alpha=0.5, n_iter=5)

        # Optional: node positions
        if hasattr(d, "pos") and d.pos is not None:
            pos = {i: d.pos[n].cpu().numpy() for i, n in enumerate(nodes_idx)}
        else:
            pos = nx.spring_layout(G, seed=42)

        fig, axes = plt.subplots(1, 2, figsize=(2 * 8, 6))
        _remote_data_dir = (
            "login:/nobackup/lab_rendeiro/projects/histopath/data/gtex/svs"
        )
        os.system(f"scp {_remote_data_dir}/{slide_name}.segmentation.png ./")
        slide_img = plt.imread(f"{slide_name}.segmentation.png")
        os.remove(f"{slide_name}.segmentation.png")
        axes[0].imshow(slide_img)
        axes[0].set(title="Slide")
        axes[0].axis("off")
        img_height, img_width = slide_img.shape[:2]
        aspect_ratio = img_width / img_height  # width / height

        ax = axes[1]
        v = smoothed_colors.max()
        v += v * 0.1
        nx.draw(
            G,
            pos,
            node_color=smoothed_colors,
            cmap="magma",
            with_labels=False,
            node_size=6 * np.sqrt(2),
            edge_color="lightgray",
            vmin=smoothed_colors.min(),
            vmax=v,
            ax=ax,
        )
        ax.set_aspect(aspect_ratio)
        sm = plt.cm.ScalarMappable(
            cmap="magma",
            norm=plt.Normalize(vmin=smoothed_colors.min(), vmax=v),
        )
        sm.set_array([])
        plt.colorbar(sm, label="Attention weight", ax=ax)
        ax.set(title="Attention weights")
        ax.yaxis.set_inverted(True)
        fig.suptitle(
            f"{slide_name} - {meta.loc[slide_name, 'Organ']} - {meta.loc[slide_name, 'Age Bracket']}"
        )
        axes[1].get_children()[0].set_rasterized(True)
        fig.savefig(output_file, **figkws)


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
        ax.set(xlabel="global step", ylabel=metric, yscale="log")
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
            dict(mae=dict())
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
