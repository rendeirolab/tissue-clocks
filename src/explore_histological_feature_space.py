"""
Unsupervised analysis of GTEx feature space per slide.
"""

from pathlib import Path

from tqdm import tqdm
import numpy as np
import pandas as pd
from anndata import AnnData
import scanpy as sc
import parmap
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from seaborn_extensions import clustermap

from src import config
from src.ops import process, plot_latent
from src.utils import get_restricted_info, prepare_gtex_adata

parallel = True
voi = [
    "Organ",
    "Tissue",
    "Age Bracket",
    "Age Decade",
    "Sex",
    "n_tiles",
    "Hardy Scale",
]


def main():
    collect_features()
    output_dir = config.results_dir / "multi_model"
    output_dir.mkdir(exist_ok=True)
    long_file = output_dir / "dataset.multi_model.pq"
    explore_features(output_dir, long_file)

    collect_fine_tuned_features()
    output_dir = config.results_dir / "fine_tuned"
    output_dir.mkdir(exist_ok=True)
    # long_file = output_dir / "fine_tuned.long_format.pq"
    long_file = output_dir / "fine_tuned_convnext_base_3-200_e63.long_format.pq"
    explore_features(output_dir, long_file)
    a = sc.read_h5ad(config.results_dir / "fine_tuned" / "anndata.h5ad")
    prepare_gtex_adata(a)
    annot, var = get_restricted_info()
    a.obs = a.obs.reset_index().merge(annot[["Subject ID", "Age"]]).set_index("index")

    compare_feature_spaces(a)
    characterize_feature_spaces(a)

    collect_uni_features()
    output_dir = config.results_dir / "uni_features"
    output_dir.mkdir(exist_ok=True)
    long_file = output_dir / "uni_features.pq"
    explore_features(output_dir, long_file)


def collect_features():
    from src.utils import get_model_categories

    output_dir = config.results_dir
    output_dir.mkdir(exist_ok=True)
    long_file = output_dir / "dataset.multi_model.pq"

    var = pd.DataFrame(index=get_model_categories("resnet50"))

    dfl = pd.DataFrame(columns=var.index)
    if long_file.exists():
        dfl = pd.read_parquet(long_file)
    d = Path("data/gtex/svs/")
    fs = list(d.glob("*.npy"))
    # tmp fix: skip vit for now as it has been trained on different number of classes
    fs = [f for f in fs if "vit_h14" not in f.name]
    # skip fine tuned model features (it's done separately)
    fs = [f for f in fs if "_fine_tuned_" not in f.name]

    # fs = [f for f in fs if "convnext" in f.name]

    if not parallel:
        dfl2 = pd.DataFrame(
            {
                n: np.load(f).mean(0)
                for f in tqdm(fs, colour="yellow")
                if (n := f.name.replace(".npy", "")) not in dfl.index
            },
            index=var.index,
        ).T
    else:

        def load(f):
            x = np.load(f)
            if ".mean" not in f.name:
                return x.mean(0)
            return x

        _fs = {f: n for f in fs if (n := f.name.replace(".npy", "")) not in dfl.index}
        print(len(_fs))
        values = parmap.map(load, _fs.keys(), pm_pbar=True)
        dfl2 = pd.DataFrame(values, index=_fs.values(), columns=var.index)

    dfl = pd.concat([dfl, dfl2]).sort_index().rename_axis(index=None)
    dfl.to_parquet(long_file)


def collect_fine_tuned_features():
    output_dir = config.results_dir / "fine_tuned"
    output_dir.mkdir(exist_ok=True)
    long_file = output_dir / "fine_tuned.long_format.pq"

    dfl = pd.DataFrame()
    if long_file.exists():
        dfl = pd.read_parquet(long_file)
    d = Path("data/gtex/svs/")
    fs = list(d.glob("*fine_tune*.npy"))

    if not parallel:
        raise NotImplementedError

    def load(f):
        v = np.load(f)
        if ".mean" in f.name:
            name = f.stem.replace(".mean", "")
            if len(v.shape) == 3:
                res = pd.Series(v.mean((1, 2)), name=name)
            elif len(v.shape) == 2:
                res = pd.Series(v.mean((0)), name=name)
            elif len(v.shape) == 1:
                res = pd.Series(v, name=name)
            res.index = "v" + (res.index + 1).astype(str).str.zfill(4)
            return res
        raise NotImplementedError

    names = pd.Series([f.stem.replace(".mean", "") for f in fs])
    _fs = np.asarray(fs)[~names.isin(dfl.index)]
    print(len(_fs))
    values = parmap.map(load, _fs, pm_pbar=True)
    dfl2 = pd.DataFrame(values)

    dfl = pd.concat([dfl, dfl2]).sort_index().rename_axis(index=None)
    dfl.to_parquet(long_file)


def collect_uni_features():
    output_dir = config.results_dir / "uni_features"
    output_dir.mkdir(exist_ok=True)
    long_file = output_dir / "uni_features.pq"

    dfl = pd.DataFrame()
    if long_file.exists():
        dfl = pd.read_parquet(long_file)
    d = config.data_dir / "svs"
    # fs = sorted(d.glob("*uni*.npz"))
    fs = sorted(d.glob("*uni*.mean.npy"))

    if not parallel:
        raise NotImplementedError

    def load(f: Path) -> pd.Series:
        v = np.load(f)
        name: str = f.stem.split(".")[0]
        if ".mean" in f.name:
            name = f.stem.replace(".mean", "")
            if len(v.shape) == 3:
                res = pd.Series(v.mean((1, 2)), name=name)
            elif len(v.shape) == 2:
                res = pd.Series(v.mean((0)), name=name)
            elif len(v.shape) == 1:
                res = pd.Series(v, name=name)
            res.index = "v" + (res.index + 1).astype(str).str.zfill(4)
            return res
        raise NotImplementedError

    names = pd.Series([f.stem.replace(".mean", "") for f in fs])
    _fs = np.asarray(fs)[~names.isin(dfl.index)]
    print(len(_fs))
    values = parmap.map(load, _fs, pm_pbar=True)
    dfl2 = pd.DataFrame(values)

    dfl = pd.concat([dfl, dfl2]).sort_index().rename_axis(index=None)
    dfl.to_parquet(long_file)


def explore_features(output_dir: Path, long_file: Path):
    dfl = pd.read_parquet(long_file)
    # df.index = df.index.str.replace(".sample256t", ".resnet50.sample256t")

    # Expand variables
    q = dfl.index.to_series().str.split(".").apply(pd.Series)
    q = q.drop(q.columns[(q.dropna() == "mean").all()], axis=1)
    feat_sets = sorted(
        q.iloc[:, 1:].drop_duplicates().apply(lambda x: ".".join(x), axis=1)
    )

    print(feat_sets)
    _dff = list()
    for feat_set in tqdm(feat_sets):
        d = dfl.loc[dfl.index.str.contains(feat_set)]
        d.index = d.index.str.split(".").to_series().apply(lambda x: x[0])
        d.columns = feat_set + "." + d.columns
        _dff.append(d)
    df = pd.concat(_dff, axis=1)  # .dropna()

    # df = df.loc[:, df.columns.str.contains(r".*convnext_base.*")].dropna()

    # Set up AnnData
    a = AnnData(df.astype("float32"))
    prepare_gtex_adata(a)
    a.raw = a
    a.write_h5ad(output_dir / "anndata.h5ad")
    # a = sc.read(output_dir / "anndata.h5ad")

    # Plot basic stats
    for var in ["Organ", "Tissue"]:
        cl = pd.Series(
            a.uns[f"{var}_colors"], index=a.obs[var].cat.categories, name="color"
        )
        fig, ax = plt.subplots(figsize=(4, 8))
        c = a.obs[var].value_counts(sort=True)
        sns.barplot(
            x=c,
            y=c.index,
            orient="horiz",
            order=c.index,
            palette=c.to_frame().join(cl)["color"].values,
        )
        fig.savefig(
            output_dir / f"slide_distribution.{var}.barplot.svg", **config.figkws
        )

        fig, ax = plt.subplots()
        o = a.obs.groupby(var)["n_tiles"].mean().sort_values(ascending=False)
        sns.barplot(
            data=a.obs,
            x="n_tiles",
            y=var,
            orient="horiz",
            order=o.index,
            palette=o.to_frame().join(cl)["color"].values,
        )
        fig.savefig(
            output_dir / f"tile_distribution.{var}.barplot.svg", **config.figkws
        )

    fig, axes = plt.subplots(1, 2, figsize=(3 * 2, 3))
    sns.histplot(a.obs["n_tiles"], ax=axes[0], log_scale=False)
    sns.histplot(a.obs["n_tiles"], ax=axes[1], log_scale=True)
    fig.savefig(output_dir / "n_tiles.distplot.svg", **config.figkws)

    # Process
    a = process(a, confounder=None)
    a.write_h5ad(output_dir / "anndata.h5ad")
    # a = sc.read(output_dir / "anndata.h5ad")

    # Plot learned latents
    output_prefix = output_dir / "feature_extraction.SUFFIX"
    plot_latent(a, output_prefix=output_prefix, voi=voi)

    # Re-run for each tissue independently
    for tissue in [
        "Uterus",
        "Testis",
        "Liver",
        "Lung",
        "Spleen",
        "Stomach",
        "Pancreas",
    ]:
        b = a[a.obs["Tissue"] == tissue].copy()
        b.obsm = None
        b = process(b, short=True)
        plot_latent(b, prefix=prefix, voi=voi, label=tissue)

    # Repeat with mean of tissues
    vois = [v for v in voi if v not in ["Age Decade", "n_tiles"]]
    df = a.to_df()  # .astype('float16')
    # Dropna below is to remove non-existing cases in the dataset, e.g. Male ovary
    x = df.join(a.obs[vois]).groupby(vois).mean().dropna()
    s = np.log(
        df.join(a.obs[vois]).groupby(vois).size().rename("sample_number")
    ).reindex(x.index)
    t = a.obs.groupby(vois)["n_tiles"].mean().rename("n_tiles").reindex(x.index)
    am = AnnData(x.astype("float32"), obs=x.index.to_frame().join([s, t]))
    n_tissues = am.obs["Tissue"].nunique()
    n_ages = am.obs["Age Bracket"].nunique()
    am.obs["Age Decade"] = am.obs["Age Bracket"].str.slice(0, 1).astype(int)
    am.uns["Tissue_colors"] = [
        matplotlib.colors.rgb2hex(c)
        for c in plt.get_cmap("tab20", n_tissues)(range(n_tissues))
    ]
    am.uns["Age Bracket_colors"] = [
        matplotlib.colors.rgb2hex(c)
        for c in plt.get_cmap("inferno", n_ages + 1)(range(n_ages))
    ]
    assert am.var.shape[0] == a.var.shape[0]

    # For .obs category conversion
    # For serialization
    am.obs.index = am.obs.index.to_frame().apply(lambda x: " - ".join(x), axis=1)
    import os

    am.write(os.devnull)

    am = process(am, confounder=None)
    plot_latent(am, prefix=prefix, voi=vois, label="mean")

    for v in vois:
        if len(am.obs[v].cat.categories) <= 2:
            continue
        corr = am.to_df().join(am.obs[v]).groupby(v).mean().T.corr()
        grid = clustermap(corr, cmap="RdBu_r", center=0, dendrogram_ratio=0.1)
        grid.savefig(prefix + f".mean.{v}_correlation.svg", **config.figkws)

    corr = df.join(a.obs[vois[:-1]]).groupby(vois[:-1]).mean().dropna().T.corr()
    grid = clustermap(
        corr,
        cmap="RdBu_r",
        # center=0,
        row_colors=corr.index.to_frame().drop("Tissue", axis=1),
        dendrogram_ratio=0.1,
        xticklabels=True,
        yticklabels=True,
        figsize=(20, 20),
    )
    grid.ax_heatmap.set_rasterized(True)
    grid.savefig(prefix + ".mean.almost_all_variable_correlation.svg", **config.figkws)

    corr = df.join(a.obs[vois]).groupby(vois).mean().dropna().T.corr()
    grid = clustermap(
        corr,
        cmap="RdBu_r",
        # center=0,
        row_colors=corr.index.to_frame().drop("Tissue", axis=1),
        dendrogram_ratio=0.1,
        xticklabels=True,
        yticklabels=True,
        figsize=(20, 20),
    )
    grid.ax_heatmap.set_rasterized(True)
    grid.savefig(prefix + ".mean.all_variable_correlation.svg", **config.figkws)

    # TODO: write regression models fit per variable that have technical, demographic, etc variables as predictors


def compare_feature_spaces(a):
    from sklearn.metrics import silhouette_score

    output_dir = config.results_dir
    output_dir.mkdir(exist_ok=True)

    metrics_csv = output_dir / "feature_set_comparison.csv"

    metrics = pd.DataFrame(columns=["feature_set", "variable", "silhouette_score"])
    if metrics_csv.exists():
        metrics = pd.read_csv(metrics_csv)

    q = a.var.index.str.split(".").to_series().apply(pd.Series)
    feat_sets = sorted((q[0] + "." + q[1]).unique())

    # Mean, variance relationship
    for feat_set in feat_sets:
        of = output_dir / f"feature_set.statistics.{feat_set}.svg"
        if of.exists():
            continue
        b = a.raw[:, a.var.index.str.startswith(feat_set)].to_adata()
        assert b.var.shape[0] <= a.var.shape[0]
        b.var["mean"] = b.X.mean(0)
        b.var["var"] = b.X.var(0)
        b.var["std"] = b.X.std(0)
        b.var["qv2"] = (b.var["std"] / (b.var["mean"] + abs(b.var["mean"].min()))) ** 2

        fig, axes = plt.subplots(1, 3, figsize=(3.3 * 3, 3))
        for metric, ax in zip(["var", "std", "qv2"], axes):
            ax.scatter(data=b.var, x="mean", y=metric, alpha=0.2, s=5)
            ax.set(xlabel="Mean", ylabel=metric)
            if metric == "qv2":
                ax.set_yscale("log")
        fig.tight_layout()
        fig.savefig(of, **config.figkws)

    # Sample grouping separation
    _metrics = list()
    for feat_set in feat_sets:
        b = a[:, a.var.index.str.startswith(feat_set)].copy()
        b.obsm = None
        b = process(b, short=True)
        prefix = output_dir / "feature_extraction"
        plot_latent(b, prefix=prefix, voi=voi, label=feat_set)

        for v in ["Tissue", "Age Bracket", "Sex"]:
            score = silhouette_score(b.obsm["X_umap"], b.obs[v])
            _metrics.append([feat_set, v, score])
            print([feat_set, v, score])

    metrics2 = pd.DataFrame(
        _metrics, columns=["feature_set", "variable", "silhouette_score"]
    )
    metrics = pd.concat([metrics, metrics2], ignore_index=True)
    metrics.to_csv(output_dir / "feature_set_comparison.csv", index=False)

    # Inspect
    p = metrics.pivot_table(
        index="variable", columns="feature_set", values="silhouette_score"
    ).dropna()

    fig, axes = plt.subplots(1, 2, figsize=(12 * 2, 3), sharex=True, sharey=True)
    sns.heatmap(p, ax=axes[0], vmin=-1, vmax=1, cmap="coolwarm", annot=True)
    sns.heatmap(
        ((p.T - p.mean(1)) / p.std(1)).T,
        ax=axes[1],
        vmin=-1,
        vmax=1,
        cmap="coolwarm",
        annot=True,
    )
    fig.savefig(output_dir / "feature_set_comparison.svg", **config.figkws)

    n = len(feat_sets)
    fig, axes = plt.subplots(n, n, figsize=(n * 4, n * 3))
    i = 0
    for feat_set1 in feat_sets:
        b1 = a[:, a.var.index.str.startswith(feat_set1)]
        for feat_set2 in feat_sets:
            b2 = a[:, a.var.index.str.startswith(feat_set2)]
            ax = axes.flatten()[i]
            ax.scatter(b1.X.mean(1), b2.X.mean(1), alpha=0.1, s=1, rasterized=True)
            ax.set(xlabel=feat_set1, ylabel=feat_set2)
            i += 1
    fig.tight_layout()
    fig.savefig(
        output_dir / "feature_set_comparison.pairwise_mean_obs.svg", **config.figkws
    )

    fig, axes = plt.subplots(n, n, figsize=(n * 4, n * 3))
    i = 0
    for feat_set1 in feat_sets:
        b1 = a.raw[:, a.var.index.str.startswith(feat_set1)].to_adata()
        for feat_set2 in feat_sets:
            b2 = a.raw[:, a.var.index.str.startswith(feat_set2)].to_adata()
            ax = axes.flatten()[i]
            ax.scatter(b1.X.mean(0), b2.X.mean(0), alpha=0.25, s=5, rasterized=True)
            ax.set(xlabel=feat_set1, ylabel=feat_set2)
            i += 1
    fig.tight_layout()
    fig.savefig(
        output_dir / "feature_set_comparison.pairwise_mean_var.svg", **config.figkws
    )

    fig, axes = plt.subplots(n, n, figsize=(n * 4, n * 3))
    i = 0
    for feat_set1 in feat_sets:
        b1 = a.raw[:, a.var.index.str.startswith(feat_set1)].to_adata()
        for feat_set2 in feat_sets:
            b2 = a.raw[:, a.var.index.str.startswith(feat_set2)].to_adata()
            ax = axes.flatten()[i]
            ax.scatter(b1.X.max(0), b2.X.max(0), alpha=0.25, s=5, rasterized=True)
            ax.set(xlabel=feat_set1, ylabel=feat_set2)
            i += 1
    fig.tight_layout()
    fig.savefig(
        output_dir / "feature_set_comparison.pairwise_max_var.svg", **config.figkws
    )

    fig, axes = plt.subplots(n, n, figsize=(n * 4, n * 3))
    i = 0
    for feat_set1 in feat_sets:
        b1 = a[:, a.var.index.str.startswith(feat_set1)].copy()
        b1.var["mean"] = b1.X.mean(0)
        b1.var["std"] = b1.X.std(0)
        for feat_set2 in feat_sets:
            b2 = a[:, a.var.index.str.startswith(feat_set2)].copy()
            b2.var["mean"] = b2.X.mean(0)
            b2.var["std"] = b2.X.std(0)
            ax = axes.flatten()[i]
            ax.errorbar(
                x=b1.var["mean"],
                y=b2.var["mean"],
                xerr=b1.var["std"],
                yerr=b2.var["std"],
                alpha=0.5,
                rasterized=True,
            )
            ax.set(xlabel=feat_set1, ylabel=feat_set2)
            i += 1
    fig.savefig(
        output_dir / "feature_set_comparison.pairwise_var-mean_std.svg", **config.figkws
    )

    # r = {f.name.replace(".npy", ""): pd.DataFrame(np.load(f)) for f in list(d.glob('GTEX-11VI4-2026*.npy'))}

    # for model, df in r.items():
    #     cv = df.std(0) / df.mean(0)
    #     print(model, cv.mean())

    #     grid = clustermap(df.loc[:, cv.sort_values().tail(100).index], config='z')
    #     grid.savefig("results/" + model + ".clustermap.pdf", bbox_inches='tight')

    #     a = AnnData(df)
    #     sc.pp.scale(a)
    #     sc.pp.pca(a)
    #     fig = sc.pl.pca(a, color='10', show=False).figure
    #     fig.savefig("results/" + model + ".pca.pdf", bbox_inches='tight')

    #     sc.pp.neighbors(a)
    #     sc.tl.umap(a)
    #     fig = sc.pl.umap(a, color='10', show=False).figure
    #     fig.savefig("results/" + model + ".umap.pdf", bbox_inches='tight')


def characterize_feature_spaces(a):
    output_dir = config.results_dir
    output_dir.mkdir(exist_ok=True)

    q = a.var.index.str.split(".").to_series().apply(pd.Series)
    feat_sets = sorted((q[0] + "." + q[1]).unique())

    tissues = a.obs["Tissue"].cat.categories
    paths = pd.Index(
        a.obs["Pathology Categories"].str.split(", ").apply(pd.Series).stack().unique()
    )
    path_exclude = ["clean_specimens", "no_abnormalities", "tma"]
    # paths = [p for p in paths if p not in path_exclude]
    path_df = pd.DataFrame(
        [
            a.obs["Pathology Categories"].str.contains(path).rename(path).fillna(False)
            for path in paths
        ]
    ).T

    _res_path = dict()
    for tissue in tqdm(a.obs["Organ"].cat.categories):
        df = a[a.obs["Organ"] == tissue, :].to_df()
        df = (df - df.mean()) / df.std()
        for path in paths.drop(path_exclude):
            sel = path_df.reindex(df.index)[path]
            if sel.sum() <= 10:
                continue
            d = df.groupby(sel).mean()
            if d.shape[0] == 1:
                continue
            pos_mean = d.loc[True]
            # neg_mean = d.loc[False]
            ctr = path_df.reindex(df.index)[path_exclude[:-1]].any(axis=1)
            ctr_mean = df.groupby(ctr).mean().loc[True]
            # _res_path[f"{tissue} - {path}"] = (d.loc[True] - d.loc[False]) / (
            #     1 / np.log1p(path_df[path].sum())
            # )
            # _res_path[f"{tissue} - {path}"] = pos_mean - neg_mean
            _res_path[f"{tissue} - {path}"] = pos_mean - ctr_mean
            _res_path[f"{tissue} - {path}"].loc["log_n_pos"] = np.log1p(sel.sum())
            _res_path[f"{tissue} - {path}"].loc["frac_pos"] = sel.sum() / sel.shape[0]

    # t = df.groupby(a.obs["Tissue"]).mean()
    # res_path = pd.DataFrame(_res_path).join(t.T)
    res_path = pd.DataFrame(_res_path)
    annot = res_path.loc[["log_n_pos", "frac_pos"]]
    res_path = res_path.drop(["log_n_pos", "frac_pos"])

    for feat_set in feat_sets:
        x = res_path.loc[res_path.index.str.startswith(feat_set)]
        grid = clustermap(
            x.T,
            config="z",
            square=False,
            row_colors=annot.T.join(
                annot.columns.to_series()
                .str.split(" - ")
                .apply(lambda x: x[0])
                .rename("Tissue")
            ),
            figsize=(12, 16),
        )
        grid.ax_heatmap.set(rasterized=True)
        grid.savefig(
            output_dir / f"feature_set_charaterization.{feat_set}.clustermap.svg",
            **config.figkws,
        )


if __name__ == "__main__" and "get_ipython" not in locals():
    main()
