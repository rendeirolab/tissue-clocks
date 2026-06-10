"""
Extract PLIP embeddings from a WSI.

# Run with:
cd /nobackup/lab_rendeiro/projects/histopath
mkdir -p logs/plip_inference
for N in {0..3}; do
sbatch \
-J plip_inference.gpu:l4_gpu.${N} \
-D /nobackup/lab_rendeiro/projects/histopath \
-o logs/plip_inference/plip_inference.gpu:l4_gpu.${N}.log \
-p gpu --qos gpu -c 8 --gres=gpu:l4_gpu:1 --mem=32G -t 3-00:00:00 --wrap "python -m fire src/plip_inference.py main"
done

cd /nobackup/lab_rendeiro/projects/histopath
mkdir -p logs/plip_inference
for QUEUE in tinyq shortq mediumq longq; do
for N in {0..19}; do
sbatch \
-J plip_inference.${QUEUE}.${N} \
-D /nobackup/lab_rendeiro/projects/histopath \
-o logs/plip_inference/plip_inference.${QUEUE}.${N}.log \
-p ${QUEUE} --qos ${QUEUE} -c 8 --mem=32G -t 02:00:00 --wrap "python -m fire src/plip_inference.py main"
done
done

# Test interactively:
# srun \
# --x11 --pty \
# -J plip_inference.try \
# -p gpu --qos gpu -c 8 --gres=gpu:l4_gpu:1 --mem=32G -t 3-00:00:00 bash
"""

from pathlib import Path
import tempfile
import shutil

import numpy as np
import pandas as pd
import torch
from transformers import CLIPModel, CLIPProcessor
from wsi import WholeSlideImage

metadata_dir = Path("metadata")
data_dir = Path("data") / "gtex" / "svs"
output_dir = Path("data") / "gtex" / "PLIP_histopath_150term_512px"
output_dir.mkdir(parents=True, exist_ok=True)
figkws = dict(bbox_inches="tight", dpi=300)

tmp_dir = Path(tempfile.mkdtemp())

text_classes = (
    (metadata_dir / "histology_and_histopathology_terms.txt")
    .read_text()
    .strip()
    .split("\n")
)

model = CLIPModel.from_pretrained("vinid/plip")
processor = CLIPProcessor.from_pretrained("vinid/plip")


def main():
    wsi_paths = sorted(data_dir.glob("*.svs"))
    np.random.shuffle(wsi_paths)
    for wsi_path in wsi_paths:
        do_one(wsi_path)

    shutil.rmtree(tmp_dir)


def do_one(wsi_path: Path):
    output_file1 = output_dir / (wsi_path.stem + ".feats.csv.gz")
    output_file2 = output_dir / (wsi_path.stem + ".probs.csv.gz")
    if output_file2.exists():
        return
    print(f"Doing {wsi_path.stem}")
    hdf5_file = tmp_dir / (wsi_path.stem + ".h5")
    wsi = WholeSlideImage(wsi_path, hdf5_file=hdf5_file)
    try:
        wsi.segment()
    except AssertionError:
        print(f"Failed for {wsi_path.stem}")
        return
    wsi.tile(patch_size=512, step_size=512)

    coords = wsi.get_tile_coordinates()
    n = min(200, len(coords))
    sel = np.random.choice(range(coords.shape[0]), n, replace=False)

    _res = list()
    _probs = list()
    for coord in coords[sel]:
        img = wsi.wsi.read_region(coord, 0, (512, 512))
        inputs = processor(
            text=text_classes, images=img, return_tensors="pt", padding=True
        )
        with torch.no_grad():
            out = model(**inputs)
        _res.append(out.image_embeds.squeeze().numpy())
        _probs.append(out.logits_per_image.softmax(dim=1).numpy().squeeze())

    spatial = pd.DataFrame(coords[sel], columns=["y", "x"])
    spatial.join(pd.DataFrame(_res)).to_csv(output_file1, index=False)
    spatial.join(pd.DataFrame(_probs, columns=text_classes)).to_csv(
        output_file2, index=False
    )
    print(f"Finished {wsi_path.stem}")


def spatial_analysis(slide_id):
    # Spatial analysis
    from anndata import AnnData
    import scanpy as sc

    coords = wsi.get_tile_coordinates()
    df = pd.DataFrame(np.asarray(_probs).squeeze(), columns=text_classes)
    a = AnnData(df)
    a.obsm["spatial"] = coords

    a = a[:, (a.X.mean(0) > 1e-5) & (a.X.max(0) > 0.1)]

    sc.pp.scale(a)
    sc.pp.pca(a)
    sc.pp.neighbors(a)
    sc.tl.umap(a)
    sc.tl.diffmap(a)
    a.obsm["X_diffmap"] = a.obsm["X_diffmap"][:, 1:]
    sc.tl.draw_graph(a)

    pca = pd.DataFrame(a.varm["PCs"], index=a.var.index)
    n = 6
    voi = (
        a.var.sort_values("mean").tail(n).index.tolist()
        + a.var.sort_values("std").tail(n).index.tolist()
        + pca.sort_values(0).tail(n).index.tolist()
        + pca.sort_values(1).tail(n).index.tolist()
    )
    # voi = np.unique(voi)
    for emb in a.obsm.keys():
        fig = sc.pl.embedding(a, basis=emb, color=voi, ncols=n, return_fig=True)
        for ax in fig.axes:
            ax.yaxis.set_inverted(True)


def analysis():
    from anndata import AnnData
    import scanpy as sc
    import matplotlib.pyplot as plt
    import seaborn as sns
    from tqdm import tqdm
    import sklearn
    import sklearn.feature_selection
    import statsmodels.api as sm
    from seaborn_extensions import clustermap
    from src.utils import rasterize_scanpy, prepare_gtex_adata
    from src.utils import get_restricted_info

    results_dir = Path("results") / "plip_inference"
    results_dir.mkdir(exist_ok=True)

    meta = pd.read_csv(data_dir.parent / "GTEx Portal.csv", index_col=0)
    meta["Organ"] = meta["Tissue simple"] = meta["Tissue"].str.replace(
        r" - .*", "", regex=True
    )
    files = sorted(output_dir.glob("*.probs.csv.gz"))
    ids = [f.stem.split(".")[0] for f in files]
    print(len(files))
    if (results_dir / "probs.csv.gz").exists():
        _x = pd.read_csv(results_dir / "probs.csv.gz", index_col=0)
    else:
        _x = pd.DataFrame()

    x = pd.DataFrame(
        [
            pd.read_csv(f).drop(["y", "x"], axis=1).mean(0).rename(_id)
            for _id, f in tqdm(zip(ids, files))
            if _id not in _x.index
        ]
    ).rename_axis(index="Slide")
    x = pd.concat([_x, x])
    x.to_csv(results_dir / "probs.csv.gz")

    a = AnnData(x)
    prepare_gtex_adata(a)
    a.raw = a
    sc.write(results_dir / "probs.h5ad", a)

    a = sc.read_h5ad(results_dir / "probs.h5ad")
    # # batch correction within tissue
    # _as = list()
    # for tissue in sorted(a.obs["Tissue"].unique()):
    #     _a = a[a.obs["Tissue"] == tissue].copy()
    #     sc.pp.combat(_a, "Hardy Scale")
    #     if not _a.to_df().isnull().sum().any():
    #         sc.pp.scale(_a)
    #         _as.append(_a)
    # a = sc.concat(_as)
    sc.pp.scale(a)
    sc.pp.pca(a)
    # sc.external.pp.bbknn(a, "Hardy Scale")
    sc.external.pp.harmony_integrate(a, "Hardy Scale")
    sc.pp.neighbors(a, use_rep="X_pca_harmony")
    sc.tl.umap(a)
    sc.tl.diffmap(a)
    a.obsm["X_diffmap"] = a.obsm["X_diffmap"][:, 1:]
    sc.tl.draw_graph(a)

    pca = pd.DataFrame(a.varm["PCs"], index=a.var.index)
    n = 6
    voi = (
        a.var.sort_values("mean").tail(n).index.tolist()
        + a.var.sort_values("std").tail(n).index.tolist()
        + pca.sort_values(0).tail(n).index.tolist()
        + pca.sort_values(1).tail(n).index.tolist()
    )
    vmin = [np.percentile(a[:, v].X, 2) for v in voi] + [None] * 4
    vmax = [np.percentile(a[:, v].X, 95) for v in voi] + [None] * 4
    voi += ["Organ", "Tissue", "Sex", "Age Decade", "Hardy Scale"]
    for emb in a.obsm.keys():
        _a = a[a.obs.sample(frac=1).index].copy()
        fig = sc.pl.embedding(
            _a,
            basis=emb,
            color=voi,
            ncols=n,
            return_fig=True,
            vmin=vmin,
            vmax=vmax,
            use_raw=False,
        )
        for ax in fig.axes:
            ax.yaxis.set_inverted(True)
        rasterize_scanpy(fig)
        fig.savefig(
            results_dir / f"dimres.all_terms.{emb}.svg", bbox_inches="tight", dpi=300
        )

    p = a.raw.to_adata().to_df().groupby(a.obs["Organ"]).mean()
    p = p.iloc[:, :74]
    p = p.loc[:, (p > 0.1).any(axis=0)]
    sel = []
    # sel = np.unique(p.mean(0).sort_values().tail(p.shape[0] // 2).index).tolist()
    sel += np.unique(p.std(0).sort_values().tail(p.shape[0] // 4).index).tolist()
    # sel = np.unique(p.idxmax(1).values.tolist()).tolist()
    sel += np.unique(p.drop(sel, axis=1).idxmax(1).values.tolist()).tolist()
    # sel += np.unique(p.drop(sel, axis=1).idxmax(1).values.tolist()).tolist()
    # sel += np.unique(p.drop(sel, axis=1).idxmax(1).values.tolist()).tolist()
    sel = np.unique(sel).tolist()
    g = clustermap(
        p.loc[:, sel],
        metric="correlation",
        dendrogram_ratio=0.035,
        figsize=(9, 7),
        square=False,
        vmin=0,
        vmax=3,
    )
    g.savefig(results_dir / "top_terms.per_organ.abs.svg", bbox_inches="tight", dpi=300)
    g = clustermap(
        (((p.loc[:, sel]).T - p.loc[:, sel].mean(1)) / p.loc[:, sel].std(1)).T,
        # p.loc[:, sel],
        metric="correlation",
        cmap="Blues",
        dendrogram_ratio=0.035,
        figsize=(6.5, 6),
        square=False,
        vmin=0,
        vmax=3,
    )
    g.savefig(
        results_dir / "top_terms.per_organ.z_abs.svg", bbox_inches="tight", dpi=300
    )
    g = clustermap(
        p.loc[:, sel],
        metric="correlation",
        dendrogram_ratio=0.035,
        figsize=(9, 7),
        square=False,
        config="z",
    )
    g.savefig(results_dir / "top_terms.per_organ.z.svg", bbox_inches="tight", dpi=300)

    # Add exact age
    d, _ = get_restricted_info()
    a.obs["Age"] = a.obs.merge(
        d[["Age"]], how="left", left_on="Subject ID", right_index=True
    )["Age"]
    exclude_entities = [
        "Bladder",
        "Cervix",
        "Fallopian Tube",
        "Kidney",
    ]

    # Regress on age
    _coefs = list()
    for organ in tqdm(a.obs["Organ"].unique()):
        aa = a[a.obs["Organ"] == organ].raw.to_adata().copy()
        if (organ in exclude_entities) or (aa.shape[0] < 100):
            continue
        _x = aa.to_df()
        _x = _x.loc[:, (_x.mean(0) > 1e-10) & (_x.max(0) > 1e-8)]
        _x = (_x - _x.mean()) / _x.std()
        _y = aa.obs["Age"]
        fit = sklearn.linear_model.GammaRegressor(alpha=5).fit(_x, _y)
        coef = pd.Series(fit.coef_, index=_x.columns)
        p = sklearn.feature_selection.f_regression(_x.assign(intercept=1), _y)[1][:-1]
        r = coef.to_frame("coef").assign(p=p, Organ=organ, score=fit.score(_x, _y))
        _coefs.append(r)
    res = pd.concat(_coefs)
    res.to_csv(results_dir / "changes_with_age.pathology_terms.csv")
    res = pd.read_csv(results_dir / "changes_with_age.pathology_terms.csv", index_col=0)

    # # all together
    _x = a.to_df()
    _x = (_x - _x.mean()) / _x.std()
    _x = _x.join(a.obs["Organ"].cat.codes.rename("Organ"))
    _y = a.obs["Age"]
    fit = sklearn.linear_model.GammaRegressor(alpha=1).fit(_x, _y)
    coef = pd.Series(fit.coef_, index=_x.columns)
    p = sklearn.feature_selection.f_regression(_x.assign(intercept=1), _y)[1][:-1]
    r = coef.to_frame("coef").assign(p=p, Organ="All", score=fit.score(_x, _y))
    r["padj"] = sm.stats.multipletests(r["p"], method="bonferroni")[1]

    fig, ax = plt.subplots(figsize=(2.5, 2.5))
    ax.scatter(r["coef"] * 10, -np.log10(r["padj"]), s=1, alpha=0.75)
    ax.axvline(0, linestyle="--", color="grey")
    for t in r.query("coef > 0").sort_values("p").head(10).index:
        ax.text(r.loc[t, "coef"] * 10, -np.log10(r.loc[t, "padj"]), t, ha="left")
    for t in r.query("coef < 0").sort_values("p").head(10).index:
        ax.text(r.loc[t, "coef"] * 10, -np.log10(r.loc[t, "padj"]), t, ha="right")
    ax.set(xlabel="beta", ylabel="-log10(p)")
    fig.savefig(results_dir / "changes_with_age.pathology_terms.all.svg", **figkws)

    c = res.reset_index().pivot(columns="Organ", index="index", values="coef") * 10
    p = res.reset_index().pivot(columns="Organ", index="index", values="p")
    for col in p.columns:
        p[col] = sm.stats.multipletests(p[col], method="fdr_bh")[1]
    # p = -np.log10(p)
    c = c.loc[text_classes[74:]]
    p = p.loc[text_classes[74:]]

    v = c.abs().max().max()
    v -= v * 0.05
    c2 = c.loc[(p < 1e-30).any(axis=1) & ((p < 1e-10).sum(axis=1) >= 2)]
    c2 = c2.loc[((p < 1e-10).sum(axis=1) >= 3)]
    g = clustermap(
        c2,
        cmap="coolwarm",
        vmin=-v,
        vmax=v,
        pvalues=p.reindex(c2.index),
        first_pvalue_threshold=1e-5,
        second_pvalue_threshold=1e-10,
        dendrogram_ratio=0.05,
        metric="cosine",
        figsize=(6, 4),
        col_colors=c2.mean(axis=0)
        .to_frame("Mean")
        .join((p < 1e-10).sum(0).rename("Number of significant terms")),
        row_colors=(p < 1e-10)
        .sum(1)
        .to_frame("Number of significant organs")
        .join(c.mean(axis=1).rename("Mean")),
    )
    g.savefig(results_dir / "changes_with_age.pathology_terms.svg", **figkws)

    fig, axes = plt.subplots(
        5, 6, figsize=(6 * 1.5, 5 * 1.5), sharex=True, sharey=False
    )
    for organ, ax in zip(c.columns, axes.flatten()):
        ax.axvline(0, c="k", linestyle="--", zorder=-1)
        ax.scatter(
            c[organ],
            -np.log10(p[organ]),
            alpha=0.75,
            s=2,
            c=c[organ],
            cmap="coolwarm",
            vmin=-0.08,
            vmax=0.08,
        )
        sel = p.sort_values(organ).head(6).index
        for x, y, v in zip(c[organ], -np.log10(p[organ]), c.index):
            if v in sel:
                ax.text(x, y, v, fontsize="xx-small")
        ax.set(title=organ)
        v = (-np.log10(p[organ])).max()
        v += v * 0.05
        ax.set_ylim(top=max(5, v))
    for ax in axes.flatten():
        if ax.get_title() == "":
            ax.axis("off")
    fig.savefig(
        results_dir / "changes_with_age.pathology_terms.volcano_plot.svg",
    )


def replot_umaps_text_terms():
    import scanpy as sc
    from src.utils import rasterize_scanpy

    input_morpho = Path("results") / "gtex" / "fine_tuned"
    results_dir = Path("results") / "plip_inference"
    output_dir = results_dir / "text_terms_in_umap_space"
    output_dir.mkdir(exist_ok=True)
    meta = pd.read_csv("data/gtex/GTEx Portal.csv", index_col=0)

    x = pd.read_csv(results_dir / "probs.csv.gz", index_col=0)
    a = sc.AnnData(x, obs=meta.reindex(x.index))
    ha = sc.read_h5ad(input_morpho / "anndata.h5ad")
    umap = pd.DataFrame(ha.obsm["X_umap"], index=ha.obs.index).reindex(x.index).dropna()
    a = a[a.obs.index.isin(umap.index)]
    a.obsm["X_umap"] = umap.reindex(a.obs.index).values

    m = a.to_df().groupby(a.obs["Tissue"]).mean()
    # sel = m.idxmax(1).unique().tolist()
    # sel += m.drop(sel, axis=1).idxmax(1).unique().tolist()
    # sel += m.drop(sel, axis=1).idxmax(1).unique().tolist()
    sel = m.columns.tolist()
    sel = [
        "Muscle tissue",
        "Endothelium",
        "Adipose tissue",
        "Mucous membrane",
        "Connective tissue",
        "Neurons",
    ]
    vmin = [0 for v in sel]
    vmax = [np.percentile(a[:, v].X, 98) for v in sel]
    for t, vmi, vma in zip(sel, vmin, vmax):
        fig = sc.pl.umap(
            a,
            color=t,
            show=False,
            return_fig=True,
            vmin=vmi,
            vmax=vma,
            cmap="Blues",
            add_outline=True,
            outline_width=(0.1, 0.01),
        )
        rasterize_scanpy(fig)
        fig.savefig(
            output_dir / f"top_terms.umap_all_tissues.{t}.svg",
            bbox_inches="tight",
            dpi=300,
        )


def get_harmony_loadings(a):
    c = np.empty_like(a.varm["PCs"])
    pc = pd.DataFrame(a.obsm["X_pca_harmony"], index=a.obs.index)
    pc = (pc - pc.mean()) / pc.std()
    _x = a.to_df()
    for p in range(c.shape[1]):
        # c[:, p] = pc[p] @ _x
        c[:, p] = _x.corrwith(pc[p])
    c = pd.DataFrame(c, index=a.var.index)
    a.varm["X_pca_harmony"] = c
    return a


def gene_expression():
    from pathlib import Path

    import pandas as pd
    from tqdm import tqdm
    from anndata import AnnData
    import scanpy as sc
    import statsmodels.api as sm
    import sklearn.feature_selection
    from src.utils import rasterize_scanpy
    from seaborn_extensions import clustermap
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pingouin as pg

    metadata_dir = Path("metadata")
    data_dir = Path("data")
    expr_dir = data_dir / "gtex" / "gene_expression"
    results_dir = Path("results") / "gtex" / "gene_expression" / "signature_level"
    results_dir.mkdir(exist_ok=True, parents=True)

    #
    norm = pd.read_parquet(expr_dir / "log_cpm.pq")
    obs = pd.read_parquet(expr_dir / "log_cpm.obs.pq").rename(
        columns={"Tissue Simple": "Organ"}
    )
    # fixed = pd.read_parquet(expr_dir / "log_cpm.age_regressed.clipped.pq")
    # obs = pd.read_parquet(expr_dir / "log_cpm.age_regressed.clipped.obs.pq")

    # meta = pd.read_csv(data_dir / "gtex" / "GTEx Portal.csv", index_col=0)

    a = AnnData(norm, obs=obs)

    sig_file = metadata_dir / "gene_50signatures_merge.gmt"
    sig_genes = {
        g.split("\t")[0]: g.split("\t")[2:]
        for g in sig_file.open().read().strip().split("\n")
    }
    for sig, genes in tqdm(sig_genes.items()):
        print(sig, len(genes))
        if sig in a.obs.columns:
            continue
        sel = [g for g in genes if g in norm.columns]
        sc.tl.score_genes(a, sel, score_name=sig)
    a.obs[sig_genes.keys()].to_parquet(expr_dir / "log_cpm.gene_50signatures_merge.pq")

    sig_file = metadata_dir / "h.all.v7.5.1.symbols.gmt"
    sig_genes = {
        g.split("\t")[0]: g.split("\t")[2:]
        for g in sig_file.open().read().strip().split("\n")
    }
    for sig, genes in tqdm(sig_genes.items()):
        print(sig, len(genes))
        if sig in a.obs.columns:
            continue
        sel = [g for g in genes if g in norm.columns]
        sc.tl.score_genes(a, sel, score_name=sig)
    a.obs[sig_genes.keys()].to_parquet(expr_dir / "log_cpm.h.all.v7.5.1.symbols.pq")

    x = pd.concat(
        [
            pd.read_parquet(expr_dir / "log_cpm.gene_50signatures_merge.pq"),
            pd.read_parquet(expr_dir / "log_cpm.h.all.v7.5.1.symbols.pq"),
        ],
        axis=1,
    )
    voi = [
        "Tissue",
        "Organ",
        "Subject ID",
        "Age",
        "Sex",
        "Cohort",
        "Ischemic Time (Minutes)",
    ]
    a = AnnData(x, obs=obs[voi])
    a.raw = a
    a.write_h5ad(results_dir / "signature_space.h5ad")

    a = sc.read_h5ad(results_dir / "signature_space.h5ad")
    sc.pp.scale(a)
    sc.pp.pca(a)
    sc.pp.neighbors(a)
    sc.tl.umap(a)
    sc.tl.diffmap(a)
    a.obsm["X_diffmap"] = a.obsm["X_diffmap"][:, 1:]
    sc.tl.draw_graph(a)
    a.write_h5ad(results_dir / "signature_space.h5ad")
    a = sc.read_h5ad(results_dir / "signature_space.h5ad")

    pca = pd.DataFrame(a.varm["PCs"], index=a.var.index)
    n = 6
    voi2 = (
        a.var.sort_values("mean").tail(n).index.tolist()
        + a.var.sort_values("std").tail(n).index.tolist()
        + pca.sort_values(0).tail(n).index.tolist()
        + pca.sort_values(1).tail(n).index.tolist()
    )
    vmin = [np.percentile(a[:, v].X, 2) for v in voi2] + [None] * 4
    vmax = [np.percentile(a[:, v].X, 95) for v in voi2] + [None] * 4
    voi2 += voi
    for emb in a.obsm.keys():
        _a = a[a.obs.sample(frac=1).index].copy()
        fig = sc.pl.embedding(
            _a,
            basis=emb,
            color=voi2,
            ncols=n,
            return_fig=True,
            vmin=vmin,
            vmax=vmax,
            use_raw=False,
        )
        for ax in fig.axes:
            ax.yaxis.set_inverted(True)
        rasterize_scanpy(fig)
        fig.savefig(
            results_dir / f"dimres.all_terms.{emb}.svg", bbox_inches="tight", dpi=300
        )

    p = a.raw.to_adata().to_df().groupby(a.obs["Organ"]).mean()
    p = p.loc[:, ~p.columns.str.startswith("HALLMARK_")].drop(["Cells", "Whole"])
    # p.columns = p.columns.str.replace("HALLMARK_", "").str.replace("_", " ").str.title()
    p = p.loc[:, (p > 0.1).any(axis=0)]
    sel = []
    # sel = np.unique(p.mean(0).sort_values().tail(p.shape[0] // 2).index).tolist()
    sel += np.unique(p.std(0).sort_values().tail(p.shape[0] // 4).index).tolist()
    # sel = np.unique(p.idxmax(1).values.tolist()).tolist()
    sel += np.unique(p.drop(sel, axis=1).idxmax(1).values.tolist()).tolist()
    sel += np.unique(p.drop(sel, axis=1).idxmax(1).values.tolist()).tolist()
    sel += np.unique(p.drop(sel, axis=1).idxmax(1).values.tolist()).tolist()
    sel += np.unique(p.drop(sel, axis=1).idxmax(1).values.tolist()).tolist()
    sel += np.unique(p.drop(sel, axis=1).idxmax(1).values.tolist()).tolist()
    sel += np.unique(p.drop(sel, axis=1).idxmax(1).values.tolist()).tolist()
    sel = np.unique(sel).tolist()
    g = clustermap(
        p.loc[:, sel],
        metric="correlation",
        dendrogram_ratio=0.035,
        figsize=(9, 7),
        square=False,
        vmin=0,
        vmax=3,
    )
    g.savefig(
        results_dir / "top_terms.per_organ.sigs.abs.svg", bbox_inches="tight", dpi=300
    )
    g = clustermap(
        (((p.loc[:, sel]).T - p.loc[:, sel].mean(1)) / p.loc[:, sel].std(1)).T,
        # p.loc[:, sel],
        metric="correlation",
        cmap="Blues",
        dendrogram_ratio=0.035,
        figsize=(6.5, 6),
        square=False,
        vmin=0,
        vmax=3,
    )
    g.savefig(
        results_dir / "top_terms.per_organ.sigs.z_abs.svg", bbox_inches="tight", dpi=300
    )
    g = clustermap(
        p.loc[:, sel],
        metric="correlation",
        dendrogram_ratio=0.035,
        figsize=(5, 6),
        square=False,
        config="z",
        cmap="PuOr_r",
    )
    g.savefig(
        results_dir / "top_terms.per_organ.sigs.z.svg", bbox_inches="tight", dpi=300
    )

    exclude_entities = [
        "Bladder",
        "Cervix",
        "Fallopian Tube",
        "Kidney",
        "Whole",
        "Cells",
    ]

    # Regress on age
    _coefs = list()
    for organ in tqdm(a.obs["Organ"].unique()):
        aa = a.raw.to_adata()[a.obs["Organ"] == organ].copy()
        if (organ in exclude_entities) or (aa.shape[0] < 100):
            continue
        _x = aa.to_df()
        _x = (_x - _x.mean()) / _x.std()
        _y = aa.obs["Age"]
        fit = sklearn.linear_model.GammaRegressor(alpha=5).fit(_x, _y)
        coef = pd.Series(fit.coef_, index=_x.columns)
        p = sklearn.feature_selection.f_regression(_x.assign(intercept=1), _y)[1][:-1]
        r = coef.to_frame("coef").assign(p=p, Organ=organ, score=fit.score(_x, _y))
        r["padj"] = pg.multicomp(r["p"], method="fdr_bh")[1]
        r["coef"] *= 10
        _coefs.append(r)
    res = pd.concat(_coefs)
    res.to_csv(results_dir / "changes_with_age.signatures.csv")
    res = pd.read_csv(results_dir / "changes_with_age.signatures.csv", index_col=0)

    # # all together
    _x = a.to_df()
    _x = (_x - _x.mean()) / _x.std()
    _x = _x.join(a.obs["Organ"].cat.codes.rename("Organ"))
    _y = a.obs["Age"]
    fit = sklearn.linear_model.GammaRegressor(alpha=1).fit(_x, _y)
    coef = pd.Series(fit.coef_, index=_x.columns)
    p = sklearn.feature_selection.f_regression(_x.assign(intercept=1), _y)[1][:-1]
    r = coef.to_frame("coef").assign(p=p, Organ="All", score=fit.score(_x, _y))
    r["padj"] = pg.multicomp(r["p"], method="fdr_bh")[1]

    fig, ax = plt.subplots(figsize=(2.5, 2.5))
    ax.scatter(r["coef"] * 10, -np.log10(r["padj"]), s=1, alpha=0.75)
    ax.axvline(0, linestyle="--", color="grey")
    for t in r.query("coef > 0").sort_values("p").head(10).index:
        ax.text(r.loc[t, "coef"] * 10, -np.log10(r.loc[t, "padj"]), t, ha="left")
    for t in r.query("coef < 0").sort_values("p").head(10).index:
        ax.text(r.loc[t, "coef"] * 10, -np.log10(r.loc[t, "padj"]), t, ha="right")
    ax.set(xlabel="beta", ylabel="-log10(p)")
    fig.savefig(results_dir / "changes_with_age.signatures.all.svg", **figkws)

    #
    c = res.reset_index().pivot(columns="Organ", index="index", values="coef").dropna()
    p = res.reset_index().pivot(columns="Organ", index="index", values="p").dropna()

    v = c.abs().max().max()
    v -= v * 0.05
    c2 = c.loc[c.index.str.startswith("HALLMARK")]
    c2.index = (
        c2.index.str.replace("HALLMARK_", "").str.replace("_", " ").str.capitalize()
    )
    p2 = p.copy()
    p2.index = (
        p2.index.str.replace("HALLMARK_", "").str.replace("_", " ").str.capitalize()
    )
    g = clustermap(
        c2,
        cmap="coolwarm",
        vmin=-v,
        vmax=v,
        pvalues=p2.reindex(c2.index),
        first_pvalue_threshold=1e-5,
        second_pvalue_threshold=1e-10,
        dendrogram_ratio=0.05,
        metric="cosine",
        figsize=(6, 10),
        col_colors=c2.mean(axis=0)
        .to_frame("Mean")
        .join((p < 1e-10).sum(0).rename("Number of significant terms")),
        row_colors=(p2 < 1e-10)
        .sum(1)
        .to_frame("Number of significant organs")
        .join(c.mean(axis=1).rename("Mean")),
    )
    g.savefig(results_dir / "changes_with_age.signatures.all.svg", **figkws)

    c2 = c2.loc[(p.reindex(c2.index) < 1e-5).sum(axis=1).sort_values().tail(12).index]
    p2 = p2.reindex(c2.index)
    # c2 = c.loc[(p < 1e-10).any(axis=1) & ((p < 1e-10).sum(axis=1) >= 2)]
    # c2 = c2.loc[((p < 1e-8).sum(axis=1) >= 3)]
    g = clustermap(
        c2,
        cmap="coolwarm",
        vmin=-v,
        vmax=v,
        pvalues=p2.reindex(c2.index),
        first_pvalue_threshold=1e-5,
        second_pvalue_threshold=1e-10,
        dendrogram_ratio=0.05,
        metric="cosine",
        figsize=(6, 4),
        col_colors=c2.mean(axis=0)
        .to_frame("Mean")
        .join((p < 1e-10).sum(0).rename("Number of significant terms")),
        row_colors=(p2 < 1e-10)
        .sum(1)
        .to_frame("Number of significant organs")
        .join(c.mean(axis=1).rename("Mean")),
    )
    g.savefig(results_dir / "changes_with_age.signatures.selected.svg", **figkws)

    fig, axes = plt.subplots(
        5, 6, figsize=(6 * 1.5, 5 * 1.5), sharex=True, sharey=False
    )
    for organ, ax in zip(c.columns, axes.flatten()):
        ax.axvline(0, c="k", linestyle="--", zorder=-1)
        ax.scatter(
            c[organ],
            -np.log10(p[organ]),
            alpha=0.75,
            s=2,
            c=c[organ],
            cmap="coolwarm",
            vmin=-0.08,
            vmax=0.08,
        )
        sel = p.sort_values(organ).head(6).index
        for x, y, v in zip(c[organ], -np.log10(p[organ]), c.index):
            if v in sel:
                ax.text(x, y, v, fontsize="xx-small")
        ax.set(title=organ)
        v = (-np.log10(p[organ])).max()
        v += v * 0.05
        ax.set_ylim(top=max(5, v))
    for ax in axes.flatten():
        if ax.get_title() == "":
            ax.axis("off")
    fig.savefig(results_dir / "changes_with_age.signatures.volcano_plot.svg", **figkws)

    # Get age gaps
    gaps_dir = Path("results") / "gtex" / "fine_tuned" / "_pre_2024-01-19_age_X_frac1.0"
    exclude_entities = [
        "Bladder",
        "Cervix",
        "Fallopian Tube",
        "Kidney",
    ]
    exclude_entities += ["mean", "max", "sum", "std"]

    # target_var = "residuals_adj"
    target_var = "prediction_adj"
    preds = pd.read_parquet(
        gaps_dir / "tissue-specific_clocks.Ridge.GroupKFold.predictions_residuals.pq"
    )
    preds = preds.query("Tissue not in @exclude_entities")
    preds.index = preds.index.to_series().str.extract(r"(GTEX-\w+)-\d{4}", expand=False)
    ind = preds.drop("Tissue", axis=1).groupby(level=0).mean().assign(Tissue="mean")
    preds = pd.concat([preds, ind])
    preds["Organ"] = preds["Tissue"].apply(lambda x: x.split(" - ")[0])

    _coefs = list()
    for organ in tqdm(preds["Organ"].unique()):
        if organ in exclude_entities:
            continue
        _y = preds.query("Organ == @organ")[target_var]
        _y = _y.groupby(level=0).mean()
        if target_var == "residuals_adj":
            _y = _y.loc[_y.abs() < 50]
            _y += 100

        _y = _y.loc[_y > 0]

        aa = a.raw.to_adata()[a.obs["Organ"] == organ].copy()
        _x = aa.to_df()
        _x.index = _x.index.to_series().str.extract(r"(GTEX-\w+)-\d{4}", expand=False)
        _x = _x.groupby(level=0).mean().reindex(_y.index).dropna()
        _y = _y.reindex(_x.index)

        if _x.empty or (_x.shape[0] < 100):
            continue

        # _x = _x.loc[:, (_x.mean(0) > 1e-10) & (_x.max(0) > 1e-8)]
        _x = (_x - _x.mean()) / _x.std()
        fit = sklearn.linear_model.GammaRegressor(alpha=5).fit(_x, _y)
        coef = pd.Series(fit.coef_, index=_x.columns)
        p = sklearn.feature_selection.f_regression(_x.assign(intercept=1), _y)[1][:-1]
        r = coef.to_frame("coef").assign(p=p, Organ=organ, score=fit.score(_x, _y))
        r["coef"] *= 10
        r["padj"] = pg.multicomp(r["p"], method="fdr_bh")[1]
        _coefs.append(r)
    res2 = pd.concat(_coefs)
    res2.to_csv(results_dir / "changes_with_age-gap.signatures.csv")
    res2 = pd.read_csv(results_dir / "changes_with_age-gap.signatures.csv", index_col=0)

    c = res2.reset_index().pivot(columns="Organ", index="index", values="coef").dropna()
    p = res2.reset_index().pivot(columns="Organ", index="index", values="p").dropna()

    v = c.abs().max().max()
    v -= v * 0.05
    c2 = c.loc[c.index.str.startswith("HALLMARK")]
    c2 = c2.loc[(p.reindex(c2.index) < 1e-5).sum(axis=1).sort_values().tail(12).index]
    c2.index = (
        c2.index.str.replace("HALLMARK_", "").str.replace("_", " ").str.capitalize()
    )
    p2 = p.copy()
    p2 = p2.loc[p2.index.str.startswith("HALLMARK")]
    p2.index = (
        p2.index.str.replace("HALLMARK_", "").str.replace("_", " ").str.capitalize()
    )
    p2 = p2.reindex(c2.index)
    # c2 = c.loc[(p < 1e-10).any(axis=1) & ((p < 1e-10).sum(axis=1) >= 2)]
    # c2 = c2.loc[((p < 1e-8).sum(axis=1) >= 3)]
    g = clustermap(
        c2,
        cmap="coolwarm",
        vmin=-v,
        vmax=v,
        pvalues=p2.reindex(c2.index),
        first_pvalue_threshold=1e-5,
        second_pvalue_threshold=1e-10,
        dendrogram_ratio=0.05,
        metric="cosine",
        figsize=(6, 4),
        col_colors=c2.mean(axis=0)
        .to_frame("Mean")
        .join((p < 1e-10).sum(0).rename("Number of significant terms")),
        row_colors=(p2 < 1e-10)
        .sum(1)
        .to_frame("Number of significant organs")
        .join(c.mean(axis=1).rename("Mean")),
    )
    g.savefig(results_dir / "changes_with_age-gap.signatures.svg", **figkws)

    fig, axes = plt.subplots(
        5, 6, figsize=(6 * 1.5, 5 * 1.5), sharex=True, sharey=False
    )
    for organ, ax in zip(c.columns, axes.flatten()):
        ax.axvline(0, c="k", linestyle="--", zorder=-1)
        ax.scatter(
            c[organ],
            -np.log10(p[organ]),
            alpha=0.75,
            s=2,
            c=c[organ],
            cmap="coolwarm",
            vmin=-0.08,
            vmax=0.08,
        )
        sel = p.sort_values(organ).head(6).index
        for x, y, v in zip(c[organ], -np.log10(p[organ]), c.index):
            if v in sel:
                ax.text(x, y, v, fontsize="xx-small")
        ax.set(title=organ)
        v = (-np.log10(p[organ])).max()
        v += v * 0.05
        ax.set_ylim(top=max(5, v))
    for ax in axes.flatten():
        if ax.get_title() == "":
            ax.axis("off")
    fig.savefig(
        results_dir / "changes_with_age-gap.signatures.volcano_plot.svg",
    )

    res = pd.read_csv(results_dir / "changes_with_age.signatures.csv", index_col=0)
    res2 = pd.read_csv(results_dir / "changes_with_age-gap.signatures.csv", index_col=0)

    c1 = res.copy()  # .loc[res.index.str.startswith("HALLMARK")]
    c2 = res2.copy()  # .loc[res2.index.str.startswith("HALLMARK")]

    _x = c1.drop("Organ", axis=1).groupby(level=0).mean()
    _y = c2.drop("Organ", axis=1).groupby(level=0).mean()

    # # waterfall
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(6, 5),
        gridspec_kw={"height_ratios": [1, 4], "hspace": 0},
        sharex=True,
    )
    _x2 = _x.copy()["coef"]
    _y2 = _y.copy()["coef"]
    mean = pd.concat([_x2, _y2], axis=1).mean(axis=1).sort_values()
    _x2 = _x2.reindex(mean.index)
    _y2 = _y2.reindex(mean.index)
    sign = (mean > 0).astype(int).replace(0, -1)
    d = pd.concat(
        [_y2.sub(_x2).loc[sign == -1].abs(), _y2.sub(_x2).loc[sign == 1]]
    ).reindex(mean.index)
    v = d.abs().max()
    v += v * 0.05
    ax = axes[0]
    ax.axhline(0, c="k", linestyle="--", zorder=-1, linewidth=0.35)
    ax.plot(
        mean,
        d.rolling(window=10, center=True).mean(),
        color="grey",
        linewidth=1.5,
        linestyle="--",
    )
    ax.scatter(mean, d, color="white", edgecolor="grey", s=7.5, linewidth=0.35)
    ax.set(ylabel="Biological / Chronological\n(signed difference)", ylim=(-v, v))

    ax = axes[1]
    ax.plot(
        _x2.rolling(window=10, center=True).mean(),
        np.arange(_x2.shape[0]),
        color=sns.color_palette()[0],
        linewidth=1,
    )
    ax.plot(
        _y2.rolling(window=10, center=True).mean(),
        np.arange(_x2.shape[0]),
        color=sns.color_palette()[1],
        linewidth=1,
    )
    ax.axvline(0, c="k", linestyle="--", zorder=-1, linewidth=0.1)
    for i, t in enumerate(_x2.index):
        ax.plot((_y2[t], _x2[t]), (i, i), color="grey", linewidth=0.35)
        # ax.axhline(i, _x2[t], _y2[t], color="grey", linewidth=0.35)
    ax.scatter(
        _x2,
        np.arange(_x2.shape[0]),
        alpha=0.75,
        s=25,
        linewidth=1,
        edgecolor="grey",
        label="Chronological age",
    )
    ax.scatter(
        _y2,
        np.arange(_x2.shape[0]),
        alpha=0.75,
        s=25,
        linewidth=1,
        edgecolor="grey",
        label="Biological age",
    )

    ax.legend()
    ax.set(xlabel=r"Change with age ($\beta$/decade)", ylabel="Signatures (ranked)")
    ax.set_yticks(np.arange(0, _x2.shape[0]))
    ax.set_yticklabels(
        _x2.index.str.replace("HALLMARK_", "").str.replace("_", " ").str.capitalize(),
        rotation=0,
    )
    fig.savefig(results_dir / "changes_with_age-gap.waterfall.svg", **figkws)

    fig, ax = plt.subplots(figsize=(3, 3))
    sns.kdeplot(d, ax=ax)
    sns.rugplot(d, ax=ax)
    ax.axvline(0, c="k", linestyle="--", zorder=-1, linewidth=0.1)
    ax.set(xlim=(-v, v), title=f"Wilcoxon p = {pg.wilcoxon(d)['p-val'].squeeze():.2e}")
    fig.savefig(
        results_dir / "changes_with_age-gap.waterfall.differences_pvalue.svg", **figkws
    )

    fig, ax = plt.subplots(figsize=(3, 3))
    slopes = pd.Series(slopes)
    v = slopes.abs().max()
    v += v * 0.05
    v += 1
    sns.kdeplot(slopes, ax=ax)
    sns.rugplot(slopes, ax=ax)
    ax.axvline(1, c="k", linestyle="--", zorder=-1, linewidth=0.1)
    ax.set(
        xlim=(-v, v),
        title=f"Wilcoxon p = {pg.wilcoxon(np.log(slopes))['p-val'].squeeze():.2e}",
    )
    fig.savefig(
        results_dir / "changes_with_age-gap.waterfall.slopes_pvalue.svg", **figkws
    )

    mean.to_frame("mean").join(
        c1.groupby(level=0)["coef"].mean().rename("chronological age"),
    ).join(
        c2.groupby(level=0)["coef"].mean().rename("biological age"),
    ).join(
        d.rename("difference"),
    ).to_csv(
        results_dir / "changes_with_age-gap.waterfall.differences.csv"
    )

    # # scatter
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.axvline(0, c="k", linestyle="--", zorder=-1, linewidth=0.1)
    ax.axhline(0, c="k", linestyle="--", zorder=-1, linewidth=0.1)
    sns.regplot(
        x=_x["coef"],
        y=_y["coef"],
        ax=ax,
        line_kws={"linewidth": 0.5, "color": "k"},
        scatter=False,
    )
    ax.scatter(_x["coef"], _y["coef"], alpha=0.75, s=2)
    ax.plot(
        [_x["coef"].min(), _x["coef"].max()],
        [_x["coef"].min(), _x["coef"].max()],
        linestyle="--",
        linewidth=0.5,
        c="k",
    )
    s1 = _x["score"].iloc[0]
    s2 = _y["score"].iloc[0]
    ax.set(
        title=f"All, S= {s1:.2f}-{s2:.2f}, r = {_x['coef'].corr(_y['coef']):.2f}",
        xlabel="Chronological age",
        ylabel="Biological age\n(derived from histology)",
    )
    mean = pd.concat([_x, _y])["coef"].groupby(level=0).mean()
    d = (_y - _x)["coef"].abs().sort_values()
    tp = (
        mean.sort_values().head(3).index.tolist()
        + mean.sort_values().tail(3).index.tolist()
        + _x["coef"].sort_values().head(3).index.tolist()
        + _x["coef"].sort_values().tail(3).index.tolist()
        + _y["coef"].sort_values().head(3).index.tolist()
        + _y["coef"].sort_values().tail(3).index.tolist()
        + d.tail(18).index.tolist()
    )
    tp = list(set(tp))
    for t in tp:
        ax.text(
            _x.loc[t, "coef"],
            _y.loc[t, "coef"],
            t.replace("HALLMARK_", "").replace("_", " ").capitalize(),
            fontsize="xx-small",
        )
    for t in _x.query("p < 1e-3").index:
        ax.text(
            _x.loc[t, "coef"],
            _y.loc[t, "coef"],
            "*",
            fontsize="xx-small",
            ha="right",
            c="orange",
        )
        if t not in tp:
            ax.text(
                _x.loc[t, "coef"],
                _y.loc[t, "coef"],
                t.replace("HALLMARK_", "").replace("_", " ").capitalize(),
                fontsize="xx-small",
            )
    for t in _y.query("p < 1e-3").index:
        ax.text(
            _x.loc[t, "coef"],
            _y.loc[t, "coef"],
            "*",
            fontsize="xx-small",
            ha="right",
            c="r",
        )
        if t not in tp:
            ax.text(
                _x.loc[t, "coef"],
                _y.loc[t, "coef"],
                t.replace("HALLMARK_", "").replace("_", " ").capitalize(),
                fontsize="xx-small",
            )
    fig.savefig(
        results_dir
        / "comparison_of_changes_with_age_and_age-gap.signatures.scatter.mean_across_organs.svg",
        **figkws,
    )

    fig, axes = plt.subplots(
        5, 6, figsize=(6 * 3, 5 * 3), gridspec_kw={"wspace": 0.2, "hspace": 0.2}
    )
    slopes = dict()
    r2s = dict()
    for organ, ax in zip(sorted(c1["Organ"].unique()), axes.flatten()):
        _x = c1.query("Organ == @organ")
        _y = c2.query("Organ == @organ").reindex(_x.index)
        # get slope
        slope = sm.OLS(_y["coef"], _x["coef"]).fit().params["coef"]
        slopes[organ] = slope
        r2 = _x["coef"].corr(_y["coef"]) ** 2
        r2s[organ] = r2
        pmin = pd.concat([_x["p"], _y["p"]]).groupby(level=0).min().reindex(_x.index)
        pmin = -np.log10(pmin)
        if _x["coef"].isnull().all() or _y["coef"].isnull().all():
            continue
        ax.axvline(0, c="k", linestyle="--", zorder=-1, linewidth=0.1)
        ax.axhline(0, c="k", linestyle="--", zorder=-1, linewidth=0.1)
        sns.regplot(
            x=_x["coef"],
            y=_y["coef"],
            scatter=False,
            ax=ax,
            line_kws={"linewidth": 0.5, "color": "k"},
        )
        ax.scatter(
            _x["coef"],
            _y["coef"],
            s=2 + 6 * pmin,
            alpha=0.75,
            color="white",
            edgecolor=sns.color_palette()[0],
            linewidth=0.5,
        )
        ax.plot(
            [_x["coef"].min(), _x["coef"].max()],
            [_x["coef"].min(), _x["coef"].max()],
            linestyle="--",
            linewidth=0.5,
            c="k",
        )
        s1 = _x["score"].iloc[0]
        s2 = _y["score"].iloc[0]
        ax.set(title=f"{organ}, m = {slope:.3f}, R2 = {r2:.2f}")
        mean = pd.concat([_x, _y])["coef"].groupby(level=0).mean()
        tp = (
            mean.sort_values().head(3).index.tolist()
            + mean.sort_values().tail(3).index.tolist()
            # + _x["coef"].sort_values().head(3).index.tolist()
            # + _x["coef"].sort_values().tail(3).index.tolist()
            # + _y["coef"].sort_values().head(3).index.tolist()
            # + _y["coef"].sort_values().tail(3).index.tolist()
        )
        tp = list(set(tp))
        for t in tp:
            ax.text(
                _x.loc[t, "coef"],
                _y.loc[t, "coef"],
                t.replace("HALLMARK_", "").replace("_", " ").capitalize(),
                fontsize="xx-small",
            )
        # for t in _x.query("p < 1e-3").index:
        #     ax.text(
        #         _x.loc[t, "coef"],
        #         _y.loc[t, "coef"],
        #         "*",
        #         fontsize="xx-small",
        #         ha="right",
        #         c="orange",
        #     )
        #     if t not in tp:
        #         ax.text(
        #             _x.loc[t, "coef"],
        #             _y.loc[t, "coef"],
        #             t.replace("HALLMARK_", "").replace("_", " ").capitalize(),
        #             fontsize="xx-small",
        #         )
        # for t in _y.query("p < 1e-3").index:
        #     ax.text(
        #         _x.loc[t, "coef"],
        #         _y.loc[t, "coef"],
        #         "*",
        #         fontsize="xx-small",
        #         ha="right",
        #         c="red",
        #     )
        #     if t not in tp:
        #         ax.text(
        #             _x.loc[t, "coef"],
        #             _y.loc[t, "coef"],
        #             t.replace("HALLMARK_", "").replace("_", " ").capitalize(),
        #             fontsize="xx-small",
        #         )
    fig.savefig(
        results_dir
        / "comparison_of_changes_with_age_and_age-gap.signatures.scatter.svg",
        **figkws,
    )

    fig, axes = plt.subplots(
        5,
        6,
        figsize=(6 * 3, 5 * 3),
        gridspec_kw={"wspace": 0.2, "hspace": 0.2},
        sharex=True,
        sharey=True,
    )
    for organ, ax in zip(sorted(c1["Organ"].unique()), axes.flatten()):
        _x = c1.query("Organ == @organ")
        _y = c2.query("Organ == @organ").reindex(_x.index)
        if _x["coef"].isnull().all() or _y["coef"].isnull().all():
            continue
        ax.axvline(0, c="k", linestyle="--", zorder=-1, linewidth=0.1)
        ax.axhline(0, c="k", linestyle="--", zorder=-1, linewidth=0.1)
        ax.scatter(_x["coef"], _y["coef"], alpha=0.75, s=2)
        ax.plot(
            [_x["coef"].min(), _x["coef"].max()],
            [_x["coef"].min(), _x["coef"].max()],
            linestyle="--",
            linewidth=0.5,
            c="k",
        )
        s1 = _x["score"].iloc[0]
        s2 = _y["score"].iloc[0]
        ax.set(
            title=f"{organ}, S= {s1:.2f}-{s2:.2f}, r = {_x['coef'].corr(_y['coef']):.2f}"
        )
        mean = pd.concat([_x, _y])["coef"].groupby(level=0).mean()
        tp = (
            mean.sort_values().head(3).index.tolist()
            + mean.sort_values().tail(3).index.tolist()
            + _x["coef"].sort_values().head(3).index.tolist()
            + _x["coef"].sort_values().tail(3).index.tolist()
            + _y["coef"].sort_values().head(3).index.tolist()
            + _y["coef"].sort_values().tail(3).index.tolist()
        )
        tp = list(set(tp))
        for t in tp:
            ax.text(
                _x.loc[t, "coef"],
                _y.loc[t, "coef"],
                t.replace("HALLMARK_", "").replace("_", " ").capitalize(),
                fontsize="xx-small",
            )
        for t in _x.query("p < 1e-3").index:
            ax.text(
                _x.loc[t, "coef"],
                _y.loc[t, "coef"],
                "*",
                fontsize="xx-small",
                ha="right",
                c="orange",
            )
            if t not in tp:
                ax.text(
                    _x.loc[t, "coef"],
                    _y.loc[t, "coef"],
                    t.replace("HALLMARK_", "").replace("_", " ").capitalize(),
                    fontsize="xx-small",
                )
        for t in _y.query("p < 1e-3").index:
            ax.text(
                _x.loc[t, "coef"],
                _y.loc[t, "coef"],
                "*",
                fontsize="xx-small",
                ha="right",
                c="r",
            )
            if t not in tp:
                ax.text(
                    _x.loc[t, "coef"],
                    _y.loc[t, "coef"],
                    t.replace("HALLMARK_", "").replace("_", " ").capitalize(),
                    fontsize="xx-small",
                )
    fig.savefig(
        results_dir
        / "comparison_of_changes_with_age_and_age-gap.signatures.scatter.fixed.svg",
        **figkws,
    )
