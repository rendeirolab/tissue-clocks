import typing as tp
from pathlib import Path
import os

from tqdm import tqdm
import numpy as np
import pandas as pd
from shapely import Polygon
from wsi_core import WholeSlideImage
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")
# set font svg
plt.rcParams["svg.fonttype"] = "none"


MicronsPerPixel = tp.NewType("MicronsPerPixel", float)

metadata_dir = Path("metadata")
data_dir = Path("data") / "gtex"
results_dir = Path("results")
input_dir = results_dir / "gtex" / "fine_tuned" / "age_X_frac1.0"
output_dir = (
    results_dir / "gtex" / "fine_tuned" / "age_X_frac1.0" / "age_gap_visualization"
)
output_dir.mkdir(parents=True, exist_ok=True)

figkws = dict(bbox_inches="tight", dpi=300)

meta = pd.read_csv(data_dir / "GTEx Portal.csv", index_col=0)
meta["Tissue Simple"] = meta["Tissue"].str.extract(r"(\w+).*", expand=False)


# Get histology residuals
model_name = "Ridge"
cv_name = "GroupKFold"
s = f"tissue-specific_clocks.{model_name}.{cv_name}."
df = pd.read_parquet(input_dir / (s + "predictions_residuals.pq")).query(
    "shuffled == False"
)
df["Subject ID"] = df.index.str.extract(r"(GTEX-.*)-\d+")[0].values
df = df.join(meta["Age Bracket"])
# res = df.pivot_table(index="Subject ID", columns="Tissue", values="residuals_adj")
# s = f"pan-tissue_clock.{model_name}.{cv_name}."
# dfg = pd.read_parquet(input_dir / (s + "predictions_residuals.pq"))
# dfg["Subject ID"] = dfg.index.str.extract(r"(GTEX-.*)-\d+")[0].values


def get_largest_tissue_piece_image(
    slide_id: str,
) -> tuple[np.ndarray, np.ndarray, MicronsPerPixel]:
    sf = data_dir / "svs" / (slide_id + ".svs")
    slide = WholeSlideImage(sf)
    slide.initSegmentation()
    areas = [Polygon(path.squeeze()).area for path in slide.contours_tissue]
    contour = slide.contours_tissue[np.argmax(areas)].squeeze()
    topleft = contour.min(0)
    bottomright = contour.max(0)
    level = 2
    scale_factor = (
        slide.wsi.level_dimensions[level][0] / slide.wsi.level_dimensions[0][0]
    )
    mpp = MicronsPerPixel(float(slide.wsi.properties["openslide.mpp-x"]) / scale_factor)
    # size = slide.wsi.level_dimensions[level]
    size = np.ceil((bottomright - topleft) * scale_factor).astype(int)
    region = np.asarray(slide.wsi.read_region(topleft, level, size).convert("RGB"))

    xtrim = (region == 0).all(0).any(1)
    xtrim = np.searchsorted(xtrim.cumsum(), 1)
    ytrim = (region == 0).all(1).any(1)
    ytrim = np.searchsorted(ytrim.cumsum(), 1)
    region = region[:ytrim, :xtrim]
    return (
        region,
        (contour - topleft).T * scale_factor,
        mpp,
    )
    # aspect = region.shape[1] / region.shape[0]
    # fig, ax = plt.subplots(figsize=(4 * aspect, 4))
    # ax.imshow(region)
    # ax.plot(*(contour - topleft).T * scale_factor, color="black", linewidth=3)


def get_slides(slides: list[str]):
    for slide in slides:
        if not (data_dir / "svs" / (slide + ".svs")).exists():
            cmd = f"scp arendeiro@login:projects/histopath/data/gtex/svs/{slide}.svs data/gtex/svs/"
            os.system(cmd)
        if not (data_dir / "svs" / (slide + ".segmentation.pickle")).exists():
            cmd = f"scp arendeiro@login:projects/histopath/data/gtex/svs/{slide}.segmentation.pickle data/gtex/svs/"
            try:
                assert os.system(cmd) == 0
            except (FileNotFoundError, AssertionError):
                cmd = f"scp arendeiro@login:projects/histopath/data/gtex/svs/{slide}.contours_tissue.pickle data/gtex/svs/{slide}.segmentation.pickle"
                os.system(cmd)


var = "residuals_adj"
n = 3
tissues: list[str] = [
    "Brain - Cerebellum",
    "Liver",
    "Lung",
    "Pancreas",
    "Skin - Not Sun Exposed (Suprapubic)",
    "Thyroid",
    "Testis",
    "Spleen",
    "Stomach",
    "Colon - Sigmoid",
]
age_brackets: list[str] = []

if not tissues:
    tissues = sorted(df["Tissue"].unique())

if not age_brackets:
    age_brackets = sorted(meta["Age Bracket"].unique())

for labels, extensions in zip([True, False], ["png", "pdf", "svgz"], ["svgz"]):
    for tissue in tissues:
        if (output_dir / f"{tissue}_{var}.{labels=}.svgz").exists():
            continue
        fig, axes = plt.subplots(
            n * 3,
            len(age_brackets),
            figsize=(len(age_brackets) * 4, n * 3 * 4),
            gridspec_kw=dict(hspace=0.01, wspace=0.1),
        )
        for axs, bracket in tqdm(zip(axes.T, age_brackets), total=len(age_brackets)):
            r = df.query("Tissue == @tissue & `Age Bracket` == @bracket")
            top = r[var].nlargest(n)
            middle = r.loc[
                (r[var] - r[var].mean()).abs().nsmallest(n).index
            ].sort_values(var)[var]
            bottom = r[var].nsmallest(n)[::-1]
            slides = top.index.tolist() + middle.index.tolist() + bottom.index.tolist()
            get_slides(slides)
            for ax, slide in zip(axs, slides):
                img, contour, mpp = get_largest_tissue_piece_image(slide)
                if img.shape[0] > img.shape[1]:
                    img = img.transpose(1, 0, 2)
                    contour = contour[[1, 0]]
                ax.imshow(img)
                ax.plot(*contour, color="black", linewidth=3)
                ax.axis("off")
                # add scale bar
                bar_length = 1000 / mpp
                ax.plot([10, 10 + bar_length], [10, 10], color="black", linewidth=3)
                if labels:
                    ax.set_title(
                        slide
                        + f"\nAge:{r.loc[slide, 'Age']:.0f}, "
                        + f"Predicted:{r.loc[slide, 'prediction_adj']:.1f}, "
                        + f"Gap:{r.loc[slide, 'residuals_adj']:.2f}"
                        + "\n"
                        + f"Pathology: {meta.loc[slide, 'Pathology Categories']}"
                    )
        if labels:
            fig.suptitle(f"{tissue} {var}", y=0.92)
        for end in extensions:
            fig.savefig(
                output_dir / f"{tissue}_{var}.{labels=}.{end}",
                bbox_inches="tight",
                dpi=200,
            )


n = 1
for tissue in tissues:
    slides = []
    for bracket in age_brackets:
        r = df.query("Tissue == @tissue & `Age Bracket` == @bracket")
        top = r[var].nlargest(n)
        middle = r.loc[(r[var] - r[var].mean()).abs().nsmallest(n).index].sort_values(
            var
        )[var]
        bottom = r[var].nsmallest(n)[::-1]
        slides += top.index.tolist() + middle.index.tolist() + bottom.index.tolist()

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    r = df.query("Tissue == @tissue")
    v = r[var].abs().max()
    vm = r[var].std()

    # Scatter of all samples
    ax.scatter(
        r["Age"],
        r[var],
        c=r[var],
        alpha=0.5,
        vmin=-vm,
        vmax=vm,
        cmap="coolwarm",
        # zorder=-100,
    )
    ax.axhline(0, linestyle="--", color="grey")
    for slide in tqdm(slides):
        # Add slide name in location
        loc = (r.loc[slide, "Age"], r.loc[slide, var])
        ax.annotate(slide, loc, fontsize=6)

        # Get image
        img, contour, mpp = get_largest_tissue_piece_image(slide)
        if img.shape[0] > img.shape[1]:
            img = img.transpose(1, 0, 2)
            contour = contour[[1, 0]]
        aspect = img.shape[1] / img.shape[0]
        y = 10 // 2
        x = y * aspect / 2

        # Add image to scatter
        ax.imshow(
            img,
            origin="upper",
            # (left, right, bottom, top)
            extent=[loc[0] - x, loc[0] + x, loc[1] - y, loc[1] + y],
            aspect="auto",
            interpolation="none",
            alpha=0.85,
        )
        # TODO: add tissue contour
        # ax.plot(*contour, color="black", linewidth=3, extent=?)
        ax.plot(
            [loc[0] - x, loc[0] + x, loc[0] + x, loc[0] - x, loc[0] - x],
            [loc[1] - y, loc[1] - y, loc[1] + y, loc[1] + y, loc[1] - y],
            color="black",
            linewidth=0.25,
        )
        # Scale bar
        bar_frac = 1000 / mpp / img.shape[0]
        ax.plot(
            [
                loc[0] - x + 0.1,
                loc[0] - x + (((loc[0] + x) - (loc[0] - x)) * bar_frac) + 0.1,
            ],
            [
                loc[1] - y + 10 - 0.3,
                loc[1] - y + 10 - 0.3,
            ],
            color="black",
            linewidth=3,
        )
    # Scale bar legend
    ax.plot(
        [
            r["Age"].min(),
            r["Age"].min() + 5,
        ],
        [
            r[var].min(),
            r[var].min(),
        ],
        color="black",
        linewidth=3,
    )
    ax.text(r["Age"].min() + 2.5, r[var].min() - 3.5, "1 mm", fontsize=8, ha="center")

    # Customs
    ax.set(
        xlim=(15, 75),
        ylim=(-v - vm, v + vm),
    )
    ax.set_xlabel("Chronological age", fontsize=16)
    ax.set_ylabel("Age gap", fontsize=16)
    ax.set_title(tissue, fontsize=16)

    # Save
    for end in ["png", "pdf", "svgz"]:
        fig.savefig(
            output_dir / f"{tissue}_{var}_scatter_overlay.{end}",
            dpi=200,
            bbox_inches="tight",
        )
