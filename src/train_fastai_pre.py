import click
from tqdm import tqdm
import pandas as pd
from wsi_core import WholeSlideImage

from src.utils import Path


@click.command()
@click.argument("n_slides", type=int)
@click.argument("n_tiles", type=int)
@click.argument("allow_less", type=bool)
def create_jpg_dataset(
    n_slides: int = 3, n_tiles: int = 2000, allow_less: bool = False
):
    data_dir = Path("data/gtex/svs")
    output_dir = Path(
        f"data/gtex/datasets/gtex_balanced_stratified_{n_slides}_slides_{n_tiles}_tiles"
    ).mkdir()
    slide_files = list(data_dir.glob("*.svs"))
    slide_annot_f = output_dir / "slide_annotation.csv"

    attrs = ["Tissue_simple", "Sex", "Age Bracket"]

    if not slide_annot_f.exists():
        available_slides = [
            f.stem for f in slide_files if f.replace_(".svs", ".h5").exists()
        ]

        meta = pd.read_csv(data_dir.parent / "GTEx Portal.csv", index_col=0)
        meta = meta.loc[meta.index.isin(available_slides)]

        # Simplify tissue classes by joining different tissues of same organ
        meta["Tissue_simple"] = meta["Tissue"].str.split(" - ").apply(lambda x: x[0])

        # Subset ages: use extremes and center for train
        ages = ["20-29", "40-49", "60-69"]
        meta = meta.loc[meta["Age Bracket"].isin(ages)]

        # Balance slides
        # # get slides independendently for somatic and germ tissues as balancing has to be different

        # # figure out which tissues are germ vs somatic
        sel = (meta.groupby(["Tissue_simple", "Sex"]).size() == 0).groupby(
            "Tissue_simple"
        ).size() == 1
        germ = meta["Tissue_simple"].value_counts()[sel].index

        meta_soma = meta.loc[~meta["Tissue_simple"].isin(germ)]

        meta_soma_sel = sample(meta_soma, attrs, n_slides, not allow_less)

        meta_germ = meta.loc[meta["Tissue_simple"].isin(germ)]
        meta_germ_sel = sample(
            meta_germ, ["Tissue_simple", "Age Bracket"], n_slides, not allow_less
        )

        # Join both somatic and germ tissue
        train = (
            pd.concat([meta_soma_sel, meta_germ_sel])
            .sort_index()
            .assign(split="train", n_tiles=n_tiles)
        )
        # meta_sel = pd.concat([meta_soma, meta_germ]).sort_index()

        meta = pd.read_csv(data_dir.parent / "GTEx Portal.csv", index_col=0)
        meta = meta.loc[meta.index.isin(available_slides)]

        # Simplify tissue classes by joining different tissues of same organ
        meta["Tissue_simple"] = meta["Tissue"].str.split(" - ").apply(lambda x: x[0])

        # Subset ages: use middle ones for validation
        ages = ["20-29", "40-49", "60-69"]
        meta = meta.loc[~meta["Age Bracket"].isin(ages)]

        # Balance slides
        # # get slides independendently for somatic and germ tissues as balancing has to be different
        meta_soma = meta.loc[~meta["Tissue_simple"].isin(germ)]
        meta_soma_sel = sample(meta_soma, attrs, 1, not allow_less)
        meta_germ = meta.loc[meta["Tissue_simple"].isin(germ)]
        meta_germ_sel = sample(
            meta_germ, ["Tissue_simple", "Age Bracket"], 1, not allow_less
        )

        # Join both somatic and germ tissue
        valid = (
            pd.concat([meta_soma_sel, meta_germ_sel])
            .sort_index()
            .assign(split="valid", n_tiles=n_tiles // 10)
        )

        # Join both train and valid in same dataframe
        meta_sel = pd.concat([train, valid]).sort_index()

        # Save file with splits
        meta_sel.to_csv(slide_annot_f)
    meta_sel = pd.read_csv(slide_annot_f, index_col=0)

    (output_dir / "train").mkdir()
    (output_dir / "valid").mkdir()
    attributes = ["Tissue_simple", "Sex", "Age Bracket"]
    for slide_name, row in tqdm(meta_sel.iterrows(), total=meta_sel.shape[0]):
        slide_f = data_dir / slide_name + ".svs"
        slide = WholeSlideImage(slide_f)
        slide.attributes = row[attributes].to_dict()
        slide.save_tile_images(output_dir / row["split"], n=row["n_tiles"], frac=None)


def sample(df: pd.DataFrame, attrs: list[str], n: int, strict: bool) -> pd.DataFrame:
    # Previous, more strict sampling:
    if strict:
        return df.groupby(attrs).sample(n=n)

    # Allow sampling with unbalanced classes if class has less than n:
    nl = df.groupby(attrs).size()
    _df_sel = list()
    for row in nl.index:
        s = df.query(
            " & ".join([f"`{a}` == '{b}'" for a, b in dict(zip(attrs, row)).items()])
        )
        ts = min(n, s.shape[0])
        if ts != n:
            print(f"{row} combination has less than {n}.")
        _df_sel.append(s.sample(n=ts))
    df_sel = pd.concat(_df_sel)
    return df_sel


if __name__ == "__main__" and "get_ipython" not in locals():
    import sys

    try:
        sys.exit(create_jpg_dataset())
    except KeyboardInterrupt:
        sys.exit()

# create_jpg_dataset(n_slides=3, n_tiles=200)
# create_jpg_dataset(n_slides=3, n_tiles=2000)

# create_jpg_dataset(n_slides=10, n_tiles=100, allow_less=True)
# create_jpg_dataset(n_slides=10, n_tiles=800, allow_less=True)

# create_jpg_dataset(n_slides=50, n_tiles=50, allow_less=True)

# # To submit jobs:
# cd ~/projects/histopath/
# export PYTHONPATH=.

# N_SLIDES=3
# N_TILES=200
# sbatch \
# --time 30-00:00:00 --qos longq --partition longq \
# --mem 32000 -c 8 \
# -o logs/datasets/gtex_balanced_stratified_${N_SLIDES}_slides_${N_SLIDES}_tiles_pre.out \
# -J gtex_balanced_stratified_${N_SLIDES}_slides_${N_SLIDES}_tiles_pre \
# --wrap "python -u src/train_fastai_pre.py ${N_SLIDES} ${N_TILES} False"

# N_SLIDES=3
# N_TILES=2000
# sbatch \
# --time 30-00:00:00 --qos longq --partition longq \
# --mem 32000 -c 8 \
# -o logs/datasets/gtex_balanced_stratified_${N_SLIDES}_slides_${N_SLIDES}_tiles_pre.out \
# -J gtex_balanced_stratified_${N_SLIDES}_slides_${N_SLIDES}_tiles_pre \
# --wrap "python -u src/train_fastai_pre.py ${N_SLIDES} ${N_TILES} False"

# N_SLIDES=10
# N_TILES=100
# sbatch \
# --time 0-04:00:00 --qos interactiveq --partition interactiveq \
# --mem 16000 -c 2 \
# -o logs/datasets/gtex_balanced_stratified_${N_SLIDES}_slides_${N_TILES}_tiles_pre.out \
# -J gtex_balanced_stratified_${N_SLIDES}_slides_${N_TILES}_tiles_pre \
# --wrap "python -u src/train_fastai_pre.py ${N_SLIDES} ${N_TILES} True"

# N_SLIDES=10
# N_TILES=800
# sbatch \
# --time 1-12:00:00 --qos mediumq --partition mediumq \
# --mem 16000 -c 2 \
# -o logs/datasets/gtex_balanced_stratified_${N_SLIDES}_slides_${N_TILES}_tiles_pre.out \
# -J gtex_balanced_stratified_${N_SLIDES}_slides_${N_TILES}_tiles_pre \
# --wrap "python -u src/train_fastai_pre.py ${N_SLIDES} ${N_TILES} True"

# N_SLIDES=3
# N_TILES=200
# cd data/gtex/datasets/gtex_balanced_stratified_${N_SLIDES}_slides_${N_TILES}_tiles
# tar --exclude models -czf ../gtex_balanced_stratified_${N_SLIDES}_slides_${N_TILES}_tiles.tar.gz *
# N_SLIDES=3
# N_TILES=2000
# cd data/gtex/datasets/gtex_balanced_stratified_${N_SLIDES}_slides_${N_TILES}_tiles
# tar --exclude models -czf ../gtex_balanced_stratified_${N_SLIDES}_slides_${N_TILES}_tiles.tar.gz *
# N_SLIDES=10
# N_TILES=100
# cd data/gtex/datasets/gtex_balanced_stratified_${N_SLIDES}_slides_${N_TILES}_tiles
# tar --exclude models -czf ../gtex_balanced_stratified_${N_SLIDES}_slides_${N_TILES}_tiles.tar.gz *
# N_SLIDES=10
# N_TILES=800
# cd data/gtex/datasets/gtex_balanced_stratified_${N_SLIDES}_slides_${N_TILES}_tiles
# tar --exclude models -czf ../gtex_balanced_stratified_${N_SLIDES}_slides_${N_TILES}_tiles.tar.gz *
# sbatch \
# --time 1-12:00:00 --qos mediumq --partition mediumq \
# --mem 16000 -c 1 \
# -o logs/datasets/gtex_balanced_stratified_${N_SLIDES}_slides_${N_TILES}_tiles_package.out \
# -J gtex_balanced_stratified_${N_SLIDES}_slides_${N_TILES}_tiles_package \
# -D data/gtex/datasets/gtex_balanced_stratified_${N_SLIDES}_slides_${N_TILES}_tiles \
# --wrap "tar --exclude models -czf ../gtex_balanced_stratified_${N_SLIDES}_slides_${N_TILES}_tiles.tar.gz *"


# # Make a dataset with all slides (later to be renamed to "full" or "all_slides" or something)
# N_SLIDES=30_000
# N_TILES=2000
# sbatch \
# --time 30-00:00:00 --qos longq --partition longq \
# --mem 32000 -c 8 \
# -o logs/datasets/gtex_balanced_stratified_${N_SLIDES}_slides_${N_SLIDES}_tiles_pre.out \
# -J gtex_balanced_stratified_${N_SLIDES}_slides_${N_SLIDES}_tiles_pre \
# --wrap "python -u src/train_fastai_pre.py ${N_SLIDES} ${N_TILES} True"
