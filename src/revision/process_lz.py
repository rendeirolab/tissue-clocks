#!/usr/bin/env uv --script

# /// script
# dependencies = [
#   "numpy",
#   "pandas",
#   "spatialdata>=0.4.0",
#   "lazyslide>=0.7.2",
#   "scanpy",
#   "conch",
# ]
# [tool.uv.sources]
# conch = { git = "https://github.com/mahmoodlab/CONCH.git" }
# ///

"""
Analysis of skin H&E images

Use `uv` to manage dependencies, virtualenv and run.

Run in cluster with:

PROJECT_NAME=histopath
cd ~/projects/${PROJECT_NAME}
mkdir -p logs/processing

for ID in {00..11}; do
sbatch \
--job-name ${PROJECT_NAME}.processing.gpu:l4.${ID} \
--partition=gpu --qos=gpu --gres=gpu:l4_gpu:1 --time 3-00:00:00 -c 8 --mem 96G --comment="skip_dcgm" \
--output logs/processing/gpu:l4.${ID}.log \
--wrap "uv run --frozen --no-sync src/revision/process_lz.py"
done

for ID in {00..06}; do
sbatch \
--job-name ${PROJECT_NAME}.processing.gpu:h100pcie.${ID} \
--partition=gpu --qos=gpu --gres=gpu:h100pcie:1 --time 3-00:00:00 -c 8 --mem 96G --comment="skip_dcgm" \
--output logs/processing/gpu:h100pcie.${ID}.log \
--wrap "uv run --frozen --no-sync python src/revision/process_lz.py"
done

for ID in {00..03}; do
sbatch \
--job-name ${PROJECT_NAME}.processing.gpu:h100hgx.${ID} \
--partition=gpu --qos=gpu --gres=gpu:h100hgx:1 --time 3-00:00:00 -c 8 --mem 96G --comment="skip_dcgm" \
--output logs/processing/gpu:h100hgx.${ID}.log \
--wrap "uv run --frozen --no-sync python src/revision/process_lz.py"
done

ll processed/histopathology/*.zarr/tables/titan_tiles/X/0/0 | wc -l

srun --qos interactiveq --partition interactiveq --mem 96000 -c 16 --x11 --pty -J IPython \
uv run --with pandas --with scanpy --with lazyslide --with ipython ipython

"""

from pathlib import Path
import logging

import numpy as np
import pandas as pd
import lazyslide as zs
import scanpy as sc
import matplotlib.pyplot as plt


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

logger.info("Starting process")

metadata_dir = Path("metadata")
terms = metadata_dir / "histology_and_histopathology_terms.txt"
data_dir = Path("data") / "gtex" / "svs"
locks_dir = Path("locks")
locks_dir.mkdir(exist_ok=True, parents=True)
processed_dir = Path("processed") / "histopathology"
processed_dir.mkdir(exist_ok=True, parents=True)
results_dir = Path("results") / "tissue_clocks_revision"
results_dir.mkdir(exist_ok=True, parents=True)
figkws = dict(dpi=300, bbox_inches="tight")

metadata_file = Path("data") / "gtex" / "GTEx Portal.csv"
metadata = pd.read_csv(metadata_file, index_col=0)
metadata = metadata.query(
    "Tissue.str.startswith('Skin') | (Tissue == 'Brain - Cortex') | Tissue.str.contains('Colon') | Tissue.str.contains('Lung')"
)
files = sorted([data_dir / (f + ".svs") for f in metadata.index])
files = [f for f in files if f.exists()]
model_names = [
    "ccnbg63",
    "uni",
    "uni2",
    "conch",
    "virchow",
    "virchow2",
    "hibou-b",
    "hibou-l",
    "midnight",
    "gigapath",
    "h0-mini",
    "phikon",
    "phikonv2",
    "ctranspath",
    "chief",
    "h-optimus-0",
    "h-optimus-1",
    "titan",
]


def main() -> None:
    np.random.shuffle(files)
    for file in files:
        np.random.shuffle(model_names)
        for model_name in model_names:
            logger.info(f"Doing '{file}' with model '{model_name}'.")
            extract(file, model_name)


def extract(
    file: Path,
    model_name: str = "virchow2",
    mpp: float = 0.5,
    tile_width: int = 224,
) -> None:
    # mpp= 0.5; tile_width = 224
    import fasteners

    (processed_dir / "qc" / "tissue").mkdir(exist_ok=True, parents=True)
    output_file = (
        processed_dir / f"{file.stem}.{mpp}mpp.{tile_width}px.{model_name}.h5ad"
    )

    if output_file.exists():
        return

    if model_name in zs.models.list_models("multimodal"):
        term_embeddings = terms.with_suffix(f".{model_name}.pq")
        if not term_embeddings.exists():
            texts = terms.open().read().strip().splitlines()
            text_embeddings = zs.tl.text_embedding(texts, model_name)
            text_embeddings.to_parquet(term_embeddings)
        text_embeddings = pd.read_parquet(term_embeddings)

    if model_name in ["ccnbg63"]:
        from torchvision import transforms

        model_path = (
            Path("/nobackup")
            / "lab_rendeiro"
            / "projects"
            / "histopath"
            / "data"
            / "gtex"
            / "models"
            / "cemm-convnext_base_fine_tune_63.pkl"
        )
        transform = transforms.Compose([transforms.ToTensor()])

    zarr_file = processed_dir / (file.stem + ".zarr")
    lock_file = locks_dir / zarr_file.with_suffix(".lock").name
    s = zs.open_wsi(file, store=zarr_file)
    if "tiles" not in s.shapes:
        zs.pp.find_tissues(s)
        zs.pp.tile_tissues(s, tile_px=tile_width, mpp=mpp)
        with fasteners.InterProcessLock(lock_file):
            s.write()

    if f"{model_name}_tiles" not in s.tables:
        if model_name not in ["ccnbg63"]:
            zs.tl.feature_extraction(s, model_name)
        else:
            zs.tl.feature_extraction(
                s, model_name=model_name, model_path=model_path, transform=transform
            )
        s.fetch.features_anndata(f"{model_name}_tiles").write(output_file)
    if model_name in zs.models.list_models("multimodal"):
        if f"{model_name}_tiles_text_similarity" not in s.tables:
            zs.tl.text_image_similarity(s, text_embeddings, model=model_name)
        s.fetch.features_anndata(f"{model_name}_tiles_text_similarity").write(
            processed_dir
            / f"{file.stem}.{mpp}mpp.{tile_width}px.{model_name}_text_similarity.h5ad"
        )
    with fasteners.InterProcessLock(lock_file):
        s.write()

    a = sc.read_h5ad(
        processed_dir / f"{file.stem}.{mpp}mpp.{tile_width}px.{model_name}.h5ad"
    )
    assert s.shapes["tiles"].shape[0] == a.shape[0]

    f = processed_dir / "qc" / "tissue" / f"{file.stem}.tissue.png"
    if not f.exists():
        fig, ax = plt.subplots(figsize=(8, 8))
        zs.pl.tissue(s, ax=ax, mark_origin=False)
        fig.savefig(f, **figkws)

    f = processed_dir / "qc" / "tissue" / f"{file.stem}.tiles.png"
    if not f.exists():
        fig, ax = plt.subplots(figsize=(8, 8))
        zs.pl.tiles(s, ax=ax, mark_origin=False)
        fig.savefig(f, **figkws)


if __name__ == "__main__":
    main()
