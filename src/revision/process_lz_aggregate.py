#!/usr/bin/env uv


# SPDX-FileCopyrightText: Copyright (c) 2026 Rendeiro Lab, CeMM - Research Center for Molecular Medicine, Austrian Academy of Sciences
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#
# Licensed under the PolyForm Noncommercial License 1.0.0 (see LICENSE).

"""
Aggregate features. Run in cluster with:

PROJECT_NAME=histopath
cd ~/projects/${PROJECT_NAME}
mkdir -p logs/processing

for ID in {00..119}; do
sbatch \
--job-name ${PROJECT_NAME}.aggregating.tinyq.${ID} \
--qos tinyq -p tinyq --time 02:00:00 -c 8 --mem 32G --comment="skip_dcgm" \
--output logs/aggregating/tinyq.${ID}.log \
--wrap "uv run --frozen --no-sync python src/revision/process_lz_aggregate.py"
done

ll processed/histopathology/*.prism_aggregated.npy | wc -l
ll processed/histopathology/*.titan_aggregated.npy | wc -l
"""

from pathlib import Path

import numpy as np
import pandas as pd
import parmap
import fasteners
import anndata as ad
import lazyslide as zs


data_dir = Path("data") / "gtex" / "svs"
locks_dir = Path("locks")
locks_dir.mkdir(exist_ok=True, parents=True)
processed_dir = Path("processed") / "histopathology"
processed_dir.mkdir(exist_ok=True, parents=True)


def main() -> None:
    # For vision models at tile level
    collect("h0-mini", agg_func="mean")

    # For tile aggregator models
    collect_aggregated("prism")
    collect_aggregated("titan")


def collect(model_name: str, agg_func: str = "mean"):
    """
    uv run --with fire fire src/revision/process_lz.py collect h0-mini mean
    """
    assert agg_func in ["mean"], f"Unsupported agg func: {agg_func}"
    annot = pd.read_csv(data_dir.parent / "GTEx Portal.csv", index_col=0)
    zarrs = sorted(processed_dir.glob("*.zarr"))
    to_do = [z for z in zarrs if (z / "tables" / f"{model_name}_tiles").exists()]
    output_file = (
        processed_dir / f"gtex_anndata.0.5mpp.224px.{model_name}.{agg_func}.h5ad"
    )
    if output_file.exists():
        return

    def process_zarr(zarr_file):
        a = ad.read_zarr(zarr_file / "tables" / f"{model_name}_tiles")
        return zarr_file.stem, a.X.mean(axis=0)

    results = parmap.map(process_zarr, to_do, pm_pbar=True, pm_processes=24)

    _feats = dict(results)
    x = pd.DataFrame.from_dict(_feats, orient="index")
    x = x.rename_axis(index=annot.index.name)
    x.columns = x.columns.astype(str)
    adata = ad.AnnData(X=x, obs=annot.loc[x.index])
    adata.write_h5ad(output_file)


def collect_aggregated(model_name: str = "prism"):
    """
    uv run --with fire fire src/revision/process_lz.py collect_aggregated prism
    """
    # !uv add environs protobuf sacremoses
    assert model_name in ["prism", "titan"], f"Unsupported model name: {model_name}"
    encoders = {"prism": "virchow", "titan": "titan"}
    vision_model = encoders[model_name]

    zarrs = sorted(processed_dir.glob("*.zarr"))
    to_do = [z for z in zarrs if (z / "tables" / f"{vision_model}_tiles").exists()]
    np.random.shuffle(to_do)

    for zarr_file in to_do:
        output_file = processed_dir / f"{zarr_file.stem}.{model_name}_aggregated.npy"
        if output_file.exists():
            continue
        s = zs.open_wsi(
            data_dir / (zarr_file.stem + ".svs"), store=zarr_file.as_posix()
        )
        assert f"{vision_model}_tiles" in s.tables
        zs.tl.feature_aggregation(s, feature_key=vision_model, encoder=model_name)
        feats = s[f"{vision_model}_tiles"].uns["agg_ops"]["agg_slide"]["features"]

        lock_file = locks_dir / zarr_file.with_suffix(".lock").name
        with fasteners.InterProcessLock(lock_file):
            s.write()


def _fix_titan_zarr():
    from tqdm import tqdm
    import pandas as pd
    import scanpy as sc

    # if h5ad exists but table is not in zarr, add to zarr, save zarr
    files = pd.Series(sorted(processed_dir.glob("GTEX-*.titan.h5ad")))
    # extract modified dates as datetime objects
    times = (
        pd.Series([f.stat().st_mtime for f in files])
        .apply(pd.to_datetime, unit="s")
        .sort_values(ascending=False)
    )

    # subset for files modified before 2025-08-22, 10 am
    files = files[times < pd.to_datetime("2025-08-22 10:00:00")]

    for h5ad_file in tqdm(files):
        slide_name = h5ad_file.stem.split(".")[0]
        svs_file = data_dir / (slide_name + ".svs")
        zarr_file = processed_dir / (slide_name + ".zarr")
        table_file = zarr_file / "tables" / "titan_tiles"

        if not table_file.exists():
            print(f"Adding {slide_name} to zarr")
            a = sc.read_h5ad(h5ad_file)
            s = zs.open_wsi(svs_file, store=zarr_file)
            s.tables["titan_tiles"] = a
            lock_file = locks_dir / zarr_file.with_suffix(".lock").name
            with fasteners.InterProcessLock(lock_file):
                s.write()
        else:
            s = zs.open_wsi(svs_file, store=zarr_file)
            a = sc.read_h5ad(h5ad_file)
            if (s.tables["titan_tiles"].X == a.X).all():
                print(f"{slide_name} already in zarr, skipping")
            else:
                print(f"Updating {slide_name} in zarr")
                s.tables["titan_tiles"] = a

    # This should now match:
    # ll processed/histopathology/*.titan.h5ad | wc -l
    # ll processed/histopathology/GTEX-*.zarr/tables/titan_tiles/X/0/0 | wc -l


if __name__ == "__main__":
    main()
