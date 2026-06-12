# SPDX-FileCopyrightText: Copyright (c) 2026 Rendeiro Lab, CeMM - Research Center for Molecular Medicine, Austrian Academy of Sciences
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#
# Licensed under the PolyForm Noncommercial License 1.0.0 (see LICENSE).

"""

NOTE: The Slurm submission blocks below reference machine-specific paths
and are included for transparency only. They are not expected to run
outside the original machine.

# Launch:
cd /nobackup/lab_rendeiro/projects/histopath
mkdir -p logs/prepare_graphs

for N in {0..9}; do
sbatch -p tinyq --qos tinyq -c 1 --mem 2000 --time 02:00:00 \
-D /nobackup/lab_rendeiro/projects/histopath \
-J prepare_graphs_${N} -o /nobackup/lab_rendeiro/projects/histopath/logs/prepare_graphs/prepare_graphs_${N}.log \
--wrap "python -m fire /nobackup/lab_rendeiro/projects/histopath/src/_prepare_graphs.py main"
done

# Monitor:
fd graph.pt /nobackup/lab_rendeiro/projects/histopath/data/gtex/graphs | wc -l
"""

from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from scipy.spatial import cKDTree
from torch_geometric.data import Data
from src.utils import get_restricted_info


target_variable = "Age"

# Info
input_dir = Path("data") / "gtex" / "svs"
graph_dir = Path("data") / "gtex" / "graphs"
graph_dir.mkdir(exist_ok=True)
meta = pd.read_csv(input_dir.parent / "GTEx Portal.csv")
meta["Organ"] = meta["Tissue"].str.extract(r"(\w+ ?\w+ ?\w+)-?.*", expand=False)
indi, _ = get_restricted_info()
meta = meta.merge(
    indi[[target_variable]], how="left", left_on="Subject ID", right_index=True
).set_index("Tissue Sample ID")


def main():
    slide_files = sorted(input_dir.glob("*.h5"))
    slide_names = [f.stem for f in slide_files if f.name.count(".") == 1]
    slide_names = np.random.permutation(slide_names)
    feature_space_names = ["cemm-convnext_base_fine_tune_63.224px", "resnet50.224px"]
    for slide_name in slide_names:
        target_value = meta.loc[slide_name, target_variable]
        for feature_space_name in feature_space_names:
            tile_width = int(feature_space_name.split(".")[1].replace("px", ""))
            tile_distance = tile_width * np.sqrt(2) + 1
            try:
                make_graph(slide_name, target_value, feature_space_name, tile_distance)
            except:
                pass


def make_graph(
    slide_name: str,
    target_value: float,
    feature_space_name: str,
    tile_distance: float = 316.23,
):
    processed_file = graph_dir / f"{slide_name}.{feature_space_name}.graph.pt"
    lock_file = processed_file.with_suffix(".pt.lock")
    failed_file = processed_file.with_suffix(".pt.failed")
    missing_input_file = processed_file.with_suffix(".pt.missing_input")
    coords_file = input_dir / (slide_name + ".h5")
    feats_file = input_dir / f"{slide_name}.{feature_space_name}.npy"

    if processed_file.exists():
        return
    if (not coords_file.exists()) or (not feats_file.exists()):
        missing_input_file.touch()
        return
    if lock_file.exists():
        return
    lock_file.touch()

    coords = h5py.File(coords_file)["coords"][:]
    edge_index = get_edges(coords, tile_distance)

    x = torch.tensor(np.load(feats_file))
    data = Data(
        x=x,
        edge_index=edge_index,
        y=target_value,
        pos=coords,
        slide_name=slide_name,
        tile_distance=tile_distance,
    )
    if (data.edge_index.max() + 1) > data.x.shape[0]:
        failed_file.touch()
        return
    torch.save(data, processed_file)
    try:
        lock_file.unlink()
    except FileNotFoundError:
        processed_file.unlink()


def get_edges(coords, radius) -> torch.Tensor:
    tree = cKDTree(coords)
    pairs = tree.query_pairs(radius)
    edge_index = torch.tensor(np.array(list(pairs)).T)
    return edge_index
