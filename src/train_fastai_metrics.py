# SPDX-FileCopyrightText: Copyright (c) 2026 Rendeiro Lab, CeMM - Research Center for Molecular Medicine, Austrian Academy of Sciences
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#
# Licensed under the PolyForm Noncommercial License 1.0.0 (see LICENSE).

from argparse import ArgumentParser
import re
from timeit import default_timer as timer

import numpy as np
import pandas as pd
import torch
import fastai
from fastai.vision.all import (
    L,
    DataBlock,
    ImageBlock,
    CategoryBlock,
    aug_transforms,
    vision_learner,
    error_rate,
)
from src.utils import Path


def splitter(items):
    train = [i for i, f in enumerate(items) if f.parent.name == "train"]
    valid = [i for i, f in enumerate(items) if f.parent.name == "valid"]
    test = [i for i, f in enumerate(items) if f.parent.name == "test"]
    return train, valid, test


def get_individual_id(f: Path | str):
    if isinstance(f, Path):
        return f.stem.split(".")[0]
    if isinstance(f, str):
        return f.split(".")[0]
    raise ValueError()


def get_tissue_class(f: Path | str):
    if isinstance(f, Path):
        return f.stem.split(".")[1]
    if isinstance(f, str):
        return f.split(".")[1]
    raise ValueError()


def sorted_nicely(ls):
    def convert(text):
        return int(text) if text.isdigit() else text

    def alphanum_key(key):
        return [convert(c) for c in re.split("([0-9]+)", key)]

    return sorted(ls, key=alphanum_key)


def load_checkpoint(learn, checkpoint_file: Path = None, epoch: int = None):
    if checkpoint_file is None:
        files = sorted_nicely([p.as_posix() for p in (path / "models").glob("*.pth")])
        if epoch is not None:
            checkpoint_file = [f for f in files if f.endswith(f"_{epoch}.pth")][0]
        else:
            checkpoint_file = files[-1]
    else:
        assert epoch is None
    sd = torch.load(checkpoint_file)
    if "model" in sd:
        learn.model.load_state_dict(sd["model"])
        learn.opt.load_state_dict(sd["opt"])
    else:
        learn.model.load_state_dict(sd)


# class Args:
#     path: Path
#     model: str
#     epoch: int
#     checkpoint: Path
#     learn_rate: float


# args = Args()
# args.path = Path("data/gtex/datasets/gtex_balanced_stratified_3_slides_200_tiles")
# args.model = "convnext_base"
# args.epoch = 63
# args.checkpoint = Path(
#     f"data/gtex/datasets/gtex_balanced_stratified_3_slides_200_tiles/models/\
# {args.model}_gtex_balanced_stratified_3_slides_200_tiles_fine_tune_{args.epoch}.pth"
# )

parser = ArgumentParser()
parser.add_argument(dest="path", type=Path, help="Path to dataset")
parser.add_argument(dest="model", type=str, help="Model to use", default="resnet50")
parser.add_argument(
    "--only-valid",
    dest="only_valid",
    action="store_true",
    help="Only evaluate on validation set.",
)
args = parser.parse_args()


path = args.path
model = getattr(fastai.vision.all, args.model)

block = DataBlock(
    blocks=[ImageBlock, CategoryBlock],
    get_y=get_tissue_class,
    splitter=splitter,
    batch_tfms=aug_transforms(size=224),
)
dls = block.dataloaders(L(path.glob("*/*.jpg")), path=path, num_workers=4, bs=64)

print(f"Using dataset from {path}")
print(f"Train size: {len(dls.train_ds)}")
print(f"Validation size: {len(dls.valid_ds)}")

learn = vision_learner(dls, model, metrics=error_rate)

models_dir = path / "models"
checkpoints = sorted(models_dir.glob(f"{args.model}_{path.name}_fine_tune_*.pth"))
epochs = [int(f.stem.split("_")[-1]) for f in checkpoints]

metrics_file = models_dir / f"metrics_{args.model}.csv"
if metrics_file.exists():
    metrics = pd.read_csv(metrics_file)
    max_epoch = metrics["epoch"].max()
else:
    metrics = pd.DataFrame(
        columns=[
            "model",
            "dataset",
            "epoch",
            "train_loss",
            "train_error_rate",
            "train_time",
            "valid_loss",
            "valid_error_rate",
            "valid_time",
        ]
    )
    max_epoch = 0


iterator = sorted(
    [(c, e) for c, e in zip(checkpoints, epochs) if e > max_epoch], key=lambda x: x[1]
)

print("Starting inference.")
for checkpoint, epoch in iterator:
    # print(f"Doing epoch {epoch} with checkpoint file: '{checkpoint}'")
    load_checkpoint(learn, checkpoint)

    if not args.only_valid:
        start = timer()
        with learn.no_bar():
            tprobs, ty, tpreds, tloss = learn.get_preds(
                ds_idx=0, with_decoded=True, with_loss=True
            )
        ttime = timer() - start
        tloss = tloss.mean().item()
        ter = (tpreds != ty).sum() / ty.shape[0]
        ter = ter.item()
    else:
        tprobs = ty = tpreds = tloss = ttime = ter = np.nan

    start = timer()
    with learn.no_bar():
        vprobs, vy, vpreds, vloss = learn.get_preds(
            ds_idx=1, with_decoded=True, with_loss=True
        )
    vtime = timer() - start
    vloss = vloss.mean().item()
    ver = (vpreds != vy).sum() / vy.shape[0]
    ver = ver.item()
    res = pd.Series(
        [
            args.model,
            args.path.name,
            epoch,
            tloss,
            ter,
            ttime,
            vloss,
            ver,
            vtime,
        ],
        index=metrics.columns,
    )
    print(res)
    metrics = pd.concat([metrics, res.to_frame().T], axis=0, ignore_index=True)
    metrics.to_csv(metrics_file, index=False)

print("Finished inference.")

# # To migrate from metrics scraped from logs to CSV:
# metrics = pd.read_parquet(
#     "results/train_gtex_balanced_stratified_3_slides_fine_tune.pq"
# )
# for model in metrics["model"].unique():
#     for tiles in metrics.query(f"model == '{model}'")["tiles"].unique():
#         df = metrics.query(f"model == '{model}' & tiles == {tiles}")
#         df["dataset"] = f"gtex_balanced_stratified_3_slides_{tiles}_tiles"
#         df = df[
#             [
#                 "model",
#                 "dataset",
#                 "epoch",
#                 "train_loss",
#                 "train_error_rate",
#                 "train_time",
#                 "valid_loss",
#                 "valid_error_rate",
#                 "valid_time",
#             ]
#         ]
#         df.to_csv(path / "models" / f"metrics_{model}.csv", index=False)


# # To submit jobs:
# cd ~/projects/histopath/
# mkdir -p logs/model_metrics
# export PYTHONPATH=.

# MODELS=(
# # alexnet
# # googlenet
# vgg16
# # vgg19
# # densenet121
# # densenet201
# xresnet50
# resnet50
# # resnet152
# # efficientnet_v2_m
# # efficientnet_v2_l
# convnext_tiny
# convnext_base
# convnext_large
# # vit_h14_in1k
# # maxvit_t
# )
# TILES=(200)
# for MODEL in ${MODELS[@]}; do
#     for TILE in ${TILES[@]}; do
#         sbatch \
#         --time 2-00:00:00 --qos mediumq --partition mediumq \
#         --mem 32000 -c 8 \
#         -o logs/model_metrics/metrics_${MODEL}_gtex_balanced_stratified_3_slides_${TILE}_tiles_fine_tune.out \
#         -J metrics_${MODEL}_gtex_balanced_stratified_3_slides_${TILE}_tiles_fine_tune \
#         --wrap "python -u src/train_fastai_metrics.py --only-valid data/gtex/datasets/gtex_balanced_stratified_3_slides_${TILE}_tiles ${MODEL}"
#     done
# done
