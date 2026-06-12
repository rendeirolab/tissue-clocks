# SPDX-FileCopyrightText: Copyright (c) 2026 Rendeiro Lab, CeMM - Research Center for Molecular Medicine, Austrian Academy of Sciences
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#
# Licensed under the PolyForm Noncommercial License 1.0.0 (see LICENSE).

from pathlib import Path
import subprocess

import numpy as np
import pandas as pd
from tqdm import tqdm

model_checkpoint_dir = Path("~/.cache/torch/hub/checkpoints/").expanduser()
metadata_dir = Path("metadata")
data_dir = Path("data/gtex/svs")

models = [
    "alexnet",
    "googlenet",
    "vgg16",
    "vgg19",
    "densenet121",
    "densenet201",
    "resnet50",
    "resnet152",
    "efficientnet_v2_m",
    "efficientnet_v2_l",
    "convnext_base",
    "convnext_large",
    "vit_h14_in1k",
]


def get_hub(model: str):
    return "pytorch/vision" if model != "vit_h14_in1k" else "facebookresearch/SWAG"


def get_checkpoint_file(model: str) -> Path:
    return [f for f in model_checkpoint_dir.glob("*.pth") if model in f.name][0]


def get_checkpoint_files():
    for model in tqdm(models):
        get_checkpoint_file(model)


def download_model(model: str):
    import torchvision

    model = getattr(torchvision.models, model)(pretrained=True)


def download_models():
    for model in tqdm(models):
        download_model(model)


def check_outputs(model: str, only_store_mean: bool = False) -> pd.Series:
    files = list(data_dir.glob(f"*{model}.*.npy"))
    rep = (".mean", "") if only_store_mean else ("", "")
    return (
        pd.Series([f.stem.replace(*rep).split(".")[2] for f in files], dtype="object")
        .value_counts()
        .rename("model")
    )


def write_slide_list(
    only_missing: bool = True,
    model: str | None = None,
    n_resolutions: int | None = None,
    only_store_mean: bool = False,
):
    if only_missing:
        assert model is not None
        assert n_resolutions is not None
    df = pd.read_csv("data/gtex/GTEx Portal.csv", index_col=0)
    mis = pd.read_csv(metadata_dir / "failed_slides.txt", header=None).squeeze()
    sel = ~df.index.isin(mis)
    df = df.loc[sel]
    if not only_missing:
        print(df.shape[0])
        df.sample(frac=1).index.to_series().to_csv(
            "slides.txt", index=False, header=False
        )

    fs = sorted(data_dir.glob("*.npy"))
    if fs:
        fdf = (
            pd.DataFrame(
                [[f] + f.stem.replace(".mean", "").split(".") for f in fs],
                columns=["file", "slide_id", "model", "res"],
            )
            .set_index("slide_id")
            .query(f"model == '{model}'")
        )
        fdf["mean"] = fdf["file"].astype(str).str.contains(".mean", regex=False)
        fdf = fdf.query(f"mean == {only_store_mean}")
        df = df.loc[df.join(fdf).groupby(level=0).size() < n_resolutions]

    print(df.shape[0])
    df.sample(frac=1).index.to_series().to_csv("slides.txt", index=False, header=False)


def submit(
    model: str,
    checkpoint: Path | None = None,
    n_jobs: int = 5,
    max_jobs: int = 400,
    job_params: dict = dict(queue="tinyq", memory=8000, cpus=8),
    only_store_mean: bool = False,
    only_missing: bool = False,
):
    write_slide_list(only_missing, model, 3, only_store_mean)
    hub = get_hub(model)
    if checkpoint is None:
        checkpoint = get_checkpoint_file(model)
    log_dir = Path("logs") / "gtex" / "feature_extraction"
    log_dir.mkdir(exist_ok=True, parents=True)

    sub_dir = Path("submission") / "gtex" / "feature_extraction"
    sub_dir.mkdir(exist_ok=True, parents=True)

    slides = open("slides.txt", "r").readlines()
    additional = "--only-store-mean" if only_store_mean else ""
    job_ids = list()
    for job, slide_ids in enumerate(np.array_split(slides, n_jobs)):
        sub_f = sub_dir / f"{model}.slides{job}.txt"
        params = f"-p {job_params['queue']} \
--qos {job_params['queue']} \
--mem {job_params['memory']} \
-o {log_dir / f'extract_GTEx-{model}.%A.%a.log'} \
-c {job_params['cpus']}"
        with open(sub_f, "w") as fh:
            fh.write("".join(slide_ids))
        cmd = f"sbatch --array=1-{len(slide_ids)}%{max_jobs} \
{params} -J extract_GTEx-{model} \
src/feature_extractor.job.sh \
{sub_f} {model} {checkpoint} {hub} {additional}"

        o = subprocess.check_output(cmd.split(" ")).decode().strip()
        job_ids.append(int(o.split(" ")[-1]))

    print(f"Submitted {n_jobs} arrayed jobs for '{model}' with ids: {job_ids}")

    # TODO: submit job epilog concatenating logs and parsing out what worked, what didn't


def check_results():
    res = dict()
    for model in tqdm(models):
        res[model] = check_outputs(model)
    res = pd.DataFrame(res).fillna(0).astype(int).T
    print(res)
    return res


# python -m fire src/feature_extractor.submit.py submit \
#     --model=cemm-convnext_base_fine_tune_63 --n_jobs=20 --max_jobs=400 \
#     --checkpoint=data/gtex/models/cemm-convnext_base_fine_tune_63.pkl \
#     --job_params='{"queue": "tinyq", "memory": 16000, "cpus": 8}'
