# SPDX-FileCopyrightText: Copyright (c) 2026 Rendeiro Lab, CeMM - Research Center for Molecular Medicine, Austrian Academy of Sciences
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#
# Licensed under the PolyForm Noncommercial License 1.0.0 (see LICENSE).

from pathlib import Path
import typing as tp
import argparse
import json
from timeit import default_timer as timer
import os
import time
import resource

from tqdm_loggable.auto import tqdm
import requests
import paramiko
import h5py
import numpy as np
import pandas as pd
from shapely.geometry import Polygon
import torch
import torchvision
from torchvision import transforms
import PIL
import wandb

from wsi_core import WholeSlideImage

torch.set_float32_matmul_precision("high")

# # To test:
# python -u src/feature_extractor.py --slide-ids GTEX-1HCU8-0726 \
# --output-suffix pt63 -s extract \
# -m convnext_base_fine_tune_63 \
# -w data/gtex/datasets/gtex_balanced_stratified_3_slides_200_tiles/convnext_base_gtex_balanced_stratified_3_slides_200_tiles_fine_tune_63.model.pkl \
# -t 224 data/gtex/svs


# cli = dict(
#     model_hub="pytorch/vision",
#     model_name="convnext_tiny",
#     tile_size=224 * 2,
#     coord_shift=-112,
#     device="cpu",
#     output_suffix="resnet50_448px",
#     copy_from_ssh_address='login',
#     data_dir=Path("data") / "gtex" / "svs",
#     allow_download=True,
#     remove_slide=True,
#     slide_ids="slides.txt",
# )
# args: argparse.Namespace = argparse.Namespace(**cli)


def main():
    if args.slide_ids is None:
        tqdm.write("Loading slide IDs from directory.")
        # Directory with SVS files (slide-ids inferred from file names)
        # In this case it will do feature extraction only
        wsi_paths = list_slides_from_path(args.data_dir)
        tqdm.write(f"Found {len(wsi_paths)} files.")
    elif Path(args.slide_ids).is_dir():
        tqdm.write("Loading slide IDs from directory.")
        # Directory with SVS files (slide-ids inferred from file names)
        # In this case it will do feature extraction only
        wsi_paths = list_slides_from_path(Path(args.slide_ids))
        tqdm.write(f"Found {len(wsi_paths)} files.")
    elif Path(args.slide_ids).is_file():
        tqdm.write("Loading slide IDs from file.")
        # Slide IDs given as a path to a text file
        # Slides will be downloaded if not already present in `data_dir`
        slide_ids = pd.read_csv(args.slide_ids, header=None).squeeze().tolist()
        # If inside Slurm array job, only process the current task
        if not os.environ.get("SLURM_ARRAY_TASK_ID", None) is None:
            task = int(os.environ.get("SLURM_ARRAY_TASK_ID"))
            tqdm.write(f"Inside Slurm Array job, doing task '{task}'.")
            slide_ids = [slide_ids[task]]
            time.sleep(np.random.randint(0, 30))  # stagger tasks
        wsi_paths = [args.data_dir / s for s in slide_ids]

        if not args.allow_download:
            wsi_paths = [f for f in wsi_paths if f.exists()]
        tqdm.write(f"Found {len(wsi_paths)} files.")
    elif isinstance(args.slide_ids, str):
        tqdm.write("Loading slide IDs from given CLI string.")
        # Slide IDs given as one string, comma-delimited
        slide_ids = args.slide_ids.split(",")
        wsi_paths = [args.data_dir / s for s in slide_ids]
        if not args.allow_download:
            wsi_paths = [f for f in wsi_paths if f.exists()]
        tqdm.write(f"Found {len(wsi_paths)} files.")
    else:
        raise ValueError(f"Invalid value for `slide_ids`: '{args.slide_ids}'.")

    if args.log_to_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run,
            job_type="eval",
            group=args.wandb_project + " - " + args.wandb_run.split(".")[0],
            id=args.wandb_project + " - " + args.wandb_run,
            resume="auto",
            config=args,
        )

    if args.shuffle_slide_order:
        wsi_paths = np.random.permutation(wsi_paths)

    if not args.parallel:
        for wsi_path in tqdm(wsi_paths):
            tqdm.write(f"Doing slide '{wsi_path.stem}'.")
            extract_features_from_slide([wsi_path])
    else:
        # raise NotImplementedError(
        #     "Parallel processing across slides not implemented yet."
        # )
        tqdm.write(f"Doing {len(wsi_paths)} slides.")
        extract_features_from_slide(wsi_paths)

    max_mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    tqdm.write(f"Max memory used: {max_mem / 1024:.2f} MB")
    tqdm.write("Finished.")


def get_args(cli=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        dest="data_dir", default=Path("data") / "gtex" / "svs", type=Path
    )
    _help = (
        "Slide IDs. Can be a path to a file with one ID per line, "
        "a string with IDs comma separated, or not given which will process "
        "all .svs files in `data_dir`."
    )
    parser.add_argument("--slide-ids", default=None, help=_help)
    parser.add_argument(
        "--slide-file-types", default=["svs", "ndpi"], nargs="+", type=str
    )
    parser.add_argument("--shuffle-slide-order", action="store_true")
    parser.add_argument("--output-suffix", default="")
    parser.add_argument(
        "-n", "--sample-n-tiles", dest="n_tiles", default=None, type=int
    )
    parser.add_argument(
        "-f", "--sample-fraction-tiles", dest="f_tiles", default=None, type=float
    )
    parser.add_argument(
        "-s", "--steps", default="all", choices=["all", "download", "extract"]
    )
    parser.add_argument("-m", "--model-name", default="resnet50")
    parser.add_argument("--model-hub", default="pytorch/vision")
    parser.add_argument("-w", "--model-weights", default=None, type=Path)
    parser.add_argument("-t", "--tile-size", default=224, type=int)
    parser.add_argument("--coord-shift", default=0, type=int)
    parser.add_argument(
        "-d", "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    # parser.add_argument("--n-cpus", default=-1, type=int)
    parser.add_argument("-b", "--batch-size", default=64, type=int)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--copy-from-ssh-address", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--remove-slide", action="store_true")
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument("--compile-model", action="store_true")
    parser.add_argument("--only-store-mean", action="store_true")
    parser.add_argument("--log-to-wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default=None)
    parser.add_argument("--wandb-run", type=str, default=None)
    args = parser.parse_args(cli)

    if args.n_tiles is not None and args.f_tiles is not None:
        raise ValueError("Can't specify both `n_tiles` and `f_tiles`.")

    if args.steps == "all":
        args.steps = ["download", "extract"]
    else:
        args.steps = [args.steps]
    if args.output_suffix != "":
        args.output_suffix += "."
    if args.only_store_mean:
        args.output_suffix += "mean."

    tqdm.write("CLI:\n" + "\n".join([f"{k}: {v}" for k, v in args.__dict__.items()]))
    return args


def list_slides_from_path(path: Path) -> list[Path]:
    slides = []
    for ftype in args.slide_file_types:
        slides += list(path.rglob(f"*.{ftype}"))
    return sorted(slides)


def extract_features_from_slide(wsi_paths: list[Path]) -> None:
    # Load model
    if args.model_weights is None:
        model = torch.hub.load(args.model_hub, args.model_name, weights="DEFAULT")
    else:
        tqdm.write(f"Using model weights from file: '{args.model_weights}'")
        if args.model_weights.endswith(".pkl"):
            model = torch.load(args.model_weights)
        elif hasattr(torchvision.models, args.model_name):
            # N.B. Calling torch.hub.load pings some server even if weights are already local.
            # To avoid DDoSing a random server, instantiating a model as below,
            # (from a local file explicitely) avoids this
            model = getattr(torchvision.models, args.model_name)()
            model.load_state_dict(torch.load(args.model_weights))
        else:
            # Insecure
            model = torch.jit.load(args.model_weights)
    model = model.to(args.device)
    model.eval()
    if args.compile_model:
        model = torch.compile(model, mode="max-autotune")  # requires torch>=2.0.0

    model.input_width = guess_input_dims(model)
    tqdm.write(
        f"Determined input size of model '{args.model_name}' to be '{model.input_width}'."
    )

    for i, wsi_path in tqdm(
        enumerate(wsi_paths),
        desc="slide",
        disable=len(wsi_paths) <= 1,
        position=0,
        leave=False,
    ):
        extract_features(wsi_path, model)

    # parmap.map(
    #     extract_features,
    #     wsi_paths,
    #     model=model,
    #     pm_pbar=len(wsi_paths) > 1,
    #     pm_processes=args.n_cpus,
    # )


def extract_features(wsi_path: Path, model) -> None:
    feat_file = wsi_path.with_suffix(f".{args.output_suffix}npy")
    if feat_file.exists() and not args.overwrite:
        tqdm.write(f"File '{feat_file}' already exists. Not overwriting.")
        return

    # Create lock file
    lock_file = wsi_path.with_suffix(".lock")
    if lock_file.exists():
        tqdm.write(f"File '{lock_file}' being used. Skipping.")
        return
    lock_file.parent.mkdir(exist_ok=True, parents=True)
    lock_file.touch()

    time_download = 0.0
    if "download" in args.steps and (not wsi_path.exists()):
        tqdm.write(f"\nDownloading slide '{wsi_path}'.")
        start = timer()
        get_he_file(wsi_path)
        time_download = timer() - start
        tqdm.write(f"Downloading '{wsi_path.as_posix()}' took {time_download:.2f}s.")

    if "extract" not in args.steps:
        return

    if not wsi_path.exists():
        tqdm.write(f"File '{wsi_path}' does not exist. Skipping.")
        return

    tqdm.write(f"Doing '{wsi_path}'.")

    try:
        slide = WholeSlideImage(wsi_path)
    except PIL.UnidentifiedImageError:
        tqdm.write(f"Failed for file '{wsi_path}'. Deleting file.")
        wsi_path.unlink()
        return

    # Segment and tile if not done already
    if not slide.has_tile_coords():
        slide.segment(method="manual")
        slide.tile()
    coords = slide.get_tile_coordinates()
    coords += args.coord_shift
    tqdm.write(f"Slide '{wsi_path}' has {coords.shape[0]} tiles.")
    # TODO: keep/get information on which tissue piece each tile is.

    # Sample tiles if requested
    tqdm.write(f"Will save output as '{feat_file}'.")
    if args.n_tiles is None and args.f_tiles is None:
        pass
    else:
        n = (
            int(np.round(args.f_tiles * coords.shape[0]))
            if args.f_tiles is not None
            else args.n_tiles
        )
        n = min(n, coords.shape[0])
        coords = coords[np.random.choice(coords.shape[0], n, replace=False)]
        tqdm.write(f"Will extract features for only {n} random tiles.")

    tile_dims = (args.tile_size, args.tile_size)

    # Function (pipe) to transform a PIL Image (or numpy array) to tensor of shape (3, X, Y)
    # normalized as the input data for model training on ImageNet
    norm = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
        + (
            [transforms.Resize(model.input_width)]
            if args.tile_size != model.input_width
            else []
        )
    )

    start = timer()
    _feats = list()
    n = 0
    t = (coords.shape[0] // args.batch_size) + 1
    for batch in tqdm(
        batched(coords, args.batch_size), total=t, desc="batch", position=1, leave=False
    ):
        t = torch.stack(
            [
                norm(slide.wsi.read_region(c, level=0, size=tile_dims).convert("RGB"))
                for c in batch
            ]
        ).to(args.device)

        with torch.no_grad():
            feat = model(t).cpu().numpy()
        if args.only_store_mean:
            n += feat.shape[0]
            feat += feat.sum(0)
        else:
            _feats.append(feat)

    if args.only_store_mean:
        feats = feat / n
    else:
        feats = np.concatenate(_feats)
    if args.only_store_mean:
        # Some models may output high-dimensions
        if feats.ndim == 2:
            np.save(feat_file, feats.mean(0))
        elif feats.ndim == 4:
            np.save(feat_file, feats.mean((2, 3)))
        else:
            tqdm.write("Model output dimensions not clear, saving all.")
            np.save(feat_file, feats)
    else:
        np.save(feat_file, feats)

    time_elapsed = timer() - start
    tqdm.write(f"Extracting '{wsi_path.as_posix()}' took {time_elapsed:.2f}s.")

    if args.log_to_wandb:
        wandb.log(
            {
                "slides": wandb.run.summary.get("slides", 0) + 1,
                "n_tiles": coords.shape[0],
                "time_download_s": time_download,
                "time_inference_s": time_elapsed,
            }
        )

    # write_data_per_tile(slide, data={args.model_name: feats})

    slide.wsi.close()

    if args.remove_slide:
        tqdm.write(f"Removing slide '{wsi_path}'.")
        wsi_path.unlink()

    # Remove lock file
    lock_file.unlink(missing_ok=True)


def get_he_file(wsi_path: Path) -> None:
    import sys

    slide_id = wsi_path.stem
    if wsi_path.exists():
        tqdm.write("File already exists. Skipping...")
        return

    if not args.copy_from_ssh_address:
        tqdm.write("Will download from GTEx portal.")
        u = f"https://gtexportal.org/rest/v1/histology/image?format=json&tissueSampleId={slide_id}"
        url = f"https://brd.nci.nih.gov/brd/imagedownload/{slide_id}"
        with requests.get(u) as r:
            if r.ok:
                try:
                    with requests.get(url) as req:
                        with wsi_path.open("wb") as out:
                            out.write(req.content)
                    return
                except:
                    pass
    else:
        tqdm.write(f"Will copy from '{args.copy_from_ssh_address}'.")
        with paramiko.SSHClient() as ssh:
            ssh.load_host_keys(Path("~/.ssh/known_hosts").expanduser().as_posix())
            ssh.connect(args.copy_from_ssh_address)
            wsi_path.parent.mkdir(exist_ok=True, parents=True)
            with ssh.open_sftp() as sftp:
                prj_dir = Path("~/projects/histopath").expanduser()
                tqdm.write(f"Will copy '{prj_dir / wsi_path}' to '{wsi_path}'.")
                sftp.get((prj_dir / wsi_path).as_posix(), wsi_path.as_posix())
                try:
                    sftp.get((prj_dir / wsi_path).as_posix(), wsi_path.as_posix())
                    return
                except:
                    pass
    tqdm.write(f"Interrupted getting of '{wsi_path}'. Deleting it.")
    if wsi_path.exists():
        wsi_path.unlink()
    sys.exit(1)


def tissue_segment_slide(
    wsi_path: Path, plot: bool = True
) -> tp.Optional[list[list[tuple[int, int]]]]:
    clam_params_file = "https://raw.githubusercontent.com/mahmoodlab/CLAM/master/presets/bwh_biopsy.csv"
    segmentation_json = wsi_path.with_suffix(".contours_tissue.json")
    segmentation_pickle = wsi_path.with_suffix(".contours_tissue.pickle")
    if segmentation_json.exists() and not args.overwrite:
        return

    slide = WholeSlideImage(wsi_path)
    # TODO: optimize parameters for adipose tissue and other tissues with sparse cellularity
    params = pd.read_csv(clam_params_file).squeeze()
    slide.segmentTissue(
        seg_level=slide.wsi.level_count - 1, filter_params=params.to_dict()
    )
    slide.saveSegmentation(segmentation_pickle)
    contour_to_geojson(slide.contours_tissue, segmentation_json)

    if not plot:
        return slide.contours_tissue

    slide.visWSI(-1).save(wsi_path.with_suffix(".contours_tissue.png"))

    # from openslide import OpenSlide
    # oslide = OpenSlide(wsi_path)
    # thumb = oslide.get_thumbnail(slide.level_dim[-1])
    # fig, ax = plt.subplots()
    # ax.imshow(thumb)
    # diff = (np.asarray(slide.level_dim[0]) / np.asarray(slide.level_dim[-1]))[0]
    # ax.plot(*slide.contours_tissue[0].squeeze().T / diff)
    # ax.plot(*slide.contours_tissue[1].squeeze().T / diff)
    # for hole in slide.holes_tissue:  # not working
    #     for col in hole:
    #         ax.plot(*col.T / diff, color='black', linestyle='--')
    # fig.savefig('tissue_segmentation.png')
    return slide.contours_tissue


def contour_to_geojson(
    feats: list[list[tuple[int, int]]], json_file: Path, kind: str = "Tissue"
):
    features = list()
    for feat in feats:
        features.append(
            {
                "type": "Feature",
                "geometry": {
                    "type": "LineString",
                    # "coordinates": [[int(y) for y in x[0]] for x in feat],
                    "coordinates": [
                        [int(x[0]), int(x[1])] for x in feat.squeeze().tolist()
                    ],
                },
                "properties": {"name": kind, "area": Polygon(feat.squeeze()).area},
            }
        )
    output = {"type": "FeatureCollection", "features": features}
    json.dump(output, json_file.open("w"), indent=4)


def get_tile_coordinates(slide):
    hdf5_file = slide.wsi._filename.with_suffix(".h5")
    with h5py.File(hdf5_file) as h5:
        return h5["coords"][()]


def write_data_per_tile(slide, data: dict):
    hdf5_file = slide.wsi._filename.with_suffix(".h5")
    for k, v in data.items():
        with h5py.File(hdf5_file, "a") as h5:
            h5.create_dataset(k, data=v)


def batched(iterable, n):
    """Batch data into lists of length n. The last batch may be shorter."""
    from itertools import islice

    # batched('ABCDEFG', 3) --> ABC DEF G
    it = iter(iterable)
    while True:
        batch = list(islice(it, n))
        if not batch:
            return
        yield batch


def guess_input_dims(model):
    for i in range(224, 519):
        try:
            with torch.no_grad():
                _ = model(torch.randn(2, 3, i, i).to(args.device))
        except (AssertionError, RuntimeError):
            pass
        else:
            return i
    raise RuntimeError("Could not guess input dimensions.")


if __name__ == "__main__" and "quit" not in locals():
    args = get_args()
    main()
