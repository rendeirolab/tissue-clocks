from pathlib import Path
import argparse

import fastai
from fastai.vision.all import (
    L,
    DataBlock,
    ImageBlock,
    CategoryBlock,
    aug_transforms,
    vision_learner,
    error_rate,
    Resize,
)
from fastai.distributed import rank_distrib
from fastai.callback.wandb import WandbCallback
from fastai.callback.all import SaveModelCallback
from torch.distributed.elastic.multiprocessing.errors import record

import wandb


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(dest="path", type=Path, help="Path to dataset")
    parser.add_argument(dest="model", type=str, help="Name of model to use")
    parser.add_argument(
        "--tile-size",
        type=int,
        default=224,
        help="Width of tile to use.",
    )
    parser.add_argument(
        "-c",
        "--checkpoint",
        type=Path,
        default=None,
        help="Optional checkpoint file to load.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=12,
        help="Number of epochs to train for.",
    )
    parser.add_argument(
        "--learn-rate",
        type=float,
        default=None,
        help="Optional learning rate to use.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size to use.",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Use mixed precision for training.",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="Which GPU to use in not in distributed mode.",
    )
    parser.add_argument(
        "--cpus",
        type=int,
        default=4,
        help="CPUs to use.",
    )
    parser.add_argument(
        "--only-find-learn-rate",
        action="store_true",
        help="Only find learning rate. Run without distributed mode.",
    )
    parser.add_argument(
        "--write-learn-rate-to",
        type=Path,
        default=None,
        help="File to write learning rate to.",
    )
    args = parser.parse_args()
    args.precision = "fp16" if args.fp16 else "fp32"
    args.name = f"{args.path.name}_{args.model}_fine_tune.{args.tile_size}px.{args.batch_size}b.{args.precision}.{args.epochs}e"
    if args.only_find_learn_rate:
        if args.write_learn_rate_to is None:
            args.write_learn_rate_to = (
                args.path / "models" / (f"{args.name}.learn_rate.txt")
            )
    return args


# cli = dict(
#     path=Path("data/gtex/datasets/gtex_balanced_stratified_3_slides_200_tiles/"),
#     tile_size=64,
#     model="resnet50",
#     epochs=12,
#     checkpoint=None,
#     learn_rate=None,
#     batch_size=64,
#     gpu=0,
#     cpus=4,
#     fp16=True,
#     only_find_learn_rate=False,
# )
# args = argparse.Namespace(**cli)

wandb.require(experiment="service")


@record
def train():
    args = get_args()
    cbs = []
    if rank_distrib() == 0 and (not args.only_find_learn_rate):
        run = wandb.init(
            project="histopath_tile_level_supervised_training",
            name=args.name,
            id=args.name,
            resume="auto",
            config=args,
        )
        cbs = [
            WandbCallback(log_model=True),
            SaveModelCallback(monitor="error_rate", every_epoch=True, fname=args.name),
        ]

    block = DataBlock(
        blocks=[ImageBlock, CategoryBlock],
        get_y=get_tissue_class,
        splitter=splitter,
        item_tfms=Resize(args.tile_size),
        batch_tfms=aug_transforms(),
    )
    dls = block.dataloaders(
        L(args.path.glob("*/*.jpg")),
        path=args.path,
        num_workers=args.cpus,
        bs=args.batch_size,
    )

    if rank_distrib() == 0:
        print(f"Using dataset from {args.path}")
        print(f"Train size: {len(dls.train_ds)}")
        print(f"Validation size: {len(dls.valid_ds)}")

    if hasattr(fastai.vision.all, args.model):
        model = getattr(fastai.vision.all, args.model)
    else:
        model = args.model
    learn = vision_learner(dls, model, metrics=error_rate, cbs=cbs)

    if args.fp16:
        learn = learn.to_fp16()

    epoch = 0
    if args.checkpoint is not None:
        load_checkpoint(learn, args.checkpoint)
        epoch = int(args.checkpoint.stem.split("_")[-1])
        print(
            f"Continuing at epoch {epoch} with previous model checkpoint file: '{args.checkpoint}'"
        )

    # lr = 0.0004786300996784121
    # lr = 0.002511886414140463
    if args.learn_rate is None:
        sug = learn.lr_find()
        lr = sug.valley
        print(f"\nFound learning rate: {lr}\n")
    else:
        lr = args.learn_rate
        print(f"Using learning rate: {lr}")

    if args.only_find_learn_rate:
        with open(args.write_learn_rate_to, "w") as f:
            f.write(f"{lr}")
        print(f"Wrote learn rate to '{args.write_learn_rate_to}'.")
        print("Finished.")
        return

    if rank_distrib() == 0:
        print("Training.")
    # print(["epoch", "train_loss", "valid_loss", "error_rate", "time"])
    with learn.distrib_ctx(sync_bn=False):
        # with learn.no_bar():  # for log-friendly output
        learn.fine_tune(
            epochs=args.epochs, freeze_epochs=1, base_lr=lr, start_epoch=epoch
        )
    wandb.finish()
    if rank_distrib() == 0:
        print("Finished training.")

    # Export model for inference
    if rank_distrib() == 0:
        checkpoint_file = args.path / "models" / f"{args.name}_{args.epochs - 1}.pth"
        export_model(learn.model, checkpoint_file)
        print("Exported model for inference.")


def visualize_model(model):
    import torch
    from torchviz import make_dot

    yhat = model(torch.randn(2, 3, 224, 224))
    make_dot(yhat, params=dict(list(model.named_parameters()))).render(
        "torchviz", format="png"
    )


def export_model(model, checkpoint_file):
    import torch

    model.load_state_dict(torch.load(checkpoint_file, map_location=torch.device("cpu")))
    idx = find_last_linear_layer(model)
    nm = model[-2] + model[-1][:-idx]
    torch.save(nm, checkpoint_file.with_suffix(".pkl"))


def find_last_linear_layer(model):
    import torch

    for i, layer in enumerate(model[-1]):
        if isinstance(layer, torch.nn.modules.linear.Linear):
            return i


def sorted_nicely(ls):
    import re

    def convert(text):
        return int(text) if text.isdigit() else text

    def alphanum_key(key):
        return [convert(c) for c in re.split("([0-9]+)", key)]

    return sorted(ls, key=alphanum_key)


def load_checkpoint(learn, checkpoint_file: Path = None, epoch: int = None):
    import torch

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

    print(f"Loaded checkpoint file: {checkpoint_file}")


def splitter(items):
    train = [i for i, f in enumerate(items) if f.parent.name == "train"]
    valid = [i for i, f in enumerate(items) if f.parent.name == "valid"]
    test = [i for i, f in enumerate(items) if f.parent.name == "test"]
    return train, valid, test


def get_individual_id(f):
    if isinstance(f, Path):
        return f.stem.split(".")[0]
    if isinstance(f, str):
        return f.split(".")[0]
    raise ValueError()


def get_tissue_class(f):
    if isinstance(f, Path):
        return f.stem.split(".")[1]
    if isinstance(f, str):
        return f.split(".")[1]
    raise ValueError()


if __name__ == "__main__" and ("quit" not in locals()):
    train()

# # Running
# accelerate launch src/train_fastai.py \
#     --tile-size 64 data/gtex/datasets/gtex_balanced_stratified_10_slides_800_tiles resnet50 --batch-size 512 --cpus 16

# DATASET_DIR=data/gtex/datasets
# DATASET=gtex_balanced_stratified_10_slides_800_tiles
# MODELS=(
# alexnet
# resnet18
# resnet50
# convnext_tiny
# convnext_large
# # levit_128
# )
# # set -e
# for MODEL in ${MODELS[@]}; do
#     echo "Running model: ${MODEL}"
#     LR_F=${DATASET_DIR}/${DATASET}/models/${DATASET}_${MODEL}_fine_tune.64px.512b.fp16.12e.learn_rate.txt
#     if [ ! -f ${LR_F} ]; then
#         python3 src/train_fastai.py \
#             --tile-size 64 --batch-size 512 --fp16 --cpus 48 --only-find-learn-rate data/gtex/datasets/${DATASET} $MODEL
#     fi
#     LR=`cat ${LR_F}`
#     echo "Learning rate: ${LR}"
#     accelerate launch src/train_fastai.py --tile-size 64 --batch-size 512 --fp16 --cpus 48 --epochs 100 data/${DATASET} $MODEL --learn-rate $LR
# done

# # Saving models
# models = [
#     "alexnet",
#     "resnet18",
#     "resnet34",
#     "resnet152",
#     "convnext_tiny",
#     "convnext_base",
#     "convnext_large"]
# for model_name in models:
#     model = getattr(fastai.vision.all, model_name)
#     learn = vision_learner(dls, model, metrics=error_rate, cbs=cbs)
#     checkpoint_file = (
#         args.path
#         / "models"
#         / (args.path.name + "_" + model_name + "_fine_tune_11.pth")
#     )
#     export_model(learn.model, checkpoint_file)
#     print("Exported model for inference.")
