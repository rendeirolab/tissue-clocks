import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.utils import Path

plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["text.usetex"] = False


datasets_dir = Path("data") / "gtex" / "datasets"
results_dir = (Path("results") / "gtex" / "model_training").mkdir()
fs = list(datasets_dir.glob("*/models/*.csv"))

dataset_sizes = {
    ds.name: len(list((ds / "train").glob("*.jpg")))
    for ds in datasets_dir.glob("*")
    if ds.is_dir()
}

# Gather metrics
df = pd.concat([pd.read_csv(f) for f in fs])
df = df.sort_values(["model", "dataset", "epoch"])
for time in ["train_time", "valid_time"]:
    df[f"cum_{time}"] = df.groupby(["model", "dataset"])[time].cumsum()
df["examples"] = df["dataset"].replace(dataset_sizes)
df["cum_examples"] = df.groupby(["model", "dataset"])["examples"].cumsum()
# TODO: convert epochs to GFLOPS
df.to_csv(results_dir / "train_metrics.csv", index=False)


# Plot epochs vs loss/metrics
for xvar in ["epoch", "cum_examples", "cum_train_time"]:
    for metric in [
        "train_loss",
        "train_error_rate",
        "valid_loss",
        "valid_error_rate",
    ]:
        df = pd.read_csv(results_dir / "train_metrics.csv")
        df = df.loc[df["epoch"] <= 100]
        # df = df.query("model != 'convnext_large'")
        n_datasets = df["dataset"].nunique()
        fig, axes = plt.subplots(
            2,
            n_datasets,
            figsize=(6 * n_datasets, 4 * 2),
            sharex="row",
            sharey=True,
            gridspec_kw=dict(wspace=0.025, hspace=0.1),
            squeeze=False,
        )
        df["epoch"] += 2
        for axs, dataset in zip(axes.T, df["dataset"].unique()):
            for ax in axs:
                for model in df["model"].unique():
                    df_ = df.query(f"model == '{model}' & dataset == '{dataset}'")
                    if df_.empty:
                        continue
                    ax.plot(df_[xvar], df_[metric], ".-", label=model)
                    ax.set(facecolor="none")
                ax.set(xlabel=xvar.replace("cum_examples", "Examples"))
            axs[0].set(title=dataset)
            axs[0].legend()
            axs[-1].set(xscale="log")
        for ax in axes[:, 0]:
            ax.set(ylabel=metric.replace("_", " ").capitalize())
        sns.despine(fig)
        fig.savefig(
            results_dir / f"train_fine_tune.{xvar}_{metric}.svg",
            dpi=300,
            bbox_inches="tight",
        )

# Get best epoch
df = pd.read_csv(results_dir / "train_metrics.csv")
df2 = df.loc[df.groupby(["model", "dataset"])["valid_error_rate"].idxmin()]
# df2 = df2.query("model != 'convnext_large'")

# Plot time training vs error rate
fig, ax = plt.subplots(figsize=(4.2, 4.0))
ax.scatter(x=df2["cum_train_time"] / 60 / 60, y=df2["valid_error_rate"])
for _, row in df2.iterrows():
    ax.text(
        x=row["cum_train_time"] / 60 / 60,
        y=row["valid_error_rate"],
        s=f"{row['model']} - {row['dataset']}",
    )
ax.set(xlabel="Time (hours)", ylabel="Error rate")
fig.savefig(
    results_dir / "train_fine_tune.time_vs_error.svg",
    dpi=300,
    bbox_inches="tight",
)

# # TODO plot confusion matrices etc for best models
# svg_stack.py --direction=h --margin=20 train_*train_loss*.svg > train_loss.svg
# svg_stack.py --direction=h --margin=20 train_*valid_loss*.svg > valid_loss.svg
# svg_stack.py --direction=h --margin=20 train_*valid_error*.svg > valid_error_rate.svg

# inkscape -d 300 -o train_loss.png train_loss.edited.svg
# inkscape -d 300 -o valid_loss.png valid_loss.edited.svg
# inkscape -d 300 -o valid_error_rate.png valid_error_rate.edited.svg
