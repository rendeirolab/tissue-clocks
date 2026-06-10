from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

results_dir = Path("results") / "gtex" / "predict_gaps_from_blood_expression"
output_dir = (
    Path("results") / "tissue_clocks_revision" / "predict_gaps_from_blood_expression"
)
output_dir.mkdir(parents=True, exist_ok=True)
metrics_file = (
    results_dir
    / "tissue-specific_clocks.Ridge.KFold.predictions_residuals.predicted_from_blood_expression.age_regressed.residuals_adj.metrics.csv"
)
metrics = pd.read_csv(metrics_file, index_col=0).query(
    "~Tissue.isin(['std', 'sum', 'max'])"
)

# Rename tissue 'mean' to "Systemic"
metrics = metrics.replace({"Tissue": {"mean": "Systemic"}})

# Make Pearson r metric derived from R2
metrics["all_r"] = metrics["all_r2"].apply(
    lambda r2: r2**0.5 if r2 >= 0 else -((-r2) ** 0.5)
)

# Make new metric which is ratio of MAE for Shuffled=False over Shuffled=True for each tissue
real = metrics.query("Shuffled == False").set_index("Tissue")
shuffled = metrics.query("Shuffled == True").set_index("Tissue")
metrics["MAE_ratio"] = [
    shuffled.loc[tissue, "MAE"] / real.loc[tissue, "MAE"]
    for tissue in metrics["Tissue"]
]

# Make new metric which is ratio of R2 for Shuffled=False over Shuffled=True for each tissue
metrics["R2_ratio"] = [
    real.loc[tissue, "all_r2"] / shuffled.loc[tissue, "all_r2"]
    for tissue in metrics["Tissue"]
]

# Order tissues by performance
metrics = metrics.sort_values(by="MAE_ratio", ascending=False).reset_index(drop=True)


# Make figure with subplots, barplots for each metric, y-axis is tissue, x-axis is metric value
# Overlay bars for Shuffled=True and Shuffled=False
names = ["n_samples", "MAE", "all_r", "all_r2", "MAE_ratio", "R2_ratio"]
fig, axes = plt.subplots(
    1,
    len(names),
    figsize=(len(names) * 1.5, 5),
    sharey=True,
    gridspec_kw={"wspace": 0.05},
)
for i, name in enumerate(names):
    ax = axes[i]
    sns.barplot(
        data=metrics,
        x=name,
        y="Tissue",
        hue="Shuffled" if name in ["MAE", "all_r", "all_r2"] else None,
        palette="Set2" if name in ["MAE", "all_r", "all_r2"] else None,
        # color="blue" if name in ["MAE_ratio", "R2_ratio"] else None,
        ax=ax,
    )
    if name in ["MAE_ratio", "R2_ratio"]:
        ax.axvline(1, color="black", linestyle="--", linewidth=1)
    ax.set_title(name)
    ax.legend(title="Shuffled", loc="lower right")
fig.tight_layout()
fig.savefig(output_dir / "metrics_comparison.svg", dpi=300, bbox_inches="tight")
