#!/usr/bin/env uv --script

# /// script
# dependencies = [
#   "numpy",
#   "pandas",
#   "matplotlib",
#   "seaborn",
#   "scipy",
# ]
# ///

"""
Bland-Altman analysis for external cohort validation.

For each external cohort, compute:
- Mean of predicted and chronological age (x-axis)
- Difference: predicted - chronological age (y-axis)
- Bias (mean difference) with 95% CI
- Limits of agreement: bias ± 1.96 * SD

NOTE: This script was developed on a remote machine. Results paths may
need to be adapted for local use.

Usage:
    uv run python src/revision/bland_altman.py --cohort lungaging-schiller --model virchow2 --organ Lung
    uv run python src/revision/bland_altman.py --cohort all --model virchow2 --organ all
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from src.utils import get_restricted_info


results_dir = Path("results") / "tissue_clocks_revision"
clock_dir = results_dir / "clock"
output_dir = results_dir / "bland_altman"
output_dir.mkdir(exist_ok=True, parents=True)

cohorts = [
    "skinpath",
    "neuropath",
    "lungaging-schiller",
    "colonaging-fennell",
    "brain-healthy-histo",
]

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
]

organs = ["Brain", "Colon", "Lung", "Skin"]


def load_predictions(cohort, model="virchow2"):
    pattern = f"cohort_{cohort}.predictions.ridgecv.0.5mpp.224px.{model}.all_models.csv"
    file = clock_dir / pattern
    if not file.exists():
        # Try alternate pattern with leading underscore
        file = clock_dir / f"_{pattern}"
    if not file.exists():
        print(f"Warning: No prediction file found for cohort {cohort} ({pattern})")
        return None
    df = pd.read_csv(file, index_col=0)
    df["Cohort"] = cohort
    return df


def load_gtex_predictions(model="virchow2"):
    dfs = []
    for organ in organs:
        file = clock_dir / f"ridgecv-cv_preds.0.5mpp.224px.{model}.{organ}.csv"
        if file.exists():
            df = pd.read_csv(file, index_col=0)
            df["Organ"] = organ
            df["Cohort"] = "gtex"
            dfs.append(df)
    if not dfs:
        return None
    return pd.concat(dfs, ignore_index=False)


def compute_bland_altman(pred, true):
    mean = (pred + true) / 2
    diff = pred - true

    n = len(diff)
    bias = np.mean(diff)
    sd = np.std(diff, ddof=1)
    se = sd / np.sqrt(n)

    t_crit = stats.t.ppf(0.975, df=n - 1)
    bias_ci_lower = bias - t_crit * se
    bias_ci_upper = bias + t_crit * se

    loa_upper = bias + 1.96 * sd
    loa_lower = bias - 1.96 * sd

    # Calibration: linear regression of predicted on chronological age
    slope, intercept, r_value, p_value, std_err = stats.linregress(true, pred)

    # 95% CI for slope
    slope_se = std_err
    t_crit_reg = stats.t.ppf(0.975, df=n - 2)
    slope_ci_lower = slope - t_crit_reg * slope_se
    slope_ci_upper = slope + t_crit_reg * slope_se

    # 95% CI for intercept (need to compute SE manually)
    x_mean = np.mean(true)
    ss_x = np.sum((true - x_mean) ** 2)
    intercept_se = sd * np.sqrt(1 / n + x_mean**2 / ss_x)
    intercept_ci_lower = intercept - t_crit_reg * intercept_se
    intercept_ci_upper = intercept + t_crit_reg * intercept_se

    return {
        "n": n,
        "bias": bias,
        "bias_ci_lower": bias_ci_lower,
        "bias_ci_upper": bias_ci_upper,
        "sd": sd,
        "loa_lower": loa_lower,
        "loa_upper": loa_upper,
        "mean_x": mean,
        "diff_y": diff,
        "slope": slope,
        "slope_ci_lower": slope_ci_lower,
        "slope_ci_upper": slope_ci_upper,
        "intercept": intercept,
        "intercept_ci_lower": intercept_ci_lower,
        "intercept_ci_upper": intercept_ci_upper,
        "r_squared": r_value**2,
    }


def plot_bland_altman(ba_result, cohort, model, organ):
    fig, ax = plt.subplots(figsize=(8, 6))

    mean_x = ba_result["mean_x"]
    diff_y = ba_result["diff_y"]
    bias = ba_result["bias"]
    loa_upper = ba_result["loa_upper"]
    loa_lower = ba_result["loa_lower"]

    ax.scatter(mean_x, diff_y, alpha=0.5, s=30)

    ax.axhline(
        y=bias, color="black", linestyle="-", linewidth=1.5, label=f"Bias = {bias:.2f}"
    )
    ax.axhline(
        y=loa_upper,
        color="gray",
        linestyle="--",
        linewidth=1,
        label=f"LOA = {loa_upper:.2f}",
    )
    ax.axhline(
        y=loa_lower,
        color="gray",
        linestyle="--",
        linewidth=1,
        label=f"LOA = {loa_lower:.2f}",
    )

    ax.set_xlabel("Mean of Predicted and Chronological Age", fontsize=12)
    ax.set_ylabel("Predicted - Chronological Age", fontsize=12)
    ax.set_title(
        f"Bland-Altman: {cohort} | {model} | {organ}\n"
        f"n={ba_result['n']}, Bias={bias:.2f} ({ba_result['bias_ci_lower']:.2f} to {ba_result['bias_ci_upper']:.2f}), "
        f"SD={ba_result['sd']:.2f}, LOA=[{loa_lower:.2f}, {loa_upper:.2f}]\n"
        f"Slope={ba_result['slope']:.2f} ({ba_result['slope_ci_lower']:.2f} to {ba_result['slope_ci_upper']:.2f}), "
        f"Intercept={ba_result['intercept']:.2f}, R2={ba_result['r_squared']:.2f}"
    )
    ax.legend(loc="upper right")

    plt.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", default="all", help="Cohort name or 'all'")
    parser.add_argument("--model", default="virchow2", help="Model name or 'all'")
    parser.add_argument("--organ", default="all", help="Organ or 'all'")
    args = parser.parse_args()

    cohorts_to_run = cohorts if args.cohort == "all" else [args.cohort]
    models_to_run = model_names if args.model == "all" else [args.model]
    organs_to_run = organs if args.organ == "all" else [args.organ]

    results = []

    for cohort in cohorts_to_run:
        print(f"\n{'=' * 60}")
        print(f"Processing cohort: {cohort}")
        print(f"{'=' * 60}")

        df = load_predictions(cohort, args.model)
        if df is None:
            continue

        for model in models_to_run:
            for organ in organs_to_run:
                subset = df[(df["Model"] == model) & (df["Organ"] == organ)]
                if len(subset) < 10:
                    print(
                        f"Skipping {cohort}/{model}/{organ}: insufficient samples ({len(subset)})"
                    )
                    continue

                pred = subset["pred"].values
                true = subset["Age"].values

                ba = compute_bland_altman(pred, true)

                result = {
                    "cohort": cohort,
                    "model": model,
                    "organ": organ,
                    "n": ba["n"],
                    "bias": ba["bias"],
                    "bias_ci_lower": ba["bias_ci_lower"],
                    "bias_ci_upper": ba["bias_ci_upper"],
                    "sd": ba["sd"],
                    "loa_lower": ba["loa_lower"],
                    "loa_upper": ba["loa_upper"],
                    "slope": ba["slope"],
                    "slope_ci_lower": ba["slope_ci_lower"],
                    "slope_ci_upper": ba["slope_ci_upper"],
                    "intercept": ba["intercept"],
                    "intercept_ci_lower": ba["intercept_ci_lower"],
                    "intercept_ci_upper": ba["intercept_ci_upper"],
                    "r_squared": ba["r_squared"],
                }
                results.append(result)

                fig = plot_bland_altman(ba, cohort, model, organ)
                fig.savefig(
                    output_dir / f"{cohort}_{model}_{organ}_bland_altman.svg", dpi=300
                )
                plt.close(fig)

                print(
                    f"{cohort}/{model}/{organ}: n={ba['n']}, "
                    f"bias={ba['bias']:.2f} [{ba['bias_ci_lower']:.2f}, {ba['bias_ci_upper']:.2f}], "
                    f"LOA=[{ba['loa_lower']:.2f}, {ba['loa_upper']:.2f}], "
                    f"slope={ba['slope']:.2f} [{ba['slope_ci_lower']:.2f}, {ba['slope_ci_upper']:.2f}], "
                    f"intercept={ba['intercept']:.2f} [{ba['intercept_ci_lower']:.2f}, {ba['intercept_ci_upper']:.2f}], "
                    f"R2={ba['r_squared']:.2f}"
                )

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_dir / "summary.csv", index=False)
    print(f"\nSummary saved to {output_dir / 'summary.csv'}")

    # Generate combined BA grid plot
    generate_combined_ba_grid(args.model, args.cohort)


def generate_combined_ba_grid(model="virchow2", exclude_cohort="all"):
    # Load GTex predictions
    gtex_df = load_gtex_predictions(model)
    if gtex_df is None:
        print("Warning: No GTEx predictions found for combined grid")
        return

    # Load all external cohort predictions
    external_dfs = {}
    for cohort in cohorts:
        if exclude_cohort != "all" and cohort == exclude_cohort:
            continue
        df = load_predictions(cohort, model)
        if df is not None:
            external_dfs[cohort] = df

    # Build combined dataframe for all organ/cohort combinations
    ba_data = []

    # Add GTEx (as first column)
    for organ in organs:
        subset = gtex_df[gtex_df["Organ"] == organ]
        if len(subset) >= 10:
            pred = subset["pred"].values
            true = subset["Age"].values
            ba = compute_bland_altman(pred, true)
            ba_data.append({"cohort": "GTEx (train)", "organ": organ, **ba})

    # Add external cohorts
    for cohort, df in external_dfs.items():
        for organ in organs:
            subset = df[df["Organ"] == organ]
            if len(subset) >= 10:
                pred = subset["pred"].values
                true = subset["Age"].values
                ba = compute_bland_altman(pred, true)
                ba_data.append({"cohort": cohort, "organ": organ, **ba})

    if not ba_data:
        print("Warning: No BA data for combined grid")
        return

    ba_all = pd.DataFrame(ba_data)

    # Create grid: rows = organs (GTEx tissues), cols = cohorts
    cohort_order = ["GTEx (train)"] + [c for c in cohorts if c in external_dfs]

    n_rows = len(organs)
    n_cols = len(cohort_order)
    fig, axes = plt.subplots(
        nrows=n_rows,
        ncols=n_cols,
        figsize=(3 * n_cols, 2.5 * n_rows),
        squeeze=False,
    )

    for i, organ in enumerate(organs):
        for j, cohort in enumerate(cohort_order):
            ax = axes[i, j]
            subset = ba_all[(ba_all["cohort"] == cohort) & (ba_all["organ"] == organ)]

            if len(subset) == 0:
                ax.text(
                    0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes
                )
                ax.set_title(f"{cohort}\n{organ}")
                continue

            row = subset.iloc[0]
            mean_x = row["mean_x"]
            diff_y = row["diff_y"]

            # Calculate symmetric color limits for RdBu_r
            vmax = max(abs(diff_y.max()), abs(diff_y.min()))
            scatter = ax.scatter(
                mean_x,
                diff_y,
                c=diff_y,
                cmap="coolwarm",
                vmin=-vmax,
                vmax=vmax,
                alpha=0.75,
                s=20,
                edgecolors="none",
            )
            ax.axhline(y=row["bias"], color="black", linestyle="-", linewidth=1)
            ax.axhline(y=row["loa_upper"], color="gray", linestyle="--", linewidth=0.8)
            ax.axhline(y=row["loa_lower"], color="gray", linestyle="--", linewidth=0.8)

            if j == 0:
                ax.set_ylabel(f"{organ}\nError (years)", fontsize=9)
            if i == 0:
                ax.set_title(f"{cohort}", fontsize=10)
            if i == n_rows - 1:
                ax.set_xlabel("Mean age", fontsize=9)

            # Add metrics as text
            n = row["n"]
            bias = row["bias"]
            loa_lower = row["loa_lower"]
            loa_upper = row["loa_upper"]
            ax.text(
                0.05,
                0.95,
                f"n={n:.0f}\nb={bias:.1f}\nLOA=[{loa_lower:.0f}, {loa_upper:.0f}]",
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=6,
            )

            # Set consistent axis limits
            all_means = (
                ba_all["mean_x"]
                .apply(
                    lambda x: np.array(x) if isinstance(x, (list, np.ndarray)) else x
                )
                .apply(len)
                .max()
            )
            all_diffs = ba_all.apply(
                lambda r: max(abs(r["loa_lower"]), abs(r["loa_upper"])), axis=1
            ).max()
            ax.set_xlim(0, 90)
            ax.set_ylim(-all_diffs * 1.1, all_diffs * 1.1)

    plt.tight_layout()
    fig.savefig(output_dir / f"combined_bland_altman_{model}.svg", dpi=300)
    plt.close()
    print(
        f"\nCombined BA grid saved to {output_dir / f'combined_bland_altman_{model}.svg'}"
    )


if __name__ == "__main__":
    main()
