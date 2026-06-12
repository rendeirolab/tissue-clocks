#!/usr/bin/env uv --script

# /// script
# dependencies = [
#   "tqdm",
#   "joblib",
#   "numpy",
#   "pandas",
#   "anndata",
#   "matplotlib",
#   "seaborn",
#   "scikit-learn",
#   "pingouin",
# ]
# ///

# SPDX-FileCopyrightText: Copyright (c) 2026 Rendeiro Lab, CeMM - Research Center for Molecular Medicine, Austrian Academy of Sciences
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#
# Licensed under the PolyForm Noncommercial License 1.0.0 (see LICENSE).

"""
ICC analysis script - computes ICC across sampling fractions.

NOTE: This script was run on a remote machine and references machine-specific
paths. It is included for transparency and is not expected to run outside
the original machine.

Usage:
    uv run python src/revision/icc_apply_clocks.py --model virchow2 --organ Lung
"""

import argparse
from pathlib import Path

from tqdm import tqdm
from joblib import load
import numpy as np
import pandas as pd
import anndata as ad
import matplotlib.pyplot as plt
import seaborn as sns
import pingouin as pg

from src.utils import get_restricted_info


processed_dir = Path("processed/histopathology")
icc_sampled_dir = processed_dir / "icc_sampled_v2"
results_dir = Path("results") / "tissue_clocks_revision"
clock_dir = results_dir / "clock"
icc_dir = results_dir / "icc_v2"
icc_dir.mkdir(exist_ok=True, parents=True)

mpp = 0.5
tile_width = 224

fractions = [0.0001, 0.001, 0.01, 0.1, 0.5]
n_seeds = 10


def load_meta():
    meta_file = Path("data") / "gtex" / "GTEx Portal.csv"
    meta = pd.read_csv(meta_file, index_col=0)
    rest, _ = get_restricted_info()
    meta = meta.merge(
        rest[["Age", "Cohort"]], left_on="Subject ID", right_index=True, how="left"
    )
    return meta


def load_clocks(model_name: str, organ: str, ml_type: str = "ridgecv"):
    suffix = f"{mpp}mpp.{tile_width}px.{model_name}"
    scaler = load(clock_dir / f"{ml_type}-scaler.{suffix}.{organ}.joblib")
    model = load(clock_dir / f"{ml_type}-final_model.{suffix}.{organ}.joblib")
    calibrator = load(clock_dir / f"{ml_type}-calibrator.{suffix}.{organ}.joblib")
    return scaler, model, calibrator


def predict_age(features, scaler, model, calibrator):
    x_scaled = scaler.transform(features)
    pred_raw = model.predict(x_scaled)
    pred = calibrator.predict(pred_raw.reshape(-1, 1))
    return pred.flatten()


def compute_icc(predictions_df, n_seeds=10):
    """Compute ICC3 for predictions DataFrame with seed column."""
    # Reshape to long format for pingouin
    samples = predictions_df["sample_id"].unique()
    n_samples = len(samples)

    # For each sample, get predictions across seeds
    long_data = []
    for sample_id in samples:
        subset = predictions_df[predictions_df["sample_id"] == sample_id]
        true_age = subset["true_age"].iloc[0]
        for _, row in subset.iterrows():
            long_data.append(
                {
                    "subject": sample_id,
                    "rater": row["seed"],
                    "rating": row["pred"],
                    "true_age": true_age,
                }
            )

    long_df = pd.DataFrame(long_data)

    icc = pg.intraclass_corr(
        data=long_df, targets="subject", raters="rater", ratings="rating"
    )
    return icc


def run_icc_analysis(
    model_name: str,
    organ: str,
    ml_type: str = "ridgecv",
):
    suffix = f"{mpp}mpp.{tile_width}px.{model_name}"

    if not (clock_dir / f"{ml_type}-scaler.{suffix}.{organ}.joblib").exists():
        print(f"Skipping {model_name} {organ}: no trained clocks found")
        return None

    print(f"\n{'=' * 60}")
    print(f"Running ICC analysis: {model_name} - {organ}")
    print(f"{'=' * 60}")

    scaler, model, calibrator = load_clocks(model_name, organ, ml_type)
    meta = load_meta()

    results = []

    for frac in fractions:
        h5ad_path = icc_sampled_dir / f"{organ}.{model_name}.fraction{frac}.h5ad"
        if not h5ad_path.exists():
            print(f"Skipping fraction {frac}: file not found")
            continue

        print(f"\nProcessing fraction {frac}...")
        a = ad.read_h5ad(h5ad_path)

        sample_ids = a.obs["sample_id"].values
        seeds = a.obs["seed"].values

        # Predict age for all samples
        preds = predict_age(a.X, scaler, model, calibrator)

        # Build predictions DataFrame
        pred_df = pd.DataFrame(
            {
                "sample_id": sample_ids,
                "seed": seeds,
                "pred": preds,
            }
        )

        # Get true ages
        true_ages = []
        for sid in pred_df["sample_id"]:
            if sid in meta.index:
                true_ages.append(meta.loc[sid, "Age"])
            else:
                true_ages.append(np.nan)
        pred_df["true_age"] = true_ages

        # Count valid samples (unique sample IDs with valid age)
        n_valid = pred_df["true_age"].notna().sum()
        n_unique_samples = pred_df[pred_df["true_age"].notna()]["sample_id"].nunique()
        n_samples_per_seed = n_unique_samples

        print(
            f"  Total: {len(pred_df)} predictions, {n_unique_samples} unique samples per seed"
        )

        if len(pred_df) < 10 or n_unique_samples < 10:
            print(f"  Skipping: insufficient data")
            continue

        # Compute ICC
        icc_results = compute_icc(pred_df, n_seeds)

        icc_row = icc_results[icc_results["Type"] == "ICC3"].iloc[0]
        icc_value = icc_row["ICC"]
        ci_lower = icc_row["CI95%"][0]
        ci_upper = icc_row["CI95%"][1]

        results.append(
            {
                "fraction": frac,
                "n_predictions": len(pred_df),
                "n_unique_samples": n_unique_samples,
                "icc": icc_value,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
            }
        )

        print(f"  ICC = {icc_value:.3f} (95% CI: {ci_lower:.3f}-{ci_upper:.3f})")

        # Save predictions for this fraction
        pred_df.to_csv(
            icc_dir / f"{model_name}_{organ}_fraction{frac}.predictions.csv",
            index=False,
        )

    if not results:
        print("No results generated")
        return None

    results_df = pd.DataFrame(results)
    results_df.to_csv(
        icc_dir / f"{model_name}_{organ}_icc_by_fraction.csv", index=False
    )

    # Plot ICC curve
    fig, ax = plt.subplots(figsize=(5, 4))

    ax.plot(
        results_df["fraction"], results_df["icc"], "o-", markersize=10, linewidth=2.5
    )
    ax.fill_between(
        results_df["fraction"],
        results_df["ci_lower"],
        results_df["ci_upper"],
        alpha=0.2,
    )

    ax.set_xscale("log")
    ax.set_xlabel("Sampling Fraction", fontsize=14)
    ax.set_ylabel("ICC", fontsize=14)
    ax.set_title(f"ICC vs Sampling Fraction: {model_name} - {organ}", fontsize=14)

    # Add grey vertical lines at each sampling point
    for frac in results_df["fraction"]:
        ax.axvline(x=frac, color="gray", linestyle="-", linewidth=0.5, alpha=0.5)

    # Set x-axis ticks to match fractions and add minor ticks
    ax.set_xticks(results_df["fraction"])
    ax.set_xticklabels([str(f) for f in results_df["fraction"]], fontsize=11)
    ax.tick_params(axis="x", which="minor", length=3)

    # Grid synchronized with ticks
    ax.grid(True, which="major", alpha=0.4)
    ax.grid(True, which="minor", alpha=0.15)

    # Add annotations for sample sizes
    for i, row in results_df.iterrows():
        ax.annotate(
            f"n={row['n_unique_samples']}",
            (row["fraction"], row["icc"]),
            textcoords="offset points",
            xytext=(0, 12),
            ha="center",
            fontsize=11,
        )

    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig(icc_dir / f"{model_name}_{organ}_icc_curve.svg", dpi=300)
    plt.close()

    print(f"\nResults saved to {icc_dir}")
    print(results_df.to_string(index=False))

    return results_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="virchow2")
    parser.add_argument("--organ", default="Lung")
    parser.add_argument("--ml-type", default="ridgecv")
    args = parser.parse_args()

    result = run_icc_analysis(
        model_name=args.model,
        organ=args.organ,
        ml_type=args.ml_type,
    )


if __name__ == "__main__":
    main()
