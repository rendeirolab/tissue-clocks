"""
ICC reproducibility analysis for tissue clocks.

Randomly sample 50% of tiles per slide, aggregate, and predict age.
Repeat 5 times per slide to assess within-slide reproducibility.

NOTE: This script uses hardcoded machine-specific paths and is included
for transparency. It is not expected to run outside the original machine.

Usage:
    uv run python src/revision/icc_reproducibility.py --model virchow2 --organ Lung
    uv run python src/revision/icc_reproducibility.py --model virchow2 --organ all
    uv run python src/revision/icc_reproducibility.py --model all --organ all
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
from sklearn.preprocessing import StandardScaler
import pingouin as pg

from src.utils import get_restricted_info


processed_dir = Path("/data/projects/tissueclocks/processed/histopathology")
results_dir = Path("results") / "tissue_clocks_revision"
clock_dir = results_dir / "clock"
icc_dir = results_dir / "icc"
icc_dir.mkdir(exist_ok=True, parents=True)
results_dir.mkdir(exist_ok=True, parents=True)

mpp = 0.5
tile_width = 224

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

meta_file = Path("data") / "gtex" / "GTEx Portal.csv"
meta = pd.read_csv(meta_file, index_col=0)
meta = meta.query(
    "Tissue.str.startswith('Skin') | (Tissue == 'Brain - Cortex') | Tissue.str.contains('Colon') | Tissue.str.startswith('Lung')"
)
rest, _ = get_restricted_info()
meta = meta.merge(
    rest[["Age", "Cohort"]], left_on="Subject ID", right_index=True, how="left"
)


def load_clocks(model_name: str, organ: str, ml_type: str = "ridge"):
    suffix = f"{mpp}mpp.{tile_width}px.{model_name}"
    scaler = load(clock_dir / f"scaler.{suffix}.{organ}.joblib")
    model = load(clock_dir / f"{ml_type}_final_model.{suffix}.{organ}.joblib")
    calibrator = load(clock_dir / f"calibrator.{suffix}.{organ}.joblib")
    return scaler, model, calibrator


def get_gtex_zarrs(organ: str):
    zarrs = sorted(processed_dir.glob("GTEX-*.zarr"))
    organ_tissue_map = {
        "Brain": "Brain - Cortex",
        "Colon": "Colon - Transverse",
        "Lung": "Lung",
        "Skin": "Skin - Lower leg",
    }
    target_tissue = organ_tissue_map.get(organ, organ)
    valid_subjects = meta[meta["Tissue"] == target_tissue].index.tolist()
    valid_zarrs = [z for z in zarrs if z.stem in valid_subjects]
    return valid_zarrs


def predict_age(features, scaler, model, calibrator):
    x_scaled = scaler.transform(features)
    pred_raw = model.predict(x_scaled)
    pred = calibrator.predict(pred_raw.reshape(-1, 1))
    return pred.flatten()


def compute_icc(predictions_df):
    predictions_df = predictions_df.reset_index(drop=True)
    long_df = predictions_df.melt(
        var_name="iteration", value_name="predicted_age", ignore_index=False
    )
    long_df["subject"] = long_df.index
    long_df = long_df.reset_index(drop=True)
    icc = pg.intraclass_corr(
        data=long_df, targets="subject", raters="iteration", ratings="predicted_age"
    )
    return icc


def run_icc_analysis(
    model_name: str,
    organ: str,
    sample_fraction: float = 0.5,
    n_iterations: int = 5,
    ml_type: str = "ridge",
):
    suffix = f"{mpp}mpp.{tile_width}px.{model_name}"

    if not (clock_dir / f"scaler.{suffix}.{organ}.joblib").exists():
        print(f"Skipping {model_name} {organ}: no trained clocks found")
        return None

    print(f"\n{'=' * 60}")
    print(f"Running ICC analysis: {model_name} - {organ}")
    print(f"{'=' * 60}")

    scaler, model, calibrator = load_clocks(model_name, organ, ml_type)
    zarrs = get_gtex_zarrs(organ)
    print(f"Found {len(zarrs)} GTEx samples for {organ}")

    all_preds = []
    for zarr_path in tqdm(zarrs, desc=f"Processing {organ} samples"):
        sample_id = zarr_path.stem
        try:
            a = ad.read_zarr(zarr_path, observer=f"{model_name}_tiles")
        except Exception as e:
            print(f"Warning: Could not load {zarr_path.name}: {e}")
            continue

        if a.n_obs < 10:
            continue

        tile_indices = np.arange(a.n_obs)
        n_tiles_to_sample = max(1, int(len(tile_indices) * sample_fraction))

        iteration_preds = []
        for i in range(n_iterations):
            sampled_idx = np.random.choice(
                tile_indices, size=n_tiles_to_sample, replace=False
            )
            features = a.X[sampled_idx].toarray().mean(axis=0).reshape(1, -1)
            pred = predict_age(features, scaler, model, calibrator)[0]
            iteration_preds.append(pred)

        true_age = meta.loc[sample_id, "Age"]
        all_preds.append(
            {
                "sample_id": sample_id,
                "true_age": true_age,
                **{f"iter_{i}": p for i, p in enumerate(iteration_preds)},
            }
        )

    if not all_preds:
        print(f"No valid predictions for {model_name} {organ}")
        return None

    pred_df = pd.DataFrame(all_preds)
    iteration_cols = [f"iter_{i}" for i in range(n_iterations)]

    pred_df.to_csv(
        icc_dir / f"{model_name}_{organ}.predictions.csv",
        index=False,
    )

    icc_results = compute_icc(pred_df[iteration_cols])
    icc_value = icc_results.loc[icc_results["Type"] == "ICC3", "ICC"].values[0]
    ci_lower = (
        icc_results.loc[icc_results["Type"] == "ICC3", "CI95%"]
        .values[0]
        .split("-")[0]
        .strip("[] ")
    )
    ci_upper = (
        icc_results.loc[icc_results["Type"] == "ICC3", "CI95%"]
        .values[0]
        .split("-")[1]
        .strip("[] ")
    )

    result = pd.DataFrame(
        {
            "model": [model_name],
            "organ": [organ],
            "n_samples": [len(pred_df)],
            "icc": [icc_value],
            "ci_lower": [ci_lower],
            "ci_upper": [ci_upper],
        }
    )
    result.to_csv(icc_dir / f"{model_name}_{organ}.icc.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    pred_df_melted = pred_df.melt(
        id_vars=["sample_id", "true_age"],
        value_vars=iteration_cols,
        var_name="iteration",
        value_name="predicted_age",
    )
    sns.boxplot(data=pred_df_melted, x="sample_id", y="predicted_age", ax=axes[0])
    axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=90, fontsize=6)
    axes[0].set_xlabel("Sample")
    axes[0].set_ylabel("Predicted Age")
    axes[0].set_title(f"{model_name} - {organ}\nICC={icc_value:.3f}")

    axes[1].scatter(
        pred_df["true_age"], pred_df[iteration_cols].mean(axis=1), alpha=0.6
    )
    min_val = min(pred_df["true_age"].min(), pred_df[iteration_cols].mean().min())
    max_val = max(pred_df["true_age"].max(), pred_df[iteration_cols].mean().max())
    axes[1].plot([min_val, max_val], [min_val, max_val], "k--", label="y=x")
    axes[1].set_xlabel("Chronological Age")
    axes[1].set_ylabel("Mean Predicted Age")
    axes[1].set_title(f"Mean prediction vs True age")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(icc_dir / f"{model_name}_{organ}.icc_plot.svg", dpi=300)
    plt.close()

    print(f"ICC = {icc_value:.3f} (95% CI: {ci_lower}-{ci_upper})")
    print(f"Results saved to {icc_dir}")

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="virchow2", help="Model name")
    parser.add_argument("--organ", default="Lung", help="Organ (or 'all')")
    parser.add_argument(
        "--sample-fraction", type=float, default=0.5, help="Fraction of tiles to sample"
    )
    parser.add_argument(
        "--n-iterations", type=int, default=5, help="Number of iterations per sample"
    )
    parser.add_argument(
        "--ml-type", default="ridge", help="ML type used for clocks (e.g., ridge, lgbm)"
    )
    args = parser.parse_args()

    if args.organ == "all":
        organs_to_run = organs
    else:
        organs_to_run = [args.organ]

    if args.model == "all":
        models_to_run = model_names
    else:
        models_to_run = [args.model]

    results = []
    for model_name in models_to_run:
        for organ in organs_to_run:
            result = run_icc_analysis(
                model_name=model_name,
                organ=organ,
                sample_fraction=args.sample_fraction,
                n_iterations=args.n_iterations,
                ml_type=args.ml_type,
            )
            if result is not None:
                results.append(result)

    if results:
        all_results = pd.concat(results, ignore_index=True)
        all_results.to_csv(icc_dir / "all_icc_results.csv", index=False)
        print("\n" + "=" * 60)
        print("Summary of all ICC results:")
        print(all_results.to_string())


if __name__ == "__main__":
    main()
