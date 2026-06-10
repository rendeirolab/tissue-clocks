#!/usr/bin/env python3
"""
ICC tile sampling script v2 - optimized for IO with parallel processing.

For each GTEx zarr file:
1. Load once
2. Sample at multiple fractions with multiple seeds
3. Store aggregated features per fraction/seed

Output: one h5ad per fraction (containing all seeds for all samples)

Usage:
    cd /data/projects/histopath
    python src/revision/icc_sample_tiles_v2.py --model virchow2 --organ Lung
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import anndata as ad
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading


processed_dir = Path("processed/histopathology")
output_dir = processed_dir / "icc_sampled_v2"
output_dir.mkdir(exist_ok=True, parents=True)

mpp = 0.5
tile_width = 224


def load_meta():
    meta_file = Path("data/gtex/GTEx Portal.csv")
    meta = pd.read_csv(meta_file, index_col=0)
    return meta


def get_gtex_zarrs(meta, organ):
    zarrs = sorted(processed_dir.glob("GTEX-*.zarr"))
    organ_tissue_map = {
        "Brain": "Brain - Cortex",
        "Colon": "Colon - Transverse",
        "Lung": "Lung",
        "Skin": "Skin - Sun Exposed (Lower leg)",
    }
    target_tissue = organ_tissue_map.get(organ, organ)
    valid_subjects = meta.loc[meta["Tissue"] == target_tissue].index.tolist()
    valid_zarrs = [z for z in zarrs if z.stem in valid_subjects]
    return valid_zarrs


def process_zarr_file(args):
    """Process a single zarr file - returns aggregated features for all fractions/seeds."""
    zarr_path, model_name, fractions, n_seeds, min_tiles = args

    result = {
        "sample_id": None,
        "features": {f: {s: None for s in range(n_seeds)} for f in fractions},
        "n_tiles": 0,
        "error": None,
    }

    try:
        a = ad.read_zarr(zarr_path.as_posix() + f"/tables/{model_name}_tiles")
        tile_matrix = a.X
        n_tiles = tile_matrix.shape[0]

        if n_tiles < min_tiles:
            result["error"] = f"only {n_tiles} tiles"
            return result

        result["n_tiles"] = n_tiles
        result["sample_id"] = zarr_path.stem

        for frac in fractions:
            n_sample = max(1, int(n_tiles * frac))

            if n_sample < min_tiles:
                continue

            for seed_idx in range(n_seeds):
                rng = np.random.default_rng(seed=seed_idx)
                idx = rng.choice(n_tiles, size=n_sample, replace=False)
                aggregated = tile_matrix[idx].mean(axis=0)
                result["features"][frac][seed_idx] = aggregated

    except Exception as e:
        result["error"] = str(e)

    return result


def process_zarr_chunk(args):
    """Process a chunk of zarr files - for memory management."""
    zarr_paths, model_name, fractions, n_seeds, min_tiles = args

    chunk_results = []
    chunk_errors = []

    for zarr_path in zarr_paths:
        result = process_zarr_file(
            (zarr_path, model_name, fractions, n_seeds, min_tiles)
        )
        if result["sample_id"] is not None:
            chunk_results.append(result)
        elif result["error"] is not None:
            chunk_errors.append((zarr_path.stem, result["error"]))

    return chunk_results, chunk_errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="virchow2")
    parser.add_argument("--organ", default="Lung")
    parser.add_argument("--fractions", default="0.0001,0.001,0.01,0.1,0.5")
    parser.add_argument("--n-seeds", type=int, default=10)
    parser.add_argument("--min-tiles", type=int, default=3)
    parser.add_argument(
        "--n-workers", type=int, default=8, help="Number of parallel workers"
    )
    parser.add_argument(
        "--chunk-size", type=int, default=100, help="Samples per chunk for memory"
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Limit number of samples for testing"
    )
    args = parser.parse_args()

    fractions = [float(f) for f in args.fractions.split(",")]
    n_seeds = args.n_seeds
    min_tiles = args.min_tiles

    print(f"Sampling tiles for {args.model} - {args.organ}")
    print(f"Fractions: {fractions}")
    print(f"Seeds per fraction: {n_seeds}")
    print(f"Min tiles to sample: {min_tiles}")
    print(f"Workers: {args.n_workers}, Chunk size: {args.chunk_size}")

    meta = load_meta()
    zarrs = get_gtex_zarrs(meta, args.organ)

    if args.limit:
        zarrs = zarrs[: args.limit]

    print(f"Found {len(zarrs)} GTEx samples for {args.organ}")

    # Storage for all results
    all_results = []
    total_errors = []

    # Split into chunks for memory management
    chunk_size = args.chunk_size
    chunks = [zarrs[i : i + chunk_size] for i in range(0, len(zarrs), chunk_size)]
    print(f"Processing in {len(chunks)} chunks of up to {chunk_size} samples...")

    # Process chunks in parallel
    with ThreadPoolExecutor(max_workers=args.n_workers) as executor:
        futures = {
            executor.submit(
                process_zarr_chunk, (chunk, args.model, fractions, n_seeds, min_tiles)
            ): i
            for i, chunk in enumerate(chunks)
        }

        for future in tqdm(as_completed(futures), total=len(futures), desc="Chunks"):
            chunk_results, chunk_errors = future.result()
            all_results.extend(chunk_results)
            total_errors.extend(chunk_errors)

    print(f"\nProcessed {len(all_results)} samples successfully")
    if total_errors:
        print(f"Errors ({len(total_errors)}):")
        for sample_id, error in total_errors[:5]:
            print(f"  {sample_id}: {error}")
        if len(total_errors) > 5:
            print(f"  ... and {len(total_errors) - 5} more")

    # Aggregate results by fraction/seed
    print("\nAggregating results...")
    results_by_frac_seed = {f: {s: [] for s in range(n_seeds)} for f in fractions}
    valid_sample_ids = {f: {s: [] for s in range(n_seeds)} for f in fractions}
    skipped_counts = {f: 0 for f in fractions}

    for result in all_results:
        if result["sample_id"] is None:
            continue

        sample_id = result["sample_id"]

        for frac in fractions:
            n_tiles = result["n_tiles"]
            n_sample = max(1, int(n_tiles * frac))

            if n_sample < min_tiles:
                skipped_counts[frac] += 1
                continue

            for seed_idx in range(n_seeds):
                if result["features"][frac][seed_idx] is not None:
                    results_by_frac_seed[frac][seed_idx].append(
                        result["features"][frac][seed_idx]
                    )
                    valid_sample_ids[frac][seed_idx].append(sample_id)

    print(f"\nSkipped samples (too few tiles):")
    for frac in fractions:
        print(f"  fraction={frac}: {skipped_counts[frac]} samples")

    # Write one h5ad per fraction
    for frac in fractions:
        print(f"\nWriting fraction {frac}...")

        all_features = []
        all_sample_ids = []
        all_seeds = []

        for seed_idx in range(n_seeds):
            if len(results_by_frac_seed[frac][seed_idx]) > 0:
                features = np.vstack(results_by_frac_seed[frac][seed_idx])
                all_features.append(features)
                all_sample_ids.extend(valid_sample_ids[frac][seed_idx])
                all_seeds.extend([seed_idx] * len(valid_sample_ids[frac][seed_idx]))

        if not all_features:
            print(f"  No valid data for fraction {frac}")
            continue

        X = np.vstack(all_features)
        n_total = X.shape[0]
        n_features = X.shape[1]
        print(
            f"  Total: {n_total} predictions ({n_seeds} seeds × ~{n_total // n_seeds} samples)"
        )

        # Create obs
        obs = pd.DataFrame(
            {
                "sample_id": all_sample_ids,
                "seed": all_seeds,
            }
        )
        obs["sample_id"] = obs["sample_id"].astype(str)

        # Add metadata
        obs = obs.merge(
            meta[["Tissue", "Sex", "Age Bracket"]],
            left_on="sample_id",
            right_index=True,
            how="left",
        )

        a = ad.AnnData(X=X, obs=obs)
        output_file = output_dir / f"{args.organ}.{args.model}.fraction{frac}.h5ad"
        a.write(output_file)
        print(f"  Saved to: {output_file}")

    print(f"\nDone! Output files in: {output_dir}")


if __name__ == "__main__":
    main()
