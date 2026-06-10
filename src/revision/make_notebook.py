# This cell programmatically creates a didactic Jupyter notebook that walks through building
# a tissue-clock from WSI features (e.g., Virchow2). It is parameterized and robust to missing data.
# The notebook will be saved as tissue_clock_walkthrough.ipynb.

import nbformat as nbf
from pathlib import Path
from textwrap import dedent

nb = nbf.v4.new_notebook()
cells = []

# 1) Title / Overview
cells.append(
    nbf.v4.new_markdown_cell(
        dedent(
            """
# Build a tissue-clock from histopathology features

This notebook demonstrates how to construct **tissue-specific age predictors** ("tissue-clocks") from **whole-slide image (WSI)** features.
We assume that **slide-level features** were extracted by a vision model (e.g., Virchow2) and aggregated per slide (e.g., mean over tiles).

The workflow:
1. Load or generate an `AnnData` with slide-level features (`.X`) and metadata (`.obs`) that includes tissue labels and donor/subject identifiers.
2. For each organ, perform **grouped cross-validation** (grouped by donor) to predict chronological age from features using Ridge regression.
3. Record performance (MAE, R2), save predictions, and generate diagnostic plots.
4. Keep it **simple**: no calibration or final model fitting—just cross-validated estimates, per organ.
"""
        )
    )
)

# 2) Requirements and parameters
cells.append(
    nbf.v4.new_markdown_cell(
        dedent(
            """
## setup and parameters

- Dependencies: `scanpy`, `anndata`, `numpy`, `pandas`, `scikit-learn`, `matplotlib`
- Inputs:
  - `results_dir`: directory where consolidated AnnData (per model/mpp/tile size) is written
  - `processed_dir`: directory where per-slide `.h5ad` files live (only needed if you want to (re)build the consolidated AnnData)
  - `meta`: a dataframe (CSV/TSV) with at least `Tissue`, `Age`, and `Subject ID` indexed by slide/sample ID
  
> If you already have a consolidated file `gtex_anndata.{mpp}mpp.{tile_width}px.{model_name}.h5ad` in `results_dir`, you can skip the collection step.
"""
        )
    )
)

cells.append(
    nbf.v4.new_code_cell(
        dedent(
            """
from pathlib import Path

# Set your paths here
results_dir = Path(os.environ.get("RESULTS_DIR", "/data/projects/tissueclocks/results"))
processed_dir = Path(os.environ.get("PROCESSED_DIR", "/data/projects/tissueclocks/processed"))
meta_path = Path(os.environ.get("META_PATH", "/data/projects/tissueclocks/meta.tsv"))

# Model / feature parameters
model_name = "virchow2"
mpp = 0.5
tile_width = 224

# Plotting defaults
FIGSIZE = (4, 4)
DPI = 180

results_dir.mkdir(parents=True, exist_ok=True)
print("results_dir:", results_dir)
print("processed_dir:", processed_dir)
print("meta_path:", meta_path)
"""
        )
    )
)

# 3) Utilities: data I/O, safe loaders
cells.append(
    nbf.v4.new_markdown_cell(
        dedent(
            """
## data loading utilities
"""
        )
    )
)

cells.append(
    nbf.v4.new_code_cell(
        dedent(
            """
import pandas as pd
import numpy as np

def load_meta(meta_path: Path) -> pd.DataFrame:
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata file not found at {meta_path}.")
    # robust parser: supports CSV or TSV
    sep = "," if meta_path.suffix.lower() == ".csv" else "\\t"
    meta = pd.read_csv(meta_path, sep=sep, index_col=0)
    # minimal sanity checks
    required = {"Tissue", "Age", "Subject ID"}
    missing = required - set(meta.columns)
    if missing:
        raise ValueError(f"Metadata is missing required columns: {missing}")
    return meta
"""
        )
    )
)

# 4) Feature collection (provided by user, lightly wrapped)
cells.append(
    nbf.v4.new_markdown_cell(
        dedent(
            """
## collect slide-level features into a single AnnData (optional)

This step scans `processed_dir` for per-slide `.h5ad` files matching the parameter suffix and collapses tile-level features by the **mean**.
If the consolidated file exists already, we skip this step.
"""
        )
    )
)

cells.append(
    nbf.v4.new_code_cell(
        dedent(
            """
import scanpy as sc

def collect_features(model_name: str = "virchow2", mpp: float = 0.5, tile_width: int = 224):
    suffix = f".{mpp}mpp.{tile_width}px.{model_name}"
    output_file = results_dir / f"gtex_anndata.{suffix}.h5ad"
    if output_file.exists():
        print("Consolidated AnnData already exists:", output_file)
        return output_file

    files = sorted(processed_dir.glob(f"*{suffix}.h5ad"))
    if not files:
        raise FileNotFoundError(f"No per-slide h5ad files found with suffix '{suffix}' in {processed_dir}")

    meta = load_meta(meta_path)
    ds = {}
    for f in files:
        d = sc.read_h5ad(f).to_df().mean().rename(f.stem.replace(suffix, ""))
        ds[d.name] = d.values.tolist()
    df = pd.DataFrame(ds).T

    # Align with metadata and drop rows without tissue
    obs = meta.reindex(df.index).dropna(subset=["Tissue"])
    a = sc.AnnData(df.loc[obs.index], obs=obs)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    a.write(output_file)
    print("Wrote consolidated AnnData to:", output_file)
    return output_file

# Try to produce (or locate) the consolidated AnnData
suffix = f".{mpp}mpp.{tile_width}px.{model_name}"
consolidated_path = results_dir / f"gtex_anndata.{suffix}.h5ad"
if not consolidated_path.exists():
    try:
        consolidated_path = collect_features(model_name=model_name, mpp=mpp, tile_width=tile_width)
    except Exception as e:
        print("Feature collection skipped (likely missing inputs). You can set env vars or edit paths above.")
        print("Reason:", e)
else:
    print("Using existing consolidated AnnData:", consolidated_path)
"""
        )
    )
)

# 5) Load AnnData and prepare
cells.append(
    nbf.v4.new_markdown_cell(
        dedent(
            """
## load consolidated features
"""
        )
    )
)

cells.append(
    nbf.v4.new_code_cell(
        dedent(
            """
import scanpy as sc

def load_consolidated(path: Path):
    a = sc.read_h5ad(path)
    # Derive Organ column from Tissue if applicable (e.g., "Colon - Sigmoid" -> "Colon")
    if "Organ" not in a.obs:
        a.obs["Organ"] = a.obs["Tissue"].astype(str).str.split(" - ").str[0]
    return a

adata = None
if consolidated_path.exists():
    adata = load_consolidated(consolidated_path)
    display(adata)
else:
    # Fallback: create a tiny synthetic dataset to demonstrate pipeline
    print("Consolidated AnnData not found; building a small synthetic example.")
    rng = np.random.default_rng(42)
    n = 400
    p = 128
    X = rng.normal(size=(n, p))
    # Two organs with different age structure; 40 donors repeated 5 slides each
    donors = np.repeat([f"D{i:03d}" for i in range(40)], 10)
    tissues = np.where(np.arange(n) % 2 == 0, "Colon - Transverse", "Lung - Upper lobe")
    organs = np.where(np.arange(n) % 2 == 0, "Colon", "Lung")
    # A linear age signal + noise
    age = (X[:, :3].sum(axis=1) * 2 + rng.normal(scale=5, size=n) + 55).clip(20, 90)
    obs = pd.DataFrame({"Tissue": tissues, "Organ": organs, "Subject ID": donors[:n], "Age": age}, index=[f"S{i:04d}" for i in range(n)])
    adata = sc.AnnData(X, obs=obs)
    display(adata)

# quick sanity
assert "Age" in adata.obs, "Age column required in .obs"
assert "Subject ID" in adata.obs, "Subject ID column required in .obs"
assert "Organ" in adata.obs, "Organ column is required (derived if missing)."
"""
        )
    )
)

# 6) CV training function (stripped minimal version)
cells.append(
    nbf.v4.new_markdown_cell(
        dedent(
            """
## cross-validated ridge per organ

- Grouped 5-fold CV on `Subject ID` to avoid leakage across slides from the same donor.
- Standardize features inside the pipeline.
- Save per-organ predictions and a diagnostic scatter plot.
"""
        )
    )
)

cells.append(
    nbf.v4.new_code_cell(
        dedent(
            """
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib.pyplot as plt

def fit_cv_per_organ(a, outdir: Path, model_name: str, mpp: float, tile_width: int):
    outdir.mkdir(parents=True, exist_ok=True)
    suffix = f".{mpp}mpp.{tile_width}px.{model_name}"
    cv = GroupKFold(5)

    organs = sorted(a.obs["Organ"].unique())
    summary_rows = []

    for organ in organs:
        _a = a[a.obs["Organ"] == organ]
        if _a.n_obs < 40 or _a.n_vars < 5:
            # skip small organs to keep CV stable
            continue

        # Aggregate per slide ID if the matrix has multi-index columns. If already per-slide, take as-is.
        X = _a.to_df()
        if isinstance(X.index, pd.MultiIndex):
            X = X.groupby(level=0).mean()

        # Align y (Age) to X
        y = _a.obs.reindex(X.index)["Age"]
        groups = _a.obs.reindex(X.index)["Subject ID"]

        alphas = np.logspace(-2, 3, 20)
        ridgecv = RidgeCV(alphas=alphas, fit_intercept=True)
        model = make_pipeline(StandardScaler(with_mean=True, with_std=True), ridgecv)

        y_pred = cross_val_predict(model, X, y, cv=cv, groups=groups, n_jobs=-1)
        r2 = r2_score(y, y_pred)
        mae = mean_absolute_error(y, y_pred)

        # Save predictions
        pred_df = pd.DataFrame({"y_true": y, "y_pred": y_pred, "age_gap": y_pred - y})
        pred_path = outdir / f"ridgecv{suffix}.{organ}.csv"
        pred_df.to_csv(pred_path, index=True)

        # Plot
        vmin, vmax = pd.Series(pd.concat([y, pd.Series(y_pred, index=y.index)])).agg(["min", "max"]).values
        fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
        ax.plot([vmin, vmax], [vmin, vmax], linestyle="--")
        ax.scatter(y, y_pred, alpha=0.5)
        ax.set_xlabel("chronological age")
        ax.set_ylabel("predicted age")
        ax.set_title(f"{organ} | MAE={mae:.2f}, R2={r2:.2f}")
        fig.tight_layout()
        fig.savefig(outdir / f"ridgecv{suffix}.{organ}.png", dpi=DPI)
        plt.close(fig)

        summary_rows.append({"Organ": organ, "n": len(y), "p": X.shape[1], "MAE": mae, "R2": r2})

    return pd.DataFrame(summary_rows).sort_values(["R2", "MAE"], ascending=[False, True]).reset_index(drop=True)

clock_outdir = results_dir / "clock_cv_only"
summary = fit_cv_per_organ(adata, clock_outdir, model_name=model_name, mpp=mpp, tile_width=tile_width)
summary_path = clock_outdir / f"summary.{mpp}mpp.{tile_width}px.{model_name}.csv"
summary.to_csv(summary_path, index=False)
summary
"""
        )
    )
)

# 7) Quick quality controls
cells.append(
    nbf.v4.new_markdown_cell(
        dedent(
            """
## quick quality control

Inspect per-organ performance. High R2 and low MAE indicate stronger age signals in morphology for that organ.
"""
        )
    )
)

cells.append(
    nbf.v4.new_code_cell(
        dedent(
            """
from caas_jupyter_tools import display_dataframe_to_user
display_dataframe_to_user("tissue_clock_summary", summary)
summary.describe()
"""
        )
    )
)

# 8) Inspect one organ's predictions and age gap distribution
cells.append(
    nbf.v4.new_markdown_cell(
        dedent(
            """
## inspect predictions for a specific organ
"""
        )
    )
)

cells.append(
    nbf.v4.new_code_cell(
        dedent(
            """
# Choose the top-performing organ (if any), else fall back to the first available
if len(summary):
    organ_pick = summary.iloc[0]["Organ"]
    print("Inspecting organ:", organ_pick)

    suffix = f".{mpp}mpp.{tile_width}px.{model_name}"
    pred_path = clock_outdir / f"ridgecv{suffix}.{organ_pick}.csv"
    pred_df = pd.read_csv(pred_path, index_col=0)
    pred_df.head()
else:
    print("No organs available for inspection (likely synthetic or minimal data).")
"""
        )
    )
)

cells.append(
    nbf.v4.new_code_cell(
        dedent(
            """
if len(summary):
    import matplotlib.pyplot as plt

    # Age-gap distribution
    fig, ax = plt.subplots(figsize=(5, 3), dpi=DPI)
    ax.hist(pred_df["age_gap"], bins=30)
    ax.set_xlabel("age gap (pred - true)")
    ax.set_ylabel("count")
    ax.set_title(f"{organ_pick} | distribution of age gaps")
    fig.tight_layout()
    fig
"""
        )
    )
)

# 9) Notes / next steps
cells.append(
    nbf.v4.new_markdown_cell(
        dedent(
            """
## notes and next steps

- This notebook intentionally **does not** fit a final model or export scalers/coefficients.
- To stabilize and interpret models further you might:
  - Add **covariates** (e.g., ischemic time) and perform residualization prior to CV.
  - Explore regularization paths, nested CV, or robust regressors.
  - Compare different feature backbones (Virchow2, PRISM, TITAN, etc.).
  - Evaluate **cross-cohort** transfer by applying clocks trained in one dataset to another.
- For mechanistic links, integrate matched RNA-seq to test pathway associations vs biological age gaps.
"""
        )
    )
)

nb.cells = cells

out_path = Path("src/revision/tissue_clock_walkthrough.ipynb")
with open(out_path, "w") as f:
    nbf.write(nb, f)

out_path
