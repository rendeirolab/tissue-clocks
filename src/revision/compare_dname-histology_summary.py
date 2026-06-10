from pathlib import Path

import pandas as pd

metadata_dir = Path("metadata")
data_dir = Path("data") / "gtex"
input_dir = Path("results") / "tissue_clocks_revision" / "clock"
results_dir = Path("results") / "tissue_clocks_revision" / "dname_comparison"
results_dir.mkdir(parents=True, exist_ok=True)
figkws = dict(dpi=300, bbox_inches="tight")


files = sorted(results_dir.glob("*/*.stats.csv"))

_res = list()
for file in files:
    stats = pd.read_csv(file)
    stats = stats.assign(
        Modality=file.stem.split(".")[0],
        Outcome=file.parent.stem,
        Transformation=file.stem.split(".")[-2],
    )
    _res.append(stats)
res = pd.concat(_res, axis=0).query("Transformation == 'residuals_adj'")

# Summarize some metrics
# # Best model per outcome/modality
# # Worst model per outcome/modality
# # Performance of best model per outcome/modality
# # Average of all models per outcome/modality
# # Number of models per outcome/modality with significant association
summary = list()
for (outcome, modality, organ), d in res.groupby(["Outcome", "Modality", "Organ"]):
    if outcome == "telomeres":
        best_model = d.loc[d["Coef."].idxmin()]
        worst_model = d.loc[d["Coef."].idxmax()]
    else:
        best_model = d.loc[d["Coef."].idxmax()]
        worst_model = d.loc[d["Coef."].idxmin()]
    avg_r2 = d["Coef."].mean()
    n_significant = (d["P>|t|"] < 0.05).sum()
    total_models = d.shape[0]
    avg_ci_width = (d["0.975]"] - d["[0.025"]).mean()
    coef_std = d["Coef."].std()
    n_positive = (d["Coef."] > 0).sum()
    pct_positive = n_positive / total_models * 100
    avg_n = d["n"].mean()
    summary.append(
        pd.Series(
            {
                "Outcome": outcome,
                "Modality": modality,
                "Organ": organ,
                "Best model": best_model["model_name"],
                "Best coef.": best_model["Coef."],
                "Worst model": worst_model["model_name"],
                "Worst coef.": worst_model["Coef."],
                "Average coef.": avg_r2,
                "Coef. std dev": coef_std,
                "Significant models": n_significant,
                "Total models": total_models,
                "% significant models": n_significant / total_models * 100,
                "% positive coef.": pct_positive,
                "Avg CI width": avg_ci_width,
                "Avg n": avg_n,
            }
        )
    )
summary_df = (
    pd.concat(summary, axis=1)
    .T.set_index(["Organ", "Outcome", "Modality"])
    .sort_index()
)

with pd.ExcelWriter(results_dir / "histology_pathology_summary.xlsx") as writer:
    summary_df.to_excel(writer, sheet_name="summary")
    ci_comparison = summary_df["Avg CI width"].unstack("Modality")
    ci_comparison["CI width difference (dname - histology)"] = (
        ci_comparison["dname"] - ci_comparison["histology"]
    )
    ci_comparison.to_excel(writer, sheet_name="ci_comparison")

    std_comparison = summary_df["Coef. std dev"].unstack("Modality")
    std_comparison["Std diff (dname - histology)"] = (
        std_comparison["dname"] - std_comparison["histology"]
    )
    std_comparison.to_excel(writer, sheet_name="coef_std_comparison")

    pct_pos_comparison = summary_df["% positive coef."].unstack("Modality")
    pct_pos_comparison["Pct diff (dname - histology)"] = (
        pct_pos_comparison["dname"] - pct_pos_comparison["histology"]
    )
    pct_pos_comparison.to_excel(writer, sheet_name="pct_positive_comparison")

    n_comparison = summary_df["Avg n"].unstack("Modality")
    n_comparison["n diff (dname - histology)"] = (
        n_comparison["dname"] - n_comparison["histology"]
    )
    n_comparison.to_excel(writer, sheet_name="sample_size_comparison")

print("\n=== Summary ===")
print(summary_df)
print("\n=== CI Width Comparison (dname vs histology) ===")
print(ci_comparison)
print("\n=== Coefficient Std Dev Comparison ===")
print(std_comparison)
print("\n=== % Positive Coefficient Comparison ===")
print(pct_pos_comparison)
print("\n=== Sample Size Comparison ===")
print(n_comparison)
