# SPDX-FileCopyrightText: Copyright (c) 2026 Rendeiro Lab, CeMM - Research Center for Molecular Medicine, Austrian Academy of Sciences
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#
# Licensed under the PolyForm Noncommercial License 1.0.0 (see LICENSE).

from pathlib import Path

import pandas as pd

metadata_dir = Path("metadata")
gaps_dir = Path("results") / "gtex" / "fine_tuned" / "_pre_2024-01-19_age_X_frac1.0"
output_dir = Path("manuscript")
output_dir.mkdir(parents=True, exist_ok=True)

metrics = pd.read_parquet(
    gaps_dir / "tissue-specific_clocks.Ridge.GroupKFold.metrics.pq"
)
metrics["Fold"] = list(range(1, 6)) * 40

vs = [
    "r_squared",
    "mean_absolute_error",
    "explained_variance",
    "pearson",
    "coefficient",
    "intercept",
]

q = metrics[pd.Series(vs) + "_adj"].rename(
    columns=lambda x: "valid_" + x.replace("_adj", "")
)
metrics = pd.concat([metrics, q], axis=1).drop(
    vs + (pd.Series(vs) + "_adj").tolist(), axis=1
)
metrics = metrics[
    ["Tissue", "Fold"] + metrics.columns.drop(["Tissue", "Fold"]).tolist()
]
# metrics.to_csv(output_dir / "Supplementary Table 1.csv", index=False)

preds = pd.read_parquet(
    gaps_dir / "tissue-specific_clocks.Ridge.GroupKFold.predictions_residuals.pq"
)
preds["residuals_abs"] = preds["residuals"].abs()
preds["residuals_abs-shuffled"] = preds["residuals-shuffled"].abs()
preds["residuals_abs_adj"] = preds["residuals_adj"].abs()

vs = ["prediction", "residuals", "residuals_abs"]
preds = preds.drop(vs, axis=1)
q = preds[pd.Series(vs) + "_adj"].rename(columns=lambda x: x.replace("_adj", ""))
preds = pd.concat([preds, q], axis=1).drop((pd.Series(vs) + "_adj").tolist(), axis=1)
preds = preds.groupby("Tissue").mean()
preds = preds[vs + (pd.Series(vs) + "-shuffled").tolist()]
preds.columns = "mean_" + preds.columns
# preds.to_csv(output_dir / "Supplementary Table 2.csv", index=True)


metrics.groupby("Tissue").mean().join(preds).drop(["Fold", "time"], axis=1).to_csv(
    output_dir / "Supplementary Table 1.csv", index=True
)
