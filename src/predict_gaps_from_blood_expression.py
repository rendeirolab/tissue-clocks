"""
Predict age gaps from gene expression data in blood samples.
"""

from pathlib import Path

from timeit import default_timer as timer
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV


metadata_dir = Path("metadata")
data_dir = Path("data")
expr_dir = Path("data") / "gtex" / "gene_expression"
gaps_dir = Path("results") / "gtex" / "fine_tuned" / "_pre_2024-01-19_age_X_frac1.0"
output_dir = Path("results") / "gtex" / "predict_gaps_from_blood_expression"
output_dir.mkdir(parents=True, exist_ok=True)


exclude_entities = [
    "Bladder",
    "Cervix - Ectocervix",
    "Cervix - Endocervix",
    "Fallopian Tube",
    "Kidney - Medulla",
]

# Load data
meta = pd.read_csv(data_dir / "gtex" / "GTEx Portal.csv", index_col=0)
# x = pd.read_parquet(expr_dir / "log_cpm.pq")
# x = pd.read_parquet(expr_dir / "log_cpm.age_regressed.pq")
annot = pd.read_parquet(expr_dir / "log_cpm.age_regressed.obs.pq")
gaps = pd.read_parquet(
    gaps_dir / "tissue-specific_clocks.Ridge.GroupKFold.predictions_residuals.pq"
)

gaps.loc[gaps["residuals_adj"].abs() > 50, "residuals_adj"] = np.nan
gaps["Individual"] = gaps.index.str.extract(r"(GTEX-\w+)-\d{4}.*")[0].values
for op in ["mean", "max", "std", "sum"]:
    m = (
        gaps.groupby("Individual")["residuals_adj"]
        .apply(op)
        .to_frame("residuals_adj")
        .assign(Tissue=op)
        .reset_index()
    )
    m.index = m["Individual"].rename("Tissue Sample ID") + "-0000"
    gaps = pd.concat([gaps, m], axis=0)
gaps = gaps.drop(["Individual"], axis=1)
gaps = gaps.query("~Tissue.isin(@exclude_entities)")

# # Map IDs and remove duplicates
# x.index = x.index.str.extract(r"(GTEX-\w+-\d{4}).*")[0]
# x = x.groupby(level=0).mean()


# y = gaps.reindex(x.index)["residuals_adj"].dropna()
# x = x.loc[y.index]


# # Train model on tissue expression, predicting age gaps

# individual = x.index.str.extract(r"(GTEX-\w+)-\d{4}")[0]
# cv = GroupKFold(n_splits=5)
# split = cv.split(x, y, groups=individual)
# model = RidgeCV(alphas=np.logspace(-3, 3, 10), cv=split)
# model.fit(x, y)

# # Predict gaps
# y_pred = pd.Series(model.predict(x), index=y.index)

# # Calculate MAE
# mae = mean_absolute_error(y, y_pred)
# print(f"MAE: {mae:.2f}")

# # Plot predictions
# fig, ax = plt.subplots()
# ax.scatter(y, y_pred, s=1, alpha=0.5, rasterized=True)
# sns.regplot(x=y, y=y_pred, ax=ax, scatter=False)
# ax.plot([y.min(), y.max()], [y.min(), y.max()], color="black", linestyle="--")
# ax.set(xlabel="True age gap", ylabel="Predicted age gap")
# fig.savefig(
#     output_dir
#     / "tissue-specific_clocks.Ridge.GroupKFold.predictions_residuals.predicted_from_tissue_expression.svg",
#     bbox_inches="tight",
#     dpi=300,
# )

# # Plot residuals
# fig, ax = plt.subplots()
# ax.scatter(y, y_pred - y, s=1)
# ax.axhline(0, color="black", linestyle="--")
# ax.set(xlabel="True age gap", ylabel="Residuals")
# fig.savefig(
#     output_dir
#     / "tissue-specific_clocks.Ridge.GroupKFold.predictions_residuals.residuals_from_tissue_expression.svg",
#     bbox_inches="tight",
#     dpi=300,
# )

# # Save results
# results = pd.DataFrame({"y": y, "y_pred": y_pred})
# results.to_parquet(
#     output_dir
#     / "tissue-specific_clocks.Ridge.GroupKFold.predictions_residuals.predicted_from_tissue_expression.pq"
# )
# pd.Series(model.coef_, index=x.columns).to_csv(
#     output_dir
#     / "tissue-specific_clocks.Ridge.GroupKFold.predictions_residuals.predicted_from_tissue_expression.coef.csv"
# )

# Train model on blood expression, predicting age
for suffix in ["", ".age_regressed"]:
    x = pd.read_parquet(expr_dir / f"log_cpm{suffix}.pq")
    x2 = x.loc[annot.query("SMTS == 'Blood'").index]
    x2.index = x2.index.str.extract(r"(GTEX-\w+)-\d{4}.*")[0]
    x2 = x2.groupby(level=0).mean()
    x2 = x2.loc[:, x2.mean() > 1]

    for target in ["residuals_adj", "Age"]:
        output_prefix = (
            output_dir
            / f"tissue-specific_clocks.Ridge.KFold.predictions_residuals.predicted_from_blood_expression{suffix}.{target}.xyz"
        )
        if output_prefix.with_suffix(".metrics.csv").exists():
            continue

        _res = list()
        _preds = list()
        _coefs = list()
        tissues = sorted(gaps["Tissue"].unique())
        # for tissue in tqdm(tissues[tissues.index(tissue) :]):
        for tissue in tqdm(tissues):
            obs = gaps.query("Tissue == @tissue").drop(["Tissue"], axis=1)
            obs.index = obs.index.str.extract(r"(GTEX-\w+)-\d{4}.*")[0]
            _y = obs.groupby(level=0).mean().reindex(x2.index)[target].dropna()
            _x = x2.loc[_y.index].groupby(level=0).mean().dropna()
            _y = _y.reindex(_x.index).dropna()
            _x = _x.loc[_y.index]

            if _y.empty or _x.empty:
                continue

            steps = [
                ("scaler", StandardScaler()),
                ("ridge", RidgeCV(alphas=np.logspace(-1, 7, 20))),
                # ("ridge", BayesianRidge()),
                # ("ridge", RandomForestRegressor(n_jobs=16)),
            ]
            pipeline = Pipeline(steps)
            cv = KFold(n_splits=5)

            # Real
            start = timer()
            ests = cross_validate(
                pipeline,
                _x,
                _y,
                cv=cv,
                scoring="neg_mean_absolute_error",
                return_estimator=True,
                return_indices=True,
                # n_jobs=16,
            )
            time = timer() - start
            alphas = [e.named_steps["ridge"].alpha_ for e in ests["estimator"]]
            coefs = pd.DataFrame(
                [e.named_steps["ridge"].coef_ for e in ests["estimator"]],
                columns=_x.columns,
            )
            scores = [e.named_steps["ridge"].best_score_ for e in ests["estimator"]]
            all_r2 = [e.score(_x, _y) for e in ests["estimator"]]

            # Save
            _res.append(
                {
                    "Tissue": tissue,
                    "Shuffled": False,
                    "MAE": np.mean(-ests["test_score"]),
                    "n_samples": len(_y),
                    "time": time,
                    "alpha": np.mean(alphas),
                    "n_alphas": len(alphas),
                    "n_features": len(_x.columns),
                    "n_splits": cv.n_splits,
                    "estimator_name": "Ridge",
                    "scores": np.mean(scores),
                    "all_r2": np.mean(all_r2),
                }
            )
            predictions = pd.Series(index=_y.index, name="predicted_gap")
            for idx, (train_idx, test_idx) in enumerate(
                zip(ests["indices"]["train"], ests["indices"]["test"])
            ):
                estimator = ests["estimator"][idx]
                predictions.loc[_x.iloc[test_idx].index] = estimator.predict(
                    _x.iloc[test_idx]
                )
            _preds.append(
                predictions.to_frame("prediction").assign(Tissue=tissue, Shuffled=False)
            )
            _coefs.append(coefs.assign(Tissue=tissue, Shuffled=False))

            # Shuffled
            _y = pd.Series(np.random.permutation(_y), index=_y.index)
            ests = cross_validate(
                pipeline,
                _x,
                _y,
                cv=cv,
                scoring="neg_mean_absolute_error",
                return_estimator=True,
                return_indices=True,
                # n_jobs=16,
            )
            time = timer() - start
            alphas = [e.named_steps["ridge"].alpha_ for e in ests["estimator"]]
            coefs = pd.DataFrame(
                [e.named_steps["ridge"].coef_ for e in ests["estimator"]],
                columns=_x.columns,
            )
            scores = [e.named_steps["ridge"].best_score_ for e in ests["estimator"]]
            all_r2 = [e.score(_x, _y) for e in ests["estimator"]]

            # Save
            _res.append(
                {
                    "Tissue": tissue,
                    "Shuffled": True,
                    "MAE": np.mean(-ests["test_score"]),
                    "n_samples": len(_y),
                    "time": time,
                    "alpha": np.mean(alphas),
                    "n_alphas": len(alphas),
                    "n_features": len(_x.columns),
                    "n_splits": cv.n_splits,
                    "estimator_name": "Ridge",
                    "scores": np.mean(scores),
                    "all_r2": np.mean(all_r2),
                }
            )
            predictions = pd.Series(index=_y.index, name="predicted_gap")
            for idx, (train_idx, test_idx) in enumerate(
                zip(ests["indices"]["train"], ests["indices"]["test"])
            ):
                estimator = ests["estimator"][idx]
                predictions.loc[_x.iloc[test_idx].index] = estimator.predict(
                    _x.iloc[test_idx]
                )
            _preds.append(
                predictions.to_frame("prediction").assign(Tissue=tissue, Shuffled=True)
            )
            _coefs.append(coefs.assign(Tissue=tissue, Shuffled=True))

        res = pd.DataFrame(_res)
        res.to_csv(output_prefix.with_suffix(".metrics.csv"))
        preds = pd.concat(_preds)
        preds.to_csv(output_prefix.with_suffix(".predictions.csv"))
        coefs = pd.concat(_coefs)
        coefs.to_csv(output_prefix.with_suffix(".coefs.csv"))


# # Try training on all tissues
# _xs = list()
# _ys = list()
# _tissues = list()
# tissues = sorted(gaps["Tissue"].unique())
# for tissue in tqdm(tissues):
#     obs = gaps.query("Tissue == @tissue").drop(["Tissue"], axis=1)
#     obs.index = obs.index.str.extract(r"(GTEX-\w+)-\d{4}.*")[0]
#     obs = obs.groupby(level=0).mean()
#     _y = obs.reindex(x2.index)["residuals_adj"].dropna()
#     _x = x2.loc[_y.index].groupby(level=0).mean().dropna()
#     _y = _y.reindex(_x.index).dropna()
#     _x = _x.loc[_y.index]
#     _xs.append(_x)
#     _ys.append(_y)
#     _tissues.append([tissue] * len(_y))
# xs = pd.concat(_xs)
# ys = pd.concat(_ys).reindex(xs.index)
# ts = pd.Series(np.concatenate(_tissues), index=ys.index)
# individual = xs.index

# i = np.random.choice(range(xs.shape[0]), 10000, replace=False)
# xs = xs.iloc[i]
# ys = ys.iloc[i]
# ts = ts.iloc[i]
# individual = individual[i]

# steps = [
#     ("scaler", StandardScaler()),
#     ("ridge", RidgeCV(alphas=np.logspace(-1, 7, 20))),
# ]
# pipeline = Pipeline(steps)
# cv = GroupKFold(n_splits=5)
# ys_pred = cross_val_predict(pipeline, xs, ys, cv=cv, groups=individual)
# predictions = pd.Series(ys_pred, index=ys.index, name="predicted_gap")
# mae = mean_absolute_error(ys, predictions)

# ys_s = pd.Series(np.random.permutation(ys), index=ys.index)
# ys_pred_s = cross_val_predict(pipeline, xs, ys_s, cv=cv, groups=individual)
# predictions_s = pd.Series(ys_pred_s, index=ys_s.index, name="predicted_gap")
# mae_s = mean_absolute_error(ys, predictions_s)


# res = pd.DataFrame(
#     {
#         "Tissue": ts,
#         "Gap": ys,
#         "Real": predictions,
#         "Shuffled": predictions_s,
#     }
# )
# res["real_e"] = (res["Real"] - res["Gap"]).abs()
# res["shuffled_e"] = (res["Shuffled"] - res["Gap"]).abs()
# # res.to_csv(
# #     output_dir
# #     / "tissue-specific_clocks.Ridge.KFold.predictions_residuals.predicted_from_blood_expression.age_regressed.jointly.csv"
# # )
# m = res.groupby(["Tissue"]).mean().sort_index()


# fig, axes = plt.subplots(1, 2, figsize=(8, 4))
# axes[0].scatter(m["real_e"], m["shuffled_e"])
# axes[0].plot([0, 20], [0, 20], color="black", linestyle="--")
# axes[0].set(
#     xscale="log",
#     yscale="log",
#     xlabel="Mean MAE (True)",
#     ylabel="Mean MAE (Shuffled)",
#     title="All tissues jointly",
# )
# for tissue in m.index:
#     axes[0].annotate(tissue, (m.loc[tissue, "real_e"], m.loc[tissue, "shuffled_e"]))

# f = m["real_e"] / m["shuffled_e"]
# axes[1].scatter(m["real_e"], f)
# axes[1].axhline(1, color="black", linestyle="--")
# axes[1].set(
#     xlabel="Mean MAE (True)", ylabel="log(Mean MAE (True) / Mean MAE (Shuffled))"
# )
# for tissue in m.index:
#     axes[1].annotate(tissue, (m.loc[tissue, "real_e"], f.loc[tissue]))
# fig.savefig(
#     output_dir
#     / "tissue-specific_clocks.Ridge.KFold.predictions_residuals.predicted_from_blood_expression.age_regressed.jointly.scatterplot.svg",
#     bbox_inches="tight",
#     dpi=300,
# )
