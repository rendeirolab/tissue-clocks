from pathlib import Path

from tqdm import tqdm
import numpy as np
import pandas as pd
from anndata import AnnData
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
from seaborn_extensions import clustermap
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, roc_auc_score
import pingouin as pg
import statsmodels.api as sm
from statsmodels.formula.api import ols

from src import config

from src.utils import get_restricted_info, get_engineered_info


feature_model_name: str = "fine_tuned"
covariates: list[str] = ["Sex", "Cohort"]

output_dir = Path("results") / "tissue_clocks_revision" / "predict_suicide"
output_dir.mkdir(parents=True, exist_ok=True)


def main():
    a = sc.read_h5ad(config.results_dir / feature_model_name / "anndata.h5ad")
    a.obs.index.name = "Tissue Sample ID"
    var, var_annot = get_restricted_info()
    info = (
        get_engineered_info()["death:Manner Of Death_Suicide"]
        .astype(float)
        .rename("Suicide")
    )
    var = var.join(info)
    a.obs = (
        a.obs.reset_index()
        .merge(
            var[["Age", "Ischemic Time (Minutes)", "Cohort", "Suicide"]],
            left_on="Subject ID",
            right_index=True,
        )
        .set_index(a.obs.index.name)
    )
    a.obs["Organ"] = a.obs["Tissue"].str.replace(" - .*", "", regex=True)

    correlate_age_gaps_with_suicide(a)

    predict_suicide(a)


def silhouette_score_with_suicide(a):
    from sklearn.metrics import silhouette_score

    silhouette_score(a.obsm["X_pca"][:, :2], a.obs["Suicide"])


def correlate_age_gaps_with_suicide(a):
    gaps_dir = Path("results") / "gtex" / "fine_tuned" / "_pre_2024-01-19_age_X_frac1.0"
    preds = pd.read_parquet(
        gaps_dir / "tissue-specific_clocks.Ridge.GroupKFold.predictions_residuals.pq"
    )
    y = preds.join(a.obs["Suicide"]).join(a.obs[covariates])
    y["Organ"] = y["Tissue"].str.replace(r" - .*", "", regex=True)

    formula_interaction = "residuals_adj ~ Sex * Suicide"
    # _res = dict()
    _res = list()
    for organ in a.obs["Organ"].unique():
        yt = y.query("Organ == @organ").copy()
        if yt.empty or yt["Suicide"].sum() == 0:
            continue

        # # t-test (no good as it's not adjusted for sex, etc)
        # tp = pg.ttest(
        #     yt.query("Suicide == 0")["residuals_adj"],
        #     yt.query("Suicide == 1")["residuals_adj"],
        # ).squeeze()["p-val"]
        # _res[organ] = {
        #     "Mean_yes": yt.query("Suicide == 1")["residuals_adj"].mean(),
        #     "Mean_no": yt.query("Suicide == 0")["residuals_adj"].mean(),
        #     "T-test_p": tp,
        #     "Pearson": yt[["Suicide", "residuals_adj"]].corr().iloc[0, 1],
        #     "R^2": r2_score(yt["Suicide"], yt["residuals_adj"]),
        #     # "ROC_AUC": roc_auc_score(yt["Suicide"], yt["residuals_adj"]),
        #     "frac_suicide": yt["Suicide"].mean(),
        #     "mean_age_gap": yt["residuals_adj"].mean(),
        # }

        # ANOVA
        model_interaction = ols(formula_interaction, data=yt).fit()
        _res.append(model_interaction.summary2().tables[1].assign(Organ=organ))

    # res = pd.DataFrame(_res).T
    # res["T-test_q"] = pg.multicomp(res["T-test_p"], method="bonf")[1]
    res = pd.concat(_res)
    res = res.loc["Suicide"].sort_values("P>|t|")
    res["P>|t| adj"] = pg.multicomp(res["P>|t|"], method="bonf")[1]
    res.to_csv(output_dir / "testing_results.csv")


def predict_suicide(a: AnnData):
    target = "Suicide"

    scaler = StandardScaler()
    model = RandomForestClassifier(n_jobs=14)
    splitter = GroupKFold(5, shuffle=True)
    _records = []
    for organ in tqdm(
        a.obs["Organ"].unique(), total=a.obs["Organ"].nunique(), position=0
    ):
        at = a[a.obs["Organ"] == organ].copy()
        at = at[~at.obs[target].isnull()].copy()
        if at.n_obs < 20:
            continue

        X = at.to_df().join(
            pd.get_dummies(a.obs[covariates], drop_first=True).astype(float)
        )
        X = X.loc[:, X.var(0) > 0]
        y = at.obs[target].values
        groups = at.obs["Subject ID"].values
        for train_idx, valid_idx in tqdm(
            splitter.split(X, y, groups), total=5, position=1, leave=False
        ):
            xt = scaler.fit_transform(X.iloc[train_idx])
            xv = scaler.fit_transform(X.iloc[valid_idx])

            model.fit(xt, y[train_idx])
            y_pred = model.predict(xv)

            random_idx = np.random.choice(train_idx, size=len(train_idx), replace=False)
            model.fit(xt, y[random_idx])
            y_rand = model.predict(xv)

            _records.append(
                {
                    "organ": organ,
                    "roc_auc_score": roc_auc_score(y[valid_idx], y_pred),
                    "roc_auc_score_random": roc_auc_score(y[valid_idx], y_rand),
                    "n_pos": y[valid_idx].sum(),
                    "n": len(y[valid_idx]),
                }
            )

    records = pd.DataFrame(_records)
    records.groupby("organ").mean().sort_values("roc_auc_score").to_csv(
        output_dir / "results.csv"
    )
