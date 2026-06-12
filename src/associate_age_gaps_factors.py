# SPDX-FileCopyrightText: Copyright (c) 2026 Rendeiro Lab, CeMM - Research Center for Molecular Medicine, Austrian Academy of Sciences
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#
# Licensed under the PolyForm Noncommercial License 1.0.0 (see LICENSE).

from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.linear_model import LinearRegression, Ridge

from src import config
from src.utils import get_individual_factors, clustermap_marsilea


from sklearn import set_config

set_config(transform_output="pandas")

feature_space: str = "X"
feature_model_name: str = "fine_tuned"
frac: float = 1.0
model_name: str = "Ridge"
cv_name: str = "GroupKFold"
# target_variable: str = "residuals_adj"
target_variable: str = "Age"
per: str = "Tissue"
exclude_entities = [
    "Bladder",
    "Cervix - Ectocervix",
    "Cervix - Endocervix",
    "Fallopian Tube",
    "Kidney - Medulla",
]
# target_variables = ["Age", "prediction_adj", "residuals_adj"]
target_variables = ["residuals_adj"]


input_dir = (
    config.results_dir
    / feature_model_name
    / (
        f"_pre_2024-01-19_age_{feature_space}"
        + (f"_frac{frac}" if feature_space == "X" else "")
    )
)
output_dir = input_dir / "age_gaps_factors-2025"
output_dir.mkdir(parents=True, exist_ok=True)


def associate_age_gaps_factors():
    output_prefix = output_dir / f"{model_name}.{cv_name}.SUFFIX"
    output_file = output_prefix.with_suffix(".statsmodels_ridge_regression.csv")

    # Join info from tissue-specific and pan-tissue clocks
    s = f"tissue-specific_clocks.{model_name}.{cv_name}"
    df = pd.read_parquet(input_dir / f"{s}.predictions_residuals.pq")
    if "shuffled" not in df.columns:
        df["shuffled"] = False
    df = df.query("not `shuffled` and `Tissue` not in @exclude_entities").drop(
        ["shuffled"], axis=1, errors="ignore"
    )
    df["Subject ID"] = df.index.str.extract(r"(GTEX-.*)-\d+")[0].values

    sel = df["residuals_adj"].abs() > 50
    df.loc[sel, "residuals_adj"] = np.nan

    # Get individual data
    y = get_individual_factors()
    y = y.replace(-1, np.nan)
    nums = ["demographics:Age", "death:Ischemic Time (Minutes)", "demographics:BMI"]
    for num in nums:
        y[num] = (y[num] - y[num].mean()) / y[num].std()
    y = y.drop(["demographics:Age", "death:Ischemic Time (Minutes)"], axis=1)
    y = y.drop(
        ["demographics:BMI", "demographics:Weight", "demographics:Height"], axis=1
    )
    y = y.loc[:, ~y.columns.str.startswith("death:")]

    # colinear with "serology:HIV I II Ab"
    y = y.drop(["serology:HIV I II Plus O Antibody"], axis=1)
    # colinear with "serology:HBcAb Total"
    y = y.drop(["serology:HBcAb IgM"], axis=1)
    # colinear with 'behaviour:Signs Of Drug Abuse'
    y = y.drop(["behaviour:Drugs For Non Medical Use In 5y"], axis=1)
    # colinear with 'serology:HIV 1 NAT'
    y = y.drop(["serology:HIV I II Ab"], axis=1)
    # colinear with 'morbidity:Nephritis, Nephrotic Syndrome and/or Nephrosis' in women only
    y = y.drop(["morbidity:Received Human Growth Hormone"], axis=1)
    # colinear with 'morbidity:Unexplained Weight Loss' in woman only
    y = y.drop(["morbidity:Night Sweats"], axis=1)
    # colinear with 'serology:HCV 1 NAT' in woman only
    y = y.drop(["serology:HIV 1 NAT"], axis=1)
    # colinear with 'morbidity:Unexplained Weight Loss' in men only
    y = y.drop(["behaviour:Tattoos Done In 12m"], axis=1)
    # colinear with 'morbidity:Systemic Lupus' in woman only
    y = y.drop(["morbidity:Nephritis, Nephrotic Syndrome and/or Nephrosis"], axis=1)
    # colinear with 'morbidity:Sarcoidosis' in woman only
    y = y.drop(["morbidity:Open Wounds"], axis=1)
    # remove as is ambiguous and no more information is known
    y = y.drop(["morbidity:Abnormal Wbc"], axis=1)

    # corr = y.corr()
    # g = clustermap_marsilea(
    #     corr.fillna(0), mask=corr.isnull(), cmap="coolwarm", center=0, square=True
    # )
    # g.savefig(output_prefix.with_suffix(".all.clustermap.pdf"), **config.figkws)

    res = fit(df, y)
    res.to_csv(output_file, index=False)

    plot_summary_heatmaps(df, y)
    plot_examples(df, y)


def fit(df, y):
    from joblib import Parallel, delayed

    def one_permutation(i, model, x3, y3, tissue, target_variable, sex):
        ry3 = y3.copy()
        np.random.shuffle(ry3.values)
        model.fit(x3, ry3)
        res = (
            get_model_fit_summary(model, x3, y3)
            .reset_index()
            .assign(
                tissue=tissue,
                target_variable=target_variable,
                sex=sex,
                shuffled=True,
                iteration=i,
            )
        )
        return res

    _res = list()
    for target_variable in target_variables:
        x2 = df.pivot_table(
            index="Subject ID", columns="Tissue", values=target_variable
        )
        if target_variable.startswith("residuals"):
            x2.values[x2.abs().values > 50] = np.nan
        y2 = y.reindex(x2.index)
        tissues = x2.columns[x2.isnull().sum() < 600].tolist()
        sexes = y2["demographics:Sex"].dropna().unique().tolist()

        for tissue in tqdm(tissues, leave=True, position=0):
            for sex in tqdm(sexes, leave=False, position=1):
                data = (
                    y2.query("`demographics:Sex` == @sex")
                    .join(x2[tissue])
                    .dropna(subset=[tissue])
                    .T.dropna()
                    .T.drop("demographics:Sex", axis=1)
                )
                x3 = data.drop(tissue, axis=1)
                x3 = x3.loc[:, x3.var() > 0]
                x3 = x3.loc[:, x3.sum() >= 3]
                if x3.empty:
                    continue
                y3 = data[tissue]

                # model = Ridge(2 * x3.shape[0])
                model = LinearRegression(fit_intercept=False)
                model.fit(x3, y3)
                res = (
                    get_model_fit_summary(model, x3, y3)
                    .reset_index()
                    .assign(
                        tissue=tissue,
                        target_variable=target_variable,
                        sex=sex,
                        shuffled=False,
                        iteration=0,
                    )
                )
                _res.append(res)

                results = Parallel(n_jobs=-1)(
                    delayed(one_permutation)(
                        i, model, x3, y3, tissue, target_variable, sex
                    )
                    for i in range(1000)
                )
                _res += results

    res = pd.concat(_res)
    res.to_parquet(
        output_dir
        / f"{model_name}.{cv_name}.SUFFIX.statsmodels_ridge_regression.permutations.pq",
        index=False,
    )
    # Call p-values empirically
    real = res.query("not shuffled").set_index(
        ["target_variable", "tissue", "sex", "index"]
    )
    permuted = res.query("shuffled").set_index(
        ["target_variable", "tissue", "sex", "index"]
    )
    _p_values = dict()
    for (target_variable, tissue, sex, index), group in real.groupby(
        level=[0, 1, 2, 3]
    ):
        real_coef = group["Coefficient"].values[0]
        perm_coefs = permuted.loc[
            (target_variable, tissue, sex, index), "Coefficient"
        ].values
        if np.isnan(real_coef):
            _p_values.append(np.nan)
            continue
        if real_coef >= 0:
            p_emp = (np.sum(perm_coefs >= real_coef) + 1) / (len(perm_coefs) + 1)
        else:
            p_emp = (np.sum(perm_coefs <= real_coef) + 1) / (len(perm_coefs) + 1)
        _p_values[(target_variable, tissue, sex, index)] = p_emp
    p_values = (
        pd.Series(_p_values, name="P value empirical")
        .reset_index()
        .rename(
            columns={
                "level_0": "target_variable",
                "level_1": "tissue",
                "level_2": "sex",
                "level_3": "index",
            }
        )
    )
    res = real.reset_index().merge(
        p_values, on=["target_variable", "tissue", "sex", "index"]
    )
    res.to_csv(
        output_dir / f"{model_name}.{cv_name}.SUFFIX.statsmodels_ridge_regression.csv",
        index=False,
    )

    p = res.pivot_table(
        index="index", columns="tissue", values="P value empirical", aggfunc="min"
    ).drop("Intercept")
    p = -np.log10(p)
    g = clustermap_marsilea(
        p.fillna(0), mask=p.isnull(), cmap="viridis", square=True, robust=True
    )
    g.fig.savefig(
        output_dir
        / f"{model_name}.{cv_name}.SUFFIX.min_empirical_pvalue.clustermap.svg",
        **config.figkws,
    )

    fig, ax = plt.subplots(figsize=(3, 3))
    ax.scatter(
        -np.log10(res["P value"]),
        -np.log10(res["P value empirical"]),
        alpha=0.1,
        rasterized=True,
    )
    ax.plot(
        [
            0,
            max(
                -np.log10(res["P value"].min()),
                -np.log10(res["P value empirical"].min()),
            ),
        ],
        [
            0,
            max(
                -np.log10(res["P value"].min()),
                -np.log10(res["P value empirical"].min()),
            ),
        ],
        color="red",
        linestyle="--",
    )
    ax.set(
        xlabel="-Log10(Standard P value)",
        ylabel="-Log10(Empirical P value)",
        title="Empirical vs Standard P values",
    )
    sns.despine(fig)
    fig.savefig(
        output_dir / f"{model_name}.{cv_name}.SUFFIX.empirical_vs_standard_pvalues.svg",
        **config.figkws,
    )

    return res


def plot_summary_heatmaps(df, y):
    output_prefix = output_dir / f"{model_name}.{cv_name}.SUFFIX"
    output_file = output_prefix.with_suffix(".statsmodels_ridge_regression.csv")

    res = pd.read_csv(output_file)
    res["sex"] = res["sex"].map({0: "female", 1: "male"})

    factor_summary = (
        (~y.isnull())
        .sum()
        .sort_values()
        .to_frame("n")
        .join(y.mean().rename("mean"))
        .join(y.std().rename("std"))
        .rename_axis(index="trait")
    )
    n_sex = (~y.isnull()).groupby(y["demographics:Sex"]).sum()
    factor_summary["sex_ratio"] = n_sex.loc[1] / n_sex.loc[0]
    factor_summary["log_mean"] = np.log(factor_summary["mean"])
    factor_summary = factor_summary.join(
        y.groupby("demographics:Sex")
        .mean()
        .T.rename(columns={1: "mean_male", 0: "mean_female"})
    )
    factor_summary["ratio_mean_sex"] = (
        factor_summary["mean_male"] / factor_summary["mean_female"]
    )
    factor_summary["std_mean_sex"] = factor_summary[["mean_male", "mean_female"]].std(1)

    tissue_summary = df.groupby("Tissue").size().to_frame("n")
    tissue_summary = tissue_summary.join(
        df.groupby("Tissue")["Age"].mean().rename("mean_age")
    )

    #
    v = 5
    for target_variable in target_variables:
        for sex, label in zip(
            ["female", "male", ["female", "male"]], ["female", "male", "both"]
        ):
            c = res.query(
                f"target_variable == '{target_variable}' & index != 'Intercept'"
            ).pivot(index="index", columns=["sex", "tissue"], values="Coefficient")
            c = c[sex]
            c = c.loc[c.var(1) > 0, c.var(0) > 0]
            g = clustermap_marsilea(
                c.fillna(0),
                mask=c.isnull(),
                vmin=-v,
                vmax=v,
                cmap="coolwarm",
                row_colors=factor_summary.reindex(c.index),
                col_colors=tissue_summary.reindex(c.columns),
                square=True,
                metric="cosine",
                robust=99.5,
            )
            g.savefig(
                output_prefix.with_suffix(f".{target_variable}.{label}.clustermap.svg"),
                **config.figkws,
            )
            c2 = c.reindex(c.mean(1).sort_values().index)
            g = clustermap_marsilea(
                c2.fillna(0),
                mask=c2.isnull(),
                vmin=-v,
                vmax=v,
                cmap="coolwarm",
                row_colors=factor_summary.reindex(c2.index),
                col_colors=tissue_summary.reindex(c2.columns),
                square=True,
                metric="cosine",
                robust=True,
                row_cluster=False,
            )
            g.savefig(
                output_prefix.with_suffix(
                    f".{target_variable}.{label}.clustermap.sorted.svg"
                ),
                **config.figkws,
            )

            if label == "both":
                bias = y["demographics:Sex"].mean()
                mean = (c["female"] + c["male"] * bias) / 2
                mean = mean.loc[mean.var(1) > 0, mean.var(0) > 0]

                g = clustermap_marsilea(
                    mean.fillna(0),
                    mask=mean.isnull(),
                    vmin=-v,
                    vmax=v,
                    cmap="coolwarm",
                    row_colors=factor_summary.reindex(mean.index),
                    col_colors=tissue_summary.reindex(mean.columns),
                    square=True,
                    metric="cosine",
                    robust=99.5,
                )
                g.savefig(
                    output_prefix.with_suffix(
                        f".{target_variable}.{label}_mean.clustermap.svg"
                    ),
                    **config.figkws,
                )
                mean2 = mean.reindex(mean.mean(1).sort_values().index)
                g = clustermap_marsilea(
                    mean2.fillna(0),
                    mask=mean2.isnull(),
                    vmin=-v,
                    vmax=v,
                    cmap="coolwarm",
                    row_colors=factor_summary.reindex(mean2.index),
                    col_colors=tissue_summary.reindex(mean2.columns),
                    square=True,
                    metric="cosine",
                    robust=True,
                    row_cluster=False,
                )
                g.savefig(
                    output_prefix.with_suffix(
                        f".{target_variable}.{label}_mean.clustermap.sorted.svg"
                    ),
                    **config.figkws,
                )

                fig, ax = plt.subplots(figsize=(1.25, 6))
                v = mean2.mean(1)
                l = mean2.index.str.split(":").str[1].str.split("(").str[0]
                s = 12 * 2 ** (5 * factor_summary.loc[mean2.index, "mean"]) * 0.5
                ax.scatter(v, l, c=v, s=s, cmap="coolwarm", vmin=-0.1, vmax=0.1)
                ax.axvline(0, linestyle="--", color="grey")
                ax.yaxis.set_inverted(True)
                sns.despine(fig)
                fig.savefig(
                    output_prefix.with_suffix(
                        f".{target_variable}.{label}_mean.clustermap.sorted.side_bubbles.svg"
                    ),
                    **config.figkws,
                )

        # Compare sexes
        diffs = np.log(c["female"] / c["male"]).rename_axis(index="trait")
        diffs = diffs.loc[diffs.var(1) > 0, diffs.var(0) > 0]

        g = clustermap_marsilea(
            diffs.fillna(0),
            mask=diffs.isnull(),
            center=0,
            cmap="BrBG",
            square=True,
        )
        g.savefig(
            output_prefix.with_suffix(
                f".{target_variable}.sex_diffences.clustermap.svg"
            ),
            **config.figkws,
        )

        fig, axes = plt.subplots(1, 2, figsize=(2 * 3, 7))
        for ax, label in zip(axes, ["tissue", "trait"]):
            diff = diffs.stack().groupby(level=label).mean().sort_values()
            sns.barplot(x=diff, y=diff.index, orient="h", ax=ax)
            ax.set(title=label, xlabel="Log(Female/male)", ylabel=label)
        fig.savefig(
            output_prefix.with_suffix(f".{target_variable}.sex_diffences.barplot.svg"),
            **config.figkws,
        )


def get_model_fit_summary(
    model: sklearn.base.RegressorMixin, X: pd.DataFrame, y: pd.Series
) -> pd.DataFrame:
    from scipy import stats

    params = np.append(model.intercept_, model.coef_)
    predictions = model.predict(X)

    newX = X.assign(Intercept=1)[["Intercept"] + X.columns.tolist()]
    MSE = (sum((y - predictions) ** 2)) / (len(newX) - len(newX.columns))

    var_b = MSE * (np.linalg.inv(np.dot(newX.T, newX)).diagonal())
    sd_b = np.sqrt(var_b)
    ts_b = params / sd_b

    shape = newX.shape[0] - newX.shape[1]
    p = pd.Series(
        [2 * (1 - stats.t.cdf(np.abs(i), shape)) for i in ts_b], index=newX.columns
    )

    return pd.DataFrame(
        {"Coefficient": params, "Standard Error": sd_b, "T value": ts_b, "P value": p}
    )


# def get_model_fit_summary(model, X, y):
#     from scipy import stats
#     import numpy as np
#     import pandas as pd

#     params = np.append(model.intercept_, model.coef_)
#     predictions = model.predict(X)

#     # Add intercept as first column
#     newX = np.column_stack([np.ones(X.shape[0]), X.values])
#     MSE = np.sum((y - predictions) ** 2) / (newX.shape[0] - newX.shape[1])

#     var_b = MSE * np.diag(np.linalg.inv(newX.T @ newX))
#     sd_b = np.sqrt(var_b)
#     ts_b = params / sd_b
#     df = newX.shape[0] - newX.shape[1]

#     p = 2 * stats.t.sf(np.abs(ts_b), df)

#     return pd.DataFrame(
#         {"Coefficient": params, "Standard Error": sd_b, "T value": ts_b, "P value": p},
#         index=["Intercept"] + X.columns.tolist(),
#     )


def plot_confidence_intervals():
    raise NotImplementedError


def plot_examples(df, y, n_top: int = 20):
    output_prefix = output_dir / f"{model_name}.{cv_name}.SUFFIX"
    output_file = output_prefix.with_suffix(".statsmodels_ridge_regression.csv")

    res = pd.read_csv(output_file)
    # res["sex"] = res["sex"].map({0: "female", 1: "male"})

    spiked = [
        ("morbidity:Unexplained Seizures", "Brain - Cerebellum"),
        ("morbidity:Alzheimer's OR Dementia", "Brain - Cortex"),
        ("morbidity:Alzheimer's OR Dementia", "Brain - Cerebellum"),
        (
            "morbidity:Chronic Respiratory Disease (Chronic Obstructive Pulmonary Syndrome (COPD) OR Chronic Lower Respiratory Disease (CLRD) (chronic bronchitis, emphysema, asthma))",
            "Lung",
        ),
        (
            "morbidity:Heart attack, acute myocardial infarction, acute coronary syndrome",
            "Adipose - Visceral (Omentum)",
        ),
    ]

    # Illustrate top associations    #
    for target_variable in target_variables:
        for sex, label in zip([[0], [1], [0, 1]], ["female", "male", "both"]):
            c = res.query(
                f"target_variable == '{target_variable}' & index != 'Intercept'"
            ).pivot(index="index", columns=["sex", "tissue"], values="Coefficient")
            c = c[sex]
            c = c.loc[c.var(1) > 0, c.var(0) > 0]
            s = c.stack()
            if s.ndim == 2:
                s = s.loc[s.mean(1).sort_values().index]
            else:
                s = s.sort_values()
            sel = s.tail(n_top).index.tolist()
            sel += s.head(n_top).index.tolist()
            sel = list(set(sel)) + spiked

            fig, axes = plt.subplots(
                3, n_top + len(spiked), figsize=(3 * n_top, 3 * (2 + 1))
            )
            for (trait, tissue), ax in zip(sel, axes.flatten()):

                if y[trait].min() < 0:
                    continue

                t = (
                    df.query("Tissue == @tissue")
                    .set_index("Subject ID")
                    .drop(["Tissue"], axis=1)
                    .groupby(level=0)
                    .mean()
                )
                can = y.loc[y["demographics:Sex"].isin(sex), trait]
                pos_s = t.reindex(can[can > 0].index)[target_variable].dropna()
                pm = pos_s.mean()
                pn = pos_s.shape[0]
                neg_s = t.reindex(can[can < 1].index)[target_variable].dropna()
                nm = neg_s.mean()
                nn = neg_s.shape[0]

                # pg.mwu(pos_s, neg_s)

                sns.histplot(
                    neg_s,
                    ax=ax,
                    kde=True,
                    stat="percent",
                    label="Negative",
                )
                ax.axvline(nm, color=sns.color_palette()[0], linestyle="--")
                sns.histplot(
                    pos_s,
                    ax=ax,
                    kde=True,
                    stat="percent",
                    label="Positive",
                )
                ax.axvline(pm, color=sns.color_palette()[1], linestyle="--")
                ax.set(
                    xlabel=target_variable,
                    ylabel="% individuals",
                    title=f"{tissue}\n{trait}",
                )
                ax.text(
                    0.5,
                    0.9,
                    f"Positive: {pm:.2f} (n={pn})\nNegative: {nm:.2f} (n={nn})",
                    transform=ax.transAxes,
                    ha="center",
                )
            ax.legend()
            fig.savefig(
                output_prefix.with_suffix(f".{target_variable}.{label}.examples.svg"),
                **config.figkws,
            )
