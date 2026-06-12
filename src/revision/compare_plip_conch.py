#!/usr/bin/env python


# SPDX-FileCopyrightText: Copyright (c) 2026 Rendeiro Lab, CeMM - Research Center for Molecular Medicine, Austrian Academy of Sciences
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#
# Licensed under the PolyForm Noncommercial License 1.0.0 (see LICENSE).

"""
Compare PLIP and CONCH text-based inference results.
"""

from pathlib import Path

from tqdm import tqdm
import pandas as pd
import anndata
import matplotlib.pyplot as plt
import seaborn as sns
import pingouin as pg

from src.utils import get_restricted_info

data_dir = Path("data")
results_dir = Path("results")
lazyslide_dir = Path("processed") / "histopathology"
output_dir = Path("results") / "tissue_clocks_revision" / "compare_plip_conch"
output_dir.mkdir(exist_ok=True)

figkws = dict(dpi=300, bbox_inches="tight")

plip_f = results_dir / "plip_inference" / "probs.csv.gz"

meta = pd.read_csv(data_dir / "gtex" / "GTEx Portal.csv", index_col=0)
rest, _ = get_restricted_info()
meta = meta.merge(rest[["Age"]], left_on="Subject ID", right_index=True, how="left")
meta["Organ"] = meta["Tissue"].str.split(" - ").str[0]


def main():
    aggregate_conch_terms()
    compare()


def compare():
    plip = pd.read_csv(plip_f, index_col=0)
    conch = (
        pd.read_parquet(results_dir / "conch_text_similarity.pq")
        .join(meta["Organ"])
        .query("Organ != 'Brain'")
        .drop("Organ", axis=1)
    )

    plip = plip.reindex(conch.index).dropna()
    conch = conch.reindex(plip.index).dropna()
    plip = plip.reindex(conch.index)

    corrs = pd.Series(
        {
            col: plip[[col]].corrwith(conch[col], method="pearson").values[0]
            for col in tqdm(plip.columns)
        }
    )

    fig, ax = plt.subplots(figsize=(3, 3))
    sns.histplot(corrs, ax=ax, bins=15, kde=False)
    ax.axvline(corrs.mean(), linestyle="--", color="red")
    ax.axvline(0, linestyle="--", color="black")
    ax.set(
        xlabel="Pearson correlation PLIP vs CONCH",
        ylabel="Number of terms",
        xlim=(-1, 1),
    )
    fig.savefig(output_dir / "all_terms.correlation.histplot.svg", **figkws)

    # Try understanding dis/agreement
    # from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LinearRegression

    val_features = (
        plip.mean(0)
        .to_frame("plip_values_mean")
        .join(plip.std(0).rename("plip_values_std"))
        .join(conch.mean(0).rename("conch_values_mean"))
        .join(conch.std(0).rename("conch_values_std"))
    )
    str_features = extract_text_features(corrs.index.to_series())
    # str_stats = extract_text_statistics(corrs.index.to_series()).drop(
    #     "text_standard", axis=1
    # )
    str_stats = pd.DataFrame(index=corrs.index.to_series())
    # vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 4), max_features=500)
    # X_tfidf = vectorizer.fit_transform(corrs.index)
    # vec_features = pd.DataFrame(X_tfidf.todense(), index=corrs.index)
    # vec_features.columns = "V" + vec_features.columns.astype(str)

    model = LinearRegression()
    _x = val_features.join(str_features).join(str_stats)  # .join(vec_features)
    x = (_x - _x.mean()) / _x.std()
    x = x.loc[:, x.var() > 0]
    y = (corrs - corrs.mean()) / corrs.std()
    model.fit(x, y)
    coef = pd.Series(model.coef_, x.columns).sort_values()
    cv = (_x.std() / _x.mean()).reindex(coef.index).rename("CV")
    # cv = _x.mean().reindex(coef.index)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(1.2 * 3, 6),
        gridspec_kw={"wspace": 0.01, "width_ratios": [1, 0.2]},
        sharey=True,
    )
    sns.barplot(x=coef, y=coef.index, orient="horiz", ax=axes[0])
    sns.barplot(x=cv, y=coef.index, orient="horiz", ax=axes[1])
    axes[0].axvline(0, linestyle="--", color="grey")
    axes[0].set(xlabel="Coefficient", ylabel="Feature")
    fig.savefig(output_dir / "all_terms.coefficient.barplot.svg", **figkws)

    # change with age per organ
    organs = sorted(meta.reindex(conch.index)["Organ"].unique())
    plip_betas = pd.DataFrame(
        {
            organ: plip.corrwith(meta.query("Organ == @organ")["Age"], method="pearson")
            for organ in organs
        }
    )
    conch_betas = pd.DataFrame(
        {
            organ: conch.corrwith(
                meta.query("Organ == @organ")["Age"], method="pearson"
            )
            for organ in organs
        }
    )

    o = len(organs)
    corrs_betas = {
        col: plip_betas[[col]].corrwith(conch_betas[col]).values[0]
        for col in tqdm(plip_betas.columns)
    }
    print(
        corrs_betas
    )  # {'Colon': 0.12542780052357722, 'Lung': 0.3736986103574104, 'Skin': 0.2359681059586509}
    fig, axes = plt.subplots(1, o, figsize=(o * 4.2, 1 * 4), squeeze=False)
    for organ, ax in zip(organs, axes.flatten()):
        ax.axvline(0, linestyle="--", color="grey")
        ax.axhline(0, linestyle="--", color="grey")
        sns.regplot(x=plip_betas[organ], y=conch_betas[organ], scatter=False, ax=ax)
        ax.scatter(
            plip_betas[organ], y=conch_betas[organ], alpha=0.75, s=1, rasterized=True
        )
        ax.set(xlabel="PLIP", ylabel="CONCH", title=organ)
    fig.savefig(output_dir / "change_with_age.scatter.svg", **figkws)

    fig, axes = plt.subplots(2, 1, figsize=(3, 3 * 2))
    axes[0].scatter(plip_betas.mean(1), corrs, s=1, alpha=0.95)
    axes[1].scatter(conch_betas.mean(1), corrs, s=1, alpha=0.95)
    r = pg.corr(plip_betas.mean(1), corrs).loc["pearson"]
    axes[0].set(
        ylim=(-1, 1),
        ylabel="PLIP vs CONCH correlation",
        xlabel="Change with age (PLIP)",
        title=f"r={r['r']:.2f}, p={r['p-val']:.2e}",
    )
    r = pg.corr(conch_betas.mean(1), corrs).loc["pearson"]
    axes[1].set(
        ylim=(-1, 1),
        ylabel="PLIP vs CONCH correlation",
        xlabel="Change with age (CONCH)",
        title=f"r={r['r']:.2f}, p={r['p-val']:.2e}",
    )
    for ax in axes:
        ax.axvline(0, linestyle="--", color="grey")
    fig.tight_layout()
    fig.savefig(output_dir / "all_terms.corr_vs_age_change.scatter.svg", **figkws)

    # c = beta.mean(1).sort_values()
    c = (
        plip_betas.reset_index()
        .melt(id_vars="index")
        .sort_values("value")
        .set_index(["index", "variable"])
    )
    i = 15
    sel = c.tail(i).index.tolist() + c.head(i).index.tolist()
    n = len(sel)
    fig, axes = plt.subplots(n, 2, figsize=(2 * 4.2, n * 5))
    for (f, organ), axs in zip(sel, axes):
        for df, ax in zip([plip, conch], axs):
            p = df.join(meta["Organ"]).query("Organ == @organ")[[f]].join(meta["Age"])
            sns.regplot(x=p["Age"], y=p[f], scatter=False, ax=ax)
            ax.scatter(p["Age"], p[f], alpha=0.75, s=1, rasterized=True)
            ax.text(20, p[f].mean(), s=organ)
    axes[0][0].set_title("PLIP")
    axes[0][1].set_title("CONCH")
    fig.savefig(output_dir / f"top_{i}_examples.change_with_age.scatter.svg", **figkws)


def extract_text_statistics(texts: pd.Series, lang: str = "en_US") -> pd.DataFrame:
    import textstat

    textstat.set_lang(lang)

    texts = pd.Series(texts, name="text").fillna("")
    features = pd.DataFrame(index=texts)
    for text in texts:
        features.loc[text, "flesch_reading_ease"] = textstat.flesch_reading_ease(text)
        features.loc[text, "flesch_kincaid_grade"] = textstat.flesch_kincaid_grade(text)
        features.loc[text, "smog_index"] = textstat.smog_index(text)
        features.loc[text, "coleman_liau_index"] = textstat.coleman_liau_index(text)
        features.loc[text, "automated_readability_index"] = (
            textstat.automated_readability_index(text)
        )
        features.loc[text, "dale_chall_readability_score"] = (
            textstat.dale_chall_readability_score(text)
        )
        features.loc[text, "difficult_words"] = textstat.difficult_words(text)
        features.loc[text, "linsear_write_formula"] = textstat.linsear_write_formula(
            text
        )
        features.loc[text, "gunning_fog"] = textstat.gunning_fog(text)
        features.loc[text, "text_standard"] = textstat.text_standard(text)
        # Language specific
        # features.loc[text, "fernandez_huerta"] = textstat.fernandez_huerta(text)
        # features.loc[text, "szigriszt_pazos"] = textstat.szigriszt_pazos(text)
        # features.loc[text, "gutierrez_polini"] = textstat.gutierrez_polini(text)
        # features.loc[text, "crawford"] = textstat.crawford(text)
        # features.loc[text, "gulpease_index"] = textstat.gulpease_index(text)
        # features.loc[text, "osman"] = textstat.osman(text)
    return features


def extract_text_features(texts: pd.Series) -> pd.DataFrame:
    """
    Extract a broad set of interpretable text-level features from a list or pd.Index of strings.
    """
    import numpy as np
    import re
    from collections import Counter

    texts = pd.Series(texts, name="text").fillna("")

    def count_pattern(text, pattern):
        return len(re.findall(pattern, text))

    features = pd.DataFrame(index=texts)

    # --- Basic length and structure ---
    features["n_chars"] = texts.str.len()
    features["n_words"] = texts.str.split().str.len()
    features["avg_word_len"] = features["n_chars"] / features["n_words"].replace(
        0, np.nan
    )
    features["n_spaces"] = texts.str.count(" ")
    features["n_punct"] = texts.str.count(r"[^\w\s]")
    features["n_digits"] = texts.str.count(r"\d")
    features["n_upper"] = texts.str.count(r"[A-Z]")
    features["n_lower"] = texts.str.count(r"[a-z]")
    features["frac_upper"] = features["n_upper"] / features["n_chars"].replace(
        0, np.nan
    )
    features["frac_digits"] = features["n_digits"] / features["n_chars"].replace(
        0, np.nan
    )
    features["frac_punct"] = features["n_punct"] / features["n_chars"].replace(
        0, np.nan
    )

    # --- Word-level statistics ---
    split_words = texts.str.split()
    features["max_word_len"] = split_words.apply(
        lambda ws: max(map(len, ws)) if ws else 0
    )
    features["min_word_len"] = split_words.apply(
        lambda ws: min(map(len, ws)) if ws else 0
    )
    features["std_word_len"] = split_words.apply(
        lambda ws: np.std(list(map(len, ws))) if len(ws) > 1 else 0
    )
    features["unique_words"] = split_words.apply(lambda ws: len(set(ws)))
    features["frac_unique_words"] = features["unique_words"] / features[
        "n_words"
    ].replace(0, np.nan)

    # --- Composition and structure cues ---
    features["starts_with_upper"] = texts.str[0].str.isupper().fillna(False).astype(int)
    features["ends_with_period"] = texts.str.endswith(".").astype(int)
    features["has_hyphen"] = texts.str.contains("-").astype(int)
    features["has_underscore"] = texts.str.contains("_").astype(int)
    features["has_number"] = texts.str.contains(r"\d").astype(int)
    features["has_parentheses"] = texts.str.contains(r"[\(\)]").astype(int)
    features["has_comma"] = texts.str.contains(",").astype(int)
    features["has_semicolon"] = texts.str.contains(";").astype(int)
    features["has_colon"] = texts.str.contains(":").astype(int)
    features["has_quote"] = texts.str.contains(r"['\"`]").astype(int)

    # --- Character diversity ---
    features["n_unique_chars"] = texts.apply(lambda x: len(set(x)))
    features["char_entropy"] = texts.apply(
        lambda x: (
            -sum((c / len(x)) * np.log2(c / len(x)) for c in Counter(x).values())
            if len(x) > 0
            else 0
        )
    )

    # --- Capitalization patterns ---
    features["is_all_upper"] = texts.str.isupper().astype(int)
    features["is_all_lower"] = texts.str.islower().astype(int)
    # features["is_title"] = texts.str.istitle().astype(int)

    # --- Whitespace and formatting ---
    features["has_multiple_spaces"] = texts.str.contains(r"  +").astype(int)
    features["leading_space"] = texts.str.startswith(" ").astype(int)
    features["trailing_space"] = texts.str.endswith(" ").astype(int)
    features["n_tabs"] = texts.str.count("\t")
    features["n_newlines"] = texts.str.count("\n")

    # --- Simple ratios for normalization ---
    features["word_char_ratio"] = features["n_words"] / features["n_chars"].replace(
        0, np.nan
    )
    features["punct_word_ratio"] = features["n_punct"] / features["n_words"].replace(
        0, np.nan
    )
    features["digit_word_ratio"] = features["n_digits"] / features["n_words"].replace(
        0, np.nan
    )

    # Replace inf/nan
    features = features.replace([np.inf, -np.inf], np.nan).fillna(0)

    return features


def aggregate_conch_terms() -> None:
    if (results_dir / "conch_text_similarity.pq").exists():
        return
    h5ad_files = sorted(lazyslide_dir.glob("*.conch_text_similarity.h5ad"))
    df = pd.DataFrame(
        {
            f.stem.split(".")[0]: anndata.read_h5ad(f).to_df().mean()
            for f in tqdm(h5ad_files)
        }
    )
    df.T.rename_axis(index="Tissue Sample ID").to_parquet(
        results_dir / "conch_text_similarity.pq"
    )


if __name__ == "__main__":
    main()
