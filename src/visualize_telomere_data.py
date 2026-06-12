# SPDX-FileCopyrightText: Copyright (c) 2026 Rendeiro Lab, CeMM - Research Center for Molecular Medicine, Austrian Academy of Sciences
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#
# Licensed under the PolyForm Noncommercial License 1.0.0 (see LICENSE).

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src import config
from src.utils import get_restricted_info


def visualize_gtex_telomere_data():
    meta = pd.read_csv(config.gtex_csv, index_col=0)
    meta["Age Bracket"] = pd.Categorical(meta["Age Bracket"], ordered=True)

    # Connect with telomere length
    tl = pd.read_csv(
        config.metadata_dir / "GTEX_TL_visualization_data_10-15-2019.csv"
    ).rename(columns=dict(TQImean="telomere_length"))
    tl.index = tl.iloc[:, 0].str.split("-SM-").apply(lambda x: x[0]).rename("slide_id")
    tl = meta.join(tl).dropna()

    # Per age bracket
    # # across all tissues
    fig, ax = plt.subplots(figsize=(4, 3))
    sns.violinplot(
        data=tl, x="Age Bracket", y="telomere_length", ax=ax, palette="inferno"
    )
    for i, (x, m) in enumerate(
        zip(
            tl["Age Bracket"].cat.categories,
            tl.groupby("Age Bracket")["telomere_length"].mean(),
        )
    ):
        ax.text(i, m * 2, s=f"{m:.3f}")
    ax.set(ylabel="Mean telomere length")
    fig.savefig(
        config.results_dir / "telomere_length.age_bracket.violinplot.svg",
        **config.figkws,
    )

    # # per tissue
    c = tl["Tissue"].value_counts()
    tissues = c[c > 32].index
    fig, axes = plt.subplots(3, 4, figsize=(4.1 * 4, 4.1 * 3))
    for tissue, ax in zip(tissues, axes.flat):
        tlt = tl.query(f"Tissue == '{tissue}'")
        sns.violinplot(
            data=tlt, x="Age Bracket", y="telomere_length", ax=ax, palette="inferno"
        )
        for i, (x, m) in enumerate(
            zip(
                tlt["Age Bracket"].cat.categories,
                tlt.groupby("Age Bracket")["telomere_length"].mean(),
            )
        ):
            n = tlt.query(f"`Age Bracket` == '{x}'").shape[0]
            ax.text(i, m + m * 0.2, s=f"n={n}\nm={m:.3f}", ha="center", fontsize=6)
        ax.set(title=tissue, xlabel="Age Bracket", ylabel="Mean telomere length")
    for ax in axes[:-1, :].flat:
        ax.set(xlabel="")
    for ax in axes[:, 1:].flat:
        ax.set(ylabel="")
    fig.savefig(
        config.results_dir / "telomere_length.age_bracket.violinplot.per_tissue.svg",
        **config.figkws,
    )

    # For age as continuous variable
    # # add age of participants
    tl["SUBJID"] = tl["Subject ID"]
    restricted_dir = config.metadata_dir / "RESTRICTED"
    r, _ = get_restricted_info()
    age = r["Age"]
    tl = tl.reset_index().merge(age.reset_index()).set_index("Tissue Sample ID")

    # # across all tissues
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.scatter(data=tl, x="AGE", y="telomere_length", color="grey", alpha=0.25)
    sns.regplot(data=tl, x="AGE", y="telomere_length", ax=ax, scatter=False)
    ax.set(ylabel="Mean telomere length")
    fig.savefig(
        config.results_dir / "telomere_length.age.scatterplot.svg",
        **config.figkws,
    )

    # # per tissue
    c = tl["Tissue"].value_counts()
    tissues = c[c > 32].index
    fig, axes = plt.subplots(3, 4, figsize=(4.1 * 4, 4.1 * 3))
    for tissue, ax in zip(tissues, axes.flat):
        tlt = tl.query(f"Tissue == '{tissue}'")
        ax.scatter(data=tlt, x="AGE", y="telomere_length", color="grey", alpha=0.25)
        sns.regplot(data=tlt, x="AGE", y="telomere_length", ax=ax, scatter=False)
        ax.set(title=tissue, xlabel="AGE", ylabel="Mean telomere length")
    for ax in axes[:-1, :].flat:
        ax.set(xlabel="")
    for ax in axes[:, 1:].flat:
        ax.set(ylabel="")
    fig.savefig(
        config.results_dir / "telomere_length.age.scatterplot.per_tissue.svg",
        **config.figkws,
    )
