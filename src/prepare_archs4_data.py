from pathlib import Path

from tqdm import tqdm
import pandas as pd
import numpy as np
import h5py
from anndata import AnnData


data_dir = Path("data") / "archs4"
data_dir.mkdir(exist_ok=True)
results_dir = Path("results") / "archs4"
results_dir.mkdir(exist_ok=True)
figkws = dict(bbox_inches="tight", dpi=300)


def main():
    prepare_archs4_data()


def prepare_archs4_data():
    res_f = results_dir / "human_gene_v2.2.sample_metadata.voi.selected_samples.npz"
    if res_f.exists():
        return

    # ARCH4 data
    "https://s3.dev.maayanlab.cloud/archs4/files/human_gene_v2.2.h5"
    "https://s3.dev.maayanlab.cloud/archs4/files/human_gene_v2.3.h5"
    "https://s3.amazonaws.com/mssm-seq-matrix/sample_human_tsne.csv"
    "https://s3.amazonaws.com/mssm-seq-matrix/gene_human_tsne.csv"
    "https://s3.amazonaws.com/mssm-seq-matrix/gtex_matrix.h5"
    "https://s3.amazonaws.com/mssm-data/geo_human_v1.h5"

    f = h5py.File(data_dir / "human_gene_v2.2.h5")

    var_data = {
        key: f["meta"]["genes"][key][:].astype(str) for key in f["meta"]["genes"].keys()
    }
    var = pd.DataFrame(var_data)

    obs_data = {
        key: f["meta"]["samples"][key][:] for key in f["meta"]["samples"].keys()
    }
    obs = pd.DataFrame(obs_data)
    for col in obs.columns:
        if obs[col].dtype == "O":
            obs[col] = obs[col].astype(str)
    obs["gsm_id"] = obs["sample"]
    obs["gsm_num"] = obs["sample"].str.extract(r"GSM(\d+)", expand=False).astype(int)
    details = (
        obs["characteristics_ch1"]
        .str.split(",")
        .explode()
        .str.split(": ", expand=True, n=1)
        .rename(columns={0: "key", 1: "value"})
    )
    details["key"] = details["key"].str.lower()
    tr = {
        "celltype": "cell type",
        "patient id": "subject id",
        "donor": "subject id",
        "individual": "subject id",
        "donor id": "subject id",
        "subjectid": "subject id",
        "subject": "subject id",
        "patient": "subject id",
        "diseaseseverity": "disease severity",
    }
    details["key"] = details["key"].replace(tr).str.strip()
    details["value"] = details["value"].str.lower()
    details = details.reset_index().drop_duplicates()
    voi = [
        "tissue",
        "tissue type",
        "source tissue",
        "site",
        "location",
        "tissue/cell type",
        "cell type",
        "cell_type",
        "cell line",
        "cell subset",
        "cell population",
        "tissue/celltype",
        "cell subtype",
        "diagnosis",
        "patient diagnosis",
        "disease",
        "disease group",
        "disease severity",
        "disease state",
        "disease timepoint",
        "disease status",
        "health state",
        "phenotype",
        "treatment",
        "treatment condition",
        "agent",
        "concentration",
        "stimulation",
        "age",
        "age_weeks",
        "gestational week",
        "gestational day",
        "donor age",
        "donor_age",
        "age at procedure",
        "age at diagnosis",
        "gender",
        "sex",
        "race",
        "ethnicity",
        "bmi",
        "cohort",
        "donor group",
        "patient type",
        "patient group",
        "classification group",
        "donor condition",
        "control",
        "diabetes",
        "tumor type",
        "cancer type",
        "histology",
    ]
    details = details.loc[details["key"].isin(voi)]
    details = (
        details.groupby(["index", "key"])["value"]
        .apply(lambda x: ", ".join([str(y) for y in x.dropna()]))
        .reset_index("key")
    )
    for attr in tqdm(voi):
        sel = details.loc[details["key"] == attr]["value"]
        sel = (
            sel.reset_index().drop_duplicates().set_index("index")["value"].rename(attr)
        )
        obs = obs.join(sel)
    obs["cell_line_or_primary"] = (
        obs["cell line"].isnull().replace({True: "primary", False: "cell line"})
    )
    harmon_voi = {
        "tissue": [
            "tissue",
            "tissue type",
            "site",
            "location",
            "tissue/cell type",
            "tissue/celltype",
        ],
        "cell type": [
            "cell type",
            "cell_type",
            "cell line",
            "cell subset",
            "cell population",
            "cell subtype",
        ],
        "disease": [
            "diagnosis",
            "patient diagnosis",
            "disease",
            "disease group",
            "disease severity",
            "disease state",
            "disease timepoint",
            "disease status",
            "health state",
            "phenotype",
        ],
        "intervention": [
            "treatment",
            "treatment condition",
            "agent",
            "concentration",
            "stimulation",
        ],
        "age": [
            "age",
            "age_weeks",
            "gestational week",
            "gestational day",
            "donor age",
            "donor_age",
            "age at procedure",
            "age at diagnosis",
        ],
        "sex": ["gender", "sex"],
        "ethnicity": ["race", "ethnicity"],
        "bmi": ["bmi"],
        "grouping": [
            "cohort",
            "donor group",
            "patient type",
            "patient group",
            "classification group",
            "donor condition",
            "control",
            "diabetes",
            "tumor type",
            "cancer type",
            "histology",
        ],
    }
    for cat, cols in tqdm(harmon_voi.items()):
        if f"harmonized:{cat}" in obs.columns:
            continue
        sel = obs[cols].isnull().all(1)
        obs[f"harmonized:{cat}"] = obs.loc[~sel, cols].apply(
            lambda x: ";".join([str(y) for y in x.dropna()]), axis=1
        )
    obs.to_parquet(results_dir / "human_gene_v2.2.sample_metadata.pq")

    # obs = pd.read_parquet(results_dir / "human_gene_v2.2.sample_metadata.pq")

    # Select only for Bulk RNA-seq
    selected_samples = obs.loc[
        ~obs["library_source"].str.contains("single cell")
    ].query("singlecellprobability <= 0.1")
    # Select only for Primary
    selected_samples = selected_samples.query("cell_line_or_primary == 'primary'")
    # Select only for Blood
    selected_samples = selected_samples.loc[
        selected_samples["harmonized:tissue"].astype(str).str.contains("blood")
    ]

    # Select only for PBMCs
    ct_sel = [
        "unknown",
        "pbmc",
        "pbmcs",
        "peripheral blood mononuclear cells",
        "peripheral blood mononuclear cell (pbmc)",
        "peripheral blood mononuclear cells (pbmc)",
        "blood cells",
        "mononuclear cell",
    ]
    selected_samples = selected_samples.loc[
        selected_samples["harmonized:cell type"]
        .fillna("unknown")
        .str.strip()
        .isin(ct_sel)
    ]
    # selected_samples = selected_samples.loc[
    #     selected_samples["harmonized:cell type"].str.strip() != "unknown"
    # ]
    selected_samples["harmonized:age"] = (
        selected_samples["harmonized:age"]
        .astype(str)
        .replace("None", "nan")
        .str.strip()
        .str.replace(r" .*", "", regex=True)
        .str.replace("-year-old", "")
        .str.replace("yrs", "")
        .str.replace("yr", "")
        .str.replace("y", "")
        .str.replace("adult", "")
        .str.replace("young", "")
        .str.replace("old", "")
        .str.replace("embryo", "")
        .str.replace("postnatal", "")
        .str.replace("<", "")
        .str.replace(">", "")
        .str.replace(r"\d+-\d+", "", regex=True)
        .str.replace(r"\d+h", "", regex=True)
        .str.replace(r"\d+hr", "", regex=True)
        .str.strip()
        .replace("unknown", "nan")
        .replace("embrionic", "nan")
        .replace("embronic", "nan")
        .replace("neonate", "nan")
        .replace("post-natal", "nan")
        .replace("development", "nan")
        .replace("pediatric", "nan")
        .replace("child", "nan")
        .replace("before", "nan")
        .replace("mean", "nan")
        .replace("ali", "nan")
        .replace("not", "nan")
        .replace("anonymous", "nan")
        .replace("na", "nan")
        .replace("nic", "nan")
        .replace("oung", "nan")
        .replace("data", "nan")
        .replace("", "nan")
        .replace("postmenstrual", "nan")
        .astype(float)
    )
    selected_samples["harmonized:bmi"] = (
        selected_samples["harmonized:bmi"]
        .replace("n/a", "nan")
        .replace("na", "nan")
        .replace("lean", "nan")
        .replace("obese", "nan")
        .replace("low", "nan")
        .replace("high", "nan")
        .astype(float)
    )
    # TODO: Remove leukemias?
    selected_samples.to_csv(res_f.with_suffix(".obs.csv.gz"))

    # idx = pd.Index(np.arange(obs.shape[0]))
    # idx = idx[idx.isin(selected_samples.index)].tolist()
    # x_data = f["data"]["expression"][:, idx]

    # np.savez_compressed(
    #     res_f,
    #     x=x_data.T,
    #     obs=selected_samples,
    #     obs_index=selected_samples.index,
    #     obs_columns=selected_samples.columns,
    #     var=var,
    #     var_index=var.index,
    #     var_columns=var.columns,
    # )

    # Keep only groups that are identified
    selected_samples2 = selected_samples.loc[
        ~selected_samples[["harmonized:disease", "harmonized:grouping"]].isnull().all(1)
    ]
    selected_samples2["harmonized:group"] = selected_samples2[
        ["harmonized:disease", "harmonized:grouping"]
    ].apply(lambda x: ";".join([str(y) for y in x.dropna()]), axis=1)
    new_cats = categorize_strings(
        selected_samples2["harmonized:group"].value_counts().index.tolist()
    )
    selected_samples2["harmonized:group"] = selected_samples2["harmonized:group"].map(
        new_cats
    )

    # Save
    res_f = (
        results_dir
        / "human_gene_v2.2.sample_metadata.voi.selected_samples.only_with_group.npz"
    )
    selected_samples2.to_csv(res_f.with_suffix(".obs.csv.gz"))

    idx = pd.Index(np.arange(obs.shape[0]))
    idx = idx[idx.isin(selected_samples2.index)].tolist()
    x_data = f["data"]["expression"][:, idx]

    np.savez_compressed(
        res_f,
        x=x_data.T,
        obs=selected_samples2,
        obs_index=selected_samples2.index,
        obs_columns=selected_samples2.columns,
        var=var,
        var_index=var.index,
        var_columns=var.columns,
    )


def categorize_strings(terms: list[str], model: str = "gpt-4") -> dict[str, str]:
    """Optional helper: categorize disease strings via OpenAI API.

    Requires a local API key file at ~/.openai.auth.json. This step is
    optional and not required for the main ARCHS4 validation pipeline.
    """
    import json
    from openai import OpenAI

    api_key = Path("~/.openai.auth.json").expanduser().open().read().strip()
    system = "You are a highly knowledgeable AI in the field of medicine."
    user = """
    The following is a list of diseases or conditions mined from gene expression datasets.
    Your job is to categorize each string into disease categories uniformly since strings may be abbreviated or misspelled.
    For abbreviations, assume the most common disease is the correct one.
    There are also controls and healthy individuals which should all be grouped together.
    If some strings are ambiguous or not relevant, set them to "Other".
    Spell out the disease names if possible (e.g. "Chronic lymphocytic leukemia" instead of "CLL").
    Output a consise JSON file without indentation with a mapping between each string and its category.

    Categories:
    """
    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user + "', '".join(terms) + "'."},
        ],
    )
    return json.loads(response.choices[0].message.content)


def load_archs4_data():
    res_f = (
        results_dir
        / "human_gene_v2.2.sample_metadata.voi.selected_samples.only_with_group.npz"
    )
    if not res_f.exists():
        prepare_archs4_data()

    with np.load(res_f, allow_pickle=True) as df:
        x_data = df["x"]
        obs = pd.DataFrame(df["obs"], df["obs_index"], df["obs_columns"])
        var = pd.DataFrame(df["var"], df["var_index"], df["var_columns"])

    a = AnnData(x_data, obs=obs, var=var)
    a.raw = a
    return a
