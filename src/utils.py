# SPDX-FileCopyrightText: Copyright (c) 2026 Rendeiro Lab, CeMM - Research Center for Molecular Medicine, Austrian Academy of Sciences
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#
# Licensed under the PolyForm Noncommercial License 1.0.0 (see LICENSE).

import typing as tp
from pathlib import Path
import json

import requests
from tqdm.auto import tqdm
import tifffile
import numpy as np
import pandas as pd
from anndata import AnnData

from src import config


def get_restricted_info() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Get information from GTEx cohort including controlled access variables.
    Returns one dataframe with the restricted information and another with
    the annotation of those same variables.

    Data is formatted to replace special values with NaN, conversion of string columns to categorical,
    transformation into categorical variables, and ordering of categorical variables when appropriate.

    Expects the following files to be present in the `metadata` directory under the current path:
    - `RESTRICTED/phs000424.v9.pht002742.v9.p2.c1.GTEx_Subject_Phenotypes.GRU.txt.gz`: restricted information
    - `gtex_variable_description.xml`: [optional] variable annotation, will be downloaded if not present
    """
    # Get variable annotation
    variable_annotation_file = config.metadata_dir / "gtex_variable_description.xml"
    if not variable_annotation_file.exists():
        url = "https://ftp.ncbi.nlm.nih.gov/dbgap/studies/phs000424/phs000424.v9.p2/pheno_variable_summaries/phs000424.v9.pht002742.v9.p2.GTEx_Subject_Phenotypes.var_report.xml"

        with variable_annotation_file.open("bw") as handle:
            handle.write(requests.get(url).content)

    var_annot = (
        pd.read_xml(variable_annotation_file, parser="etree")
        .iloc[1:, :]
        .set_index("var_name")
    )
    var_annot = var_annot.drop_duplicates(subset="description", keep="last")
    # var_annot = var_annot.loc[var_annot["id"].str.endswith("p2.c1")]

    restricted_dir = config.metadata_dir / "RESTRICTED"
    df = pd.read_table(
        restricted_dir
        / "phs000424.v9.pht002742.v9.p2.c1.GTEx_Subject_Phenotypes.GRU.txt.gz",
        skiprows=10,
    )
    df = df.reindex(var_annot.index, axis=1).set_index("SUBJID").drop("K-562")
    df.columns = df.columns.to_series().replace(var_annot["description"].to_dict())
    assert df.columns.is_unique
    df["Subject ID"] = df.index

    # Fix types
    sex = {
        1: "Male",
        2: "Female",
        99: "Unknown",
    }
    df["Sex"] = df["Sex"].replace(sex).astype(pd.CategoricalDtype())
    race = {
        1: "Asian",
        2: "Black or African American",
        3: "White",
        4: "American Indian or Alaskan native",
        98: "Unreported",
        99: "Unknown",
    }
    df["Race"] = df["Race"].replace(race).astype(pd.CategoricalDtype())
    ethnicity = {
        0: "Not Hispanic or Latino",
        1: "Hispanic or Latino",
        98: "Unreported",
        99: "Unknown",
    }
    df["Ethnicity"] = df["Ethnicity"].replace(ethnicity).astype(pd.CategoricalDtype())

    df["Age"] = df["Age"].astype(int)
    date_times = ["Time of Death"]
    for t in date_times:
        df[t] = pd.to_datetime(df[t], format="%H:%M")
    time_deltas = [
        "Ischemic Time",
        "Time Cross Clamp Applied",
        "Time of Chest Incision",
    ]
    for t in time_deltas:
        # N.B. there are
        sign = df[t].str.contains(", -").astype(float).replace({1: -1, 0: 1})
        df[t] = df[t].str.replace(", -", ", ")
        df[t] = pd.to_timedelta(df[t].str.replace("(s)", "", regex=False)) * sign
    minutes = df.columns[df.columns.str.contains("(Minutes)", regex=False)]
    for col in minutes:
        df[col] = df[col].astype(pd.Int64Dtype())

    df["Number Of Hours In Refrigeration"] = df[
        "Number Of Hours In Refrigeration"
    ].astype(float)
    df["Time of Death (hour)"] = df["Time of Death"].dt.hour + (
        df["Time of Death"].dt.minute / 60
    )

    cats = df.columns[df.convert_dtypes().dtypes == "string"].tolist()
    # cats += ["dbGaP_Subject_ID"]
    for col in cats:
        df[col] = df[col].astype(pd.CategoricalDtype())

    # Find numeric types
    prev_dtypes = df.dtypes
    new_dtypes = df.convert_dtypes().dtypes
    rev = prev_dtypes.index[prev_dtypes != new_dtypes]
    df[rev] = df[rev].astype(float)

    # # Replace 99 with NaN in categorical
    for col in rev:
        if df[col].value_counts().index.isin([0, 1, 99, 98, 97, 96]).all():
            df[col] = df[col].replace({99: np.nan, 98: np.nan, 97: np.nan, 96: np.nan})

    # # Replace 99 with NaN in numeric columns
    f = (df[rev] == 99).sum() / df.shape[0]
    sel = f[f > 0.01].index
    for col in sel:
        df[col] = df[col].replace({99: np.nan, 98: np.nan, 97: np.nan, 96: np.nan})

    new_cats = df.columns[df.isin([0, 1, np.nan]).all()]
    df[new_cats] = (
        df[new_cats]
        .replace({0: False, 1: True})
        .astype(pd.CategoricalDtype(ordered=True, categories=[False, True]))
    )

    # Fix mixture of celsius and fahrenheit in body temperature
    sel = df["Core Body Temperature - Units of measurement"] == "F"
    df.loc[sel & (df["Core Body Temperature"] == 0), "Core Body Temperature"] = np.nan
    df.loc[
        sel & (df["Core Body Temperature"] < 50),
        "Core Body Temperature - Units of measurement",
    ] = "C"
    sel = df["Core Body Temperature - Units of measurement"] == "F"
    df.loc[sel, "Core Body Temperature"] = (
        df.loc[sel, "Core Body Temperature"] - 32
    ) * (5 / 9)

    # Last underlying cause of death
    to_fix = [
        "NATURAL CAUSES",
        "natural causes",
        "Natural Causes",
        "Natural Cause",
        "Natural",
        "natural",
        "natural death",
    ]
    for tf in to_fix:
        df["Last Underlying Cause Of Death"] = df[
            "Last Underlying Cause Of Death"
        ].str.replace(tf, "Natural causes")

    # Morbidity
    sel = df["Pneumonia (acute respiratory infection affecting the lungs)"] == True
    df.loc[sel, "Pneumonia"] = True
    df = df.drop(
        columns=["Pneumonia (acute respiratory infection affecting the lungs)"]
    )

    # For practical reasons, pd.Int64 is still too cumbersome, so make float
    cols = df.columns[df.dtypes.isin([pd.Int64Dtype()])]
    df[cols] = df[cols].astype(float)
    return df, var_annot


def get_engineered_info(
    df: pd.DataFrame = None, var_annot: pd.DataFrame = None
) -> pd.DataFrame:
    """
    Get information from GTEx cohort including engineered variables.
    The two input dataframes are expected to be the output of `get_restricted_info`.

    Variables are grouped into the following categories:
    - Demographics
    - Behaviour/environmental
    - Morbidity
    - Serology
    - Death related variables

    Each group is preceded by a corresponding prefix to the variable names.
    """
    if df is None or var_annot is None:
        df, var_annot = get_restricted_info()
    # Demographics
    demographic_vars = [
        "Sex",
        "Age",
        "Race",
        "Ethnicity",
        "Height",
        # "Height Units",
        "Weight",
        # "Weight Units",
        "BMI",
    ]
    demographics = df[demographic_vars].add_prefix("demographics:")

    # Behaviour/environmental
    behaviour_vars = [
        "Signs Of Drug Abuse",
        "Documented Sepsis",
        "Spots On Skin",
        "Smoking Status",
        "Smoke Type",
        "Smoke Number",
        "Smoke Period",
        "Smoke Years",
        "Smoke Comments",
        "Drugs For Non Medical Use In 5y",
        # "Sex For Money Or Drugs",
        "Tattoos Done In 12m",
        "Non Professional Tattoos",
        "Exposure To Toxics",
        # "In Uk 3m 1980 1996",
        "Non Professional Piercing",
        "No Physical Activity 4 Weeks",
        # "Resided On Northern European Military Base",
        # "Men Sex With Men",
    ]
    df["Smoke Burden"] = df["Smoke Number"] * df["Smoke Years"]
    behaviour = df[behaviour_vars + ["Smoke Burden"]].add_prefix("behaviour:")

    # Morbidity
    morbidity_vars = var_annot.loc[
        var_annot.index.str.startswith("MH"), "description"
    ].drop_duplicates()
    morbidity_vars = morbidity_vars[morbidity_vars.isin(df.columns)]
    morbidity_exclude = [
        "Blood Donation Denial Reason",
        "Drink Period",
        "Drink Type",
        "Drink Comments",
        "Tissue Transplant Comments",
        "General Comments",
        "Drinking Status",
        "Primary History Source",
        "Men Sex With Men",
        "Sex For Money Or Drugs",
        "Resided On Northern European Military Base",
        "In Uk 3m 1980 1996",
        "In Europe 5y Since 1980",
        "In Detention Center 72h",
        "Past Blood Donations Denied",
    ]
    morbidity_vars = list(filter(lambda x: x not in morbidity_exclude, morbidity_vars))
    morbidity_vars = list(filter(lambda x: x not in behaviour_vars, morbidity_vars))

    morbidity = df[morbidity_vars].replace(99.0, np.nan).convert_dtypes()
    morbidity = morbidity.loc[
        :, ~morbidity.dtypes.isin([float, int, pd.Float64Dtype(), pd.Int64Dtype()])
    ].add_prefix("morbidity:")

    bool_cols = morbidity.dtypes.apply(lambda c: c.categories.dtype == bool)
    count = (
        morbidity.loc[:, bool_cols]
        .apply(lambda x: x.dropna().cat.codes.sum())
        .sort_values()
    )
    drop = count[count < 2].index
    morbidity = morbidity.drop(columns=drop)

    # Serology
    serology_vars = var_annot.loc[
        var_annot.index.str.startswith("LB"), "description"
    ].tolist()
    serology = df[serology_vars].add_prefix("serology:")

    # Death related variables
    death_vars = [
        "Hardy Scale",
        "Ischemic Time (Minutes)",
        "Core Body Temperature",
        "Body Refrigerated",
        "Number Of Hours In Refrigeration",
        "First Underlying Cause Of Death",
        "Immediate Cause Of Death",
        "Last Underlying Cause Of Death",
        "Manner Of Death",
        "Place Of Death",
        "Season of Death",
        "Time of Death (hour)",
    ]
    death = df[death_vars]
    death = pd.get_dummies(death).add_prefix("death:")

    # # add lifespan
    sel = df.loc[:, "Manner Of Death"] == "Natural"
    death.loc[sel, "death:Lifespan"] = df.loc[sel, "Age"]

    assert pd.Series(
        demographic_vars + behaviour_vars + morbidity_vars + death_vars + serology_vars
    ).is_unique

    ret = pd.concat([demographics, behaviour, morbidity, serology, death], axis=1)
    return ret


def get_individual_factors() -> pd.DataFrame:
    annot, _ = get_restricted_info()
    annot_e = get_engineered_info()

    for col in annot_e.dtypes[annot_e.dtypes == "bool"].index:
        annot_e[col] = annot_e[col].astype(float)
    for col in annot_e.dtypes[annot_e.dtypes == "category"].index:
        c = annot_e[col].value_counts()
        sel = c[c > 0].index
        if len(sel) != 2:
            # Exclude categories with only one level
            annot_e = annot_e.drop(col, axis=1)
        else:
            can = annot_e[col].cat.codes.astype(float)
            can.loc[annot_e[col].isnull()] = can.median()
            annot_e[col] = can

    annot_e = annot_e.T.dropna().T
    y = pd.get_dummies(annot_e.loc[:, ~annot_e.isnull().any()])
    y = y.loc[:, ~y.columns.str.contains("Cause Of Death")]  # reduce redundancy
    return y


def get_pathology_data():
    meta = pd.read_csv(config.gtex_csv, index_col=0)
    paths = sorted(
        meta["Pathology Categories"].str.split(", ").explode().dropna().unique()
    )
    path_exclude = ["clean_specimens", "no_abnormalities", "tma"]
    paths = [p for p in paths if p not in path_exclude]
    path_df = pd.DataFrame(
        [
            meta["Pathology Categories"]
            .str.contains(path)
            .astype(pd.BooleanDtype())
            .rename(path)
            .fillna(False)
            for path in paths
        ]
    ).T.astype(pd.BooleanDtype())

    # set values to NaN if no text is available for any sample of an individual
    individuals = meta.index.str.extract(r"(GTEX-\w+)-\d+", expand=False).unique()
    text_lens = dict()
    for i in individuals:
        text_lens[i] = (
            meta.loc[meta.index.str.startswith(i), "Pathology Notes"]
            .astype(str)
            .apply(len)
            .sum()
        )
        if meta.loc[meta.index.str.startswith(i), "Pathology Notes"].dropna().empty:
            print(f"Setting pathology values to NaN for {i}")
            path_df.loc[path_df.index.str.startswith(i)] = np.nan
    return path_df


def get_telomere_lengths() -> pd.DataFrame:
    """
    Get information for GTEx cohort regarding telomere lengths.

    Expects the following files to be present in the `metadata` directory under the current path:
    - `GTEX_TL_visualization_data_10-15-2019.csv`: telomere lengths per individual
    """
    tl = pd.read_csv(Path("metadata") / "GTEX_TL_visualization_data_10-15-2019.csv")
    tl.index = tl.iloc[:, 0].str.replace(r"-SM-.*", "", regex=True).rename("slide_id")
    return tl


def get_somatic_mutation_counts() -> pd.DataFrame:
    """
    Get information for GTEx cohort regarding somatic mutations called from RNA-seq data.

    Expects the following files to be present in the `data/gtex` directory under the current path:
    - `gtex_mutation_counts.csv`: mutation counts per individual
    """
    mutation_counts = pd.read_csv(
        config.data_dir / "mutation_counts.per_individual.csv", index_col=0
    ).rename_axis(index="SUBJID", columns="Tissue")
    return mutation_counts


def get_tissue_info() -> pd.DataFrame:
    """
    Get information for the tissues collected in the GTEx cohort.
    These have been compiled from several sources and are now homogenized and include ontology terms.
    Useful for cross-tissue analysis/comparisons.
    """
    tinfo = pd.read_csv("metadata/gtex_tissue_annotation.updated.csv", index_col=0)
    tinfo.loc["Pancreas", "anatomical system"] = "digestive system|endocrine system"
    tinfo.loc["Breast", "anatomical system"] = (
        "exocrine system|integumental system|reproductive system"
    )
    tinfo.loc["Esophagus - Gastroesophageal Junction", "anatomical system"] = (
        "digestive system"
    )
    tinfo.loc[tinfo.index.str.startswith("Muscle - "), "anatomical system"] = (
        "muscular system"
    )
    tinfo.loc[tinfo.index.str.startswith("Adipose - "), "anatomical system"] = (
        "integumental system|endocrine system"
    )
    # g = (
    #     tinfo["anatomical system"]
    #     .str.split("|")
    #     .apply(pd.Series)
    #     .stack()
    #     .reset_index(level=1, drop=True)
    #     .rename("anatomical system")
    # )
    # tinfo = tinfo.drop(["anatomical system"], axis=1).join(g)
    return tinfo


def prepare_gtex_adata(a: AnnData) -> AnnData:
    """
    Prepare AnnData object for GTEx cohort by adding cohort relevant variables and colors.
    This is done in-place (i.e. modifies the input object).
    Supports AnnData objects with samples as index, or individuals as index.
    """
    import matplotlib
    import matplotlib.pyplot as plt
    import h5py
    from scanpy.plotting._utils import (
        add_colors_for_categorical_sample_annotation as set_colors,
    )

    # Read up GTEx varibles and transform/derive as needed
    meta = pd.read_csv(config.gtex_csv, index_col=0)
    meta["Sex"] = meta["Sex"].str.capitalize()
    meta["Organ"] = meta["Tissue simple"] = meta["Tissue"].str.replace(
        r" - .*", "", regex=True
    )
    meta["Age Decade"] = meta["Age Bracket"].str.slice(0, 2).astype(int)

    # If index is individual's GTEX ID, then AnnData is a individual-level dataset
    if a.obs.index.to_series().str.count("-").lt(2).all():
        voi = ["Sex", "Age Bracket", "Age Decade", "Hardy Scale"]
        meta = meta[["Subject ID"] + voi].drop_duplicates().set_index("Subject ID")
    else:
        voi = [
            "Organ",
            "Tissue simple",
            "Tissue",
            "Age Bracket",
            "Age Decade",
            "Sex",
            "Hardy Scale",
        ]

    # tissue = pd.read_csv(Path("metadata") / "gtex_tissue_annotation.csv", index_col=0)

    ordered_vars = ["Age Bracket", "Hardy Scale"]
    numeric_vars = ["Age Decade"]

    # Write to `uns` slot of adata a record of variables of interest
    a.uns["variables_of_interest"] = voi
    a.uns["ordered_variables"] = ordered_vars
    a.uns["numerical_variables"] = numeric_vars
    a.uns["categorical_variables"] = [v for v in voi if v not in numeric_vars]

    # Transform to categorical
    for v in a.uns["categorical_variables"]:
        meta[v] = pd.Categorical(meta[v], ordered=v not in ordered_vars)

    already = a.obs.columns.intersection(meta.columns)
    a.obs = a.obs.join(meta.drop(columns=already), how="left")
    for col in already:
        a.obs[col] = a.obs[col].astype(meta[col].dtype)

    # Set colors for each variable
    for v in a.uns["categorical_variables"]:
        set_colors(a, v)

    # Overide colors for certain variables to set specific colormaps
    for v, cmap in [
        ("Sex", "tab10"),
        ("Tissue", "tab20"),
        ("Age Bracket", "inferno"),
        ("Hardy Scale", "rainbow"),
    ]:
        if v not in voi:
            continue
        n = a.obs[v].nunique()
        a.uns[f"{v}_colors"] = [
            matplotlib.colors.rgb2hex(c) for c in plt.get_cmap(cmap, n)(range(n))
        ]

    # Drop unused levels
    for v in a.uns["categorical_variables"]:
        a.obs[v] = a.obs[v].cat.remove_unused_categories()

    if "Tissue" not in voi:
        return a

    # Add number of tiles per slide and other variables if possible
    n_tiles_file = Path("metadata") / "gtex_slides.n_tiles.csv"
    a.obs["n_tiles"] = np.nan
    if n_tiles_file.exists():
        n_tiles = pd.read_csv(n_tiles_file, index_col=0)
        if not n_tiles["n_tiles"].isnull().all():
            a.obs["n_tiles"] = n_tiles["n_tiles"].reindex(a.obs.index)
    else:
        n_tiles = pd.DataFrame(columns=["n_tiles"])
    for s in tqdm(a.obs.index):
        if s not in n_tiles.index:
            if (config.data_dir / "svs" / f"{s}.h5").exists():
                with h5py.File(config.data_dir / "svs" / f"{s}.h5") as h5:
                    a.obs.loc[s, "n_tiles"] = h5["coords"].shape[0]
    if a.obs.shape[0] > n_tiles.shape[0]:
        a.obs[["n_tiles"]].to_csv(n_tiles_file)
    if not a.obs["n_tiles"].isnull().any():
        a.obs["n_tiles"] = a.obs["n_tiles"].astype(int)

    return a


def get_model_categories(model_name: str):
    import torchvision.models

    model_to_weights = {
        "alexnet": "AlexNet",
        "googlenet": "GoogLeNet",
        "vgg16": "VGG16",
        "vgg19": "VGG16",
        "densenet121": "DenseNet121",
        "densenet201": "DenseNet201",
        "resnet50": "ResNet50",
        "resnet152": "ResNet152",
        "efficientnet_v2_l": "EfficientNet_V2_L",
        "vit_h14_in1k": None,
    }
    weights = getattr(torchvision.models, model_to_weights[model_name] + "_Weights")
    weights.DEFAULT.meta["categories"]
    return weights.DEFAULT.meta["categories"]


def get_model_size_performance():
    import torchvision

    _res = dict()
    for name in torchvision.models.__dict__:
        if not name.endswith("_Weights"):
            continue
        member = getattr(torchvision.models, name)
        meta = member.DEFAULT.meta
        for dataset, metrics in meta["_metrics"].items():
            _res[name.replace("_Weights", "")] = dict(
                url=meta["recipe"],
                dataset=dataset,
                dataset_version=member.DEFAULT.name,
                num_params=meta["num_params"],
                dims=meta["min_size"],
                **metrics,
            )
    res = pd.DataFrame(_res).T.sort_values("acc@1", ascending=False)
    res.to_csv("metadata/model_complexity_performance.csv")

    res = pd.read_csv("metadata/model_complexity_performance.csv", index_col=0)


def convert_model_weights():
    import torch

    model_dir = Path("~/.cache/torch/hub/checkpoints/").expanduser()
    for model_name in ["vit_h14", "vit_h14_in1k"]:
        model_weights = model_dir / (model_name + ".pth")
        model = torch.hub.load("facebookresearch/SWAG", model_name, pretrained=True)
        traced_graph = torch.jit.trace(
            model, torch.randn(1, 3, model.image_size, model.image_size)
        )
        traced_graph.save(model_weights)

        model = torch.load(model_weights)


def guess_input_dims(model):
    import torch

    for i in tqdm(range(224, 519)):
        try:
            with torch.no_grad():
                _ = model(torch.randn(1, 3, i, i))
        except (AssertionError, RuntimeError):
            pass
        else:
            return i


def find_tiff_metadata_attribute(filename: Path) -> tp.Optional[str]:
    t = tifffile.TiffFile(filename)
    attrs = list(filter(lambda x: x.startswith("is_"), t.__dict__))

    for attr in attrs:
        if getattr(t, attr):
            return attr
    return


def contour_to_geojson(
    feats: list[list[tuple[int, int]]], json_file: Path, kind: str = "Tissue"
):
    from shapely.geometry import Polygon

    features = list()
    for feat in feats:
        features.append(
            {
                "type": "Feature",
                "geometry": {
                    "type": "LineString",
                    # "coordinates": [[int(y) for y in x[0]] for x in feat],
                    "coordinates": [
                        [int(x[0]), int(x[1])] for x in feat.squeeze().tolist()
                    ],
                },
                "properties": {"name": kind, "area": Polygon(feat.squeeze()).area},
            }
        )
    output = {"type": "FeatureCollection", "features": features}
    json.dump(output, json_file.open("w"), indent=4)


def mask_to_polygon(
    labeled_image: np.ndarray,
    simplify: bool = True,
    simplification_threshold: float = 5.0,
) -> tp.List[np.ndarray]:
    from imantics import Mask
    from shapely.geometry import Polygon

    polygons = Mask(labeled_image).polygons()
    shapes = list()
    for point in polygons.points:
        if not simplify:
            poly = np.asarray(point).tolist()
        else:
            poly = np.asarray(
                Polygon(point).simplify(simplification_threshold).exterior.coords.xy
            ).T.tolist()
        shapes.append(poly)
    return shapes


def get_crop(
    tiff_file: Path | tifffile.TiffFile, level: int, x: int, y: int, h: int, w: int
):
    """Extract a crop from a TIFF image file directory (IFD).

    Only the tiles englobing the crop area are loaded and not the whole page.
    This is usefull for large Whole slide images that can't fit int RAM.
    Parameters
    ----------
    tiff_file : Path | TiffFile
        Path to TIFF file or tifffile.TiffFile instance from which the crop must be extracted.
    level : int
        Level in TIFF file from which image crop must be extracted.
    x, y: int
        Coordinates of the top left corner of the desired crop.
    h: int
        Desired crop height.
    w: int
        Desired crop width.
    Returns
    -------
    out : ndarray of shape (imagedepth, h, w, sampleperpixel)
        Extracted crop.
    """
    if isinstance(tiff_file, Path):
        tiff_file = tifffile.TiffFile(tiff_file)
    page = list(tiff_file.pages)[level]
    if page.is_tiled:
        return get_crop_tiled(tiff_file, level, x, y, h, w)
    return get_crop_untiled(tiff_file, level, x, y, h, w)


def get_crop_untiled(
    tiff_file: Path | tifffile.TiffFile, level: int, x: int, y: int, h: int, w: int
):
    if isinstance(tiff_file, Path):
        tiff_file = tifffile.TiffFile(tiff_file)

    page = list(tiff_file.pages)[level]

    if page.is_tiled:
        raise ValueError("Tiff file must not be tiled.")

    im_width = page.imagewidth
    im_height = page.imagelength

    if h < 1 or w < 1:
        raise ValueError("h and w must be strictly positive.")

    i1, j1 = x + h, y + w
    if x < 0 or y < 0 or i1 >= im_height or j1 >= im_width:
        raise ValueError(
            f"Requested crop area is out of image bounds. "
            f"{x}_{i1}_{im_height}, {y}_{j1}_{im_width}"
        )

    out = np.empty((page.imagedepth, h, w, page.samplesperpixel), dtype=np.uint8)
    fh = page.parent.filehandle

    for index in range(x, i1):
        offset = page.dataoffsets[index]
        bytecount = page.databytecounts[index]

        fh.seek(offset)
        data = fh.read(bytecount)

        tile, indices, shape = page.decode(data, index)

        out[:, index - x, :, :] = tile[:, :, y:j1, :]

    return out


def get_crop_tiled(
    tiff_file: Path | tifffile.TiffFile, level: int, x: int, y: int, h: int, w: int
):
    if isinstance(tiff_file, Path):
        tiff_file = tifffile.TiffFile(tiff_file)

    page = list(tiff_file.pages)[level]

    if not page.is_tiled:
        raise ValueError("Input page must be tiled.")

    im_width = page.imagewidth
    im_height = page.imagelength

    if h < 1 or w < 1:
        raise ValueError("h and w must be strictly positive.")

    if x < 0 or y < 0 or x + h >= im_height or y + w >= im_width:
        raise ValueError("Requested crop area is out of image bounds.")

    tile_width, tile_height = page.tilewidth, page.tilelength
    i1, j1 = x + h, y + w

    tile_x, tile_y = x // tile_height, y // tile_width
    tile_i1, tile_j1 = np.ceil([i1 / tile_height, j1 / tile_width]).astype(int)

    tile_per_line = int(np.ceil(im_width / tile_width))

    out = np.empty(
        (
            page.imagedepth,
            (tile_i1 - tile_x) * tile_height,
            (tile_j1 - tile_y) * tile_width,
            page.samplesperpixel,
        ),
        dtype=page.dtype,
    )

    fh = page.parent.filehandle

    for i in range(tile_x, tile_i1):
        for j in range(tile_y, tile_j1):
            index = int(i * tile_per_line + j)

            offset = page.dataoffsets[index]
            bytecount = page.databytecounts[index]

            fh.seek(offset)
            data = fh.read(bytecount)
            tile, indices, shape = page.decode(data, index)

            im_i = (i - tile_x) * tile_height
            im_j = (j - tile_y) * tile_width
            out[:, im_i : im_i + tile_height, im_j : im_j + tile_width, :] = tile

    im_x = x - tile_x * tile_height
    im_y = y - tile_y * tile_width

    return out[:, im_x : im_x + h, im_y : im_y + w, :]


def rasterize_scanpy(fig) -> None:
    """
    Rasterize figure containing Scatter plots of single cells
    such as PCA and UMAP plots drawn by Scanpy.
    """
    import warnings
    import matplotlib

    with warnings.catch_warnings(record=False):
        warnings.simplefilter("ignore")
        yes_class = (
            matplotlib.collections.PathCollection,
            matplotlib.collections.LineCollection,
        )
        not_clss = (
            matplotlib.text.Text,
            matplotlib.axis.XAxis,
            matplotlib.axis.YAxis,
        )
        if not hasattr(fig, "axes"):
            return
        if fig.axes is None:
            return
        for axs in fig.axes:
            for __c in axs.get_children():
                if not isinstance(__c, not_clss):
                    if not __c.get_children():
                        if isinstance(__c, yes_class):
                            __c.set_rasterized(True)
                    for _cc in __c.get_children():
                        if not isinstance(_cc, not_clss):
                            if isinstance(_cc, yes_class):
                                _cc.set_rasterized(True)


def isomap(anndata: AnnData, n_components: int = 2, **kwargs) -> AnnData:
    from sklearn.manifold import Isomap

    model = Isomap(n_components=n_components, **kwargs)
    anndata.obsm["X_isomap"] = model.fit_transform(anndata.X)
    return anndata


def mds(anndata: AnnData, n_components: int = 2, **kwargs) -> AnnData:
    from sklearn.manifold import MDS

    model = MDS(n_components=n_components, **kwargs)
    anndata.obsm["X_mds"] = model.fit_transform(anndata.X)
    return anndata


def z_score(x):
    return (x - x.mean()) / x.std()


def is_datetime(x: pd.Series) -> bool:
    if "datetime" in x.dtype.name:
        return True
    return False


def is_numeric(x: tp.Union[pd.Series, tp.Any]) -> bool:
    if not isinstance(x, pd.Series):
        x = pd.Series(x)
    if x.dtype.name in [
        "float",
        "float32",
        "float64",
        "int",
        "int8",
        "int16",
        "int32",
        "int64",
        "Int64",
    ] or is_datetime(x):
        return True
    if x.dtype.name in ["object", "string", "boolean", "bool"]:
        return False
    if x.dtype.name == "category":
        if len(set(type(i) for i in x)) != 1:
            raise ValueError("Series contains mixed types. Cannot transfer to color!")
        return is_numeric(x.iloc[0])
    raise ValueError(f"Cannot transfer data type '{x.dtype}' to color!")


sequential_cmaps = [
    "Purples",
    "Oranges",
    "Greens",
    "Blues",
    "Greys",
    "Reds",
    "spring",
    "summer",
    "autumn",
    "winter",
    "copper",
    "bone",
]

categorical_cmaps = [
    "Pastel1",
    "Pastel2",
    "Paired",
    "Accent",
    "Dark2",
    "Set1",
    "Set2",
    "Set3",
    "tab10",
    "tab20",
    "tab20b",
    "tab20c",
]


def clustermap_marsilea(
    data: pd.DataFrame,
    config: str = "abs",
    robust: bool | int = False,
    square: bool = False,
    metric: str = "euclidean",
    row_colors: None | pd.Series | pd.DataFrame = None,
    col_colors: None | pd.Series | pd.DataFrame = None,
    row_cmaps: None | list[str | None] = None,
    col_cmaps: None | list[str | None] = None,
    row_cluster: bool = True,
    col_cluster: bool = True,
    xticklabels: bool = True,
    yticklabels: bool = True,
    rasterize: bool = False,
    **kwargs,
):
    """
    Clustermap with Marsilea

    Parameters
    ----------
    data: pd.DataFrame
        Data to be plotted.
    config: str
        Shorthand configuration to use: 'abs' or 'z'. Default 'abs'.
        - 'abs': optimal for continuous, non-zero values (default colormap 'Reds').
        - 'z': z-score transformation on columns, zero-centered divergent colormap (default 'coolwarm').
    robust: bool | int
        Whether to use robust clustering. Default False.
        Effectively clips colormap to (100 - X)th and Xth percentiles.
        If an integer is provided it will cap color scale to that percentile. Default is 98.
    square: bool
        Whether to make the plot square such that the aspect ratio of each cell is 1. Default False.
        Text labels will also have the height of a heatmap row for improved readability.
    metric: str
        Distance metric to compute pairwise distances. Default 'euclidean'.
        See https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise_distances.html
    {row,col}_colors: pd.Series | pd.DataFrame | None
        Factors to label rows in a color mapping. Numeric or categorical values will be mapped to colormaps automatically.
        Can be a single Series or a DataFrame with factors as columns.
    {row,col}_cmaps: list[str] | None
        Colormaps to use for each factor. Must match the number of factors in `row_colors` or `col_colors`.
        By default, colormaps are selected automatically from a list of categorical or continuous colormaps accordingly.
    {row,col}_cluster: bool
        Whether to cluster rows or columns. Default True.
    {x,y}ticklabels: bool
        Whether to show tick labels for rows or columns. Default True.
    rasterize: bool
        Whether the main central heatmap should be rasterized. Default False.
    kwargs: dict
        Additional keyword arguments to be passed to `marsilea.Heatmap` for further customization.
        If overlapping with options above, the latter take precedence.
    """
    from collections import Counter
    import marsilea as ma
    import marsilea.plotter as mp

    # Checks on input
    assert config in ["abs", "z"]
    if row_colors is not None:
        assert data.index.isin(row_colors.index).all()
        row_colors = row_colors.reindex(data.index)
        row_cmaps = [None] * row_colors.shape[1] if row_cmaps is None else row_cmaps
    if col_colors is not None:
        assert data.columns.isin(col_colors.index).all()
        col_colors = col_colors.reindex(data.columns)
        col_cmaps = [None] * col_colors.shape[1] if col_cmaps is None else col_cmaps

    if isinstance(robust, bool):
        percentile = 98
    elif isinstance(robust, int | float):
        percentile = robust
        robust = True

    # Prepare kwargs to main heatmap
    if config == "abs":
        if robust:
            vmin = np.nanpercentile(data.values, 100 - percentile)
            vmax = np.nanpercentile(data.values, percentile)
        else:
            vmin = np.nanmin(data)
            vmax = np.nanmax(data)
        _kwargs = dict(vmin=vmin, vmax=vmax, cmap="Reds")
    elif config == "z":
        data = (data - data.mean()) / data.std()
        if robust:
            v = np.nanpercentile(data.abs().values, percentile)
        else:
            v = np.nanmax(data.abs().values)
        _kwargs = dict(vmin=-v, vmax=v, cmap="coolwarm", center=0)
    kwargs = _kwargs | kwargs
    if square:
        kwargs["height"], kwargs["width"] = np.asarray(data.shape) * 0.125

    # Main heatmap
    h = ma.Heatmap(data, **kwargs)

    # Axis labels
    if yticklabels:
        h.add_right(mp.Labels(data.index))
    if xticklabels:
        h.add_bottom(mp.Labels(data.columns))

    # Col and Row colors
    tracker: dict[str, int] = Counter()
    for pt, color_df, cmaps, func in [
        ("row", row_colors, row_cmaps, "add_left"),
        ("col", col_colors, col_cmaps, "add_top"),
    ]:
        f = getattr(h, func)
        if color_df is not None:
            if isinstance(color_df, pd.Series):
                color_df = color_df.to_frame()
            for name, cmap in zip(color_df.columns, cmaps):
                d = color_df[name].fillna(0)
                if is_numeric(d):
                    if cmap is None:
                        tracker["num"] += 1
                        cmap = sequential_cmaps[tracker["num"]]
                    f(
                        mp.ColorMesh(
                            d,
                            cmap=cmap,
                            # label=name,
                            # label_props=dict(fontweight="regular"),
                        ),
                        size=0.2,
                    )
                else:
                    if cmap is None:
                        tracker["cat"] += 1
                        cmap = categorical_cmaps[tracker["cat"]]
                    d = pd.Categorical(d)
                    f(
                        mp.Colors(
                            d,
                            cmap=cmap,
                            # label=name,
                            # label_props=dict(fontweight="regular"),
                        ),
                        size=0.2,
                    )  # , label=name
    h.add_legends()

    # Dendrograms
    if row_cluster:
        h.add_dendrogram("left", size=0.25, metric=metric)
    if col_cluster:
        h.add_dendrogram("top", size=0.25, metric=metric)

    # Render
    h.render()

    if rasterize:
        import matplotlib

        for child in h.get_main_ax().get_children():
            if isinstance(child, matplotlib.collections.QuadMesh):
                child.set_rasterized(True)

    # Convenience shortcuts (for seaborn_extension compatibility)
    h.fig = h.figure
    h.savefig = h.fig.savefig
    return h


def signed_fold_change(a, b):
    fold_change = np.abs(a - b) / np.abs(b)
    sign = np.sign(a - b)
    signed_fold_change = sign * fold_change
    return signed_fold_change


from matplotlib.ticker import Locator


class MinorSymLogLocator(Locator):
    """
    Dynamically find minor tick positions based on the positions of
    major ticks for a symlog scaling.
    """

    def __init__(self, linthresh):
        """
        Ticks will be placed between the major ticks.
        The placement is linear for x between -linthresh and linthresh,
        otherwise its logarithmically
        """
        self.linthresh = linthresh

    def __call__(self):
        "Return the locations of the ticks"
        majorlocs = self.axis.get_majorticklocs()

        # iterate through minor locs
        minorlocs = []

        # handle the lowest part
        for i in range(1, len(majorlocs)):
            majorstep = majorlocs[i] - majorlocs[i - 1]
            if abs(majorlocs[i - 1] + majorstep / 2) < self.linthresh:
                ndivs = 10
            else:
                ndivs = 9
            minorstep = majorstep / ndivs
            locs = np.arange(majorlocs[i - 1], majorlocs[i], minorstep)[1:]
            minorlocs.extend(locs)

        return self.raise_if_exceeds(np.array(minorlocs))

    def tick_values(self, vmin, vmax):
        raise NotImplementedError(
            "Cannot get tick locations for a " "%s type." % type(self)
        )


class DummySplitter:
    def split(self, X, y, groups):
        yield range(len(X)), range(len(X))

    def get_n_splits(self):
        return 1
