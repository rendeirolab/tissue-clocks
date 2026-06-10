from pathlib import Path

from tqdm import tqdm
import pandas as pd
import numpy as np
from statsmodels import api as sm
import matplotlib.pyplot as plt
import seaborn as sns

from seaborn_extensions import clustermap

from src.utils import get_engineered_info, get_restricted_info, get_pathology_data


metadata_dir = Path("metadata")
data_dir = Path("data")
results_dir = Path("results") / "tissue_clocks_revision" / "cohort_description_table"
results_dir.mkdir(parents=True, exist_ok=True)
figkws = dict(dpi=300, bbox_inches="tight")


rest, _ = get_restricted_info()

# pathology
path = get_pathology_data()
path.index = path.index.str.split("-").str[:2].str.join("-")
n_tissues = path.groupby(level=0).size().rename("n_tissues")
n_pathologies = path.groupby(level=0).sum().sum(axis=1).rename("n_pathologies")
path_per_tissue = (n_pathologies / n_tissues).rename("pathologies_per_tissue")

# comorbidities
feats = get_engineered_info()
morb = feats.loc[:, feats.columns.str.startswith("morbidity")].astype(bool)
n = morb.sum(0).rename("n_individuals").sort_values()
morb = morb.loc[:, n >= 3].astype(float)
n_comorbidities = morb.sum(1).rename("n_comorbidities")

# all
df = rest.join([n_tissues, n_pathologies, path_per_tissue, n_comorbidities], how="left")


variables = [
    "Age",
    "Sex",
    "Race",
    "Ethnicity",
    "BMI",
    "Manner Of Death",
    "n_tissues",
    "n_pathologies",
    "pathologies_per_tissue",
    "n_comorbidities",
]

df[variables]
summary = dict()
for v in variables:
    if df[v].dtype.name == "category" or df[v].dtype.name == "object":
        vc = df[v].value_counts(dropna=False)
        vc.index = vc.index.astype(str)
        summary[v] = vc
    else:
        summary[v] = pd.Series(
            {
                "mean": df[v].mean(),
                "std": df[v].std(),
                "min": df[v].min(),
                "25%": df[v].quantile(0.25),
                "50%": df[v].median(),
                "75%": df[v].quantile(0.75),
                "max": df[v].max(),
                "n_missing": df[v].isna().sum(),
            }
        )

# Make excel table in a single sheet
tr = pd.DataFrame(summary).T
tr.index.name = "variable"
with pd.ExcelWriter(results_dir / "cohort_description.xlsx") as writer:
    tr.to_excel(writer, sheet_name="cohort_description")


# Now per age bracket
df["Age Bracket"] = pd.cut(
    df["Age"],
    labels=["20-29", "30-39", "40-49", "50-59", "60-70"],
    bins=[20, 30, 40, 50, 60, 70],
)
age_bracket_summary = dict()
for v in variables[4:]:
    if df[v].dtype.name == "category" or df[v].dtype.name == "object":
        vc = (
            df.groupby("Age Bracket")[v]
            .value_counts(dropna=False)
            .unstack(fill_value=0)
        )
        vc.columns = vc.columns.astype(str)
        age_bracket_summary[v] = vc
    else:
        q25 = lambda x: x.quantile(0.25)
        q75 = lambda x: x.quantile(0.75)
        age_bracket_summary[v] = df.groupby("Age Bracket")[v].agg(
            ["min", q25, "mean", "median", q75, "std", "max", "count"]
        )
# Make excel table in a single sheet
tr = pd.concat(age_bracket_summary)
with pd.ExcelWriter(results_dir / "cohort_description_per_age_bracket.xlsx") as writer:
    tr.to_excel(writer, sheet_name="cohort_description_per_age_bracket")


df["Immediate Cause Of Death"].value_counts().head(60)
df["First Underlying Cause Of Death"].value_counts().head(60)
df["Manner Of Death"].value_counts().head(60)


causes = {
    "mi": "myocardial infarction",
    "esld": "end stage liver disease",
    "arrest; cardiac": "cardiac arrest",
    "anoxia": "anoxic brain injury",
    "cardiac arrest": "cardiac arrest",
    "head trauma": "head trauma",
    "heart disease": "heart disease",
    "cva": "cerebrovascular accident",
    "cva/stroke": "cerebrovascular accident",
    "cerebral vascular accident": "cerebrovascular accident",
    "cerebrovascular/stroke": "cerebrovascular accident",
    "vascular cerebral accident": "cerebrovascular accident",
    "respiratory disease": "respiratory disease",
    "respiratory failure": "respiratory failure",
    "myocardial infarction": "myocardial infarction",
    "stroke": "cerebrovascular accident",
    "arrest": "cardiac arrest",
    "strokearrest": "cardiac arrest",
    "cardiactrauma": "trauma",
    "cardiopulmonary failure": "cardiopulmonary failure",
    "intracranial hemorrhage": "intracranial hemorrhage",
    "respiratory arrest": "respiratory arrest",
    "unknown cause of death": "unknown cause of death",
    "trauma, multiple": "multiple trauma",
    "renal failure": "renal failure",
    "cva - cerebrovascular accident": "cerebrovascular accident",
    "end stage liver disease": "end stage liver disease",
    "prescription drug overdose": "prescription drug overdose",
    "probable mi": "probable myocardial infarction",
    "trauma, head": "head trauma",
    "hepatic encephalopathy": "hepatic encephalopathy",
    "miliver failure": "liver failure",
    "unknown, do not have a copy of the death certificate or an me report": "unknown cause of death",
    "intracranial bleed": "intracranial hemorrhage",
    "liver disease, other chronic and cirrhosis": "chronic liver disease and cirrhosis",
    "heart failure": "heart failure",
    "anoxic encephalopathy": "anoxic brain injury",
    "cardiac failure nos": "cardiac failure",
    "gi bleed": "gastrointestinal bleed",
    "ichesld": "unknown cause of death",
    "esrd (end stage renal disease)": "end stage renal disease",
    "mi - myocardial infarction": "myocardial infarction",
    "multiple": "multiple trauma",
    "als": "amyotrophic lateral sclerosis",
    "pulmonary embolism": "pulmonary embolism",
    "cardiac arrest, cause unspecified": "cardiac arrest",
    "mi, myocardial infarction": "myocardial infarction",
    "head trama": "head trauma",
    "acute renal failure": "acute renal failure",
    "nontraumatic intracerebral hemorrhage": "nontraumatic intracerebral hemorrhage",
    "stroke, cerebrovascular": "cerebrovascular accident",
    "drug overdose": "drug overdose",
    "sepsis": "sepsis",
    "toxic effect of unspecified substance": "toxic effect of unspecified substance",
    "acute myocardial infarction": "myocardial infarction",
    "myocardial infarc": "myocardial infarction",
    "anoxic brain injury": "anoxic brain injury",
    "end-stage renal disease": "end stage renal disease",
    "epilepsy": "epilepsy",
    "alcoholic cirrhosis of liver": "alcoholic cirrhosis of liver",
    "cerebral vascular accident (intracranial hemorrhage)": "cerebrovascular accident (intracranial hemorrhage)",
    "kidney failure acute": "acute kidney failure",
    "trauma due to fall": "trauma from fall",
    "myocardial infarction acute": "myocardial infarction",
    "intraperitoneal bleed": "intraperitoneal hemorrhage",
    "asphyxiation due to hanging": "asphyxia from hanging",
    "biventricular heart failure": "heart failure",
    "pneumonitis due to solids and liquids": "aspiration pneumonitis",
    "arrestblunt force trauma": "blunt force trauma",
    "organ transplant+complication": "complications from organ transplant",
    "anoxia encephalopathy": "anoxic brain injury",
    "pontine bleed": "pontine hemorrhage",
    "brain cancer": "brain cancer",
    "sirs": "systemic inflammatory response syndrome",
    "cerebrovascular diseases": "cerebrovascular disease",
    "subarachnoid hemorrhage": "subarachnoid hemorrhage",
    "metabolic acidosis related to diabetic ketoacidosis": "metabolic acidosis from diabetic ketoacidosis",
    "cardiac failure": "cardiac failure",
    "blunt injury": "blunt injury",
    "seizures": "seizures",
    "head trauma secondary to mva": "head trauma secondary to motor vehicle accident",
    "intracranial hemorrage (ich)": "intracranial hemorrhage",
    "copd/pulmonary edema": "COPD and pulmonary edema",
    "trauma-blunt force trauma secondary to mva": "blunt force trauma secondary to motor vehicle accident",
    "probable mi due to hypertrophic cardiomypoathy": "probable myocardial infarction due to hypertrophic cardiomyopathy",
    "anoxic brain injury secondary to mva trauma": "anoxic brain injury",
    "head trauma/anoxia": "head trauma and anoxic brain injury",
    "mi second to pneumothorax": "myocardial infarction secondary to pneumothorax",
    "smoke inhalation-respiratory disease": "respiratory disease from smoke inhalation",
    "esrd": "end stage renal disease",
    "ich secondary to head trauma from falls": "intracranial hemorrhage from head trauma",
    "sah/ich secondary to ruptured aneurysm": "subarachnoid hemorrhage/intracranial hemorrhage from ruptured aneurysm",
    "cad": "coronary artery disease",
    "multiple blunt force trauma secondary to mva": "multiple blunt force trauma secondary to motor vehicle accident",
    "pneumonia": "pneumonia",
    "trauma sah/sdh": "subarachnoid hemorrhage/subdural hemorrhage from trauma",
    "suspected overdose": "suspected overdose",
    "seizure disorder": "seizure disorder",
    "massive heart attack": "myocardial infarction",
    "heart disease (mi)": "heart disease",
    "trauma-head bleed result of a fall": "head bleed from fall",
    "suicide-hanging": "suicide by hanging",
    "anoxic encephalopathy secondary to asphyxia/hanging": "anoxic encephalopathy secondary to asphyxia",
    "lower respiratory diseases, chronic, other": "chronic lower respiratory disease",
    "dementia": "dementia",
    "anoxic brain injury secondary to cardiac arrest": "anoxic brain injury",
    "kidney disease": "kidney disease",
    "cva traumatic brain injury": "traumatic brain injury",
    "motor vehicle accident trauma": "motor vehicle accident trauma",
    "shock secondary to burns": "shock from burns",
    "copd": "COPD",
    "anoxia secondary to hanging": "anoxic brain injury",
    "complications from als": "complications from amyotrophic lateral sclerosis",
    "gsw to head": "gunshot wound to head",
    "blunt force trauma to head": "blunt force trauma to head",
    "respiratory diseases": "respiratory disease",
    "allergic reaction": "allergic reaction",
    "cva (cerebral vascular accident)": "cerebrovascular accident",
    "acute mi": "myocardial infarction",
    "ruptured aaa": "ruptured abdominal aortic aneurysm",
    "pe": "pulmonary embolism",
    "aortic dissection": "aortic dissection",
    "intracranial haemorrhage nos": "intracranial hemorrhage",
    "liver": "liver cirrhosis",
    "cirrhosis": "liver cirrhosis",
    "gsw - gun shot wound": "gunshot wound",
    "hemorrhage; intracerebral, nontraumatic": "nontraumatic intracerebral hemorrhage",
    "dysrhythmia; cardiac": "cardiac dysrhythmia",
    "aspiration pneumonia due to regurgitated food": "aspiration pneumonia",
    "failure, renal": "renal failure",
    "hemorrhage": "hemorrhage",
    "subarachnoid, nontraumatic, sequelae": "subarachnoid hemorrhage",
    "bowel obstruction": "bowel obstruction",
    "atherosclerotic cardiovascular disease": "atherosclerotic cardiovascular disease",
    "multisystem organ failure": "multisystem organ failure",
    "probable drug overdose": "probable drug overdose",
    "asphyxia secondary to hanging": "asphyxia from hanging",
    "infarcted bowel secondary to aaa repair": "infarcted bowel from ruptured abdominal aortic aneurysm repair",
    "acute renal failure secondary to polycystic kidney disease": "acute renal failure from polycystic kidney disease",
    "aspiration": "aspiration",
    "unknown": "unknown cause of death",
    "cardiac asystole": "cardiac arrest",
    "head injury trauma multiple": "multiple head injuries",
    "contact with blunt object, undetermined intent causing accidental injury": "blunt object injury",
    "cva (cerebrovascular accident)": "cerebrovascular accident",
    "arrest pulmonary": "pulmonary arrest",
    "arrest [as an cardiac arrest]": "cardiac arrest",
    "cardiovascular": "cardiovascular collapse",
    "collapse": "cardiovascular collapse",
    "kidney failure": "kidney failure",
    "poisoning by overdose of substance[x]": "drug overdose",
    "intent self harm by hanging strangulation / suffocation": "suicide by hanging",
    "fall (event)": "fall",
    "liver disease": "liver disease",
    "parkinson's disease": "parkinson's disease",
    "scleroderma": "scleroderma",
    "stroke/cerebrovascular accident": "cerebrovascular accident",
    "failure to thrive": "failure to thrive",
    "cerebrovascular": "cerebrovascular accident",
    "cerebrovascular stroke": "cerebrovascular accident",
    "cerebrovascular accidents": "cerebrovascular accidents",
    "cancer": "cancer",
    "cardiac failure, nos": "cardiac failure",
    "alcoholism": "alcoholism",
    "abdominal aortic aneurysm, ruptured": "ruptured abdominal aortic aneurysm",
    "subarachnoid": "subarachnoid hemorrhage",
    "hemorrhage, nontraumatic": "nontraumatic hemorrhage",
    "lung disease": "lung disease",
    "pneumonia, bacterial": "bacterial pneumonia",
    "nontraumatic subarachnoid hemorrhage from right vertebral artery": "nontraumatic subarachnoid hemorrhage",
    "motor- or nonmotor-vehicle accident, type of vehicle unspecified": "motor vehicle accident",
    "hypovolemic shock": "hypovolemic shock",
    "drowning and submersion, undetermined intent": "drowning and submersion",
    "congestive failure heart": "congestive heart failure",
    "tbi (traumatic brain injury)": "traumatic brain injury",
    "diabetes mellitus due to underlying condition with unspecified complications": "diabetes mellitus complications",
    "death - cause unknown": "unknown cause of death",
    "hypernatremia": "hypernatremia",
    "natural causes": "natural causes",
    "cardiac arrest, unspecified": "cardiac arrest",
    "heart attack": "myocardial infarction",
    "accident cerebrovascular": "cerebrovascular accident",
    "cardiorespiratory failure": "cardiorespiratory failure",
    "myocardial infarction (mi)": "myocardial infarction",
    "septal def.-ventricular": "ventricular septal defect",
    "suicide by hanging": "suicide by hanging",
    "cardiac arrest - asystole": "cardiac arrest",
    "arrest cardiac": "cardiac arrest",
    "alcoholism (disorder)": "alcoholism",
}


df["Immediate Cause Of Death"].str.lower().replace(causes).value_counts()

with pd.ExcelWriter(results_dir / "cohort_manner_of_death.xlsx") as writer:
    df["Immediate Cause Of Death"].str.lower().replace(causes).value_counts().head(
        12
    ).to_excel(writer)
