# Tissue-clocks

[![Biorxiv Badge](https://img.shields.io/static/v1?label=bioRxiv&message=10.1101/2024.11.14.618081&color=red&logo=biorxiv)](https://doi.org/10.1101/2024.11.14.618081) ⬅️ read the preprint here <br>
[![Lab Badge](https://img.shields.io/static/v1?label=Lab&message=rendeiro.group&color=blue&logo=github)](http://rendeiro.group)

**Histological aging signatures enable tissue-specific disease prediction from blood.**

We develop histological aging clocks from H&E-stained tissue sections to predict biological age across multiple human tissues. These clocks capture tissue-specific aging signatures that associate with pathological outcomes, telomere length, and morbidity — providing a morphological window into the aging process. Transferring these clocks to blood gene expression profiles enables non-invasive prediction of tissue-specific aging and disease risk, highlighting the potential of histological features as biomarkers for aging and pathology.
Licensed under [PolyForm Noncommercial 1.0.0](LICENSE). Fine-tuning data: [Zenodo 10.5281/zenodo.13330659](https://doi.org/10.5281/zenodo.13330659).

---

## 🚀 Setup

This project uses [uv](https://docs.astral.sh/uv/) for Python package and environment management, and [just](https://github.com/casey/just) as a command runner.

To install dependencies:
```bash
uv sync
```

To see available commands:
```bash
just
```

## 🔬 Running the analysis

Run the full original analysis pipeline:
```bash
just analysis
```

Run specific analysis stages:
```bash
just visualize   # Cohort and dataset inspection
just train       # Model fine-tuning
just features    # Histological feature extraction
just clocks      # Histological aging clocks
just rna         # Gene expression analysis
just dname       # DNA methylation analysis
just blood       # Blood prediction and associations
just validation  # Validation steps
```

Run revision work:
```bash
just revision    # All revision steps
```

Or run everything:
```bash
just analysis_full
```

---

## 📁 Code organization

The [`src`](src) directory contains all code used in the project.
Not all steps are fully reproducible due to constraints in obtaining the data from dbGAP.
Some steps while being reproducible are not feasible to be run in serial as they were run on multiple machines and configuring these is very environment dependent.

Below is an overview of all code used in the project.

### 📥 Getting data
- [`download_slides.sh`](src/download_slides.sh)
- [`download_gtex_dbGAP_restricted.sh`](src/download_gtex_dbGAP_restricted.sh)
- [`download_gtex_dname.sh`](src/download_gtex_dname.sh)

> ⚠️ Not fully reproducible due to dbGAP constraints.

### 📊 Cohort and dataset inspection and QC
- [`visualize_sample_attributes.py`](src/visualize_sample_attributes.py)
- [`visualize_morbidities.py`](src/visualize_morbidities.py)
- [`visualize_pathology_data.py`](src/visualize_pathology_data.py)
- [`visualize_telomere_data.py`](src/visualize_telomere_data.py)

### 🧠 Model fine-tuning
- [`train_fastai_pre.py`](src/train_fastai_pre.py)
- [`train_fastai.py`](src/train_fastai.py)
- [`train_fastai_post.py`](src/train_fastai_post.py)
- [`train_fastai_metrics.py`](src/train_fastai_metrics.py)

### 🔍 Histological feature extraction
- [`feature_extractor.py`](src/feature_extractor.py)
- [`feature_extractor.job.sh`](src/feature_extractor.job.sh)
- [`feature_extractor.submit.py`](src/feature_extractor.submit.py)
- [`explore_histological_feature_space.py`](src/explore_histological_feature_space.py)
- [`variance_explained_factors.py`](src/variance_explained_factors.py)

### ⏰ Histological aging clocks (tissue-clocks)
- [`predict_histological_age.py`](src/predict_histological_age.py)
- [`visualize_age_gaps.py`](src/visualize_age_gaps.py)
- [`interpret_age_gaps_pathology.tissue.py`](src/interpret_age_gaps_pathology.tissue.py)
- [`interpret_age_gaps_telomeres.tissue.py`](src/interpret_age_gaps_telomeres.tissue.py)
- [`interpret_age_gaps_across_tissues.py`](src/interpret_age_gaps_across_tissues.py)

### 💬 Interpretation through text
- [`interpret_age_gaps_text.py`](src/interpret_age_gaps_text.py)

### 🧬 Gene expression
- [`explore_rna_feature_space.py`](src/explore_rna_feature_space.py)
- [`interpret_age_gene_expression.py`](src/interpret_age_gene_expression.py)

### 🧪 DNA methylation data
- [`explore_dname_feature_space.py`](src/explore_dname_feature_space.py)
- [`predict_dname_age.py`](src/predict_dname_age.py)
- [`compare_dname_age.py`](src/compare_dname_age.py)

### 🔗 Associations
- [`associate_age_gaps_factors.py`](src/associate_age_gaps_factors.py)

### 🩸 Prediction from blood
- [`predict_gaps_from_blood_expression.py`](src/predict_gaps_from_blood_expression.py)
- [`interpret_age_gaps_pathology.blood.py`](src/interpret_age_gaps_pathology.blood.py)
- [`interpret_age_gaps_telomeres.blood.py`](src/interpret_age_gaps_telomeres.blood.py)

### ✅ Validation
- [`prepare_archs4_data.py`](src/prepare_archs4_data.py)
- [`validate_blood_predictors.py`](src/validate_blood_predictors.py)

### 🛠️ Cross project, supporting
- [`ops.py`](src/ops.py)
- [`utils.py`](src/utils.py)

### 📝 Revision work
- **Cohorts:**
    - [`cohort_description_table.py`](src/revision/cohort_description_table.py)
- **Vision models:**
    - [`make_clocks_imagenet.py`](src/revision/make_clocks_imagenet.py)
- **Foundation models and new cohorts:**
    - [`process_lz.py`](src/revision/process_lz.py)
    - [`process_lz_aggregate.py`](src/revision/process_lz_aggregate.py)
    - [`new_cohorts_clocks.py`](src/revision/new_cohorts_clocks.py)
    - [`cross_apply_clocks.py`](src/revision/cross_apply_clocks.py)
    - [`correlate_histology_dname.py`](src/revision/correlate_histology_dname.py)
    - *Outcomes:*
        - [`interpret_age_gaps_telomeres.tissue.imagenet.py`](src/revision/interpret_age_gaps_telomeres.tissue.imagenet.py)
        - [`interpret_age_gaps_morbidity.tissue.imagenet.py`](src/revision/interpret_age_gaps_morbidity.tissue.imagenet.py)
        - [`interpret_age_gaps_telomeres.tissue.py`](src/revision/interpret_age_gaps_telomeres.tissue.py)
        - [`interpret_age_gaps_pathology.tissue.py`](src/revision/interpret_age_gaps_pathology.tissue.py)
        - [`interpret_age_gaps_morbidity.tissue.py`](src/revision/interpret_age_gaps_morbidity.tissue.py)
- **GNNs:**
    - [`gnn.prepare_graphs.py`](src/revision/gnn.prepare_graphs.py)
    - [`gnn.train.py`](src/revision/gnn.train.py)
    - [`gnn.train.per_organ.py`](src/revision/gnn.train.per_organ.py)
- **DNAme comparison:**
    - [`compare_dname-histology_telomeres.py`](src/revision/compare_dname-histology_telomeres.py)
    - [`compare_dname-histology_pathology.py`](src/revision/compare_dname-histology_pathology.py)
    - [`compare_dname-histology_morbidity.py`](src/revision/compare_dname-histology_morbidity.py)
    - [`compare_dname-histology_summary.py`](src/revision/compare_dname-histology_summary.py)
- **Specialized analysis:**
    - [`compare_plip_conch.py`](src/revision/compare_plip_conch.py)
    - [`ecm_analysis.py`](src/revision/ecm_analysis.py)
- **Validation metrics:**
    - [`icc_reproducibility.py`](src/revision/icc_reproducibility.py) - Within-tissue reproducibility via ICC (tile sampling)
    - [`icc_sample_tiles.py`](src/revision/icc_sample_tiles.py) - Sample tiles from GTEx zarrs (run on remote)
    - [`icc_apply_clocks.py`](src/revision/icc_apply_clocks.py) - Apply clocks to sampled tiles and compute ICC
    - [`bland_altman.py`](src/revision/bland_altman.py) - Bland-Altman plots + calibration (slope/intercept/R²) with combined grid visualization
- **Jupyter notebook:**
    - [`make_notebook.py`](src/revision/make_notebook.py)
    - [`tissue_clock_walkthrough.ipynb`](src/revision/tissue_clock_walkthrough.ipynb)

### 📋 Supporting material
- [`supplementary_tables.py`](src/supplementary_tables.py)

---

## 📜 Citation

If you use this work, please cite our preprint:

> **Histological aging signatures enable tissue-specific disease prediction from blood**
> Ernesto Abila, Iva Buljan, Yimin Zheng, Tamas Veres, Zhilong Weng, Maja Nackenhorst, Wolfgang Hulla, Yuri Tolkach, Adelheid Wöhrer, André F. Rendeiro
> bioRxiv 2024.11.14.618081. DOI: [10.1101/2024.11.14.618081](https://doi.org/10.1101/2024.11.14.618081)
> **TODO:** After Zenodo archives the GitHub release, set the top-level `doi` in [`CITATION.cff`](CITATION.cff) to the Zenodo code DOI (currently the bioRxiv DOI).
> **TODO:** Once the Nature Medicine DOI is available, update this block and the BibTeX entry below, and update `preferred-citation` in [`CITATION.cff`](CITATION.cff).

Full author list and metadata: [`CITATION.cff`](CITATION.cff).

```bibtex
@article {Abila2024.11.14.618081,
	author = {Abila, Ernesto and Buljan, Iva and Zheng, Yimin and Veres, Tamas and Weng, Zhilong and Nackenhorst, Maja and Hulla, Wolfgang and Tolkach, Yuri and W{\"o}hrer, Adelheid and Rendeiro, Andr{\'e} F.},
	title = {Histological aging signatures enable tissue-specific disease prediction from blood},
	elocation-id = {2024.11.14.618081},
	year = {2026},
	doi = {10.1101/2024.11.14.618081},
	publisher = {Cold Spring Harbor Laboratory},
	URL = {https://www.biorxiv.org/content/early/2026/01/15/2024.11.14.618081},
	eprint = {https://www.biorxiv.org/content/early/2026/01/15/2024.11.14.618081.full.pdf},
	journal = {bioRxiv}
}
```

---

## 📬 Contact

For questions or collaborations, visit [rendeiro.group](http://rendeiro.group) or open an issue in this repository.
