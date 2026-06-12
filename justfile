# Justfile for the tissue-clocks project
# Histopathology of human aging to predict biological age and pathology

set shell := ["bash", "-cu"]

name := `basename $(pwd)`

default: help

# Display help and available commands
help:
    @echo "Justfile for the {{name}} project"
    @echo ""
    @echo "Available commands:"
    @just --list

# Install Python requirements via uv
requirements:
    uv sync

_backup_time:
    echo "Last backup: $(date)" >> _backup_time
    chmod 700 _backup_time

_sync:
    rsync --copy-links --progress -r . arendeiro@login:projects/{{name}}

# [dev] Sync data/code to remote machine
sync: _sync _backup_time

# [dev] Start an interactive IPython session
interactive:
    @echo "Starting an interactive IPython session"
    uv run python -m IPython -i src/__init__.py

# [dev] Get datasets (warning: requires dbGAP approval)
get_data:
    @echo "Warning: this step is not meant to be run, but simply details how datasets were downloaded."
    bash src/download_slides.sh
    bash src/download_gtex_dbGAP_restricted.sh
    bash src/download_gtex_dname.sh

# === Analysis pipeline ===

# Cohort and dataset inspection and QC
visualize:
    uv run python src/visualize_sample_attributes.py
    uv run python src/visualize_morbidities.py
    uv run python src/visualize_pathology_data.py
    uv run python src/visualize_telomere_data.py

# Model fine-tuning
train:
    uv run python src/train_fastai_pre.py
    uv run python src/train_fastai.py
    uv run python src/train_fastai_post.py
    uv run python src/train_fastai_metrics.py

# Histological feature extraction
features:
    uv run python src/feature_extractor.submit.py
    uv run python src/explore_histological_feature_space.py
    uv run python src/variance_explained_factors.py

# Histological aging clocks (tissue-clocks)
clocks:
    uv run python src/predict_histological_age.py
    uv run python src/visualize_age_gaps.py
    uv run python src/interpret_age_gaps_pathology.tissue.py
    uv run python src/interpret_age_gaps_telomeres.tissue.py
    uv run python src/interpret_age_gaps_across_tissues.py
    uv run python src/interpret_age_gaps_text.py

# Gene expression analysis
rna:
    uv run python src/explore_rna_feature_space.py
    uv run python src/interpret_age_gene_expression.py

# DNA methylation analysis
dname:
    uv run python src/explore_dname_feature_space.py
    uv run python src/predict_dname_age.py
    uv run python src/compare_dname_age.py

# Associations and blood prediction
blood:
    uv run python src/associate_age_gaps_factors.py
    uv run python src/predict_gaps_from_blood_expression.py
    uv run python src/interpret_age_gaps_pathology.blood.py
    uv run python src/interpret_age_gaps_telomeres.blood.py

# Validation
validation:
    uv run python src/prepare_archs4_data.py
    uv run python src/validate_blood_predictors.py

# Run all main analysis steps
analysis: visualize train features clocks rna dname blood validation
    uv run python src/supplementary_tables.py
    @echo "Analysis complete!"

# === Revision work ===

# Vision models and cohort description
revision_cohorts:
    uv run python src/revision/cohort_description_table.py
    uv run python src/revision/make_clocks_imagenet.py

# Foundation models and new cohorts
revision_foundation:
    uv run python src/revision/process_lz.py
    uv run python src/revision/process_lz_aggregate.py
    uv run python src/revision/new_cohorts_clocks.py
    uv run python src/revision/cross_apply_clocks.py
    uv run python src/revision/correlate_histology_dname.py

# Outcome interpretation (revision)
revision_outcomes:
    uv run python src/revision/interpret_age_gaps_telomeres.tissue.imagenet.py
    uv run python src/revision/interpret_age_gaps_morbidity.tissue.imagenet.py
    uv run python src/revision/interpret_age_gaps_telomeres.tissue.py
    uv run python src/revision/interpret_age_gaps_pathology.tissue.py
    uv run python src/revision/interpret_age_gaps_morbidity.tissue.py

# GNN models
revision_gnn:
    uv run python src/revision/gnn.prepare_graphs.py
    uv run python src/revision/gnn.train.py
    uv run python src/revision/gnn.train.per_organ.py

# DNAme-histology comparison
revision_dname_comparison:
    uv run python src/revision/compare_dname-histology_telomeres.py
    uv run python src/revision/compare_dname-histology_pathology.py
    uv run python src/revision/compare_dname-histology_morbidity.py
    uv run python src/revision/compare_dname-histology_summary.py

# Specialized analysis and notebook
revision_specialized:
    uv run python src/revision/compare_plip_conch.py
    uv run python src/revision/ecm_analysis.py
    uv run python src/revision/make_notebook.py

# Run all revision steps
revision: revision_cohorts revision_foundation revision_outcomes revision_gnn revision_dname_comparison revision_specialized

# Run full analysis including revision work
analysis_full: analysis revision
    @echo "Full analysis (including revision) complete!"

# === Data management ===

# [dev] Upload processed files to Zenodo
upload_data:
    @echo "Warning: this step is not meant to be run, but simply details how datasets were uploaded."
    uv run python src/upload_to_zenodo.py

# Download processed data from Zenodo (for reproducibility)
download_data:
    @echo "Not yet implemented!"
    uv run python src/download_from_zenodo.py

# === Figures ===

# Produce figures in various formats, timestamped
figures:
    cd figures && bash process.sh