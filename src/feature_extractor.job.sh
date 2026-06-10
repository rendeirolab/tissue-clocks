#!/usr/bin/env bash

LIST_FILE=$1
MODEL=$2
WEIGHTS=$3
HUB=$4
ADDITIONAL=$5

# Stagger jobs
sleep $((RANDOM%5+2))

# Get ID from file line
ID=$(sed -n ${SLURM_ARRAY_TASK_ID}p $LIST_FILE)


# Download
echo "Processing '${ID}'."

export PYTHONPATH="${PYTHONPATH}:$(pwd)"

python -u src/feature_extractor.py \
    --allow-download --slide-id $ID data/gtex/svs \
    --model-name $MODEL --model-hub $HUB --model-weights $WEIGHTS \
    --tile-size 224 --coord-shift 0 --output-suffix ${MODEL}.224px $ADDITIONAL

python -u src/feature_extractor.py \
    --allow-download --slide-id $ID data/gtex/svs \
    --model-name $MODEL --model-hub $HUB --model-weights $WEIGHTS \
    --tile-size 448 --coord-shift -112 --output-suffix ${MODEL}.448px $ADDITIONAL

python -u src/feature_extractor.py \
    --allow-download --slide-id $ID data/gtex/svs \
    --model-name $MODEL --model-hub $HUB --model-weights $WEIGHTS \
    --tile-size 894 --coord-shift -336 --output-suffix ${MODEL}.894px $ADDITIONAL

echo "Completed '${ID}'."



# # To submit to missing slides:
# LIST_FILE=slides.txt
# python -c "\
# from pathlib import Path; \
# import pandas as pd; \
# df = pd.read_csv('data/gtex/GTEx Portal.csv', index_col=0); \
# sel = [not Path(f'data/gtex/svs/{s}.svs').exists() for s in df.index]; \
# print(sum(sel)); \
# df.loc[sel].index.to_series().to_csv('$LIST_FILE', index=False, header=False)
# "
# N=`cat  $LIST_FILE | wc -l`
# sbatch --array=1-${N}%200 -p shortq --qos shortq --mem 8000 -c 4 -J extract_GTEx-${MODEL} src/extract_features.sh $LIST_FILE

# # Compile failures
# cat slurm-*.out > logs.out
# grep "Failed for " logs.out | sed "s/Failed for file '//g" | sed "s/.svs'..*//g" > failed_slides.txt

# # To test a new model:
# export PYTHONPATH=.
# srun -c 4 --mem 32000 -p shortq --qos shortq -J test_efficientnet_v2_l \
# python -u src/feature_extractor.py \
# --allow-download --slide-id $ID data/gtex/svs \
# --model-name $MODEL --model-hub $HUB \
# --tile-size 224 --coord-shift 0 --output-suffix ${MODEL}_224px
