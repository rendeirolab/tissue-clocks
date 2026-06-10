#!/usr/bin/env bash

# Stagger jobs
sleep $((RANDOM%5+2))

# Get ID from file line
ID=$(sed -n ${SLURM_ARRAY_TASK_ID}p slides.txt)
URL="https://brd.nci.nih.gov/brd/imagedownload/${ID}"
# Download
echo "Downloading '${ID}'."
wget -O data/gtex/svs/${ID}.svs $URL
echo "Completed '${ID}'."

# # To submit:
# N=`cat  slides.txt | wc -l`
# sbatch --array=1-${N}%10 -p shortq --qos shortq --mem 8000 -c 1 -J get_GTEx download_slides.sh
