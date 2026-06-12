# SPDX-FileCopyrightText: Copyright (c) 2026 Rendeiro Lab, CeMM - Research Center for Molecular Medicine, Austrian Academy of Sciences
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#
# Licensed under the PolyForm Noncommercial License 1.0.0 (see LICENSE).

FILE=GSE213478_methylation_DNAm_noob_final_BMIQ_all_tissues_987.txt.gz

wget -O $FILE \
    https://ftp.ncbi.nlm.nih.gov/geo/series/GSE213nnn/GSE213478/suppl/$FILE

wget -O GSE213478_RAW.tar \
    https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE213478&format=file
