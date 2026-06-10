#!/usr/bin/bash

# Phenotype data
mkdir -p metadata/RESTRICTED
chmod 755 metadata/RESTRICTED
chmod 400 metadata/RESTRICTED/prj_34152.ngc
chmod 400 cart_DAR120197_202302021026.krt
prefetch --ngc ./metadata/RESTRICTED/prj_34152.ngc --cart cart_DAR120197_202302021026.krt -O metadata/RESTRICTED
chmod 400 metadata/RESTRICTED/phs000424.v9*.gz

# Genotype data (RNAMutec and Exome)
mkdir -p data/RESTRICTED
chmod 755 data/RESTRICTED
chmod 400 metadata/RESTRICTED/prj_34152.ngc
chmod 400 metadata/cart_DAR120197_202308240505.krt
prefetch --ngc metadata/RESTRICTED/prj_34152.ngc --cart metadata/cart_DAR120197_202308240505.krt --output-directory data/RESTRICTED

for F in data/RESTRICTED/*.tar; do
    tar -xvf $F
done

find data/RESTRICTED/* -type d -exec chmod 755 {} \;
find data/RESTRICTED/* -type f -exec chmod 400 {} \;


# Genotype (WGS) - 2023-12-01
chmod 400 metadata/RESTRICTED/cart_DAR120197_202312010621_1.krt
chmod 400 metadata/RESTRICTED/cart_DAR120197_202312010621_2.krt
prefetch --ngc metadata/RESTRICTED/prj_34152.ngc --cart metadata/RESTRICTED/cart_DAR120197_202312010621_1.krt --output-directory data/RESTRICTED
prefetch --ngc ../../metadata/RESTRICTED/prj_34152.ngc --cart ../../metadata/RESTRICTED/cart_DAR120197_202312010621_2.krt --output-directory . data/RESTRICTED/

# v10 changelog and additional genotype files - 2025-03-25
chmod 400 metadata/RESTRICTED/cart_DAR120197_202503250613.krt
prefetch --ngc metadata/RESTRICTED/prj_34152.ngc --cart metadata/RESTRICTED/cart_DAR120197_202503250613.krt --output-directory data/RESTRICTED
find data/RESTRICTED -type d -exec chmod 770 {} \;
find data/RESTRICTED -type f -exec chmod 440 {} \;
