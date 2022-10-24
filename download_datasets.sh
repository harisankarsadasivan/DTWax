#!/bin/bash

G1='\033[0;32m'
G2='\033[0m'
R1='\033[0;31m'
R2='\033[0m'
fail=false
# echo -e "\n${G1}installing aws s3 (for downloading datasets)${G2}"
# curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" \
#     -o "aws_cli_v2.zip" && \
#     unzip aws_cli_v2.zip && \
#     ./aws/install -i ./aws -b ./aws && \
#     rm -f aws_cli_v2.zip || \
#     fail=true
# if $fail; then echo -e "\n${R1}aws s3 installation failed${R2}"; exit 1; else echo -e "${G1}done!${G2}"; fi

# echo -e "\n${G1}downloading human data...${G2}"
# mkdir -p data/human/fast5
# ./aws/dist/aws s3 --no-sign-request sync \
#         s3://ont-open-data/gm24385_2020.09/analysis/r9.4.1/20200914_1357_1-E11-H11_PAF27462_d3c9678e/guppy_v4.0.11_r9.4.1_hac_prom/align_unfiltered/chr19/fast5/ \
#         data/human/fast5 || \
#         fail=true
# if $fail; then echo -e "\n${R1}human data download failed${R2}"; exit 1; else echo -e "${G1}done!${G2}"; fi

echo -e "\n${G1}downloading covid data...${G2}"
mkdir -p data/covid/fast5 && \
    cd data/covid/fast5 && \
    wget https://cadde.s3.climb.ac.uk/SP1-raw.tgz && \
    tar -xvf SP1-raw.tgz && \
    rm SP1-raw.tgz SP1-mapped.fastq README || \
    fail=true
cd -
if $fail; then echo -e "\n${R1}covid data download failed${R2}"; exit 1; else echo -e "${G1}done!${G2}"; fi
