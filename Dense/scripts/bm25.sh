#! /bin/bash
export HF_HOME="/datadrive5/namlh35/.cache"
export HF_DATASETS_CACHE="/datadrive5/namlh35/.cache"
export WANDB_DISABLED="true"

for split in Ruby
do
    python3 bm25_retrieval.py \
    --data_path /datadrive5/namlh35/CodeGR/data/indexed_data/numberic_index \
    --split ${split} \
    --save_dir ./results/thevault
done