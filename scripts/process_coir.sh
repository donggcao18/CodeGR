#! /bin/bash
export HF_HOME="/datadrive5/namlh35/.cache"
export HF_DATASETS_CACHE="/datadrive5/namlh35/.cache"
export WANDB_DISABLED="true"

for split in apps cosqa "synthetic-text2sql"
do
    python3 process_coir.py \
    --data_type $split \
    --save_dir "/datadrive5/namlh35/CodeGR/data/train_dense" \
    --train_samples -1 \
    --test_samples -1 \
    --summarization
done