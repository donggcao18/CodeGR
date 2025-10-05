#! /bin/bash
export HF_HOME="/datadrive5/namlh35/.cache"
export HF_DATASETS_CACHE="/datadrive5/namlh35/.cache"
export WANDB_DISABLED="true"

for LANG in C
do
    python3 process_vault.py \
    --data_path "/datadrive5/namlh35/CodeGR/data/${LANG}.hf" \
    --save_dir "/datadrive5/namlh35/CodeGR/data/indexed_data/indexed_data_metatdata_v2" \
    --train_samples -1 \
    --test_samples -1 \
    --index_retrieval_ratio 1 \
    --track_metadata
done