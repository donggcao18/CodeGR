#! /bin/bash
export HF_HOME="/datadrive5/namlh35/.cache"
export HF_DATASETS_CACHE="/datadrive5/namlh35/.cache"
export WANDB_DISABLED="true"

for LANG in Ruby
do
    python3 semantic_id.py \
    --data_path "/data/${LANG}.hf" \
    --save_dir "/datadrive5/namlh35/CodeGR/data/indexed_data/semantic_id" \
    --train_samples -1 \
    --test_samples -1 \
    --index_retrieval_ratio 1 \
    --track_metadata
done