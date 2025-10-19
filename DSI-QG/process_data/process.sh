#! /bin/bash
export WANDB_DISABLED="true"

for LANG in Ruby
do
    python3 docT5.py \
    --data_path "/content/data/${LANG}.hf" \
    --save_dir "/content/CodeGR/DSI-QG/process_data" \
    --train_samples -1 \
    --test_samples 1000 \
    --index_retrieval_ratio 1 \
    --track_metadata
done