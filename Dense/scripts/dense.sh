#! /bin/bash
export HF_HOME="/datadrive5/namlh35/.cache"
export HF_DATASETS_CACHE="/datadrive5/namlh35/.cache"
export WANDB_DISABLED="true"

for split in vault_Ruby
do
    CUDA_VISIBLE_DEVICES=1 python3 dense_retrieval.py \
    --model_name microsoft/codebert-base \
    --data_split ${split} \
    --model_path /datadrive5/namlh35/CodeGR/Dense/results/thevault/vault_C/codebert-base/model.pth \
    --save_dir ./results/thevault/${split}/codebert-base \
    --eval_batch_size 1
done