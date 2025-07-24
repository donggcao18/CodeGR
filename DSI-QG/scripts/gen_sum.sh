#! /bin/bash
export HF_HOME="/datadrive5/namlh35/.cache"
export HF_DATASETS_CACHE="/datadrive5/namlh35/.cache"
export WANDB_DISABLED="true"
for LANG in Ruby Rust C JavaScript
do
        CUDA_VISIBLE_DEVICES=0,1 accelerate launch --multi_gpu --mixed_precision fp16 --num_processes 2 run.py \
                --task generation \
                --lang ${LANG} \
                --model_name Salesforce/codet5p-770m \
                --model_path /datadrive5/namlh35/CodeGR/DSI-QG/models/codesum_codet5p-770m/checkpoint-3000 \
                --per_device_eval_batch_size 32 \
                --run_name codesum_codet5p-generation \
                --max_length 256 \
                --valid_file /datadrive5/namlh35/CodeGR/data/dsi_data/${LANG}_train_r1.0.json \
                --output_dir summarization/ \
                --dataloader_num_workers 10 \
                --report_to none \
                --logging_steps 100 \
                --num_return_sequences 10
done