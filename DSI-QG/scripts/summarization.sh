#! /bin/bash
export HF_HOME="/datadrive5/namlh35/.cache"
export HF_DATASETS_CACHE="/datadrive5/namlh35/.cache"
export WANDB_DISABLED="true"

CUDA_VISIBLE_DEVICES=0,1 accelerate launch --multi_gpu --mixed_precision fp16 --num_processes 2 run.py \
        --task "DSI" \
        --model_name "google/mt5-base" \
        --run_name "XORQA-100k-mt5-base-DSI-QG" \
        --max_length 32 \
        --train_file data/xorqa_data/100k/xorqa_corpus.tsv.q10.docTquery \
        --valid_file data/xorqa_data/100k/xorqa_DSI_dev_data.json \
        --output_dir "models/XORQA-100k-mt5-base-DSI-QG" \
        --learning_rate 0.0005 \
        --warmup_steps 100000 \
        --per_device_train_batch_size 32 \
        --per_device_eval_batch_size 32 \
        --evaluation_strategy steps \
        --eval_steps 1000 \
        --max_steps 1000000 \
        --save_strategy steps \
        --dataloader_num_workers 10 \
        --save_steps 1000 \
        --save_total_limit 2 \
        --load_best_model_at_end \
        --gradient_accumulation_steps 1 \
        --report_to wandb \
        --logging_steps 100 \
        --dataloader_drop_last False \
        --metric_for_best_model Hits@10 \
        --greater_is_better True \
        --remove_prompt True