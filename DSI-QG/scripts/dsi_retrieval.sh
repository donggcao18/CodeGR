#! /bin/bash
export HF_HOME="/datadrive5/namlh35/.cache"
export HF_DATASETS_CACHE="/datadrive5/namlh35/.cache"
export WANDB_DISABLED="true"

CUDA_VISIBLE_DEVICES=1 python3 run.py \
        --task "DSI" \
        --model_name "Salesforce/codet5-base" \
        --run_name "Ruby-codet5-base-DSI" \
        --max_length 32 \
        --train_file /datadrive5/namlh35/CodeGR/data/dsi_data/Ruby_train_r1.0.json \
        --valid_file /datadrive5/namlh35/CodeGR/data/dsi_data/Ruby_test_r1.0.json \
        --output_dir "models/Ruby-codet5-base-DSI" \
        --learning_rate 0.0005 \
        --warmup_steps 100000 \
        --per_device_train_batch_size 32 \
        --per_device_eval_batch_size 32 \
        --eval_strategy steps \
        --eval_steps 5000 \
        --max_steps 500000 \
        --save_strategy steps \
        --dataloader_num_workers 10 \
        --save_steps 5000 \
        --save_total_limit 1 \
        --load_best_model_at_end \
        --gradient_accumulation_steps 1 \
        --report_to none \
        --logging_steps 500 \
        --dataloader_drop_last False \
        --metric_for_best_model Hits@10 \
        --greater_is_better True \