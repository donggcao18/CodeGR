#! /bin/bash
export HF_HOME="/datadrive5/namlh35/.cache"
export HF_DATASETS_CACHE="/datadrive5/namlh35/.cache"
export WANDB_DISABLED="true"

for LANG in JavaScript C
do
        CUDA_VISIBLE_DEVICES=0,1 accelerate launch --multi_gpu --mixed_precision fp16 --num_processes 2 run.py \
                --task "queryTdoc" \
                --model_name "Salesforce/codet5p-770m" \
                --run_name "vault_queryTdoc_codet5p" \
                --max_length 256 \
                --train_file /datadrive5/namlh35/CodeGR/data/augmentation/query_to_code_augment/${LANG}.q10 \
                --valid_file /datadrive5/namlh35/CodeGR/data/augmentation/query_to_code_augment/${LANG}.q10 \
                --output_dir models/vault_queryTdoc_${LANG}_codet5p-770m \
                --learning_rate 0.0001 \
                --warmup_steps 1000 \
                --per_device_train_batch_size 2 \
                --per_device_eval_batch_size 2  \
                --eval_strategy steps \
                --eval_steps 1000 \
                --max_steps 10000 \
                --save_strategy steps \
                --dataloader_num_workers 10 \
                --save_steps 1000 \
                --save_total_limit 2 \
                --load_best_model_at_end \
                --gradient_accumulation_steps 8 \
                --report_to none \
                --logging_steps 100 \
                --dataloader_drop_last False \
                --gradient_checkpointing
done