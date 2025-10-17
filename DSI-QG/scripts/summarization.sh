#! /bin/bash
export HF_HOME="/datadrive5/namlh35/.cache"
export HF_DATASETS_CACHE="/datadrive5/namlh35/.cache"
export WANDB_DISABLED="true"

for LANG in Ruby
do
        CUDA_VISIBLE_DEVICES=0 python3 run.py \
                --task "queryTdoc" \
                --model_name "Salesforce/codet5p-220m" \
                --run_name "vault_queryTdoc_codet5p" \
                --max_length 256 \
                --train_file /root/workspace/CodeGR/data/docT5query_training/${LANG}_train_docT5.json \
                --valid_file /root/workspace/CodeGR/data/docT5query_training/${LANG}_train_docT5.json \
                --output_dir models/vault_queryTdoc_${LANG}_codet5p-220m \
                --learning_rate 0.0001 \
                --warmup_steps 1000 \
                --per_device_train_batch_size 16 \
                --per_device_eval_batch_size 16  \
                --eval_strategy steps \
                --eval_steps 1000 \
                --max_steps 10000 \
                --save_strategy steps \
                --dataloader_num_workers 10 \
                --save_steps 1000 \
                --save_total_limit 2 \
                --load_best_model_at_end \
                --gradient_accumulation_steps 1 \
                --report_to none \
                --logging_steps 100 \
                --dataloader_drop_last False \
                --gradient_checkpointing
done