#! /bin/bash
export HF_HOME="/datadrive5/namlh35/.cache"
export HF_DATASETS_CACHE="/datadrive5/namlh35/.cache"
export WANDB_DISABLED="true"

LANG=C

for SIZE in 100000
do
        CUDA_VISIBLE_DEVICES=1 python3 run.py \
                --task "DSI" \
                --model_name google-t5/t5-base \
                --model_path models/numid-${LANG}-t5-base-DSI_QG-${SIZE}/best_model \
                --run_name "numid-${LANG}-t5-base-DSI_QG-${SIZE}" \
                --max_length 32 \
                --id_max_length 20 \
                --train_file /datadrive5/namlh35/CodeGR/data/datasize_exp/${LANG}_train_${SIZE}.json \
                --valid_file /datadrive5/namlh35/CodeGR/data/datasize_exp/${LANG}_train_${SIZE}.json \
                --output_dir "models/numid-${LANG}-t5-base-DSI_QG-${SIZE}" \
                --learning_rate 0.0005 \
                --warmup_steps 100000 \
                --per_device_train_batch_size 32 \
                --per_device_eval_batch_size 256 \
                --eval_strategy steps \
                --eval_steps 50000 \
                --max_steps 500000 \
                --save_strategy steps \
                --dataloader_num_workers 10 \
                --save_steps 50000 \
                --save_total_limit 2 \
                --load_best_model_at_end \
                --gradient_accumulation_steps 1 \
                --report_to none \
                --logging_steps 500 \
                --dataloader_drop_last False \
                --metric_for_best_model Hits@10 \
                --greater_is_better True \
                --remove_prompt True
                # --only_code True \
                # --is_text_indexing \
done