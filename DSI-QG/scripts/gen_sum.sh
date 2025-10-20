#! /bin/bash
export HF_HOME="/datadrive5/namlh35/.cache"
export HF_DATASETS_CACHE="/datadrive5/namlh35/.cache"
export WANDB_DISABLED="true"
# for LANG in Rust C JavaScript
# do
#         CUDA_VISIBLE_DEVICES=0,1 accelerate launch --multi_gpu --mixed_precision fp16 --num_processes 2 run.py \
#                 --task generation \
#                 --lang ${LANG} \
#                 --model_name Salesforce/codet5p-770m \
#                 --model_path /datadrive5/namlh35/CodeGR/DSI-QG/models/vault_queryTdoc_${LANG}_codet5p-770m/checkpoint-10000 \
#                 --per_device_eval_batch_size 32 \
#                 --run_name codegen_codet5p-generation-${LANG} \
#                 --max_length 256 \
#                 --q_max_length 128 \
#                 --valid_file /datadrive5/namlh35/CodeGR/data/indexed_data/numberic_index/${LANG}_test_r1.0.json \
#                 --output_dir /datadrive5/namlh35/CodeGR/data/augmentation/gen_code \
#                 --dataloader_num_workers 10 \
#                 --report_to none \
#                 --logging_steps 100 \
#                 --num_return_sequences 1 \
#                 --is_reverse
# done

for LANG in Ruby
do
        CUDA_VISIBLE_DEVICES=0 python3 run.py \
                --task generation \
                --lang ${LANG} \
                --model_name Salesforce/codet5p-220m \
                --model_path DocT5_model/vault_queryTdoc_${LANG}_codet5p-220m/checkpoint-10000 \
                --per_device_eval_batch_size 64 \
                --run_name codegen_codet5p-generation-${LANG} \
                --max_length 256 \
                --q_max_length 128 \
                --valid_file /root/workspace/CodeGR/data/original_indexed_data/${LANG}_train_r1.0.json \
                --output_dir /root/workspace/CodeGR/data/augmentation \
                --dataloader_num_workers 10 \
                --report_to none \
                --logging_steps 100 \
                --num_return_sequences 10 \
                --is_reverse False
done