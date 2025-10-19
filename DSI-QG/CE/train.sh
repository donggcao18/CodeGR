#! /bin/bash

outdir=runs/Ranker
model_cfg=microsoft/codebert-base

python3 train_ranker.py \
--max_grad_norm 2.0 --warm_ratio 0.1 --seed 42 --ranker_model_cfg ${model_cfg} \
--train_file /kaggle/input/ruby-ranker/Ruby_ranker_data.json --output_dir ${outdir} --train_steps 10000 \
--train_batch_size 4 --train_num 4 --learning_rate 1e-05 --print_per_step 100 --no_title \
--save_last