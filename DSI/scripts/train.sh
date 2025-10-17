
export HF_HOME=/datadrive5/namlh35/.cache
export HF_DATASETS_CACHE=/datadrive5/namlh35/.cache

NEPOCHS=2
LR=1e-4
LEN=256

CUDA_VISIBLE_DEVICES=0 python train.py \
--model_name Salesforce/codet5p-220m \
--log_file log.txt \
--save_dir ./results/codet5p-770m-Ruby-r32-e${NEPOCHS}-lr${LR}-len${LEN} \
--train_data /root/workspace/CodeGR/data/Ruby_train_r1.0_semantic.json \
--test_data /root/workspace/CodeGR/data/Ruby_test_r1.0_semantic.json \
--max_length ${LEN} \
--batch_size 128 \
--gradient_accumulation_steps 1 \
--num_train_epochs ${NEPOCHS} \
--eval_steps 2000 \
--lr ${LR} \
# --max_steps 10 \
# --eval_samples 100 \
# --test_samples 100 \
