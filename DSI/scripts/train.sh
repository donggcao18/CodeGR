
export HF_HOME=/datadrive5/namlh35/.cache
export HF_DATASETS_CACHE=/datadrive5/namlh35/.cache

NEPOCHS=10
LR=1e-4
LEN=256

CUDA_VISIBLE_DEVICES=0 python train.py \
--model_name /datadrive5/namlh35/CodeGR/DSI/results/codet5p-770m-Ruby-r32-e10-lr1e-4-len256/checkpoint-3500 \
--log_file log.txt \
--save_dir ./results/codet5p-770m-Ruby-r32-e${NEPOCHS}-lr${LR}-len${LEN} \
--train_data /datadrive5/namlh35/CodeGR/data/dsi_data/Ruby_train.json \
--test_data /datadrive5/namlh35/CodeGR/data/dsi_data/Ruby_test.json \
--max_length ${LEN} \
--batch_size 64 \
--gradient_accumulation_steps 2 \
--num_train_epochs ${NEPOCHS} \
--eval_steps 500 \
--lr ${LR} \
# --max_steps 10 \
# --eval_samples 100 \
# --test_samples 100 \