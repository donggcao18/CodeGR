

CUDA_VISIBLE_DEVICES=0 python train.py \
--log_file log.txt \
--save_dir ./results \
--train_data ./data/Ruby_train.json \
--test_data ./data/Ruby_test.json \
--max_length 64 \
--batch_size 128 \
--num_train_epochs 1 \
--eval_steps 10