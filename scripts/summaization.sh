for DATANAME in Rust 
do
    python3 preprocess.py \
    --data_path "/datadrive5/namlh35/CodeGR/data/${DATANAME}.hf" \
    --save_dir "/datadrive5/namlh35/CodeGR/data/summarization" \
    --train_samples 20000 \
    --summarization
done