
for LANG in C "C#" Cpp Go Java JavaScript PHP Python Ruby Rust
do
    python3 preprocess.py \
    --data_path "/datadrive5/namlh35/CodeGR/data/${LANG}.hf" \
    --save_dir "/datadrive5/namlh35/CodeGR/data/dsi_data" \
    --index_retrieval_ratio 1.
done