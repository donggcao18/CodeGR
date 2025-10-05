from datasets import load_dataset
import os
import jsonlines

is_num_to_text=True
is_num=False
is_query_to_code = False

text_id_column = "url_based_id"
for LANG in ['C']:
    save_dir="/datadrive5/namlh35/CodeGR/data/augmentation/our_index_query_augment_v2"
    
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    train_data = f"/datadrive5/namlh35/CodeGR/data/indexed_data/indexed_data_metatdata_v2/{LANG}_train_r1.0.json"
    test_data = f"/datadrive5/namlh35/CodeGR/data/indexed_data/indexed_data_metatdata_v2/{LANG}_test_r1.0.json"

    if not is_num_to_text and not is_query_to_code:
        
        # summary_data = f"/datadrive5/namlh35/CodeGR/data/gen_summarization/{LANG}.q10"

        train_dataset = load_dataset("json", data_files=train_data, split="train")
        test_dataset = load_dataset("json", data_files=test_data, split="train")
        # docstring_data = load_dataset("json", data_files=summary_data, split="train")

    
    if is_query_to_code:
        text_indexed_data= f"/datadrive5/namlh35/CodeGR/data/indexed_data/numberic_index/{LANG}_train_r1.0.json"
        num_indexed_data = f"/datadrive5/namlh35/CodeGR/data/augmentation/query_augment/{LANG}.q10"
        
        text_indexed_dataset = load_dataset("json", data_files=text_indexed_data, split="train").filter(lambda example: example["text"].startswith("Code:"))
        num_indexed_dataset = load_dataset("json", data_files=num_indexed_data, split="train")
        merged_num_indexed_data = {}
        for dp in num_indexed_dataset:
            if dp["text_id"] not in merged_num_indexed_data:
                merged_num_indexed_data[dp["text_id"]] = []
            merged_num_indexed_data[dp["text_id"]].append(dp["text"])
        
        final_data= []
        for id in merged_num_indexed_data:
            text_list = merged_num_indexed_data[id]
            text_based_id = text_indexed_dataset[id]["text"][5:].strip()
            final_data.extend([{"text_id": text_based_id, "text": text} for text in text_list])
        with jsonlines.open(os.path.join(save_dir, f"{LANG}.q10"), mode='w') as writer:
            writer.write_all(final_data)
    elif is_num_to_text:
        text_indexed_data= f"/datadrive5/namlh35/CodeGR/data/indexed_data/our_based_id_v2/{LANG}_train_r1.0.json"
        num_indexed_data = f"/datadrive5/namlh35/CodeGR/data/augmentation/query_augment/{LANG}.q10"
        
        text_indexed_dataset = load_dataset("json", data_files=text_indexed_data, split="train").filter(lambda example: example["text"].startswith("Code:"))
        num_indexed_dataset = load_dataset("json", data_files=num_indexed_data, split="train")
        merged_num_indexed_data = {}
        for dp in num_indexed_dataset:
            if dp["text_id"] not in merged_num_indexed_data:
                merged_num_indexed_data[dp["text_id"]] = []
            merged_num_indexed_data[dp["text_id"]].append(dp["text"])
        
        final_data= []
        for id in merged_num_indexed_data:
            text_list = merged_num_indexed_data[id]
            text_based_id = text_indexed_dataset[id]["text_id"]
            final_data.extend([{"text_id": text_based_id, "text": text} for text in text_list])
        with jsonlines.open(os.path.join(save_dir, f"{LANG}.q10"), mode='w') as writer:
            writer.write_all(final_data)
        
    elif is_num:
        train_dataset = train_dataset.remove_columns([x for x in train_dataset.column_names if x not in ["text_id", "text"]])
        test_dataset = test_dataset.remove_columns([x for x in test_dataset.column_names if x not in ["text_id", "text"]])
        train_dataset.to_json(os.path.join(save_dir, f"{LANG}_train_r1.0.json"))
        test_dataset.to_json(os.path.join(save_dir, f"{LANG}_test_r1.0.json"))
    else:
        train_dataset = train_dataset.remove_columns([x for x in train_dataset.column_names if x not in [text_id_column, "text"]]).rename_column(text_id_column, "text_id")
        test_dataset = test_dataset.remove_columns([x for x in test_dataset.column_names if x not in [text_id_column, "text"]]).rename_column(text_id_column, "text_id")
        
        train_dataset.to_json(os.path.join(save_dir, f"{LANG}_train_r1.0.json"))
        test_dataset.to_json(os.path.join(save_dir, f"{LANG}_test_r1.0.json"))
        
    