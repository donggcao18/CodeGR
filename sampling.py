import json
import os
from datasets import load_dataset, concatenate_datasets, Dataset
import random

save_dir = "/datadrive5/namlh35/CodeGR/data/datasize_exp"
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

random.seed(42)
test_samples = 1000
test_set_path = "/datadrive5/namlh35/CodeGR/data/indexed_data/numberic_index/C_test_r1.0.json"

train_samples = 4000
train_set_path = "/datadrive5/namlh35/CodeGR/data/augmentation/query_augment/C.q10"


train_data = load_dataset("json", data_files=train_set_path, split="train")
test_data = load_dataset("json", data_files=test_set_path, split="train").take(test_samples)
init_ids = set(test_data["text_id"])
train_init_data = train_data.filter(lambda x: str(x["text_id"]) in init_ids, num_proc=10)
train_remain_data = train_data.filter(lambda x: str(x["text_id"]) not in init_ids, num_proc=10)
print(train_remain_data)
train_remain_data_ids = [str(x) for x in set(train_remain_data["text_id"])]


new_test_data = {}
new_train_init_data = {}
for dp in test_data:
    new_test_data[str(dp["text_id"])] = dp["text"][6:].strip()

for dp in train_init_data:
    if str(dp["text_id"]) not in new_train_init_data:
        new_train_init_data[str(dp["text_id"])] = [dp["text"]]
    else:
        new_train_init_data[str(dp["text_id"])].append(dp["text"])

#re_indice
re_test_data = []
re_train_data = []

cnt = 0
for id in new_test_data:
    re_test_data.append({"text_id": cnt, "text": new_test_data[id]})
    re_train_data = re_train_data + [{"text_id": cnt, "text": x} for x in new_train_init_data[id]]
    cnt += 1

test_data = Dataset.from_list(re_test_data)
train_init_data = Dataset.from_list(re_train_data)
test_data.to_json(os.path.join(save_dir, "C_test.json"))
print(train_init_data)
for total_samples in [5000, 20000, 50000, 100000]:
    train_samples = total_samples - len(init_ids)
    random_element_ids = random.sample(train_remain_data_ids, train_samples)
    add_train_data = train_remain_data.filter(lambda x: str(x["text_id"]) in random_element_ids, num_proc=10)

    new_train_data = {}
    for dp in add_train_data:
        if str(dp["text_id"]) not in new_train_data:
            new_train_data[str(dp["text_id"])] = [dp["text"]]
        else:
            new_train_data[str(dp["text_id"])].append(dp["text"])

    _new_train_data = []
    new_id = len(test_data)
    for old_id in new_train_data:
        _new_train_data += [{"text_id": new_id, "text": x} for x in new_train_data[old_id]]
        new_id += 1
    add_train_data = Dataset.from_list(_new_train_data)

    full_data = concatenate_datasets([train_init_data, add_train_data])
    full_data.to_json(os.path.join(save_dir, f"C_train_{total_samples}.json"))
