from datasets import load_dataset
from rank_bm25 import BM25Okapi
from transformers import AutoTokenizer
from tqdm import tqdm
import numpy as np
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import os, json
from time import time

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str)
parser.add_argument('--save_dir', type=str)
parser.add_argument('--tokenizer', type=str, default="Salesforce/codet5-base")
parser.add_argument('--split', type=str)
args = parser.parse_args()

if not os.path.exists(args.save_dir):
    os.mkdir(args.save_dir)


train_file = os.path.join(args.data_path, f"{args.split}_train_r1.0.json")
test_file = os.path.join(args.data_path, f"{args.split}_test_r1.0.json")

tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

dataset = load_dataset("json", data_files=train_file, split="train").filter(lambda example: example["text"].startswith("Code:")).map(lambda example: {"text": example["text"][5:].strip()})
dataset = dataset.map(lambda example: {"tokenized_text": tokenizer.tokenize(example["text"])})
tokenized_corpus = dataset["tokenized_text"]
corpus_id = dataset["text_id"]

print(args.split, len(tokenized_corpus))

test_dataset = load_dataset("json", data_files=test_file, split="train").filter(lambda example: example["text"].startswith("Query:")).map(lambda example: {"text": example["text"][6:].strip()})
test_dataset = test_dataset.map(lambda example: {"tokenized_text": tokenizer.tokenize(example["text"])})
queries = test_dataset["tokenized_text"]
labels = test_dataset["text_id"]

bm25 = BM25Okapi(tokenized_corpus)


def evaluate_query(args):
    qid, query = args
    doc_scores = bm25.get_scores(query)
    ranking = np.argsort(doc_scores)[-10:][::-1].tolist()
    filtered_rank_list = [corpus_id[x] for x in ranking]
    label_id = labels[qid]

    hit_at_1 = 0
    hit_at_10 = 0
    mrr = 0.0

    hits = np.where(np.array(filtered_rank_list) == label_id)[0]
    if len(hits) != 0:
        hit_at_10 = 1
        mrr = 1 / (hits[0] + 1)
        if hits[0] == 0:
            hit_at_1 = 1

    return hit_at_1, hit_at_10, mrr

args_list = [(qid, query) for qid, query in enumerate(queries)]

# avg_times = []
# for id, arg in enumerate(args_list):
#     time1 = time()
#     evaluate_query(arg)
#     avg_times.append(time() - time1)
#     if id >5:
#         break
# print(sum(avg_times)/len(avg_times))


# Run multi-threaded evaluation
results = []
with ProcessPoolExecutor(max_workers=20) as executor:  # use processes
    futures = [executor.submit(evaluate_query, args) for args in args_list]
    for f in tqdm(as_completed(futures), total=len(futures)):
        results.append(f.result())

# Aggregate results
hit_at_1 = sum(r[0] for r in results)
hit_at_10 = sum(r[1] for r in results)
mrr = sum(r[2] for r in results)

metrics = {"Hit@1": hit_at_1/len(queries), "Hit@10": hit_at_10/len(queries), "MRR": mrr/len(queries)}
print(metrics)

with open(os.path.join(args.save_dir, f"bm25_{args.split}_metrics.json"), "w") as f:
    json.dump(metrics, f, indent=4)