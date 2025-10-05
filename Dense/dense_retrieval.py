import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
import numpy as np
from datasets import load_dataset
import copy
from time import time
from tqdm import tqdm
import os, json
import shutil
import random
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default="microsoft/codebert-base")
parser.add_argument('--model_path', type=str, default=None)
parser.add_argument('--do_train', action="store_true")
parser.add_argument('--data_split', type=str, choices =["coir_cosqa", "coir_apps", "coir_synthetic-text2sql", "vault_C", "vault_Ruby", "vault_Rust", "vault_JavaScript"], default="coir_cosqa")
parser.add_argument('--eval_batch_size', type=int, default=256)
parser.add_argument('--train_batch_size', type=int, default=32)
parser.add_argument('--nepochs', type=int, default=15)
parser.add_argument('--lr', type=float, default=2e-5)
parser.add_argument('--eval_samples', type=int, default=5000)
parser.add_argument('--max_len', type=int, default=64)
parser.add_argument('--save_dir', type=str, default="./results")
args = parser.parse_args()

model_name = args.model_name
train_file = f"/datadrive5/namlh35/CodeGR/data/train_dense/train/{args.data_split}.json"
test_file = f"/datadrive5/namlh35/CodeGR/data/train_dense/test/{args.data_split}.json"
eval_cache = os.path.join(args.save_dir, f"cache_{args.data_split}")
eval_batch_size = args.eval_batch_size
train_batch_size = args.train_batch_size
nepochs= args.nepochs
max_len=args.max_len
lr = args.lr
eval_samples = args.eval_samples
save_dir = args.save_dir

# -------------------------------
# 1. Dataset
# -------------------------------
class RetrievalDataset(Dataset):
    def __init__(self, datafile, tokenizer, num_sample = -1, max_len=32):
        self.dataset = load_dataset("json", data_files=datafile, split="train")
        if num_sample != -1 and len(self.dataset) > num_sample:
            self.dataset = self.dataset.take(num_sample)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.queries = list(self.dataset["text_id"])
        self.docs = list(self.dataset["text"])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.queries[idx], self.docs[idx]


# -------------------------------
# 2. Bi-Encoder Model
# -------------------------------
class BiEncoder(nn.Module):
    def __init__(self, model_name, max_len):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.max_len = max_len

    def encode(self, texts, tokenizer, device):
        inputs = tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True, max_length=self.max_len
        ).to(device)
        outputs = self.encoder(**inputs)
        embeddings = outputs.last_hidden_state[:, 0]  # CLS token
        return nn.functional.normalize(embeddings, dim=-1)


# -------------------------------
# 3. InfoNCE Loss
# -------------------------------
class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.05):
        super().__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, q_embeds, d_embeds):
        sim = torch.matmul(q_embeds, d_embeds.T) / self.temperature
        labels = torch.arange(sim.size(0), device=sim.device)
        return self.criterion(sim, labels)

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)


# -------------------------------
# 4. Evaluation Metrics
# -------------------------------
def evaluate(model, tokenizer, queries, docs, labels, batch_size, device, cache="./cache"):
    if not os.path.exists(cache):
      os.makedirs(cache)

    model.eval()

    num_q_batch = int(np.ceil(len(queries) / batch_size))
    num_d_batch = int(np.ceil(len(docs) / 256))
    ranked_indices = []
    infer_times = []
    for idq in tqdm(range(num_q_batch), total=num_q_batch, desc="Evaluating"):
        batch_queries = queries[idq*batch_size: (idq+1)*batch_size]
        time1 = time()
        with torch.no_grad():
            q_embeds = model.encode(batch_queries, tokenizer, device)
        time2 = time()

        infer_time = time2 - time1

        batch_queries_sim = []
        for idb in range(num_d_batch):
            if not os.path.exists(os.path.join(cache, f"{idb}.pt")):
                batch_docs = docs[idb*batch_size: (idb+1)*batch_size]
                with torch.no_grad():
                    d_embeds = model.encode(batch_docs, tokenizer, device)
                    # Save the tensor to a file
                    torch.save(d_embeds, os.path.join(cache, f"{idb}.pt"))
            else:
                # Load the tensor back
                d_embeds = torch.load(os.path.join(cache, f"{idb}.pt")).to(device)

            time3 = time()
            sim_batch_matrix = torch.matmul(q_embeds, d_embeds.T)  # [batch_num_queries x batch_num_docs]
            
            sim_batch_matrix = sim_batch_matrix.cpu().numpy()
            batch_queries_sim.append(sim_batch_matrix)
            time4 = time()

            infer_time += (time4-time3)
        
        infer_times.append(infer_time)
        if idq == 5:
            print(infer_times, np.mean(infer_times))
        batch_q_sim = np.concatenate(batch_queries_sim, axis=1) # [batch_num_queries x num_docs]
        # get top 10
        batch_ranked_indices = np.argsort(batch_q_sim, axis = 1)[:,-10:]

        ranked_indices.append(batch_ranked_indices)

    # remove cache
    shutil.rmtree(cache)


    ranked_indices = np.concatenate(ranked_indices, axis=0) # [num_queries x num_docs]
    assert len(queries) == ranked_indices.shape[0]

    hit_at_1, hit_at_10, mrr_at_10 = 0., 0., 0

    for i in range(len(queries)):
        qranked_indices = ranked_indices[i][::-1]
        label_id = labels[i]

        hits = np.where(np.array(qranked_indices) == label_id)[0]
        if len(hits) != 0:
            hit_at_10 += 1
            mrr_at_10 += 1/(hits[0] + 1)
            if hits[0] == 0:
                hit_at_1 += 1

    hit_at_1 /= len(queries)
    hit_at_10 /= len(queries)
    mrr_at_10 /= len(queries)

    return hit_at_1*100, hit_at_10*100, mrr_at_10*100


# Arguments
device = "cuda" if torch.cuda.is_available() else "cpu"
train_test_samples = -1 # !-1 for debug
print(device)
# -------------------------------
# 5. Training Loop
# -------------------------------

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = BiEncoder(model_name, max_len).to(device)
if args.model_path:
    model.load_state_dict(torch.load(args.model_path, weights_only=True))

optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
criterion = InfoNCELoss(temperature=0.05)

dataset = RetrievalDataset(train_file, tokenizer, train_test_samples)
loader = DataLoader(dataset, batch_size=train_batch_size, shuffle=True)

test_dataset = RetrievalDataset(test_file, tokenizer, train_test_samples)
all_docs = dataset.docs + test_dataset.docs

best_hit10 = 0.0
best_model = None

if args.do_train:
    for epoch in range(nepochs):
        perBatchTime = 0
        cntBatch = 0
        time1 = time()
        model.train()
        for batch in tqdm(loader, desc=f'Training epoch {epoch+1}'):
            batchTime1 = time()
            q_texts, d_texts = batch
            q_embeds = model.encode(q_texts, tokenizer, device)
            d_embeds = model.encode(d_texts, tokenizer, device)

            loss = criterion(q_embeds, d_embeds)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batchTime2 = time()
            perBatchTime += (batchTime2 - batchTime1)
            cntBatch += 1
        time2 = time()

        # Evaluate at end of epoch
        labels = range(len(all_docs))[:eval_samples]
        hit1, hit10, mrr10 = evaluate(model, tokenizer, dataset.queries[:eval_samples], all_docs, labels, eval_batch_size, device, eval_cache)
        log = f"Epoch {epoch+1}: Loss={loss.item():.4f} | Hit@1={hit1:.3f} | Hit@10={hit10:.3f} | MRR@10={mrr10:.3f} | time={time2-time1:.3f}s | perBatchTime= {(perBatchTime/cntBatch):.3f}"

        print(log)
        with open(os.path.join(save_dir, "log.txt"), "a") as f:
            f.write(log+"\n")

        if hit10 > best_hit10:
            best_hit10 = hit10
            best_model = copy.deepcopy(model)
            print(f"✅ Best model updated at epoch {epoch+1} (Hit@10={best_hit10:.3f})")

if best_model is None:
    best_model = copy.deepcopy(model)

test_labels = range(len(dataset), len(all_docs))
test_hit1, test_hit10, test_mrr10 = evaluate(best_model, tokenizer, test_dataset.queries, all_docs, test_labels, eval_batch_size, device, eval_cache)
print(f"\nHit@1={test_hit1:.3f} | Hit@10={test_hit10:.3f} | MRR@10={test_mrr10:.3f} | time={time2-time1:.3f}s")
torch.save(best_model.state_dict(), os.path.join(save_dir, "model.pth"))

metrics = {"Hit@1": test_hit1, "Hit@10": test_hit10, "MRR@10": test_mrr10}
with open(os.path.join(save_dir, "metrics.json"), "w") as f:
    json.dump(metrics, f, indent=4)