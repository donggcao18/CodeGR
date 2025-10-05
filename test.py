from transformers import AutoTokenizer, T5ForConditionalGeneration
import torch
from datasets import load_dataset, load_from_disk
import json

dataset = load_from_disk("/datadrive5/namlh35/CodeGR/data/Ruby.hf")
dataset = dataset["test"].take(5)
print(dataset)

# -------------------------------
# 1. Load model and tokenizer
# -------------------------------
model_name = "google-t5/t5-base"   # or any causal LM
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)
print(tokenizer.bos_token, tokenizer.pad_token)

# -------------------------------
# 2. Define allowed output sentences
# -------------------------------
allowed_sentences = ["{}|{}|{}".format(x["identifier"], x["repo"], x["path"]) for x in  dataset]
print(allowed_sentences)

# -------------------------------
# 3. Build a prefix trie
# -------------------------------
def build_trie(sequences, tokenizer):
    trie = {}
    for seq in sequences:
        token_ids = tokenizer.encode(seq, add_special_tokens=False)
        node = trie
        for tok in token_ids:
            node = node.setdefault(tok, {})
        node["__end__"] = True  # mark end of sequence
    return trie

trie = build_trie(allowed_sentences, tokenizer)
print([tokenizer.decode(x) for x in trie.keys()])

for start in trie:
    if tokenizer.decode(start) == "prim":
        print([tokenizer.decode(x) for x in trie[start].keys()])
# print(json.dumps(trie, indent=2))
# exit()
# -------------------------------
# 4. Define prefix_allowed_tokens_fn
# -------------------------------
def get_next_tokens(node):
    return [tok for tok in node.keys() if tok != "__end__"]

def find_trie_node(trie, prefix):
    print(tokenizer.decode(prefix))
    node = trie
    
    if len(prefix) == 1 and prefix[0] in [tokenizer.bos_token_id, tokenizer.pad_token_id]:
        return node
    elif prefix[-1] == tokenizer.eos_token_id:
        return {tokenizer.eos_token_id: True}
    else:
        prefix = prefix[1:]
        
    for tok in prefix:
        if tok in node:
            node = node[tok]
        else:
            return {}  # dead end
    return node

def prefix_allowed_tokens_fn(batch_id, input_ids):
    # input_ids is shape [seq_len], we use the last generated tokens
    node = find_trie_node(trie, input_ids.tolist())
    
    if "__end__" in node:
        return get_next_tokens(node) + [tokenizer.eos_token_id]
    return get_next_tokens(node)

# -------------------------------
# 5. Generate constrained outputs
# -------------------------------
prompt = "Identifier for function: " + dataset[0]["code"]
print(prompt)
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(
    **inputs,
    max_new_tokens=50,
    num_beams=5,
    num_return_sequences=2,
    prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
    early_stopping=True,
)

print(outputs)

decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
print("Generated output:", decoded)
