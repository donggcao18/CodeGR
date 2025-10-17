import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from sklearn.cluster import MiniBatchKMeans
from typing import List, Tuple


class CodeBERTSentenceEncoder:
    def __init__(self, model_name="microsoft/codebert-base", device=None):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, dim=1) / \
               torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)

    def encode(self, texts: List[str], batch_size: int = 64, normalize: bool = True) -> np.ndarray:
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            encoded = self.tokenizer(batch_texts, padding=True, truncation=True,
                                     max_length=256, return_tensors='pt').to(self.device)

            with torch.no_grad():
                output = self.model(**encoded)
                pooled = self.mean_pooling(output, encoded['attention_mask'])

            if normalize:
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)

            all_embeddings.append(pooled.cpu().numpy())
        print(all_embeddings)
        return np.vstack(all_embeddings)


class SemanticIDGenerator:
    def __init__(self, k: int = 10, c: int = 100, random_state: int = 42):
        self.k = k
        self.c = c
        self.random_state = random_state

    def fit(self, X: np.ndarray) -> List[str]:
        """
        Args:
            X: (N, D) numpy array of sentence embeddings

        Returns:
            List of semantic ID strings like "231"
        """
        N = X.shape[0]
        doc_indices = np.arange(N)
        assigned = self._recursive_cluster(X, doc_indices, prefix="")

        doc_ids = [""] * N
        for idx, id_str in assigned:
            doc_ids[idx] = id_str
        return doc_ids

    def _recursive_cluster(self, X: np.ndarray, 
                           indices: np.ndarray, 
                           prefix: str) -> List[Tuple[int, str]]:
        
        if len(indices) <= self.c:
            return [(idx, prefix + str(i)) for i, idx in enumerate(indices)]

        k = min(self.k, len(indices))
        kmeans = MiniBatchKMeans(n_clusters=k, 
                                 random_state=self.random_state, 
                                 batch_size=1024)
        labels = kmeans.fit_predict(X[indices])

        results = []
        for cluster_id in range(k):
            sub_indices = indices[labels == cluster_id]
            if len(sub_indices) == 0:
                continue
            sub_prefix = prefix + str(cluster_id)
            results.extend(self._recursive_cluster(X, 
                                                   sub_indices, 
                                                   sub_prefix))
        return results



if __name__ == "__main__":
    # Sample texts (can be code snippets or NL queries)
    docs = [
        "def add(a, b): return a + b",
        "print('Hello World')",
        "class Stack: def __init__(self): self.items = []",
        "for i in range(10): print(i)",
        "def divide(x, y): return x / y",
        "while True: continue",
        "import numpy as np",
        "import torch",
        "function sum(a, b) { return a + b; }",
        "console.log('Hello')",
        "console.log('Hi')",
        "class Node { constructor(val) { this.val = val; }}"
    ]

    # Step 1: Encode with CodeBERT
    encoder = CodeBERTSentenceEncoder()
    embeddings = encoder.encode(docs)

    # Step 2: Generate semantic IDs
    ssi = SemanticIDGenerator(k=3, c=3)
    ids = ssi.fit(embeddings)

    print("\nAssigned Semantic IDs:")
    for i, (doc, docid) in enumerate(zip(docs, ids)):
        print(f"{i:02d}  ID={docid}  :: {doc}")
