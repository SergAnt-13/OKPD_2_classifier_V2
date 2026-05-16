import faiss
import numpy as np


class LeafIndex:

    def __init__(self, dim: int):
        self.index = faiss.IndexFlatIP(dim)
        self.codes = []

    def build(self, embeddings: np.ndarray, codes: list):

        self.index.add(embeddings.astype("float32"))
        self.codes = codes

    def search(self, vector: np.ndarray, k: int = 20):

        scores, idxs = self.index.search(
            vector.astype("float32"),
            k
        )

        results = []

        for i in range(k):
            results.append({
                "code": self.codes[idxs[0][i]],
                "score": float(scores[0][i])
            })

        return results