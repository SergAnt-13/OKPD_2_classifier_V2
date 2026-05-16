import faiss
import numpy as np


class FAISSIndex:

    def __init__(self, dim: int):
        self.index = faiss.IndexFlatIP(dim)
        self.dim = dim
        self.codes = []

    def build(self, embeddings: np.ndarray, codes: list):
        self.index.add(embeddings.astype("float32"))
        self.codes = codes

    def search(self, query_vector: np.ndarray, k: int = 10):

        scores, indices = self.index.search(
            query_vector.astype("float32"),
            k
        )

        results = []

        for i in range(k):
            idx = indices[0][i]
            results.append({
                "code": self.codes[idx],
                "score": float(scores[0][i])
            })

        return results