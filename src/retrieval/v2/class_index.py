import numpy as np
import faiss


class ClassIndex:

    def __init__(self, dim: int):
        self.index = faiss.IndexFlatIP(dim)
        self.class_ids = []

    def build(self, embeddings: np.ndarray, class_ids: list):

        self.index.add(embeddings.astype("float32"))
        self.class_ids = class_ids

    def search(self, vector: np.ndarray, k: int = 3):

        scores, idxs = self.index.search(
            vector.astype("float32"),
            k
        )

        results = []

        for i in range(k):
            results.append({
                "class_id": self.class_ids[idxs[0][i]],
                "score": float(scores[0][i])
            })

        return results