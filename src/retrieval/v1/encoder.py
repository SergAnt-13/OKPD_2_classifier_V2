from sentence_transformers import SentenceTransformer
import numpy as np


class EmbeddingModel:

    def __init__(self, model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        self.model = SentenceTransformer(model_name)

    def encode(self, texts):
        """
        texts: List[str]
        """

        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False
        )

        return np.array(embeddings)