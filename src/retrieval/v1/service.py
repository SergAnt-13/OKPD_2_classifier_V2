from src.retrieval.v1.encoder import EmbeddingModel
from src.retrieval.v1.faiss_index import FAISSIndex


class RetrievalService:

    def __init__(self, taxonomy_nodes: dict):
        self.encoder = EmbeddingModel()
        self.index = None
        self.taxonomy_nodes = taxonomy_nodes

    def build_index(self):

        codes = []
        texts = []

        for code, node in self.taxonomy_nodes.items():
            texts.append(node.title)
            codes.append(code)

        embeddings = self.encoder.encode(texts)

        self.index = FAISSIndex(dim=embeddings.shape[1])
        self.index.build(embeddings, codes)

    def query(self, text: str, k: int = 10):

        vector = self.encoder.encode([text])

        return self.index.search(vector, k)