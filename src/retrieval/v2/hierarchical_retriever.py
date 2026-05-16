from collections import defaultdict

from src.retrieval.v1.encoder import EmbeddingModel
from src.retrieval.v2.class_index import ClassIndex
from src.retrieval.v2.leaf_index import LeafIndex


class HierarchicalRetriever:

    def __init__(self, taxonomy_nodes: dict):

        self.encoder = EmbeddingModel()
        self.taxonomy = taxonomy_nodes

        self.class_index = None
        self.leaf_index = None

        self.class_to_leaves = defaultdict(list)

    # -------------------------
    # BUILD
    # -------------------------

    def build(self):

        class_texts = []
        class_ids = []

        leaf_texts = []
        leaf_codes = []

        for code, node in self.taxonomy.items():

            class_id = code[:2]

            if len(code) <= 5:
                class_texts.append(node.title)
                class_ids.append(class_id)

            leaf_texts.append(node.title)
            leaf_codes.append(code)

            self.class_to_leaves[class_id].append(code)

        class_emb = self.encoder.encode(class_texts)
        leaf_emb = self.encoder.encode(leaf_texts)

        self.class_index = ClassIndex(class_emb.shape[1])
        self.class_index.build(class_emb, class_ids)

        self.leaf_index = LeafIndex(leaf_emb.shape[1])
        self.leaf_index.build(leaf_emb, leaf_codes)

    # -------------------------
    # QUERY
    # -------------------------

    def search(self, text: str):

        vec = self.encoder.encode([text])

        # STEP 1: top classes
        top_classes = self.class_index.search(vec, k=3)

        candidate_codes = set()

        # STEP 2: expand to leaves
        for c in top_classes:

            class_id = c["class_id"]

            for code in self.class_to_leaves[class_id]:
                candidate_codes.add(code)

        # STEP 3: filter leaf index results
        leaf_results = self.leaf_index.search(vec, k=50)

        filtered = [
            r for r in leaf_results
            if r["code"] in candidate_codes
        ]

        return {
            "top_classes": top_classes,
            "candidates": filtered[:10]
        }