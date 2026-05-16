import pandas as pd
from typing import Dict

from src.taxonomy.node import OKPDNode


class TaxonomyLoader:

    def __init__(self, filepath: str):
        self.filepath = filepath

    def load(self) -> Dict[str, OKPDNode]:
        df = pd.read_excel(self.filepath)

        nodes = {}

        # 1. создаём все узлы
        for _, row in df.iterrows():
            code = str(row["code"])
            nodes[code] = OKPDNode(
                code=code,
                title=row["description"],
                parent_code=str(row["parent_code"]) if not pd.isna(row["parent_code"]) else None
            )

        # 2. строим дерево (children links)
        for code, node in nodes.items():
            if node.parent_code and node.parent_code in nodes:
                nodes[node.parent_code].children.append(code)

        return nodes