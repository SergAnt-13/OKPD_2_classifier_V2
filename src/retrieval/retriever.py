# src/retrieval/retriever.py
from typing import List, Dict


class Retriever:
    """
    Заглушка retrieval-компонента.
    В будущем будет использовать bi-encoder + FAISS.
    """

    def __init__(self, index_path: str = None):
        self.index_path = index_path

    def search(self, text: str, top_k: int = 5) -> Dict:
        """
        Возвращает словарь с ключом "candidates" — список dict'ов:
        {"code": str, "score": float, "title": str}
        """
        # TODO: заменить на реальный поиск по FAISS
        return {
            "candidates": [
                {"code": "00.00.00.000", "score": 0.65, "title": "Заглушка"},
            ]
        }