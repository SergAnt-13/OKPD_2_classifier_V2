# src/retrieval/retriever.py
import faiss
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional

from sentence_transformers import SentenceTransformer
from config.settings import FAISS_DIR, REFERENCE_DIR
from src.preprocessing.cleaner import TextCleaner
import torch


class Retriever:
    """
    Семантический поиск по справочнику ОКПД-2.
    Использует bi-encoder и FAISS.
    Индекс и маппинг загружаются лениво при первом поиске.
    """

    def __init__(
        self,
        model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
        model_dir: Optional[Path] = None,
        index_path: Optional[Path] = None,
        id_map_path: Optional[Path] = None,
        use_lemmatizer: bool = False,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.cleaner = TextCleaner(
            abbreviations_path=REFERENCE_DIR / "сокращения.xlsx",
            use_lemmatizer=use_lemmatizer,
        )
        if model_dir and Path(model_dir).exists():
            self.model = SentenceTransformer(str(model_dir), device=device)
        else:
            self.model = SentenceTransformer(model_name, device=device)

        self.index_path = index_path or FAISS_DIR / "okpd_index.faiss"
        self.id_map_path = id_map_path or FAISS_DIR / "id_map.csv"

        self._loaded = False
        self.index = None
        self.id_map = None
        self.codes = None
        self.parent_codes = None
        self.names = None

    def _lazy_load(self) -> None:
        if self._loaded:
            return

        if not self.index_path.exists():
            raise FileNotFoundError(
                f"FAISS-индекс не найден: {self.index_path}\n"
                f"Сначала запустите: python src/retrieval/build_index.py"
            )
        if not self.id_map_path.exists():
            raise FileNotFoundError(
                f"Файл маппинга не найден: {self.id_map_path}\n"
                f"Сначала запустите: python src/retrieval/build_index.py"
            )

        print(f"Загрузка FAISS-индекса из {self.index_path}...")
        self.index = faiss.read_index(str(self.index_path))

        print(f"Загрузка маппинга из {self.id_map_path}...")
        self.id_map = pd.read_csv(self.id_map_path)
        self.codes = self.id_map["code"].values
        self.parent_codes = self.id_map["parent_code"].values
        self.names = self.id_map["name"].values

        self._loaded = True

    def search(self, text: str, top_k: int = 5) -> Dict:
        self._lazy_load()
        query_norm = self.cleaner.retrieval_view_normalized(text)
        if not query_norm.strip():
            return {"candidates": []}

        query_emb = self.model.encode([query_norm], convert_to_numpy=True)
        faiss.normalize_L2(query_emb)

        scores, indices = self.index.search(query_emb, top_k)

        candidates = []
        for score, idx in zip(scores[0], indices[0]):
            if 0 <= idx < len(self.codes):
                candidates.append(
                    {
                        "code": self.codes[idx],
                        "parent_code": self.parent_codes[idx],
                        "score": float(score),
                        "title": self.names[idx],
                    }
                )
        return {"candidates": candidates}

    def search_normalized(self, normalized_text: str, top_k: int = 5) -> Dict:
        self._lazy_load()
        if not normalized_text.strip():
            return {"candidates": []}

        query_emb = self.model.encode([normalized_text], convert_to_numpy=True)
        faiss.normalize_L2(query_emb)

        scores, indices = self.index.search(query_emb, top_k)

        candidates = []
        for score, idx in zip(scores[0], indices[0]):
            if 0 <= idx < len(self.codes):
                candidates.append(
                    {
                        "code": self.codes[idx],
                        "parent_code": self.parent_codes[idx],
                        "score": float(score),
                        "title": self.names[idx],
                    }
                )
        return {"candidates": candidates}

    def search_with_hierarchy(
        self,
        text: str,
        top_k: int = 5,
        top_n_broad: int = 50,
        min_parent_agreement: int = 3,
    ) -> Dict:
        """
        Двухэтапный поиск с учётом иерархии.
        1. Широкий поиск (top_n_broad кандидатов).
        2. Определяем наиболее частую родительскую ветку среди top-3 результатов.
        3. Оставляем только кандидатов из этой ветки и берём top_k.
        """
        # Этап 1: широкий поиск
        broad_result = self.search_normalized(text, top_k=top_n_broad)
        candidates = broad_result["candidates"]
        if not candidates:
            return broad_result

        # Определяем родительский код среди top-3
        top_n = min(3, len(candidates))
        parent_votes = [
            cand.get("parent_code", cand["code"][:2]) for cand in candidates[:top_n]
        ]
        best_parent = max(set(parent_votes), key=parent_votes.count)

        # Этап 2: фильтруем всех кандидатов по родителю
        filtered = [
            cand
            for cand in candidates
            if cand.get("parent_code", cand["code"][:2]) == best_parent
        ]
        result = filtered[:top_k]
        return {"candidates": result}