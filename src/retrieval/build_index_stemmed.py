# src/retrieval/build_index_stemmed.py
"""
Построение FAISS-индекса на основе стеммированного (Snowball) справочника.
Работает быстро, создаёт отдельные артефакты с суффиксом '_stemmed'.
"""
import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pandas as pd
import numpy as np
import faiss
import torch
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from nltk.stem.snowball import SnowballStemmer
import re

from config.settings import REFERENCE_DIR, PROCESSED_DATA_DIR, FAISS_DIR, BIENCODER_DIR
from src.preprocessing.cleaner import TextCleaner


def fast_stem(text: str) -> str:
    """Стемминг только русских слов, остальное без изменений."""
    if not isinstance(text, str) or not text.strip():
        return ""
    stemmer = SnowballStemmer("russian")
    tokens = text.split()
    stemmed = []
    for token in tokens:
        if re.fullmatch(r"[а-яё]+", token, re.IGNORECASE):
            stemmed.append(stemmer.stem(token))
        else:
            stemmed.append(token)
    return " ".join(stemmed)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Загружаем справочник
    ref_path = REFERENCE_DIR / "okpd_2.xlsx"
    df = pd.read_excel(ref_path, dtype=str)
    print(f"Загружено {len(df)} позиций")

    # Очистка без лемматизации, затем быстрый стемминг
    cleaner = TextCleaner(abbreviations_path=REFERENCE_DIR / "сокращения.xlsx", use_lemmatizer=False)
    print("Очистка и стемминг названий...")
    tqdm.pandas(desc="Очистка+стемминг")
    df["name_normalized"] = df["name"].progress_apply(
        lambda x: fast_stem(cleaner.retrieval_view_normalized(x))
    )

    # Сохраняем обработанный справочник
    norm_path = PROCESSED_DATA_DIR / "okpd_2_normalized_stemmed.csv"
    df.to_csv(norm_path, index=False)
    print(f"Справочник сохранён: {norm_path}")

    # Модель для эмбеддингов (обученная bi-encoder, если есть)
    if BIENCODER_DIR.exists():
        model = SentenceTransformer(str(BIENCODER_DIR), device=device)
        print("Использована обученная bi-encoder модель")
    else:
        model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2", device=device)

    texts = df["name_normalized"].tolist()
    print("Получение эмбеддингов...")
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

    # FAISS индекс
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    faiss.normalize_L2(embeddings)
    index.add(embeddings)

    # Сохраняем с суффиксом _stemmed
    faiss_path = FAISS_DIR / "okpd_index_stemmed.faiss"
    faiss.write_index(index, str(faiss_path))
    print(f"Индекс сохранён: {faiss_path}")

    id_map = df[["code", "parent_code", "name", "name_normalized"]].copy()
    id_map_path = FAISS_DIR / "id_map_stemmed.csv"
    id_map.to_csv(id_map_path, index=False)
    print(f"Маппинг сохранён: {id_map_path}")
    print("Готово!")


if __name__ == "__main__":
    main()