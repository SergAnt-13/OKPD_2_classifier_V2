# src/retrieval/build_index_lemmatized.py
"""
Построение FAISS-индекса на основе лемматизированного справочника.
Запускается отдельно, не трогает основной индекс.
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

from config.settings import REFERENCE_DIR, PROCESSED_DATA_DIR, FAISS_DIR
from src.preprocessing.cleaner import TextCleaner

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Загружаем справочник
    ref_path = REFERENCE_DIR / "okpd_2.xlsx"
    df = pd.read_excel(ref_path, dtype=str)
    print(f"Загружено {len(df)} позиций")

    # Включаем лемматизацию
    cleaner = TextCleaner(abbreviations_path=REFERENCE_DIR / "сокращения.xlsx", use_lemmatizer=True)
    print("Лемматизация названий...")
    tqdm.pandas(desc="Лемматизация")
    df["name_normalized"] = df["name"].progress_apply(cleaner.retrieval_view_normalized)

    # Сохраняем нормализованный справочник (лемматизированный)
    norm_path = PROCESSED_DATA_DIR / "okpd_2_normalized_lemm.csv"
    df.to_csv(norm_path, index=False)
    print(f"Справочник сохранён: {norm_path}")

    # Модель для эмбеддингов (та же MiniLM или обученная, если есть)
    model_name = "paraphrase-multilingual-MiniLM-L12-v2"
    from config.settings import BIENCODER_DIR
    if BIENCODER_DIR.exists():
        model = SentenceTransformer(str(BIENCODER_DIR), device=device)
        print("Использована обученная bi-encoder модель")
    else:
        model = SentenceTransformer(model_name, device=device)

    texts = df["name_normalized"].tolist()
    print("Получение эмбеддингов...")
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

    # FAISS индекс
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    faiss.normalize_L2(embeddings)
    index.add(embeddings)

    # Сохраняем индекс и маппинг с отдельными именами
    faiss_path = FAISS_DIR / "okpd_index_lemm.faiss"
    faiss.write_index(index, str(faiss_path))
    print(f"Индекс сохранён: {faiss_path}")

    id_map = df[["code", "parent_code", "name", "name_normalized"]].copy()
    id_map_path = FAISS_DIR / "id_map_lemm.csv"
    id_map.to_csv(id_map_path, index=False)
    print(f"Маппинг сохранён: {id_map_path}")
    print("Готово!")

if __name__ == "__main__":
    main()