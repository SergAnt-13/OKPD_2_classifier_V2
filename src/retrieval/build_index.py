# src/retrieval/build_index.py
"""
Скрипт для подготовки нормализованного справочника ОКПД-2 и FAISS-индекса.
Запускается один раз при развёртывании системы или при обновлении справочника.
"""
import sys
from pathlib import Path

# Добавляем корень проекта в sys.path, чтобы работали импорты config и src
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pandas as pd
import torch
import numpy as np
import faiss
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

from config.settings import REFERENCE_DIR, PROCESSED_DATA_DIR, FAISS_DIR
from src.preprocessing.cleaner import TextCleaner


def main():
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using device: {device}")

    # Загружаем справочник
    ref_path = REFERENCE_DIR / "okpd_2.xlsx"
    if not ref_path.exists():
        raise FileNotFoundError(f"Справочник не найден: {ref_path}")
    df = pd.read_excel(ref_path, dtype=str)
    print(f"Загружено {len(df)} позиций справочника")

    # Предобработка названий
    cleaner = TextCleaner(abbreviations_path=REFERENCE_DIR / "сокращения.xlsx", use_lemmatizer=False)
    print("Быстрая очистка названий (примеры первых 10 строк):")
    for i, row in df.head(10).iterrows():
        original = row["name"]
        normalized = cleaner.retrieval_view_normalized(original)
        print(f"  [{i}] {original!r}  ->  {normalized!r}")

    print("\nОбработка всех названий...")
    tqdm.pandas(desc="Очистка названий")
    df["name_normalized"] = df["name"].progress_apply(cleaner.retrieval_view_normalized)

    # Сохраняем обработанный справочник
    norm_path = PROCESSED_DATA_DIR / "okpd_2_normalized.csv"
    df.to_csv(norm_path, index=False)
    print(f"Нормализованный справочник сохранён: {norm_path}")

    # Загружаем модель для эмбеддингов
    model_name = "paraphrase-multilingual-MiniLM-L12-v2"
    print(f"Загрузка модели {model_name}...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(model_name, device=device)

    # Получаем эмбеддинги для всех названий
    texts = df["name_normalized"].tolist()
    print("Получение эмбеддингов...")
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

    # Строим FAISS-индекс
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # inner product = cosine similarity
    faiss.normalize_L2(embeddings)
    index.add(embeddings)

    # Сохраняем индекс
    faiss_path = FAISS_DIR / "okpd_index.faiss"
    faiss.write_index(index, str(faiss_path))
    print(f"FAISS-индекс сохранён: {faiss_path}")

    # Сохраняем соответствие индексов
    # ВАЖНО: сохраняем parent_code для иерархии
    id_map = df[["code", "parent_code", "name", "name_normalized"]].copy()
    id_map_path = FAISS_DIR / "id_map.csv"
    id_map.to_csv(id_map_path, index=False)
    print(f"Маппинг сохранён: {id_map_path}")
    print("Готово!")


if __name__ == "__main__":
    main()