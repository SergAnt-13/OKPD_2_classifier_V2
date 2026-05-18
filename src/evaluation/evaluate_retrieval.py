# src/evaluation/evaluate_retrieval.py
"""
Оценка retrieval на золотой разметке с поддержкой A/B-тестирования:
- основной индекс (без стемминга)
- лемматизированный индекс (со стеммингом Snowball)
- опциональная мягкая иерархия
- кеширование нормализованных текстов
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import pandas as pd
from tqdm import tqdm
import re
from nltk.stem.snowball import SnowballStemmer

from config.settings import TRAINING_DATA_DIR, PROCESSED_DATA_DIR, REFERENCE_DIR, FAISS_DIR, BIENCODER_DIR
from src.retrieval.retriever import Retriever
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


def evaluate_retrieval(k_values=(1, 5, 10), use_lemmatizer=False, use_soft_hierarchy=False, use_lemmatized_index=False):
    # 1. Загружаем золотую разметку
    train_path = TRAINING_DATA_DIR / "train.xlsx"
    df = pd.read_excel(train_path, dtype=str)
    df = df[["Номенклатура", "Код ОКПД2"]].dropna()
    df.columns = ["text", "true_code"]
    print(f"Всего примеров: {len(df)}")

    # 2. Имя кеш-файла
    if use_lemmatized_index:
        cache_suffix = "_stemmed"
    else:
        cache_suffix = "_lemm" if use_lemmatizer else ""
    cache_path = PROCESSED_DATA_DIR / f"train_normalized{cache_suffix}.csv"

    # 3. Загрузка или создание кеша
    if cache_path.exists():
        cached = pd.read_csv(cache_path, dtype=str)
        if "normalized" in cached.columns and "true_code" in cached.columns:
            print(f"Загружен кеш: {cache_path}")
            texts_norm = cached["normalized"].tolist()
            true_codes = cached["true_code"].tolist()
        else:
            cached = None
    else:
        cached = None

    if cached is None:
        cleaner = TextCleaner(
            abbreviations_path=REFERENCE_DIR / "сокращения.xlsx",
            use_lemmatizer=use_lemmatizer,
        )
        texts_norm = []
        true_codes = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Нормализация"):
            text = row["text"]
            tc = row["true_code"].strip()
            if not text or not tc:
                continue
            norm_text = cleaner.retrieval_view_normalized(text)
            # Если тестируем стеммированный индекс, применяем стеммер
            if use_lemmatized_index:
                norm_text = fast_stem(norm_text)
            texts_norm.append(norm_text)
            true_codes.append(tc)

        pd.DataFrame({
            "normalized": texts_norm,
            "true_code": true_codes,
        }).to_csv(cache_path, index=False)
        print(f"Кеш сохранён: {cache_path}")

    # 4. Создаём Retriever с нужными индексами
    if use_lemmatized_index:
        index_path = FAISS_DIR / "okpd_index_stemmed.faiss"
        id_map_path = FAISS_DIR / "id_map_stemmed.csv"
        if not index_path.exists() or not id_map_path.exists():
            raise FileNotFoundError("Стеммированный индекс не найден. Сначала запустите build_index_stemmed.py")
    else:
        index_path = None   # Retriever использует стандартные
        id_map_path = None

    retriever = Retriever(
        use_lemmatizer=use_lemmatizer,
        model_dir=BIENCODER_DIR if BIENCODER_DIR.exists() else None,
        index_path=index_path,
        id_map_path=id_map_path,
    )

    # 5. Оценка
    recall = {k: 0 for k in k_values}
    reciprocal_ranks = []
    total = 0

    for norm_text, true_code in tqdm(
        zip(texts_norm, true_codes), total=len(texts_norm), desc="Оценка retrieval"
    ):
        if use_soft_hierarchy:
            result = retriever.search_with_soft_hierarchy(norm_text, top_k=max(k_values))
        else:
            result = retriever.search_normalized(norm_text, top_k=max(k_values))

        candidates = result["candidates"]
        if not candidates:
            continue

        found_at = None
        for rank, cand in enumerate(candidates, start=1):
            if cand["code"] == true_code:
                found_at = rank
                break

        total += 1
        if found_at:
            reciprocal_ranks.append(1.0 / found_at)
            for k in k_values:
                if found_at <= k:
                    recall[k] += 1
        else:
            reciprocal_ranks.append(0.0)

    print(f"\nОценка на {total} примерах "
          f"(лемматизация: {use_lemmatizer}, "
          f"soft_hierarchy: {use_soft_hierarchy}, "
          f"stemmed_index: {use_lemmatized_index}):")
    for k in k_values:
        print(f"Recall@{k}: {recall[k] / total:.4f}")
    mrr = sum(reciprocal_ranks) / total if total > 0 else 0.0
    print(f"MRR: {mrr:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Оценка retrieval на золотой разметке")
    parser.add_argument("--lemmatize", action="store_true", help="Включить лемматизацию Mystem")
    parser.add_argument("--soft-hierarchy", action="store_true", help="Мягкое иерархическое переранжирование")
    parser.add_argument("--stemmed-index", action="store_true", help="Использовать лемматизированный (стеммированный) индекс")
    args = parser.parse_args()
    evaluate_retrieval(
        use_lemmatizer=args.lemmatize,
        use_soft_hierarchy=args.soft_hierarchy,
        use_lemmatized_index=args.stemmed_index,
    )