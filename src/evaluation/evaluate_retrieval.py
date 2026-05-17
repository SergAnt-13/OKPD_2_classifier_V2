# src/evaluation/evaluate_retrieval.py
"""
Оценка retrieval на золотой разметке.
Поддерживает кеширование нормализованных текстов.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import pandas as pd
from tqdm import tqdm

from config.settings import TRAINING_DATA_DIR, PROCESSED_DATA_DIR, REFERENCE_DIR
from src.retrieval.retriever import Retriever
from src.preprocessing.cleaner import TextCleaner


def evaluate_retrieval(k_values=(1, 5, 10), use_lemmatizer=False):
    # 1. Загружаем золотую разметку
    train_path = TRAINING_DATA_DIR / "train.xlsx"
    df = pd.read_excel(train_path, dtype=str)
    df = df[["Номенклатура", "Код ОКПД2"]].dropna()
    df.columns = ["text", "true_code"]
    print(f"Всего примеров: {len(df)}")

    # 2. Имя кеш-файла
    cache_suffix = "_lemm" if use_lemmatizer else ""
    cache_path = PROCESSED_DATA_DIR / f"train_normalized{cache_suffix}.csv"

    # 3. Загрузка кеша
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

    # 4. Нормализация при необходимости
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
            texts_norm.append(norm_text)
            true_codes.append(tc)

        pd.DataFrame({
            "normalized": texts_norm,
            "true_code": true_codes,
        }).to_csv(cache_path, index=False)
        print(f"Кеш сохранён: {cache_path}")

    # 5. Создаём retriever (базовая MiniLM, без model_dir)
    retriever = Retriever(use_lemmatizer=use_lemmatizer)

    # 6. Оценка
    recall = {k: 0 for k in k_values}
    reciprocal_ranks = []
    total = 0

    for norm_text, true_code in tqdm(
        zip(texts_norm, true_codes), total=len(texts_norm), desc="Оценка retrieval"
    ):
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

    print(f"\nОценка на {total} примерах (лемматизация: {use_lemmatizer}):")
    for k in k_values:
        print(f"Recall@{k}: {recall[k] / total:.4f}")
    mrr = sum(reciprocal_ranks) / total if total > 0 else 0.0
    print(f"MRR: {mrr:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Оценка retrieval на золотой разметке")
    parser.add_argument("--lemmatize", action="store_true", help="Включить лемматизацию")
    args = parser.parse_args()
    evaluate_retrieval(use_lemmatizer=args.lemmatize)