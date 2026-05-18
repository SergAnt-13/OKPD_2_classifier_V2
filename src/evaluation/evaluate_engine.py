# src/evaluation/evaluate_engine.py
"""
Оценка end‑to‑end пайплайна (retrieval-first + classifier + decision engine)
на всей золотой выборке.
Выводит распределение AUTO/REVIEW/MANUAL и точность в каждой категории.
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
from tqdm import tqdm

from config.settings import TRAINING_DATA_DIR
from src.inference.pipeline import InferencePipeline


def main():
    # Загружаем золотую разметку
    train_path = TRAINING_DATA_DIR / "train.xlsx"
    df = pd.read_excel(train_path, dtype=str)
    df = df[["Номенклатура", "Код ОКПД2"]].dropna()
    df.columns = ["text", "true_code"]
    print(f"Всего примеров: {len(df)}")

    pipeline = InferencePipeline()

    results = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Прогон engine"):
        text = row["text"]
        true_code = row["true_code"].strip()
        pred = pipeline.predict_single(text)
        results.append({
            "text": text,
            "true_code": true_code,
            "final_code": pred["final_prediction"],
            "confidence": pred["confidence"],
            "routing": pred["routing"],
            "reasons": "; ".join(pred["reasons"]),
            "classifier_code": pred.get("classifier_code", ""),
            "classifier_conf": pred.get("classifier_confidence", 0),
            "retrieval_score": pred["top_candidates"][0]["score"] if pred["top_candidates"] else 0,
        })

    res_df = pd.DataFrame(results)

    # Распределение роутинга
    print("\n=== РАСПРЕДЕЛЕНИЕ РОУТИНГА ===")
    routing_counts = res_df["routing"].value_counts()
    for mode in ["AUTO", "REVIEW", "MANUAL"]:
        if mode in routing_counts.index:
            print(f"{mode}: {routing_counts[mode]} ({routing_counts[mode]/len(res_df):.2%})")

    # Точность в каждом режиме
    print("\n=== ТОЧНОСТЬ ПО РЕЖИМАМ ===")
    for mode in ["AUTO", "REVIEW", "MANUAL"]:
        subset = res_df[res_df["routing"] == mode]
        if len(subset) > 0:
            acc = (subset["final_code"] == subset["true_code"]).mean()
            print(f"{mode}: Accuracy = {acc:.4f} (на {len(subset)} примерах)")

    # Средняя уверенность по режимам
    print("\n=== СРЕДНЯЯ УВЕРЕННОСТЬ ===")
    for mode in ["AUTO", "REVIEW", "MANUAL"]:
        subset = res_df[res_df["routing"] == mode]
        if len(subset) > 0:
            avg_conf = subset["confidence"].mean()
            print(f"{mode}: avg_confidence = {avg_conf:.4f}")

    # Сохраняем для анализа
    out_path = TRAINING_DATA_DIR / "engine_evaluation.csv"
    res_df.to_csv(out_path, index=False)
    print(f"\nРезультаты сохранены в {out_path}")


if __name__ == "__main__":
    main()