# src/evaluation/evaluate_classifier.py
"""
Оценка классификатора (RuBERT) на валидационной выборке.
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from tqdm import tqdm

from config.settings import TRAINING_DATA_DIR, CLASSIFIER_DIR, REFERENCE_DIR
from src.preprocessing.cleaner import TextCleaner
from src.models.bert_classifier import BERTClassifier


def main():
    # Загружаем данные
    train_path = TRAINING_DATA_DIR / "train.xlsx"
    df = pd.read_excel(train_path, dtype=str)
    df = df[["Номенклатура", "Код ОКПД2"]].dropna()
    df.columns = ["text", "code"]

    # Очистка текста
    cleaner = TextCleaner(abbreviations_path=REFERENCE_DIR / "сокращения.xlsx")
    df["text_clean"] = df["text"].apply(cleaner.classifier_view)

    # Отложенная выборка (20%) без стратификации
    _, val_df = train_test_split(df, test_size=0.2, random_state=42)

    # Загружаем обученный классификатор
    clf = BERTClassifier(model_dir=CLASSIFIER_DIR)

    y_true = []
    y_pred = []
    confidences = []

    print(f"Оценка на {len(val_df)} примерах...")
    for _, row in tqdm(val_df.iterrows(), total=len(val_df)):
        true_code = row["code"]
        pred = clf.predict(row["text_clean"])
        y_true.append(true_code)
        y_pred.append(pred["code"])
        confidences.append(pred["confidence"])

    acc = accuracy_score(y_true, y_pred)
    print(f"\nОбщая Accuracy: {acc:.4f}")

    # Отчёт по всем встреченным классам (без target_names, просто строки)
    print("\nОтчет по классификации (все встреченные классы):")
    print(classification_report(y_true, y_pred, zero_division=0))

    # Доверие по частым классам
    conf_df = pd.DataFrame({"true": y_true, "pred": y_pred, "conf": confidences})
    top5 = conf_df["true"].value_counts().head(5).index
    print("\nСредняя уверенность и точность по топ-5 классам:")
    for cls in top5:
        mask = conf_df["true"] == cls
        correct = (conf_df.loc[mask, "true"] == conf_df.loc[mask, "pred"]).mean()
        avg_conf = conf_df.loc[mask, "conf"].mean()
        print(f"  {cls}: avg_confidence={avg_conf:.3f}, accuracy={correct:.3f}")


if __name__ == "__main__":
    main()