# src/evaluation/analyze_gold.py
"""
Расширенный анализ золотой выборки с моделями.
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import defaultdict

from config.settings import TRAINING_DATA_DIR, REFERENCE_DIR, PROCESSED_DATA_DIR, CLASSIFIER_DIR, FAISS_DIR
from src.preprocessing.cleaner import TextCleaner
from src.retrieval.retriever import Retriever
from src.models.bert_classifier import BERTClassifier


def main():
    # 1. Загружаем золотую разметку
    train_path = TRAINING_DATA_DIR / "train.xlsx"
    df = pd.read_excel(train_path, dtype=str)
    df = df[["Номенклатура", "Код ОКПД2"]].dropna()
    df.columns = ["text", "code"]

    # 2. Загружаем справочник для статистики
    ref_path = REFERENCE_DIR / "okpd_2.xlsx"
    ref = pd.read_excel(ref_path, dtype=str)
    ref_codes = set(ref["code"].unique())

    # 3. Статистика текстов (из EDA, но добавим свои)
    print(f"Всего примеров: {len(df)}")
    print(f"Уникальных кодов: {df['code'].nunique()}")
    print(f"Кодов, отсутствующих в справочнике: {df[~df['code'].isin(ref_codes)]['code'].nunique()}")

    # Распределение классов (head/mid/tail)
    code_counts = df["code"].value_counts()
    total_codes = len(code_counts)
    head_threshold = int(total_codes * 0.2)
    mid_threshold = int(total_codes * 0.5)
    head_codes = code_counts.head(head_threshold).index.tolist()
    mid_codes = code_counts.head(mid_threshold).index.tolist()[head_threshold:]
    tail_codes = code_counts.index.tolist()[mid_threshold:]

    print(f"\nHead классов (20%): {len(head_codes)} (поддержка: {code_counts[head_codes].sum()} примеров)")
    print(f"Mid классов (30%): {len(mid_codes)} (поддержка: {code_counts[mid_codes].sum()} примеров)")
    print(f"Tail классов (50%): {len(tail_codes)} (поддержка: {code_counts[tail_codes].sum()} примеров)")

    # 4. Загружаем модели
    cleaner = TextCleaner(abbreviations_path=REFERENCE_DIR / "сокращения.xlsx")
    retriever = Retriever(
        index_path=FAISS_DIR / "okpd_index_stemmed.faiss",
        id_map_path=FAISS_DIR / "id_map_stemmed.csv"
    )
    classifier = BERTClassifier(model_dir=CLASSIFIER_DIR)

    # 5. Оценка для каждого примера
    retrieval_correct = {1: 0, 5: 0, 10: 0}
    classifier_correct = 0
    agreements = 0
    hierarchy_oks = 0
    errors_per_class = defaultdict(int)
    total_per_class = defaultdict(int)
    both_wrong = []

    print("\nОценка моделей по классам...")
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Анализ"):
        text = row["text"]
        true_code = row["code"].strip()
        total_per_class[true_code] += 1

        # Retrieval
        query = cleaner.retrieval_view_normalized(text)
        retrieval_result = retriever.search_normalized(query, top_k=10)
        retrieved_codes = [c["code"] for c in retrieval_result["candidates"]]
        if true_code in retrieved_codes[:1]:
            retrieval_correct[1] += 1
        if true_code in retrieved_codes[:5]:
            retrieval_correct[5] += 1
        if true_code in retrieved_codes[:10]:
            retrieval_correct[10] += 1

        # Classifier
        classifier_pred = classifier.predict(cleaner.classifier_view(text))
        predicted_code = classifier_pred["code"]
        if predicted_code == true_code:
            classifier_correct += 1
        else:
            errors_per_class[true_code] += 1

        # Agreement & hierarchy
        if predicted_code == retrieved_codes[0]:
            agreements += 1
        from src.taxonomy.okpd_tree import is_same_branch
        if is_same_branch(retrieved_codes[0], predicted_code, level=2):
            hierarchy_oks += 1

        # Обе модели ошиблись
        if true_code not in retrieved_codes[:5] and predicted_code != true_code:
            both_wrong.append((text, true_code, predicted_code, retrieved_codes[0]))

    # 6. Вывод результатов
    print(f"\nRetrieval Recall@1: {retrieval_correct[1]/len(df):.4f}")
    print(f"Retrieval Recall@5: {retrieval_correct[5]/len(df):.4f}")
    print(f"Retrieval Recall@10: {retrieval_correct[10]/len(df):.4f}")
    print(f"Classifier Accuracy: {classifier_correct/len(df):.4f}")
    print(f"Agreement (top-1): {agreements/len(df):.4f}")
    print(f"Hierarchy OK: {hierarchy_oks/len(df):.4f}")

    # Топ-10 классов с наибольшим числом ошибок классификатора
    error_counts = pd.Series(errors_per_class).sort_values(ascending=False).head(10)
    print("\nТоп-10 классов по ошибкам классификатора:")
    for code, err_count in error_counts.items():
        total = total_per_class[code]
        print(f"  {code}: {err_count} ошибок из {total} примеров (точность: {(total-err_count)/total:.2f})")

    # Примеры, где обе модели ошиблись
    print(f"\nПримеров, где обе модели ошиблись: {len(both_wrong)}")
    if both_wrong:
        print("Первые 5 примеров:")
        for text, true, pred_clf, pred_ret in both_wrong[:5]:
            print(f"  Текст: {text}")
            print(f"    Истина: {true}, Классификатор: {pred_clf}, Retrieval top-1: {pred_ret}")

    # Сохраняем ошибки в файл
    pd.DataFrame(both_wrong, columns=["text", "true_code", "pred_classifier", "pred_retrieval"]).to_csv(
        TRAINING_DATA_DIR / "both_wrong_examples.csv", index=False
    )
    print(f"\nОшибки обеих моделей сохранены в {TRAINING_DATA_DIR / 'both_wrong_examples.csv'}")


if __name__ == "__main__":
    main()