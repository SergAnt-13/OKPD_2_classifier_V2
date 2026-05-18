# src/training/train_biencoder_food_focused.py
"""
Переобучение bi-encoder только на пищевых товарах (классы 01, 02, 03, 10).
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import torch
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from sentence_transformers import SentenceTransformer, InputExample
from sentence_transformers.losses import MultipleNegativesRankingLoss
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator

from config.settings import (
    REFERENCE_DIR,
    RAW_DATA_DIR,
    TRAINING_DATA_DIR,
    PROCESSED_DATA_DIR,
    BIENCODER_DIR,
)
from src.preprocessing.cleaner import TextCleaner

def main():
    # 1. Загружаем золотую выборку и полную номенклатуру
    cleaner = TextCleaner(abbreviations_path=REFERENCE_DIR / "сокращения.xlsx")

    # Золото
    train_path = TRAINING_DATA_DIR / "train.xlsx"
    gold = pd.read_excel(train_path, dtype=str)
    gold = gold[["Номенклатура", "Код ОКПД2"]].dropna()
    gold.columns = ["text", "code"]

    # Полная номенклатура
    nom_path = RAW_DATA_DIR / "all_nomenclature.xlsx"
    nom = pd.read_excel(nom_path, dtype=str)
    nom.columns = ['nomenclature', 'nomenclature_name', 'nds_rate', 'okpd2_code']
    nom = nom[nom['okpd2_code'].notna() & (nom['okpd2_code'] != '')]

    # Объединяем и оставляем только пищевые классы (01, 02, 03, 10)
    all_data = pd.concat([
        gold[["text", "code"]],
        nom[["nomenclature", "okpd2_code"]].rename(columns={"nomenclature": "text", "okpd2_code": "code"})
    ], ignore_index=True)

    # Фильтруем по первым двум цифрам кода
    food_prefixes = ('01', '02', '03', '10')
    all_data = all_data[all_data['code'].str.startswith(food_prefixes, na=False)].copy()
    print(f"Пищевых товаров: {len(all_data)}")

    # Справочник (стеммированный)
    ref = pd.read_csv(PROCESSED_DATA_DIR / "okpd_2_normalized_stemmed.csv", dtype=str)
    code_to_name = dict(zip(ref["code"], ref["name_normalized"]))

    # Строим пары
    pairs = []
    for _, row in tqdm(all_data.iterrows(), total=len(all_data), desc="Подготовка пар"):
        text = cleaner.retrieval_view_normalized(row['text'])
        target = code_to_name.get(row['code'].strip())
        if text and target:
            pairs.append((text, target))
    print(f"Всего пар: {len(pairs)}")

    # Разделяем train/eval
    train_pairs, eval_pairs = train_test_split(pairs, test_size=0.2, random_state=42)
    train_examples = [InputExample(texts=[p[0], p[1]]) for p in train_pairs]
    eval_s1 = [p[0] for p in eval_pairs]
    eval_s2 = [p[1] for p in eval_pairs]
    eval_scores = [1.0] * len(eval_pairs)

    # Модель: дообучаем существующую или с нуля
    if BIENCODER_DIR.exists():
        print("Загружаем существующую модель...")
        model = SentenceTransformer(str(BIENCODER_DIR))
    else:
        print("Загружаем базовую MiniLM...")
        model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    train_dataloader = torch.utils.data.DataLoader(train_examples, shuffle=True, batch_size=16)
    train_loss = MultipleNegativesRankingLoss(model)

    evaluator = EmbeddingSimilarityEvaluator(
        sentences1=eval_s1,
        sentences2=eval_s2,
        scores=eval_scores,
        name="eval",
        show_progress_bar=True,
    )

    BIENCODER_DIR.mkdir(parents=True, exist_ok=True)
    print("Начало обучения (1 эпоха)...")
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=evaluator,
        epochs=1,
        warmup_steps=100,
        output_path=str(BIENCODER_DIR),
        save_best_model=True,
        show_progress_bar=True,
    )
    print(f"Модель сохранена в {BIENCODER_DIR}")


if __name__ == "__main__":
    main()