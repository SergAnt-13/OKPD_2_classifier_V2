# src/training/train_biencoder.py
"""
Обучение bi-encoder на парах (текст товара, название эталонного кода ОКПД2).
Использует MultipleNegativesRankingLoss.
Совместим с sentence-transformers 2.x (подмодули losses/evaluation).
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

# Импорты для sentence-transformers 2.x
from sentence_transformers import SentenceTransformer, InputExample
from sentence_transformers.losses import MultipleNegativesRankingLoss
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator

from config.settings import (
    REFERENCE_DIR,
    TRAINING_DATA_DIR,
    PROCESSED_DATA_DIR,
    BIENCODER_DIR,
)
from src.preprocessing.cleaner import TextCleaner


def load_gold_pairs():
    """Загружает золотую разметку и справочник, строит позитивные пары."""
    train_path = TRAINING_DATA_DIR / "train.xlsx"
    gold = pd.read_excel(train_path, dtype=str)
    gold = gold[["Номенклатура", "Код ОКПД2"]].dropna()
    gold.columns = ["text", "code"]
    print(f"Загружено {len(gold)} золотых примеров")

    ref_path = PROCESSED_DATA_DIR / "okpd_2_normalized.csv"
    ref = pd.read_csv(ref_path, dtype=str)

    code_to_name = dict(zip(ref["code"], ref["name_normalized"]))

    pairs = []
    cleaner = TextCleaner(abbreviations_path=REFERENCE_DIR / "сокращения.xlsx")
    for _, row in tqdm(gold.iterrows(), total=len(gold), desc="Подготовка пар"):
        text = cleaner.retrieval_view_normalized(row["text"])
        target_name = code_to_name.get(row["code"])
        if target_name and text:
            pairs.append((text, target_name))
    print(f"Построено {len(pairs)} позитивных пар")
    return pairs


def main():
    pairs = load_gold_pairs()

    train_pairs, eval_pairs = train_test_split(pairs, test_size=0.2, random_state=42)

    # Обучающие примеры
    train_examples = [InputExample(texts=[p[0], p[1]]) for p in train_pairs]

    # Данные для оценки: пары и ожидаемая высокая похожесть (1.0)
    eval_sentences1 = [p[0] for p in eval_pairs]
    eval_sentences2 = [p[1] for p in eval_pairs]
    eval_scores = [1.0] * len(eval_pairs)

    model_name = "paraphrase-multilingual-MiniLM-L12-v2"
    print(f"Загрузка базовой модели {model_name}...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Используемое устройство: {device}")
    model = SentenceTransformer(model_name, device=device)

    train_dataloader = torch.utils.data.DataLoader(
        train_examples, shuffle=True, batch_size=16
    )
    train_loss = MultipleNegativesRankingLoss(model)

    evaluator = EmbeddingSimilarityEvaluator(
        sentences1=eval_sentences1,
        sentences2=eval_sentences2,
        scores=eval_scores,
        name="eval",
        show_progress_bar=True,
    )

    BIENCODER_DIR.mkdir(parents=True, exist_ok=True)

    print("Начало обучения...")
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=evaluator,
        epochs=5,                # увеличенное число эпох для лучшей обработки редких классов
        warmup_steps=50,
        output_path=str(BIENCODER_DIR),
        save_best_model=True,
        show_progress_bar=True,
    )
    print(f"Модель сохранена в {BIENCODER_DIR}")


if __name__ == "__main__":
    main()