# src/training/train_biencoder.py
"""
Аккуратное дообучение bi-encoder (1 эпоха, заморозка нижних слоёв).
Использует MultipleNegativesRankingLoss на позитивных парах.
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
    TRAINING_DATA_DIR,
    PROCESSED_DATA_DIR,
    BIENCODER_DIR,
)
from src.preprocessing.cleaner import TextCleaner


def main():
    # Загружаем золотую разметку и справочник
    train_path = TRAINING_DATA_DIR / "train.xlsx"
    gold = pd.read_excel(train_path, dtype=str)
    gold = gold[["Номенклатура", "Код ОКПД2"]].dropna()
    gold.columns = ["text", "code"]

    ref = pd.read_csv(PROCESSED_DATA_DIR / "okpd_2_normalized.csv", dtype=str)
    code_to_name = dict(zip(ref["code"], ref["name_normalized"]))

    cleaner = TextCleaner(abbreviations_path=REFERENCE_DIR / "сокращения.xlsx")
    pairs = []
    for _, row in tqdm(gold.iterrows(), total=len(gold), desc="Подготовка пар"):
        text = cleaner.retrieval_view_normalized(row["text"])
        target = code_to_name.get(row["code"].strip())
        if text and target:
            pairs.append((text, target))
    print(f"Позитивных пар: {len(pairs)}")

    train_pairs, eval_pairs = train_test_split(pairs, test_size=0.2, random_state=42)

    train_examples = [InputExample(texts=[p[0], p[1]]) for p in train_pairs]
    eval_s1 = [p[0] for p in eval_pairs]
    eval_s2 = [p[1] for p in eval_pairs]
    eval_scores = [1.0] * len(eval_pairs)

    model_name = "paraphrase-multilingual-MiniLM-L12-v2"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Загрузка {model_name} на {device}...")
    model = SentenceTransformer(model_name, device=device)

    # Заморозка первых 6 слоёв
    for name, param in model.named_parameters():
        if name.startswith("0.encoder.layer."):
            layer_num = int(name.split(".")[2])
            if layer_num < 6:
                param.requires_grad = False

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
        warmup_steps=20,
        output_path=str(BIENCODER_DIR),
        save_best_model=True,
        show_progress_bar=True,
    )
    print(f"Модель сохранена в {BIENCODER_DIR}")


if __name__ == "__main__":
    main()