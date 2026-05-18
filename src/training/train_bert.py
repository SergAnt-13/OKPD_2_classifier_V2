# src/training/train_bert.py
"""
Обучение BERT-классификатора на золотой разметке.
Использует DeepPavlov/rubert-base-cased, стратифицированное разбиение.
Сохраняет модель, токенизатор и маппинг меток в artifacts/classifier/.
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)
from datasets import Dataset

from config.settings import TRAINING_DATA_DIR, CLASSIFIER_DIR, REFERENCE_DIR
from src.preprocessing.cleaner import TextCleaner


def load_and_prepare_data():
    """Загружает и подготавливает данные, кодирует метки."""
    train_path = TRAINING_DATA_DIR / "train.xlsx"
    df = pd.read_excel(train_path, dtype=str)
    df = df[["Номенклатура", "Код ОКПД2"]].dropna()
    df.columns = ["text", "code"]

    # Очистка текста (classifier_view)
    cleaner = TextCleaner(abbreviations_path=REFERENCE_DIR / "сокращения.xlsx")
    df["text_clean"] = df["text"].apply(cleaner.classifier_view)

    # Кодирование меток
    le = LabelEncoder()
    df["label"] = le.fit_transform(df["code"])

    # Стратифицированное разбиение (80/20)
    train_df, val_df = train_test_split(
        df, test_size=0.2, random_state=42
    )
    return train_df, val_df, le


def main():
    train_df, val_df, label_encoder = load_and_prepare_data()
    num_classes = len(label_encoder.classes_)
    print(f"Число классов: {num_classes}")

    # Токенизатор и модель
    model_name = "DeepPavlov/rubert-base-cased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=num_classes
    )

    # Токенизация
    def tokenize_function(examples):
        return tokenizer(
            examples["text_clean"], padding="max_length", truncation=True, max_length=128
        )

    train_dataset = Dataset.from_pandas(train_df[["text_clean", "label"]])
    val_dataset = Dataset.from_pandas(val_df[["text_clean", "label"]])
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    val_dataset = val_dataset.map(tokenize_function, batched=True)
    train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    val_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

    # Аргументы обучения
    training_args = TrainingArguments(
        output_dir=str(CLASSIFIER_DIR / "checkpoints"),
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_dir=str(CLASSIFIER_DIR / "logs"),
        logging_steps=10,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=5,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        save_total_limit=1,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    print("Начало обучения...")
    trainer.train()

    # Сохраняем финальную модель и артефакты
    CLASSIFIER_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(CLASSIFIER_DIR))
    tokenizer.save_pretrained(str(CLASSIFIER_DIR))

    # Сохраняем LabelEncoder
    import joblib
    joblib.dump(label_encoder, str(CLASSIFIER_DIR / "label_encoder.joblib"))

    print(f"Модель сохранена в {CLASSIFIER_DIR}")


if __name__ == "__main__":
    main()