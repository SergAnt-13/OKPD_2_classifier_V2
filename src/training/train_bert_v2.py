# src/training/train_bert_v2.py
"""
Дообучение RuBERT с balanced class weights.
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
from sklearn.utils.class_weight import compute_class_weight

from config.settings import TRAINING_DATA_DIR, CLASSIFIER_DIR, REFERENCE_DIR
from src.preprocessing.cleaner import TextCleaner


def main():
    # Загружаем данные
    train_path = TRAINING_DATA_DIR / "train.xlsx"
    df = pd.read_excel(train_path, dtype=str)
    df = df[["Номенклатура", "Код ОКПД2"]].dropna()
    df.columns = ["text", "code"]

    cleaner = TextCleaner(abbreviations_path=REFERENCE_DIR / "сокращения.xlsx")
    df["text_clean"] = df["text"].apply(cleaner.classifier_view)

    le = LabelEncoder()
    df["label"] = le.fit_transform(df["code"])
    num_classes = len(le.classes_)  # 148

    # Разбиение (без стратификации)
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

    # Токенизатор и модель
    tokenizer = AutoTokenizer.from_pretrained(str(CLASSIFIER_DIR))
    model = AutoModelForSequenceClassification.from_pretrained(str(CLASSIFIER_DIR))

    # Токенизация
    def tokenize_function(examples):
        return tokenizer(examples["text_clean"], padding="max_length", truncation=True, max_length=128)

    train_dataset = Dataset.from_pandas(train_df[["text_clean", "label"]])
    val_dataset = Dataset.from_pandas(val_df[["text_clean", "label"]])
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    val_dataset = val_dataset.map(tokenize_function, batched=True)
    train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    val_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

    # Веса классов (только для присутствующих в train)
    present_classes = np.unique(train_df["label"])
    class_weights_present = compute_class_weight(
        class_weight='balanced',
        classes=present_classes,
        y=train_df["label"]
    )
    # Полный вектор весов: по умолчанию 1.0 для отсутствующих
    full_weights = np.ones(num_classes, dtype=np.float32)
    for cls_idx, weight in zip(present_classes, class_weights_present):
        full_weights[cls_idx] = weight

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_weights = torch.tensor(full_weights, dtype=torch.float).to(device)

    # Кастомный Trainer с взвешенной функцией потерь
    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits
            loss_fct = torch.nn.CrossEntropyLoss(weight=class_weights)
            loss = loss_fct(logits, labels)
            return (loss, outputs) if return_outputs else loss

    training_args = TrainingArguments(
        output_dir=str(CLASSIFIER_DIR / "checkpoints"),
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_dir=str(CLASSIFIER_DIR / "logs"),
        logging_steps=10,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=1,
        learning_rate=2e-5,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        save_total_limit=1,
        report_to="none",
    )

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=1)],
    )

    print("Дообучение с class weights...")
    trainer.train()

    # Сохраняем модель
    model.save_pretrained(str(CLASSIFIER_DIR))
    tokenizer.save_pretrained(str(CLASSIFIER_DIR))
    import joblib
    joblib.dump(le, str(CLASSIFIER_DIR / "label_encoder.joblib"))
    print(f"Модель обновлена и сохранена в {CLASSIFIER_DIR}")


if __name__ == "__main__":
    main()