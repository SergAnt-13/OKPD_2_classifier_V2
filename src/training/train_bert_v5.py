# src/training/train_bert_enhanced.py
"""
Enhanced-обучение BERT на золотой выборке (1475 записей, 61 класс).
Параметры как в V1: 13 эпох, early stopping, class weights, label smoothing.
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
from sklearn.utils.class_weight import compute_class_weight
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


def main():
    # 1. Загружаем золотую выборку (Merged)
    train_path = TRAINING_DATA_DIR / "train.xlsx"
    df = pd.read_excel(train_path, dtype=str)
    df = df[["Номенклатура", "Код ОКПД2"]].dropna()
    df.columns = ["text", "code"]

    # 2. Оставляем только классы с >= 5 примерами (как в V1)
    code_counts = df["code"].value_counts()
    frequent_codes = code_counts[code_counts >= 5].index
    df = df[df["code"].isin(frequent_codes)].copy()
    print(f"После фильтрации: {len(df)} записей, {len(frequent_codes)} классов")

    # 3. Очистка текста (classifier_view)
    cleaner = TextCleaner(abbreviations_path=REFERENCE_DIR / "сокращения.xlsx")
    df["text_clean"] = df["text"].apply(cleaner.classifier_view)

    # 4. Кодирование меток
    le = LabelEncoder()
    df["label"] = le.fit_transform(df["code"])
    num_classes = len(le.classes_)
    print(f"Число классов: {num_classes}")

    # 5. Разделяем на train/test (стратифицированно)
    try:
        train_df, test_df = train_test_split(
            df, test_size=0.1, random_state=42, stratify=df["label"]
        )
    except ValueError:
        train_df, test_df = train_test_split(df, test_size=0.1, random_state=42)

    # 6. Токенизатор и модель
    model_name = "DeepPavlov/rubert-base-cased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=num_classes
    )

    def tokenize_function(examples):
        return tokenizer(
            examples["text_clean"], padding="max_length", truncation=True, max_length=96
        )

    train_dataset = Dataset.from_pandas(train_df[["text_clean", "label"]])
    test_dataset = Dataset.from_pandas(test_df[["text_clean", "label"]])
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    test_dataset = test_dataset.map(tokenize_function, batched=True)
    train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    test_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

    # 7. Веса классов (balanced)
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_df["label"]),
        y=train_df["label"]
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)

    # 8. WeightedTrainer с label smoothing
    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits
            loss_fct = torch.nn.CrossEntropyLoss(weight=class_weights_tensor)
            loss = loss_fct(logits, labels)
            return (loss, outputs) if return_outputs else loss

    training_args = TrainingArguments(
        output_dir=str(CLASSIFIER_DIR / "checkpoints"),
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_dir=str(CLASSIFIER_DIR / "logs"),
        logging_steps=10,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        num_train_epochs=13,                 # как в V1
        learning_rate=2e-5,
        weight_decay=0.01,
        label_smoothing_factor=0.1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        save_total_limit=1,
        report_to="none",
    )

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    print("Enhanced-обучение BERT (как в V1)...")
    trainer.train()

    # Сохраняем модель и LabelEncoder
    model.save_pretrained(str(CLASSIFIER_DIR))
    tokenizer.save_pretrained(str(CLASSIFIER_DIR))
    import joblib
    joblib.dump(le, str(CLASSIFIER_DIR / "label_encoder.joblib"))
    print(f"Модель сохранена в {CLASSIFIER_DIR}")


if __name__ == "__main__":
    main()