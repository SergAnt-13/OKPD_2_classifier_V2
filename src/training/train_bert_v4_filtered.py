# src/training/train_bert_v4_filtered.py
"""
Обучение RuBERT только на классах с достаточным количеством примеров.
Остальные классы покрываются retrieval-first.
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

from config.settings import (
    TRAINING_DATA_DIR,
    RAW_DATA_DIR,
    CLASSIFIER_DIR,
    REFERENCE_DIR,
)
from src.preprocessing.cleaner import TextCleaner


def main():
    # 1. Загружаем данные (золото + промка)
    gold = pd.read_excel(TRAINING_DATA_DIR / "train.xlsx", dtype=str)
    gold = gold[["Номенклатура", "Код ОКПД2"]].dropna()
    gold.columns = ["text", "code"]

    nom = pd.read_excel(RAW_DATA_DIR / "all_nomenclature.xlsx", dtype=str)
    nom.columns = ['nomenclature', 'nomenclature_name', 'nds_rate', 'okpd2_code']
    nom = nom[nom['okpd2_code'].notna() & (nom['okpd2_code'] != '')]
    nom = nom[["nomenclature", "okpd2_code"]].rename(
        columns={"nomenclature": "text", "okpd2_code": "code"}
    )

    all_data = pd.concat([gold, nom], ignore_index=True)
    food_prefixes = ('01', '02', '03', '10')
    all_data = all_data[all_data['code'].str.startswith(food_prefixes, na=False)]
    print(f"Всего записей (пищевые): {len(all_data)}")

    # 2. Оставляем только классы с >= MIN_SAMPLES примерами
    MIN_SAMPLES = 5
    code_counts = all_data['code'].value_counts()
    frequent_codes = code_counts[code_counts >= MIN_SAMPLES].index
    all_data_filtered = all_data[all_data['code'].isin(frequent_codes)].copy()
    print(f"После фильтрации (>= {MIN_SAMPLES} примеров): {len(all_data_filtered)} записей, "
          f"{len(frequent_codes)} классов")

    # 3. LabelEncoder только для отфильтрованных классов
    le = LabelEncoder()
    all_data_filtered["label"] = le.fit_transform(all_data_filtered["code"])
    num_classes = len(le.classes_)
    print(f"Число классов после фильтрации: {num_classes}")

    # 4. Очистка текста
    cleaner = TextCleaner(abbreviations_path=REFERENCE_DIR / "сокращения.xlsx")
    all_data_filtered["text_clean"] = all_data_filtered["text"].apply(cleaner.classifier_view)

    # 5. Train/val разбиение
    train_df, val_df = train_test_split(all_data_filtered, test_size=0.2, random_state=42)

    # 6. Веса классов
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_df["label"]),
        y=train_df["label"]
    )
    full_weights = np.ones(num_classes, dtype=np.float32)
    for cls_idx, weight in zip(np.unique(train_df["label"]), class_weights):
        full_weights[cls_idx] = weight

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_weights_tensor = torch.tensor(full_weights, dtype=torch.float).to(device)

    # 7. Токенизатор и модель (загружаем предыдущую, меняем голову)
    tokenizer = AutoTokenizer.from_pretrained(str(CLASSIFIER_DIR))
    model = AutoModelForSequenceClassification.from_pretrained(
        str(CLASSIFIER_DIR),
        num_labels=num_classes,
        ignore_mismatched_sizes=True
    )

    def tokenize_function(examples):
        return tokenizer(
            examples["text_clean"],
            padding="max_length",
            truncation=True,
            max_length=128,
        )

    train_dataset = Dataset.from_pandas(train_df[["text_clean", "label"]])
    val_dataset = Dataset.from_pandas(val_df[["text_clean", "label"]])
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    val_dataset = val_dataset.map(tokenize_function, batched=True)
    train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    val_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

    # 8. WeightedTrainer
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
        logging_steps=50,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,               # немного увеличим, т.к. классов меньше
        learning_rate=5e-6,
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
        eval_dataset=val_dataset,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=1)],
    )

    print("Обучение на отфильтрованных классах...")
    trainer.train()

    # Сохраняем модель и LabelEncoder
    model.save_pretrained(str(CLASSIFIER_DIR))
    tokenizer.save_pretrained(str(CLASSIFIER_DIR))
    import joblib
    joblib.dump(le, str(CLASSIFIER_DIR / "label_encoder.joblib"))
    print(f"Модель сохранена в {CLASSIFIER_DIR}")


if __name__ == "__main__":
    main()