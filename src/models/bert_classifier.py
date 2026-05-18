# src/models/bert_classifier.py
import numpy as np
import torch
from pathlib import Path
from typing import Dict, Optional
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import joblib
import pandas as pd

from config.settings import CLASSIFIER_DIR, REFERENCE_DIR, PROCESSED_DATA_DIR

class BERTClassifier:
    def __init__(self, model_dir: Optional[Path] = None):
        model_dir = model_dir or CLASSIFIER_DIR
        if model_dir.exists():
            self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
            self.model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
            self.label_encoder = joblib.load(str(model_dir / "label_encoder.joblib"))
            self.model.eval()
            self._loaded = True
        else:
            self._loaded = False

        # Загружаем справочник для валидации кодов
        self.valid_codes = set()
        self.parent_map = {}
        ref_path = PROCESSED_DATA_DIR / "okpd_2_normalized.csv"
        if ref_path.exists():
            ref_df = pd.read_csv(ref_path, dtype=str)
            if 'code' in ref_df.columns:
                self.valid_codes = set(ref_df['code'].str.strip())
            if 'parent_code' in ref_df.columns:
                for _, row in ref_df.iterrows():
                    code = str(row['code']).strip()
                    parent = str(row['parent_code']).strip()
                    if parent and parent != 'nan':
                        self.parent_map[code] = parent

    def predict(self, text: str) -> Dict:
        if not self._loaded:
            return {
                "code": "00.00.00.000",
                "confidence": 0.5,
                "margin": 0.0,
                "entropy": 1.0,
                "top1_prob": 0.5,
                "top2_prob": 0.5,
            }

        inputs = self.tokenizer(
            text, return_tensors="pt", padding=True, truncation=True, max_length=128
        )
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1).numpy()[0]

        top_idx = np.argmax(probs)
        top1_prob = probs[top_idx]
        sorted_probs = np.sort(probs)[::-1]
        margin = sorted_probs[0] - sorted_probs[1] if len(sorted_probs) > 1 else 0.0
        entropy = -np.sum(probs * np.log(probs + 1e-10))

        code = self.label_encoder.inverse_transform([top_idx])[0]

        # Валидация кода по справочнику
        if code not in self.valid_codes and self.valid_codes:
            # Пытаемся найти ближайшего валидного родителя
            current = code
            while current in self.parent_map:
                current = self.parent_map[current]
                if current in self.valid_codes:
                    code = current
                    top1_prob = max(0.3, top1_prob * 0.9)  # немного снижаем уверенность
                    break
            else:
                # Если родитель не найден, возвращаем "unknown"
                code = "00.00.00.000"
                top1_prob = 0.0

        return {
            "code": str(code),
            "confidence": float(top1_prob),
            "margin": float(margin),
            "entropy": float(entropy),
            "top1_prob": float(top1_prob),
            "top2_prob": float(sorted_probs[1]) if len(sorted_probs) > 1 else 0.0,
        }