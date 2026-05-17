# src/models/bert_classifier.py
from pathlib import Path
from typing import Dict
import numpy as np


class BERTClassifier:
    """
    Заглушка BERT-классификатора.
    В будущем здесь будет загрузка transformers.AutoModelForSequenceClassification.
    """

    def __init__(self, model_dir: Path = None):
        self.model_dir = model_dir
        self.id2label = {}  # будет загружаться из конфига модели

    def predict(self, text: str) -> Dict:
        """
        Возвращает словарь с полями:
        - code: str
        - confidence: float
        - margin: float
        - entropy: float
        - top1_prob: float
        - top2_prob: float
        """
        # TODO: заменить на реальный инференс BERT
        return {
            "code": "00.00.00.000",
            "confidence": 0.75,
            "margin": 0.15,
            "entropy": 1.2,
            "top1_prob": 0.75,
            "top2_prob": 0.60,
        }