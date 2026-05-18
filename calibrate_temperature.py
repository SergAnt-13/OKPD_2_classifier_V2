# calibrate_temperature.py
"""Подбирает температуру для калибровки классификатора."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import numpy as np
from sklearn.metrics import log_loss
from src.models.bert_classifier import BERTClassifier
from config.settings import TRAINING_DATA_DIR, CLASSIFIER_DIR

# Загружаем валидационную выборку (или можно использовать evaluate_classifier.py)
# Здесь для простоты возьмём несколько примеров, но лучше полноценный val
val_texts = ["конфеты шоколадные", "молоко сгущенное", "хлеб ржаной"]  # примеры
true_labels = [0, 1, 2]  # заглушка, нужно подставить реальные

clf = BERTClassifier()

# Получаем логиты (добавим метод в bert_classifier.py, если его нет)
# Пока вручную: clf.model(**inputs).logits
# ...
# Подбираем температуру на кросс-валидации
# Оптимизируем NLL на val
# Сохраняем температуру в файл