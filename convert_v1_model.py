# convert_v1_model.py
"""Конвертирует маппинг меток из V1 (label_mappings.json) в V2 (label_encoder.joblib)."""
import json
import joblib
from sklearn.preprocessing import LabelEncoder
from pathlib import Path

# Путь к V1-модели (куда скопировали)
v1_model_dir = Path("artifacts/classifier")
mappings_path = v1_model_dir / "label_mappings.json"

with open(mappings_path, encoding='utf-8') as f:
    mappings = json.load(f)

# id_to_label: ключи — индексы, значения — коды
id_to_label = {int(k): v for k, v in mappings['id_to_label'].items()}
classes = [id_to_label[i] for i in sorted(id_to_label.keys())]

le = LabelEncoder()
le.fit(classes)
joblib.dump(le, v1_model_dir / "label_encoder.joblib")
print(f"LabelEncoder сохранён, классов: {len(classes)}")