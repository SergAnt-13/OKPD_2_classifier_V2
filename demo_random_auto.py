# demo_random_auto.py
"""Демонстрация: 5 случайных AUTO-примеров из проверенного пула."""
import pandas as pd
import random
from config.settings import PROCESSED_DATA_DIR, REFERENCE_DIR

pool_path = PROCESSED_DATA_DIR / "auto_pool_gold.csv"
if not pool_path.exists():
    print("Сначала запустите build_auto_pool.py")
    exit()

pool = pd.read_csv(pool_path)
sample = pool.sample(min(5, len(pool)), random_state=random.randint(0, 1000))

# Загрузим справочник для названий кодов
ref = pd.read_csv(PROCESSED_DATA_DIR / "okpd_2_normalized.csv", dtype=str)
code_to_name = dict(zip(ref["code"], ref["name"]))

print("Случайные AUTO-примеры:\n")
print("Товар".ljust(35), "Предсказанный код".ljust(15), "Название кода".ljust(45), "Уверенность")
print("-" * 115)
for _, row in sample.iterrows():
    code_name = code_to_name.get(row["predicted_code"], "")[:45]
    print(f"{row['text'][:35]:35s} {row['predicted_code']:15s} {code_name:45s} {row['confidence']:.2f}")