# test_pipeline_quick.py
"""Быстрый тест пайплайна на 1000 записей."""
import pandas as pd
from pathlib import Path
from src.inference.pipeline import InferencePipeline
from config.settings import RAW_DATA_DIR

# Загружаем номенклатуру
df = pd.read_excel(RAW_DATA_DIR / "all_nomenclature.xlsx", dtype=str)
df.columns = ['nomenclature', 'nomenclature_name', 'nds_rate', 'okpd2_code']

# Берём первые 1000 записей
sample = df.head(1000)

pipeline = InferencePipeline()
results = []
for _, row in sample.iterrows():
    text = row['nomenclature']
    if pd.isna(text) or not str(text).strip():
        continue
    pred = pipeline.predict_single(str(text))
    results.append(pred)

res_df = pd.DataFrame(results)
print("Распределение роутинга:")
print(res_df['routing'].value_counts())
print(f"\nСредняя уверенность: {res_df['confidence'].mean():.4f}")
print(f"Доля AUTO: {(res_df['routing'] == 'AUTO').mean():.4f}")