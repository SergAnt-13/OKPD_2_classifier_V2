# create_prior_weights.py
import pandas as pd
from config.settings import TRAINING_DATA_DIR, PROCESSED_DATA_DIR

# Загружаем золотую выборку и промышленную разметку (только пищевые классы)
gold = pd.read_excel(TRAINING_DATA_DIR / "train.xlsx", dtype=str)
gold = gold[["Код ОКПД2"]].dropna()
gold.columns = ["code"]

nom = pd.read_excel("data/raw/all_nomenclature.xlsx", dtype=str)
nom = nom[['okpd2_code']].dropna()
nom.columns = ["code"]

all_codes = pd.concat([gold, nom], ignore_index=True)
# Оставляем только пищевые классы
all_codes = all_codes[all_codes['code'].str.startswith(('01','02','03','10'), na=False)]

# Подсчитываем частоты
prior = all_codes['code'].value_counts().reset_index()
prior.columns = ['code', 'prior']
prior.to_csv(PROCESSED_DATA_DIR / "class_priors.csv", index=False)
print(f"Сохранено {len(prior)} классов в class_priors.csv")