# demo_vat_random.py
import pandas as pd
import random
from config.settings import PROCESSED_DATA_DIR, REFERENCE_DIR

pool_path = PROCESSED_DATA_DIR / "auto_pool_vat.csv"
if not pool_path.exists():
    print("Сначала запустите build_auto_pool_vat.py")
    exit()

pool = pd.read_csv(pool_path)
sample = pool.sample(min(5, len(pool)), random_state=random.randint(0, 1000))

ref = pd.read_csv(PROCESSED_DATA_DIR / "okpd_2_normalized.csv", dtype=str)
code_to_name = dict(zip(ref["code"], ref["name"]))

print("Случайные примеры с определением НДС:\n")
print("Товар".ljust(35), "Предсказанный код".ljust(15), "Название кода".ljust(40), "Уверенность".ljust(6), "НДС")
print("-" * 115)
for _, row in sample.iterrows():
    code_name = code_to_name.get(row["predicted_code"], "")[:40]
    nds = "10%" if row["predicted_exempt"] else "20%"
    print(f"{row['text'][:35]:35s} {row['predicted_code']:15s} {code_name:40s} {row['confidence']:.2f}   {nds}")