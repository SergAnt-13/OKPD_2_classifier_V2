# test_load_vat.py
from pathlib import Path
import pandas as pd

REFERENCE_DIR = Path("data/reference")

# Проверяем, какие файлы есть
print("Файлы в reference:")
for f in REFERENCE_DIR.iterdir():
    print(f"  {f.name}")

# Пробуем загрузить ручной CSV
manual_path = REFERENCE_DIR / "vat_exempt_codes.xlsx"
if manual_path.exists():
    print(f"\nЧитаем {manual_path}")
    # Пробуем с разными разделителями
    for sep in [';', ',', None]:
        try:
            df = pd.read_csv(manual_path, sep=sep, dtype=str, engine='python')
            print(f"  sep={sep}: колонки {list(df.columns)}, строк {len(df)}")
            print(df.head(3))
            if 'code' in df.columns:
                codes = df['code'].dropna().str.strip()
                print(f"  Примеры кодов: {codes.head(5).tolist()}")
                break
        except Exception as e:
            print(f"  sep={sep}: ошибка {e}")
else:
    print(f"Файл {manual_path} не найден")