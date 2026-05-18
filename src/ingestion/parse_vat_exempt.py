# src/ingestion/parse_vat_exempt.py
"""
Извлекает коды ОКПД‑2 и названия из RTF постановления 908.
Использует мультистратегический поиск, включая анализ таблиц и контекста.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import re
from striprtf.striprtf import rtf_to_text

from config.settings import REFERENCE_DIR


def normalize_code(code: str) -> str:
    """Удаляем завершающие нулевые группы, например '01.45.21.000' -> '01.45.21'."""
    parts = code.split('.')
    while parts and parts[-1] in ('0', '00', '000'):
        parts.pop()
    return '.'.join(parts)


def main():
    rtf_path = REFERENCE_DIR / "Постановление Правительства РФ от 31.12.2004 N 908 Об утверждении перечней кодов видов.rtf"
    if not rtf_path.exists():
        raise FileNotFoundError(f"Файл {rtf_path} не найден")

    with open(rtf_path, 'r', encoding='utf-8') as f:
        rtf_content = f.read()

    # Конвертируем RTF в текст
    text = rtf_to_text(rtf_content)

    # ----- Стратегия 1: строки, начинающиеся с кода (как раньше, но с табуляцией) -----
    code_pattern1 = re.compile(r'^(\d{2}(?:\.\d{1,2}){0,5})\s{2,}(.+)')
    found_codes = set()

    for line in text.splitlines():
        match = code_pattern1.match(line.strip())
        if match:
            code = normalize_code(match.group(1))
            found_codes.add(code)

    # ----- Стратегия 2: поиск кодов в таблицах RTF (из VATProcessor) -----
    # Ищем последовательности \row ... \row, содержащие код и упоминание льготы
    table_pattern = re.compile(
        r'\\row.*?(\d{2}\.\d{2}(?:\.\d{2}){0,2}(?:\.\d{3})?).*?(?:льготн|10%|десят).*?\\row',
        re.IGNORECASE | re.DOTALL
    )
    table_matches = table_pattern.findall(rtf_content)
    for code in table_matches:
        found_codes.add(normalize_code(code))

    # ----- Стратегия 3: поиск кодов рядом со словами-маркерами -----
    marker_pattern = re.compile(
        r'(\d{2}\.\d{2}(?:\.\d{2}){0,2}(?:\.\d{3})?)\s*(?:-|–|—)\s*[\w\s]{0,30}(?:льготн|10%|десят)',
        re.IGNORECASE
    )
    marker_matches = marker_pattern.findall(text)
    for code in marker_matches:
        found_codes.add(normalize_code(code))

    # ----- Убираем нули и сохраняем -----
    codes_list = sorted(list(found_codes))
    df = pd.DataFrame({'code': codes_list, 'name': ''})  # названия пока пустые

    out_path = REFERENCE_DIR / "vat_exempt_codes.csv"
    df.to_csv(out_path, index=False)

    print(f"Найдено {len(df)} льготных кодов, сохранено в {out_path}")
    print("Первые 10 кодов:")
    print(df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()