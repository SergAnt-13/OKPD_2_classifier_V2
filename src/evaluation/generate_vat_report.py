# src/evaluation/generate_vat_report.py
"""
Генерирует Excel-отчёт для анализа НДС.
Добавляет колонки к исходной номенклатуре. Ничего не меняет.
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
from tqdm import tqdm

from config.settings import REFERENCE_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR
from src.inference.pipeline import InferencePipeline


def is_exempt(code: str, exempt_set: set) -> bool:
    if not code or not isinstance(code, str):
        return False
    if code in exempt_set:
        return True
    parts = code.split('.')
    for i in range(len(parts) - 1, 0, -1):
        prefix = '.'.join(parts[:i])
        if prefix in exempt_set:
            return True
    return False


def find_closest_exempt(predicted_code: str, top_candidates: list, exempt_set: set, code_to_name: dict):
    """
    Ищет ближайший льготный код среди топ-3 кандидатов.
    Возвращает (code, name) или (None, None).
    """
    for cand in top_candidates[:3]:
        if is_exempt(cand['code'], exempt_set):
            return cand['code'], code_to_name.get(cand['code'], '')
    return None, None


def load_exempt_set(ref_dir: Path) -> set:
    xlsx_path = ref_dir / "vat_exempt_codes.xlsx"
    if not xlsx_path.exists():
        print("⚠️ Файл vat_exempt_codes.xlsx не найден")
        return set()
    try:
        print("Загрузка льготных кодов...")
        df = pd.read_excel(xlsx_path, dtype=str)
        if 'code' in df.columns:
            codes = df['code'].dropna().str.strip()
            codes = codes.str.replace(r'\.0{1,3}$', '', regex=True)
            print(f"✅ Загружено {len(set(codes))} льготных кодов")
            return set(codes)
    except Exception as e:
        print(f"❌ Ошибка чтения Excel: {e}")
    return set()


def main():
    # 1. Льготные коды
    exempt_set = load_exempt_set(REFERENCE_DIR)

    # 2. Справочник ОКПД-2 (названия)
    print("Загрузка справочника ОКПД-2...")
    ref_path = PROCESSED_DATA_DIR / "okpd_2_normalized.csv"
    code_to_name = {}
    if ref_path.exists():
        okpd_ref = pd.read_csv(ref_path, dtype=str)
        if 'code' in okpd_ref.columns and 'name' in okpd_ref.columns:
            code_to_name = dict(zip(okpd_ref['code'].str.strip(), okpd_ref['name'].str.strip()))
            print(f"✅ Загружено {len(code_to_name)} названий")
    else:
        print("⚠️ Справочник не найден")

    # 3. Полная номенклатура
    nom_path = RAW_DATA_DIR / "all_nomenclature.xlsx"
    print("Загрузка номенклатуры...")
    df = pd.read_excel(nom_path, dtype=str)
    # Ожидаем колонки: nomenclature, nomenclature_name, nds_rate, okpd2_code
    df.columns = ['nomenclature', 'nomenclature_name', 'nds_rate', 'okpd2_code']
    print(f"✅ Загружено {len(df)} записей")

    # 4. Инициализация пайплайна
    print("Инициализация пайплайна...")
    pipeline = InferencePipeline()
    print("✅ Пайплайн готов")

    results = []
    print(f"Обработка {len(df)} записей...")
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Предсказания"):
        text = str(row['nomenclature']) if pd.notna(row['nomenclature']) else ''
        current_code = str(row['okpd2_code']) if pd.notna(row['okpd2_code']) else ''
        current_rate = str(row['nds_rate']) if pd.notna(row['nds_rate']) else '22%'

        if not text:
            continue

        # Название текущего кода из справочника (если есть)
        current_code_name = code_to_name.get(current_code, '') if current_code else ''

        # Предсказание
        pred = pipeline.predict_single(text)
        pred_code = pred['final_prediction']
        top_candidates = pred['top_candidates'][:3]

        # Льготность
        current_exempt = is_exempt(current_code, exempt_set)
        pred_exempt = is_exempt(pred_code, exempt_set)
        top3_any_exempt = any(is_exempt(c['code'], exempt_set) for c in top_candidates)

        # Ближайший льготный код
        closest_code, closest_name = find_closest_exempt(pred_code, top_candidates, exempt_set, code_to_name)

        # Формируем топ-3 строку
        top3_parts = []
        for c in top_candidates:
            c_name = code_to_name.get(c['code'], '')[:40]
            top3_parts.append(f"{c['code']} ({c_name}), score={c['score']:.3f}")
        top3_str = ' | '.join(top3_parts)

        results.append({
            'nomenclature': text,
            'nomenclature_name': row['nomenclature_name'],
            'current_code': current_code,
            'current_code_name': current_code_name,
            'current_nds': current_rate,
            'current_exempt': current_exempt,
            'predicted_code': pred_code,
            'predicted_name': code_to_name.get(pred_code, '')[:60],
            'predicted_exempt': pred_exempt,
            'closest_exempt_code': closest_code,
            'closest_exempt_name': closest_name,
            'top3_candidates': top3_str,
            'top3_any_exempt': top3_any_exempt,
            'confidence': pred['confidence'],
            'routing': pred['routing']
        })

    out_df = pd.DataFrame(results)

    # Сохраняем ОДИН лист со всеми данными
    output_path = PROCESSED_DATA_DIR / "vat_report.xlsx"
    print("Сохранение отчёта...")
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        out_df.to_excel(writer, sheet_name='Общий анализ', index=False)
    print(f"✅ Отчёт сохранён в {output_path}")


if __name__ == "__main__":
    main()