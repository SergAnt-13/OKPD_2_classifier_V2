# src/evaluation/analyze_existing_codes.py
"""
Расширенный анализ уже проставленных кодов ОКПД-2 в номенклатуре.
Строит диаграммы, выявляет несоответствия НДС, даёт текстовую интерпретацию.
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from wordcloud import WordCloud
from datetime import datetime

from config.settings import RAW_DATA_DIR, PROCESSED_DATA_DIR, REFERENCE_DIR

# ---------- Вспомогательные функции ----------

def is_exempt(code: str, exempt_set: set) -> bool:
    """Проверяет, является ли код льготным (с учётом родительских групп)."""
    if not code or not isinstance(code, str):
        return False
    code = code.strip()
    if code in exempt_set:
        return True
    parts = code.split('.')
    for i in range(len(parts) - 1, 0, -1):
        prefix = '.'.join(parts[:i])
        if prefix in exempt_set:
            return True
    return False


def load_exempt_set(ref_dir: Path) -> set:
    xlsx_path = ref_dir / "vat_exempt_codes.xlsx"
    if not xlsx_path.exists():
        return set()
    try:
        df = pd.read_excel(xlsx_path, dtype=str)
        if 'code' in df.columns:
            codes = df['code'].dropna().str.strip()
            codes = codes.str.replace(r'\.0{1,3}$', '', regex=True)
            return set(codes)
    except Exception:
        pass
    return set()


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


# ---------- Функции построения графиков ----------

def plot_class_distribution(df: pd.DataFrame, output_dir: Path):
    """Круговая диаграмма распределения по классам (первые 2 цифры)."""
    class_counts = df['class'].value_counts()
    plt.figure(figsize=(12, 10))
    wedges, texts, autotexts = plt.pie(
        class_counts, labels=class_counts.index, autopct='%1.1f%%',
        startangle=140, pctdistance=0.85
    )
    for autotext in autotexts:
        autotext.set_size(9)
    plt.title('Распределение кодов по классам (первые 2 цифры)', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / 'class_distribution_pie.png', dpi=150)
    plt.close()
    total = class_counts.sum()
    top3 = class_counts.head(3)
    print("=== РАСПРЕДЕЛЕНИЕ ПО КЛАССАМ ===")
    print(f"Всего записей с кодом: {total}")
    print("Топ-3 класса:")
    for cls, cnt in top3.items():
        print(f"  Класс {cls}: {cnt} записей ({cnt/total:.1%})")


def plot_top_codes(df: pd.DataFrame, output_dir: Path, top_n: int = 20):
    """Горизонтальный бар-чарт топ-20 кодов."""
    code_counts = df['okpd2_code'].value_counts().head(top_n)
    plt.figure(figsize=(12, 10))
    ax = code_counts.plot(kind='barh')
    plt.title(f'Топ-{top_n} кодов ОКПД-2 в размеченных данных', fontsize=14)
    plt.xlabel('Количество')
    plt.tight_layout()
    plt.savefig(output_dir / 'top_codes_bar.png', dpi=150)
    plt.close()
    print("\n=== ТОП-10 КОДОВ ===")
    for i, (code, cnt) in enumerate(code_counts.head(10).items(), 1):
        print(f"  {i}. {code}: {cnt} раз")


def plot_nds_distribution(df: pd.DataFrame, output_dir: Path):
    """Круговая диаграмма ставок НДС."""
    nds_counts = df['nds_rate'].value_counts()
    plt.figure(figsize=(8, 8))
    plt.pie(nds_counts, labels=nds_counts.index, autopct='%1.1f%%', startangle=90)
    plt.title('Распределение ставок НДС')
    plt.tight_layout()
    plt.savefig(output_dir / 'nds_distribution.png', dpi=150)
    plt.close()
    print("\n=== РАСПРЕДЕЛЕНИЕ СТАВОК НДС ===")
    for rate, cnt in nds_counts.items():
        print(f"  {rate}: {cnt} записей")


def plot_vat_discrepancy(df: pd.DataFrame, output_dir: Path):
    """Записи, где код льготный, но ставка не 10% и не пустая."""
    discrepancy = df[
        df['is_exempt'] &
        (df['nds_rate'] != '10%') &
        df['nds_rate'].notna() &
        (df['nds_rate'] != '')
    ]
    if not discrepancy.empty:
        print("\n=== ЗАПИСИ С НЕСООТВЕТСТВИЕМ НДС ===")
        print(f"Всего таких записей: {len(discrepancy)}")
        print("Примеры (первые 10):")
        for _, row in discrepancy.head(10).iterrows():
            print(f"  {row['nomenclature'][:60]}: код {row['okpd2_code']}, ставка {row['nds_rate']}")
        discrepancy.to_excel(output_dir / 'vat_discrepancy.xlsx', index=False)
        print(f"Полный список сохранён в {output_dir / 'vat_discrepancy.xlsx'}")
    else:
        print("\n=== НЕСООТВЕТСТВИЙ НДС НЕ НАЙДЕНО ===")


def plot_exempt_by_class(df: pd.DataFrame, output_dir: Path):
    """Тепловая карта: класс vs льготность."""
    cross = pd.crosstab(df['class'], df['is_exempt'])
    if cross.empty:
        return
    plt.figure(figsize=(10, 8))
    sns.heatmap(cross, annot=True, fmt='d', cmap='YlOrRd')
    plt.title('Льготные vs нельготные коды по классам')
    plt.ylabel('Класс (первые 2 цифры)')
    plt.xlabel('Льготный')
    plt.tight_layout()
    plt.savefig(output_dir / 'exempt_by_class_heatmap.png', dpi=150)
    plt.close()
    print("\n=== СВОДКА ПО КЛАССАМ И ЛЬГОТАМ ===")
    print(cross.to_string())


def plot_wordclouds(df: pd.DataFrame, output_dir: Path):
    """Облака слов для топ-5 классов."""
    top5 = df['class'].value_counts().head(5).index
    for cls in top5:
        subset = df[df['class'] == cls]
        texts = subset['nomenclature'].dropna().astype(str)
        if texts.empty:
            continue
        text_str = texts.str.cat(sep=' ')
        # Убираем всё, кроме букв и пробелов (для облака)
        text_str = ' '.join([w for w in text_str.split() if any(c.isalpha() for c in w)])
        if not text_str.strip():
            print(f"⚠️ Для класса {cls} не осталось слов после очистки. Пропускаем.")
            continue
        wc = WordCloud(width=800, height=400, background_color='white', max_words=100)
        try:
            wc.generate(text_str)
        except ValueError:
            print(f"⚠️ Облако слов для класса {cls} пустое, пропускаем.")
            continue
        plt.figure(figsize=(10, 5))
        plt.imshow(wc, interpolation='bilinear')
        plt.axis('off')
        plt.title(f'Облако слов для класса {cls}')
        plt.tight_layout()
        plt.savefig(output_dir / f'wordcloud_class_{cls}.png', dpi=150)
        plt.close()
    print("✅ Облака слов для топ-5 классов сохранены")


def plot_text_length(df: pd.DataFrame, output_dir: Path):
    """Гистограмма длин названий товаров."""
    df['text_len'] = df['nomenclature'].str.len()
    plt.figure(figsize=(10, 6))
    sns.histplot(df['text_len'], bins=50, kde=True)
    plt.title('Распределение длин названий товаров')
    plt.xlabel('Длина текста (символов)')
    plt.tight_layout()
    plt.savefig(output_dir / 'text_length_hist.png', dpi=150)
    plt.close()
    print(f"\nДлина названий: средняя={df['text_len'].mean():.1f}, медиана={df['text_len'].median():.0f}, макс={df['text_len'].max()}")


# ---------- Главная функция ----------

def main():
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = Path("runs") / f"existing_codes_analysis_{timestamp}"
    ensure_dir(output_dir)

    print("Загрузка данных...")
    nom_path = RAW_DATA_DIR / "all_nomenclature.xlsx"
    df = pd.read_excel(nom_path, dtype=str)
    df.columns = ['nomenclature', 'nomenclature_name', 'nds_rate', 'okpd2_code']

    # Оставляем только с кодом
    df = df[df['okpd2_code'].notna() & (df['okpd2_code'] != '')].copy()
    print(f"Записей с кодом: {len(df)}")

    # Валидность кодов
    ref_path = PROCESSED_DATA_DIR / "okpd_2_normalized.csv"
    valid_codes = set()
    if ref_path.exists():
        okpd_ref = pd.read_csv(ref_path, dtype=str)
        if 'code' in okpd_ref.columns:
            valid_codes = set(okpd_ref['code'].str.strip())

    df['code_valid'] = df['okpd2_code'].apply(lambda c: c.strip() in valid_codes)
    print(f"Валидных кодов: {df['code_valid'].sum()} ({df['code_valid'].mean():.1%})")

    # Льготные коды
    exempt_set = load_exempt_set(REFERENCE_DIR)
    df['is_exempt'] = df['okpd2_code'].apply(lambda c: is_exempt(c, exempt_set))
    print(f"Льготных кодов: {df['is_exempt'].sum()} ({df['is_exempt'].mean():.1%})")

    # Класс (первые 2 цифры)
    df['class'] = df['okpd2_code'].str.extract(r'^(\d{2})')

    # Сохраняем полный анализ
    df.to_csv(PROCESSED_DATA_DIR / "existing_codes_analysis.csv", index=False)

    # Строим графики с интерпретацией
    print("\n=== АНАЛИЗ ===")
    plot_class_distribution(df, output_dir)
    plot_top_codes(df, output_dir)
    plot_nds_distribution(df, output_dir)
    plot_vat_discrepancy(df, output_dir)
    plot_exempt_by_class(df, output_dir)
    plot_text_length(df, output_dir)
    plot_wordclouds(df, output_dir)

    print(f"\n✅ Все результаты сохранены в {output_dir}")


if __name__ == "__main__":
    main()