# src/evaluation/aggregate_analysis.py
"""
Агрегация и визуализация результатов VAT-отчёта.
Строит: матрицу ошибок, распределение длин текстов, уверенность по классам,
sunburst-диаграмму, облака слов, распределение классов.
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
from sklearn.metrics import confusion_matrix
from datetime import datetime

from config.settings import PROCESSED_DATA_DIR


# ---------- Вспомогательные функции ----------

def filter_food_codes(df: pd.DataFrame) -> pd.DataFrame:
    """Оставляет только продуктовые коды (начинаются с 01, 02, 03, 10)."""
    food_prefixes = ('01', '02', '03', '10')
    mask = df['current_code'].str.startswith(food_prefixes, na=False)
    return df[mask].copy()


def shorten_label(label: str, max_len: int = 50) -> str:
    """Обрезает название для графика."""
    if len(label) > max_len:
        return label[:max_len-3] + '...'
    return label


# ---------- Функции построения графиков ----------

def plot_confusion_matrix(df: pd.DataFrame, output_dir: Path):
    valid = df.dropna(subset=['current_code', 'predicted_code'])
    if len(valid) < 2:
        print("⚠️ Недостаточно данных для матрицы ошибок")
        return

    top_classes = valid['current_code'].value_counts().head(20).index
    valid = valid[valid['current_code'].isin(top_classes) & valid['predicted_code'].isin(top_classes)]

    if len(valid) == 0:
        print("⚠️ Нет пересечений между текущими и предсказанными кодами для топ-20")
        return

    cm = confusion_matrix(valid['current_code'], valid['predicted_code'], labels=top_classes, normalize='true')
    plt.figure(figsize=(16, 14))
    sns.heatmap(cm, xticklabels=top_classes, yticklabels=top_classes, cmap='Blues', annot=True, fmt='.2f')
    plt.title('Нормализованная матрица ошибок (топ-20 классов)')
    plt.xlabel('Предсказанный код')
    plt.ylabel('Истинный код')
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_dir / 'confusion_matrix.png', dpi=150)
    plt.close()
    print("✅ Матрица ошибок сохранена")


def plot_text_length_distribution(df: pd.DataFrame, output_dir: Path):
    """Box plot длин текстов для топ-10 классов."""
    df = df.copy()
    df['text_len'] = df['nomenclature'].str.len()
    top10 = df['current_code'].value_counts().head(10).index
    subset = df[df['current_code'].isin(top10)]

    plt.figure(figsize=(14, 8))
    sns.boxplot(data=subset, x='current_code', y='text_len')
    plt.title('Распределение длин текстов по топ-10 классам')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_dir / 'text_length_boxplot.png', dpi=150)
    plt.close()
    print("✅ Box plot длин текстов сохранён")


def plot_confidence_by_class(df: pd.DataFrame, output_dir: Path):
    """Средняя уверенность по топ-20 классам."""
    top20 = df['current_code'].value_counts().head(20).index
    subset = df[df['current_code'].isin(top20)]
    conf_avg = subset.groupby('current_code')['confidence'].mean().sort_values()

    plt.figure(figsize=(12, 8))
    conf_avg.plot(kind='barh')
    plt.title('Средняя уверенность модели по топ-20 классам')
    plt.xlabel('Средняя уверенность')
    plt.tight_layout()
    plt.savefig(output_dir / 'confidence_by_class.png', dpi=150)
    plt.close()
    print("✅ График уверенности по классам сохранён")


def plot_sunburst(df: pd.DataFrame, output_dir: Path):
    """Sunburst-диаграмма иерархии кодов (упрощённая)."""
    # Берём первые два уровня кода
    df = df.copy()
    df['level1'] = df['predicted_code'].str.extract(r'^(\d{2})')
    df['level2'] = df['predicted_code'].str.extract(r'^(\d{2}\.\d{2})')

    # Группируем по level1 -> level2
    hierarchy = df.groupby(['level1', 'level2']).size().reset_index(name='count')

    # Строим sunburst через treemap (упрощённо, без plotly)
    fig, ax = plt.subplots(figsize=(14, 14))
    hierarchy_pivot = hierarchy.pivot(index='level2', columns='level1', values='count').fillna(0)

    # Берём топ-5 level1 и топ-10 level2
    top_level1 = df['level1'].value_counts().head(5).index
    top_level2 = df['level2'].value_counts().head(20).index

    filtered = hierarchy[hierarchy['level1'].isin(top_level1) & hierarchy['level2'].isin(top_level2)]

    colors = plt.cm.Set3(np.linspace(0, 1, len(top_level1)))
    bottom = np.zeros(len(top_level1))
    for i, l1 in enumerate(top_level1):
        l2_data = filtered[filtered['level1'] == l1].set_index('level2')['count']
        l2_data = l2_data.reindex(top_level2, fill_value=0)
        ax.bar(top_level2, l2_data, bottom=bottom, color=colors[i], label=f'Класс {l1}')
        bottom += l2_data.values

    ax.set_title('Иерархия предсказанных кодов (sunburst-замена)')
    ax.set_xticklabels(top_level2, rotation=90)
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir / 'hierarchy_sunburst.png', dpi=150)
    plt.close()
    print("✅ Sunburst-диаграмма сохранена")


def plot_wordclouds(df: pd.DataFrame, output_dir: Path):
    """Облака слов для топ-5 классов."""
    top5 = df['current_code'].value_counts().head(5).index
    for code in top5:
        texts = df[df['current_code'] == code]['nomenclature'].dropna().str.cat(sep=' ')
        if not texts.strip():
            continue
        wc = WordCloud(width=800, height=400, background_color='white', max_words=100)
        wc.generate(texts)
        plt.figure(figsize=(10, 5))
        plt.imshow(wc, interpolation='bilinear')
        plt.axis('off')
        plt.title(f'Облако слов для кода {code}')
        plt.tight_layout()
        plt.savefig(output_dir / f'wordcloud_{code.replace(".", "_")}.png', dpi=150)
        plt.close()
    print("✅ Облака слов сохранены")


def plot_class_distribution(df: pd.DataFrame, output_dir: Path, top_n: int = 30):
    """Распределение классов с названиями (два графика: код и название)."""
    code_counts = df['current_code'].value_counts().head(top_n)

    # График по кодам
    plt.figure(figsize=(14, 8))
    code_counts.plot(kind='bar')
    plt.title(f'Топ-{top_n} кодов по частоте')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(output_dir / 'class_distribution_codes.png', dpi=150)
    plt.close()

    # График по названиям (если есть колонка current_code_name)
    if 'current_code_name' in df.columns:
        name_counts = df.groupby('current_code_name').size().sort_values(ascending=False).head(top_n)
        plt.figure(figsize=(14, 8))
        name_counts.plot(kind='bar')
        plt.title(f'Топ-{top_n} классов по названиям')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(output_dir / 'class_distribution_names.png', dpi=150)
        plt.close()

    print("✅ Распределение классов сохранено")


# ---------- Главная функция ----------

def main():
    # Создаём папку для результатов
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = Path("runs") / f"vat_analysis_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Загружаем отчёт
    report_path = PROCESSED_DATA_DIR / "vat_report.xlsx"
    if not report_path.exists():
        print(f"❌ Файл {report_path} не найден. Сначала запустите generate_vat_report.py")
        return

    print("Загрузка отчёта...")
    df = pd.read_excel(report_path, dtype=str)
    # Преобразуем confidence в float
    df['confidence'] = pd.to_numeric(df['confidence'], errors='coerce')
    print(f"✅ Загружено {len(df)} записей")

    # Статистика по роутингу и уверенности
    print("\n=== СТАТИСТИКА ===")
    if 'routing' in df.columns:
        print("Распределение роутинга:")
        print(df['routing'].value_counts().to_string())
    if 'confidence' in df.columns:
        print(f"\nСредняя уверенность: {df['confidence'].mean():.4f}")
        print(f"Медианная уверенность: {df['confidence'].median():.4f}")
        print(f"Доля уверенных (conf > 0.7): {(df['confidence'] > 0.7).mean():.4f}")

    if 'current_exempt' in df.columns and 'predicted_exempt' in df.columns:
        print(f"\nТекущих льготных: {df['current_exempt'].value_counts().get('True', 0)}")
        print(f"Предсказанных льготных: {df['predicted_exempt'].value_counts().get('True', 0)}")

    # Строим графики
    print("\nПостроение графиков...")
    plot_class_distribution(df, output_dir)
    plot_confusion_matrix(df, output_dir)
    plot_text_length_distribution(df, output_dir)
    plot_confidence_by_class(df, output_dir)
    plot_sunburst(df, output_dir)
    plot_wordclouds(df, output_dir)

    print(f"\n✅ Все графики сохранены в {output_dir}")

    # Сохраняем текстовую статистику
    stats_path = output_dir / "statistics.txt"
    with open(stats_path, 'w', encoding='utf-8') as f:
        f.write(f"Всего записей: {len(df)}\n")
        if 'routing' in df.columns:
            f.write(f"Распределение роутинга:\n{df['routing'].value_counts().to_string()}\n")
        if 'confidence' in df.columns:
            f.write(f"Средняя уверенность: {df['confidence'].mean():.4f}\n")
            f.write(f"Медианная уверенность: {df['confidence'].median():.4f}\n")
    print(f"Статистика сохранена в {stats_path}")


if __name__ == "__main__":
    main()