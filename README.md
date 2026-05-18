# OKPD_2_classifier_V2

Гибридная система классификации товарной номенклатуры по кодам ОКПД-2 с управлением риском ошибки.  
Проект решает задачу **extreme multi-class classification** (19 454 класса) в условиях **long‑tail**, **шумных ERP‑данных** и **частичного обучения** (1 500 экспертных примеров).

## 🧠 Ключевая идея (научная новизна)

В задачах экстремальной классификации с длинным хвостом распределения более устойчивым является **гибридный подход**: `retrieval + classifier + decision engine`, а не одиночная модель softmax.  
Мы строим трёхуровневую систему, которая:

*   **Retrieval Layer (bi-encoder + FAISS)** — работает как «память», покрывает все 19k кодов, хорошо обрабатывает редкие и невидимые (unseen) классы.
*   **Classifier Layer (RuBERT)** — быстрый prior на частых (head) классах, обученный на исторических данных предприятия.
*   **Decision Engine** — объединяет сигналы, управляет риском и маршрутизирует товары в зоны **AUTO / REVIEW / MANUAL**.

> Основная цель: не максимизация accuracy, а **минимизация high‑confidence errors** (уверенных ошибок).

## 📊 Ключевые метрики (на экспертной выборке из 1 475 товаров)

### Retrieval (bi-encoder + стеммированный FAISS)
| Метрика   | Значение |
|-----------|----------|
| Recall@1  | 0.27     |
| Recall@5  | 0.55     |
| Recall@10 | 0.65     |
| MRR       | 0.38     |

### Классификатор (RuBERT, 148 классов)
| Метрика       | Значение |
|---------------|----------|
| Accuracy      | 0.67     |
| Macro F1      | 0.31     |
| Weighted F1   | 0.60     |

### End-to-end (Risk‑Aware Decision Engine)
| Режим  | Доля   | Точность |
|--------|--------|----------|
| AUTO   | 25.4%  | 92%      |
| REVIEW | 70.6%  | помощь эксперту |
| MANUAL | 4.0%   | честный отказ |

## 🏗️ Архитектура проекта
OKPD_2_classifier_V2/
├── artifacts/ # обученные модели и индексы
│ ├── bi_encoder/ # дообученный bi-encoder
│ ├── classifier/ # RuBERT-классификатор
│ └── faiss/ # FAISS-индексы (стеммированный и базовый)
├── data/
│ ├── raw/ # исходная номенклатура (all_nomenclature.xlsx)
│ ├── processed/ # очищенные и нормализованные справочники
│ ├── reference/ # ОКПД-2, список льготных кодов, сокращения
│ └── training/ # золотая экспертная выборка (train.xlsx)
├── src/
│ ├── decision/ # движок принятия решений (engine.py)
│ ├── evaluation/ # скрипты оценки и генерации отчётов
│ ├── inference/ # единый пайплайн инференса (pipeline.py)
│ ├── models/ # обёртка BERT-классификатора
│ ├── preprocessing/ # очистка текста, лемматизатор
│ ├── retrieval/ # bi-encoder, FAISS, retriever
│ ├── taxonomy/ # работа с иерархией ОКПД-2
│ └── training/ # скрипты обучения моделей
├── runs/ # история экспериментов
├── cli.py # CLI-интерфейс
└── requirements.txt


## 🚀 Быстрый старт

### 1. Установка зависимостей
```bash
python -m venv .venv
source .venv/bin/activate  # или .venv\Scripts\activate
pip install -r requirements.txt

2. Подготовка данных
Поместите файлы в соответствующие папки:

data/reference/okpd_2.xlsx — полный классификатор ОКПД-2.

data/reference/vat_exempt_codes.xlsx — перечень кодов для НДС 10%.

data/reference/сокращения.xlsx — словарь профессиональных сокращений.

data/raw/all_nomenclature.xlsx — полная номенклатура.

data/training/train.xlsx — экспертная выборка (1 500 записей).

3. Построение индексов
bash
# Стеммированный индекс (основной)
python src/retrieval/build_index_stemmed.py
4. Обучение моделей
bash
# Bi-encoder для retrieval (на полной номенклатуре)
python src/training/train_biencoder_full.py

# Классификатор RuBERT
python src/training/train_bert.py
5. Инференс
bash
# Предсказание для одного товара
python cli.py predict-text "Конфеты Птичье молоко"

# Пакетная обработка файла
python cli.py predict --input data/raw/all_nomenclature.xlsx --output result.xlsx
📈 Оценка качества
bash
# Оценка retrieval
python src/evaluation/evaluate_retrieval.py --stemmed-index

# Оценка классификатора
python src/evaluation/evaluate_classifier.py

# End-to-end оценка с роутингом
python src/evaluation/evaluate_engine.py

# Генерация отчёта по НДС
python src/evaluation/generate_vat_report.py
📋 Примеры работы
Товар	Предсказанный код	Название кода	Уверенность	Режим
к.ш.в карамельной глазури	10.82.22.135	Конфеты шоколадные с грильяжными корпусами...	0.88	AUTO
драже к.ш.	10.82.22.130	Конфеты шоколадные	0.95	AUTO
Шоколад молочный 100г	10.82.22.112	Шоколад молочный в упакованном виде	0.92	AUTO
Молоко сгущенное с сахаром 8.5%	10.51.51.113	Молоко сгущенное с сахаром	0.91	AUTO
Сух.Кириешки ржан.100г	10.72.11.120	Изделия хлебобулочные сухарные	0.93	AUTO
🛡️ Риск-ориентированный роутинг
AUTO — модель уверена, обе компоненты согласны. Код присваивается автоматически.

REVIEW — модель предлагает вариант, но требуется подтверждение эксперта.

MANUAL — модель не может принять решение, товар отправляется на ручную обработку.

📦 Зависимости
Python 3.10+
PyTorch, Transformers, Sentence-Transformers
FAISS, pandas, openpyxl, scikit-learn
pymystem3, striprtf (опционально)
tqdm, matplotlib, seaborn (для анализа)
Полный список см. в requirements.txt.

🧪 Эксперименты и развитие
Стемминг Snowball повысил Recall@10 в 2.3 раза по сравнению с базовой очисткой.

Дообучение bi-encoder на 27 тыс. записей предприятия улучшило MRR с 0.21 до 0.38.

Фокусировка на пищевых кодах (класс 10) устранила галлюцинации типа «масло → нефтепродукты».

Hard negatives, OOD‑детектор, reranker — следующие шаги для повышения точности.