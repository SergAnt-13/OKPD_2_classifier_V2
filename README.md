АРХИТЕКТУРА ПРОЕКТА
OKPD_2_classifier_V2/
│
├── config/
│   └── settings.py                # пути и автосоздание папок
│
├── data/
│   ├── raw/                       # исходные файлы
│   ├── processed/                 # после preprocessing
│   ├── training/                  # train/val/test сплиты
│   └── reference/                 # okpd2_full.xlsx
│
├── artifacts/
│   ├── classifier/                # BERT (веса, конфиг)
│   ├── bi_encoder/                # bi-encoder для retrieval
│   ├── faiss/                     # FAISS индекс
│   └── metrics/                   # сводные метрики (опционально)
│
├── runs/                          # история экспериментов
│
├── src/
│   ├── ingestion/
│   │   └── loader.py              # загрузка сырых данных
│   │
│   ├── preprocessing/
│   │   └── cleaner.py             # бывший text_views.py: classifier_view, retrieval_view
│   │
│   ├── taxonomy/
│   │   └── okpd_tree.py           # работа с иерархией ОКПД-2
│   │
│   ├── retrieval/
│   │   ├── embedder.py            # bi-encoder, эмбеддинги
│   │   ├── faiss_index.py         # построение/загрузка FAISS
│   │   └── retriever.py           # поиск top-K
│   │
│   ├── models/
│   │   └── bert_classifier.py     # класс BERT-классификатора
│   │
│   ├── decision/
│   │   └── engine.py              # единственный DecisionEngine (вобрал confidence/engine_v2 и risk_engine)
│   │
│   ├── inference/
│   │   └── pipeline.py            # единый пайплайн инференса
│   │
│   ├── evaluation/
│   │   ├── analysis.py            # EDA (перенесённый eda_report)
│   │   ├── metrics.py             # все метрики
│   │   └── reporter.py            # генерация отчётов/графиков
│   │
│   ├── training/
│   │   ├── train_bert.py          # обучение классификатора
│   │   └── train_biencoder.py     # обучение bi-encoder
│   │
│   ├── common/
│   │   └── schemas.py             # dataclass'ы (ClassifierOutput, DecisionResult и т.п.)
│   │
│   └── utils/
│       └── helpers.py             # мелкие утилиты (softmax, загрузка csv и пр.)
│
├── cli.py                         # точка входа: train-classifier, predict, evaluate, eda
├── requirements.txt
└── README.md



🧠 Точки входа

TRAINING
python cli.py train-biencoder
python cli.py train-classifier

INDEXING
python cli.py build-faiss

PREDICTION
python cli.py predict --input file.xlsx

SAFE PSEUDO LABELS
python cli.py generate-pseudo-labels

EVALUATION
python cli.py evaluate

SINGLE SAMPLE
python cli.py predict-text "Йогурт клубничный 5%"

🧠 PREDICTION FLOW

STEP 1
preprocess.

STEP 2
retrieval:
top_k candidates

STEP 3
classifier validation.

STEP 4
hierarchy scoring.

STEP 5
risk scoring.

STEP 6
routing:

🟢 AUTO
🟡 REVIEW
🔴 MANUAL
🧠 OUTPUT STRUCTURE

Это очень важно для аналитика.

Результат НЕ должен быть:
one code
Должен быть:
{
  "text": "...",
  "top_candidates": [...],
  "final_prediction": "...",
  "retrieval_score": 0.91,
  "margin": 0.42,
  "risk_level": "medium",
  "requires_review": true
}