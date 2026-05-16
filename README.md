АРХИТЕКТУРА ПРОЕКТА v2

📁 1. PROJECT ROOT
project/
📁 2. DATA LAYER
data/
├── raw/  --Номенклатурная выгрузка
├── processed/ --результат preprocessing.
├── training/
├── pseudo_labels/
├── taxonomy/  --Классификатор ОКПД-2 (codes, hierarchy, aliases, metadata, inference/входные файлы для предсказания.)
├── evaluation/
└── inference/

taxonomy/

ОКПД-2:


📁 3. ARTIFACTS LAYER
   artifacts/
   ├── biencoder/
   ├── classifier/
   ├── faiss/
   ├── reranker/
   ├── metrics/
   └── runs/

📁 4. SRC STRUCTURE

   📁 retrieval/
   retrieval/
   отвечает ТОЛЬКО за:
   embeddings
   FAISS
   retrieval
   similarity
   
   📁 classifier/
   classifier/
   отвечает ТОЛЬКО за:
   BERT
   logits
   entropy
   classifier inference
   
   📁 hierarchy/
   hierarchy/
   отвечает ТОЛЬКО за:
   tree
   parent-child logic
   hierarchy scoring
   
   📁 preprocessing/
   preprocessing/
   разделить: raw preprocessing и embedding preprocessing
   
   📁 confidence/
   confidence/
   отдельный модуль!
   отвечает за:
   margin
   entropy
   agreement
   risk scoring
   OOD detection
   
   📁 pseudo_labeling/
   pseudo_labeling/
   отдельный isolated pipeline
   
   📁 evaluation/
   evaluation/
   считает:
   Recall@K
   MRR
   coverage
   uncertainty stats
   confusion clusters
   
   📁 pipelines/
   pipelines/
   только orchestration.



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