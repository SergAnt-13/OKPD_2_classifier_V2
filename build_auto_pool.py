# build_auto_pool.py

import pandas as pd
from tqdm import tqdm
from config.settings import TRAINING_DATA_DIR, FAISS_DIR, REFERENCE_DIR, PROCESSED_DATA_DIR, BIENCODER_DIR
from src.preprocessing.cleaner import TextCleaner
from src.retrieval.retriever import Retriever
from src.models.bert_classifier import BERTClassifier
from src.decision.engine import DecisionEngine

# Инициализация (стеммированный индекс, обученная модель)
cleaner = TextCleaner(abbreviations_path=REFERENCE_DIR / "сокращения.xlsx")
retriever = Retriever(
    model_dir=BIENCODER_DIR if BIENCODER_DIR.exists() else None,
    index_path=FAISS_DIR / "okpd_index_stemmed.faiss",
    id_map_path=FAISS_DIR / "id_map_stemmed.csv"
)
classifier = BERTClassifier()
engine = DecisionEngine(auto_threshold=0.85, review_threshold=0.30)

# Загружаем выборку
gold = pd.read_excel(TRAINING_DATA_DIR / "train.xlsx", dtype=str)
gold = gold[["Номенклатура", "Код ОКПД2"]].dropna()
gold.columns = ["text", "true_code"]

results = []
for _, row in tqdm(gold.iterrows(), total=len(gold), desc="Поиск AUTO"):
    text = str(row["text"])
    true_code = row["true_code"].strip()

    # Предсказание
    query = cleaner.retrieval_view_normalized(text)
    retrieval_result = retriever.search_normalized(query)
    classifier_result = classifier.predict(cleaner.classifier_view(text))
    decision = engine.decide(classifier_result, retrieval_result)

    if (decision.mode == "AUTO" or (decision.mode == "REVIEW" and decision.confidence > 0.7)):
        results.append({
            "text": text,
            "true_code": true_code,
            "predicted_code": decision.code,
            "confidence": decision.confidence,
            "reasons": "; ".join(decision.reasons)
        })

out_df = pd.DataFrame(results)
output_path = PROCESSED_DATA_DIR / "auto_pool_gold.csv"
out_df.to_csv(output_path, index=False)
print(f"Найдено {len(out_df)} AUTO-примеров, сохранено в {output_path}")