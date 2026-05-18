# quick_test_stemmed.py
"""Гарантированные примеры для демонстрации на защите."""
import pandas as pd
from config.settings import RAW_DATA_DIR, FAISS_DIR, REFERENCE_DIR, PROCESSED_DATA_DIR, BIENCODER_DIR
from src.preprocessing.cleaner import TextCleaner
from src.retrieval.retriever import Retriever
from src.models.bert_classifier import BERTClassifier
from src.decision.engine import DecisionEngine

# Компоненты на стеммированном индексе
cleaner = TextCleaner(abbreviations_path=REFERENCE_DIR / "сокращения.xlsx")
retriever = Retriever(
    model_dir=BIENCODER_DIR if BIENCODER_DIR.exists() else None,
    index_path=FAISS_DIR / "okpd_index_stemmed.faiss",
    id_map_path=FAISS_DIR / "id_map_stemmed.csv"
)
classifier = BERTClassifier()
engine = DecisionEngine(auto_threshold=0.85, review_threshold=0.30)

ref = pd.read_csv(PROCESSED_DATA_DIR / "okpd_2_normalized.csv", dtype=str)
code_to_name = dict(zip(ref["code"], ref["name"]))

# 10 примеров — смесь проверенных и читаемых из номенклатуры
samples = [
    "к.ш.в карамельной глазури",
    "драже к.ш.",
    "Шоколад молочный 100г",
    "Конфеты Птичье молоко",
    "Молоко сгущенное с сахаром 8.5%",
    "Масло растительное",
    "0 Битки с луком",
    "0 Голубцы",
    "Печ.сах. Земляничное НТКФ"
]

print("Демонстрация работы системы (лучшие примеры):\n")
print("Товар".ljust(35), "Код".ljust(15), "Название кода".ljust(45), "Уверенность".ljust(6), "Режим")
print("-" * 120)

for text in samples:
    query = cleaner.retrieval_view_normalized(text)
    retrieval_result = retriever.search_normalized(query)
    classifier_result = classifier.predict(cleaner.classifier_view(text))
    decision = engine.decide(classifier_result, retrieval_result)

    code = decision.code
    conf = decision.confidence
    mode = decision.mode
    code_name = code_to_name.get(code, "")[:45]

    print(f"{text[:35]:35s} {code:15s} {code_name:45s} {conf:.2f}   {mode}")