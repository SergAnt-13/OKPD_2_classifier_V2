# src/inference/pipeline.py
from typing import Dict, List
import pandas as pd
from pathlib import Path

from src.preprocessing.cleaner import TextCleaner
from src.retrieval.retriever import Retriever
from src.models.bert_classifier import BERTClassifier
from src.decision.engine import DecisionEngine, DecisionResult


class InferencePipeline:
    """
    Единый пайплайн инференса.
    Принимает текст → возвращает структурированный результат.
    """

    def __init__(
        self,
        classifier: BERTClassifier = None,
        retriever: Retriever = None,
        engine: DecisionEngine = None,
    ):
        self.cleaner = TextCleaner()
        self.classifier = classifier or BERTClassifier()
        self.retriever = retriever or Retriever()
        self.engine = engine or DecisionEngine()

    def predict_single(self, text: str) -> Dict:
        """
        Предсказание для одного товара.

        Returns:
            {
                "text": str,
                "top_candidates": list,
                "final_prediction": str,
                "confidence": float,
                "margin": float,
                "routing": str,
                "risk_level": str,
                "requires_review": bool,
                "reasons": list,
            }
        """
        # 1. Preprocessing
        text_for_bert = self.cleaner.classifier_view(text)
        text_for_retrieval = self.cleaner.retrieval_view(text)

        # 2. Retrieval
        retrieval_result = self.retriever.search(text_for_retrieval)

        # 3. Classification
        classifier_result = self.classifier.predict(text_for_bert)

        # 4. Decision
        decision = self.engine.decide(classifier_result, retrieval_result)

        return {
            "text": text,
            "top_candidates": retrieval_result["candidates"],
            "final_prediction": decision.code,
            "confidence": decision.confidence,
            "margin": classifier_result.get("margin", 0.0),
            "routing": decision.mode,
            "risk_level": "low" if decision.mode == "AUTO" else (
                "medium" if decision.mode == "REVIEW" else "high"
            ),
            "requires_review": decision.mode != "AUTO",
            "reasons": decision.reasons,
        }

    def predict_batch(self, texts: List[str]) -> List[Dict]:
        """Предсказание для списка товаров."""
        return [self.predict_single(text) for text in texts]

    def predict_file(self, file_path: Path, text_column: str = "Номенклатура") -> pd.DataFrame:
        """Предсказание для файла (CSV/Excel). Возвращает DataFrame с результатами."""
        if file_path.suffix == ".csv":
            df = pd.read_csv(file_path)
        else:
            df = pd.read_excel(file_path)

        results = self.predict_batch(df[text_column].astype(str).tolist())

        # Добавляем колонки с результатами
        df["predicted_code"] = [r["final_prediction"] for r in results]
        df["confidence"] = [r["confidence"] for r in results]
        df["routing"] = [r["routing"] for r in results]
        df["requires_review"] = [r["requires_review"] for r in results]

        return df