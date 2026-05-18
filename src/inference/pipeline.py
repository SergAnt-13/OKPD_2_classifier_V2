# src/inference/pipeline.py
from typing import Dict, List
import pandas as pd
from pathlib import Path

from config.settings import REFERENCE_DIR, FAISS_DIR, BIENCODER_DIR
from src.preprocessing.cleaner import TextCleaner
from src.retrieval.retriever import Retriever
from src.models.bert_classifier import BERTClassifier
from src.decision.engine import DecisionEngine
from src.taxonomy.okpd_tree import is_same_branch


class InferencePipeline:
    """
    Единый пайплайн инференса.
    Использует стеммированный индекс и дообученный bi-encoder.
    """

    def __init__(
        self,
        classifier: BERTClassifier = None,
        retriever: Retriever = None,
        engine: DecisionEngine = None,
    ):
        self.cleaner = TextCleaner(abbreviations_path=REFERENCE_DIR / "сокращения.xlsx")
        self.classifier = classifier or BERTClassifier()
        self.retriever = retriever or Retriever(
            model_dir=BIENCODER_DIR if BIENCODER_DIR.exists() else None,
            index_path=FAISS_DIR / "okpd_index_stemmed.faiss",
            id_map_path=FAISS_DIR / "id_map_stemmed.csv",
        )
        self.engine = engine or DecisionEngine()

    def predict_single(self, text: str) -> Dict:
        # 1. Preprocessing
        text_for_bert = self.cleaner.classifier_view(text)
        text_for_retrieval = self.cleaner.retrieval_view_normalized(text)

        # 2. Retrieval
        retrieval_result = self.retriever.search(text_for_retrieval)

        # 3. Classification
        classifier_result = self.classifier.predict(text_for_bert)

        # 4. Decision
        decision = self.engine.decide(classifier_result, retrieval_result, original_text=text)

        retrieval_top1 = retrieval_result["candidates"][0]["code"] if retrieval_result["candidates"] else ""
        agreement = (classifier_result.get("code") == retrieval_top1)
        hierarchy_ok = is_same_branch(retrieval_top1, classifier_result.get("code"), level=2)

        return {
            "text": text,
            "normalized_text": text_for_retrieval,
            "top_candidates": retrieval_result["candidates"],
            "final_prediction": decision.code,
            "confidence": decision.confidence,
            "margin": classifier_result.get("margin", 0.0),
            "entropy": classifier_result.get("entropy", 0.0),
            "routing": decision.mode,
            "risk_level": "low" if decision.mode == "AUTO" else (
                "medium" if decision.mode == "REVIEW" else "high"
            ),
            "requires_review": decision.mode != "AUTO",
            "reasons": decision.reasons,
            "classifier_code": classifier_result.get("code"),
            "classifier_confidence": classifier_result.get("confidence"),
            "agreement": agreement,
            "hierarchy_ok": hierarchy_ok,
        }

    def predict_batch(self, texts: List[str]) -> List[Dict]:
        return [self.predict_single(text) for text in texts]

    def predict_file(self, file_path: Path, text_column: str = "nomenclature") -> pd.DataFrame:
        if file_path.suffix == ".csv":
            df = pd.read_csv(file_path)
        else:
            df = pd.read_excel(file_path)

        results = self.predict_batch(df[text_column].astype(str).tolist())

        df["predicted_code"] = [r["final_prediction"] for r in results]
        df["confidence"] = [r["confidence"] for r in results]
        df["routing"] = [r["routing"] for r in results]
        df["requires_review"] = [r["requires_review"] for r in results]

        return df