from dataclasses import dataclass
from typing import List, Optional


@dataclass
class RetrievalCandidate:
    code: str
    title: str
    score: float


@dataclass
class ClassifierPrediction:
    code: str
    probability: float


@dataclass
class RiskReport:
    risk_score: int
    risk_level: str
    requires_review: bool
    reasons: List[str]


@dataclass
class PredictionResult:
    input_text: str
    final_code: str
    confidence_zone: str

    retrieval_candidates: List[RetrievalCandidate]

    classifier_prediction: Optional[ClassifierPrediction]

    risk_report: RiskReport