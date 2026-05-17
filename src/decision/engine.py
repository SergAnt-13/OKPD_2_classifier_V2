# src/decision/engine.py
from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np


@dataclass
class DecisionResult:
    """Финальное решение по товару."""
    code: Optional[str]
    mode: str  # AUTO / REVIEW / MANUAL
    confidence: float
    reasons: List[str] = field(default_factory=list)
    risk_score: int = 0


class DecisionEngine:
    """
    Центр управления риском.
    Объединяет сигналы retrieval и classifier, принимает решение о роутинге.
    """

    def __init__(
        self,
        auto_threshold: float = 0.85,
        review_threshold: float = 0.65,
        alpha: float = 0.5,      # вес retrieval
        beta: float = 0.3,       # вес classifier
        gamma: float = 0.2,      # вес margin
    ):
        self.auto_threshold = auto_threshold
        self.review_threshold = review_threshold
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def decide(
        self,
        classifier_pred: dict,
        retrieval_pred: dict,
        hierarchy_consistent: bool = True,
    ) -> DecisionResult:
        """
        Принимает решение на основе сигналов.

        Args:
            classifier_pred: {"code": str, "confidence": float, "margin": float, "entropy": float}
            retrieval_pred: {"candidates": [{"code": str, "score": float}, ...]}
            hierarchy_consistent: согласована ли иерархия

        Returns:
            DecisionResult с кодом, модой, причинами
        """
        reasons = []

        # --- 1. RETRIEVAL ---
        top_retrieval = retrieval_pred["candidates"][0]
        retrieval_code = top_retrieval["code"]
        retrieval_score = top_retrieval["score"]

        # --- 2. CLASSIFIER ---
        clf_code = classifier_pred.get("code")
        clf_conf = classifier_pred.get("confidence", 0.0)
        margin = classifier_pred.get("margin", 0.0)
        entropy = classifier_pred.get("entropy", 0.0)

        # --- 3. AGREEMENT ---
        agreement = (clf_code == retrieval_code)
        if agreement:
            reasons.append("classifier == retrieval")
        else:
            reasons.append("model disagreement")

        # --- 4. COMBINED SCORE ---
        final_score = (
            self.alpha * retrieval_score +
            self.beta * clf_conf +
            self.gamma * margin
        )

        # --- 5. PENALTIES ---
        if not agreement:
            final_score *= 0.85
        if retrieval_score < 0.4:
            final_score *= 0.8
            reasons.append("low retrieval similarity")
        if clf_conf < 0.5:
            final_score *= 0.85
            reasons.append("low classifier confidence")
        if margin < 0.2:
            final_score *= 0.9
            reasons.append("low margin")
        if entropy > 1.5:
            final_score *= 0.85
            reasons.append("high entropy")
        if not hierarchy_consistent:
            final_score *= 0.8
            reasons.append("hierarchy inconsistent")

        # --- 6. RISK SCORE (из risk_engine) ---
        risk_score = 0
        if retrieval_score < 0.75:
            risk_score += 2
        if margin < 0.15:
            risk_score += 2
        if entropy > 1.5:
            risk_score += 1
        if not agreement:
            risk_score += 3
        if not hierarchy_consistent:
            risk_score += 3

        # --- 7. ROUTING ---
        if risk_score <= 2 and final_score >= self.auto_threshold:
            mode = "AUTO"
        elif risk_score <= 5 or final_score >= self.review_threshold:
            mode = "REVIEW"
        else:
            mode = "MANUAL"

        return DecisionResult(
            code=retrieval_code,
            mode=mode,
            confidence=round(float(final_score), 4),
            reasons=reasons,
            risk_score=risk_score,
        )