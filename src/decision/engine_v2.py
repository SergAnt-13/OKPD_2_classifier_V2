from dataclasses import dataclass


@dataclass
class DecisionResult:
    code: str | None
    mode: str  # AUTO / REVIEW / MANUAL
    confidence: float
    reason: list


class DecisionEngineV2:

    def __init__(self):
        self.auto_threshold = 0.85
        self.review_threshold = 0.65

    def decide(self, classifier_pred: dict, retrieval_pred: dict):

        reasons = []

        # -------------------------
        # RETRIEVAL
        # -------------------------
        top_retrieval = retrieval_pred["candidates"][0]
        retrieval_code = top_retrieval["code"]
        retrieval_score = top_retrieval["score"]

        # -------------------------
        # CLASSIFIER
        # -------------------------
        clf_code = classifier_pred["code"]
        clf_conf = classifier_pred["confidence"]
        top1 = classifier_pred.get("top1_prob", clf_conf)
        top2 = classifier_pred.get("top2_prob", 0.0)

        # -------------------------
        # AGREEMENT
        # -------------------------
        agreement = (clf_code == retrieval_code)
        if agreement:
            reasons.append("agreement")

        # -------------------------
        # MARGIN
        # -------------------------
        margin = top1 - top2
        if margin < 0.2:
            reasons.append("low_margin")

        # -------------------------
        # BASE CONFIDENCE
        # -------------------------
        base_conf = 0.6 * retrieval_score + 0.4 * clf_conf

        # penalties
        if not agreement:
            base_conf *= 0.85
            reasons.append("disagreement")

        if retrieval_score < 0.4:
            base_conf *= 0.8
            reasons.append("low_retrieval")

        if clf_conf < 0.5:
            base_conf *= 0.85
            reasons.append("low_classifier")

        if margin < 0.2:
            base_conf *= 0.9

        # -------------------------
        # DECISION
        # -------------------------
        if base_conf >= self.auto_threshold and agreement:
            mode = "AUTO"
        elif base_conf >= self.review_threshold:
            mode = "REVIEW"
        else:
            mode = "MANUAL"

        return DecisionResult(
            code=retrieval_code,
            mode=mode,
            confidence=float(base_conf),
            reason=reasons
        )