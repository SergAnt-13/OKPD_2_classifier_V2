from dataclasses import dataclass


@dataclass
class DecisionResult:
    code: str | None
    mode: str  # AUTO / REVIEW / MANUAL
    confidence: float
    reason: list


class DecisionEngineV2:

    def __init__(self,
                 alpha=0.5,
                 beta=0.3,
                 gamma=0.2):

        """
        alpha = retrieval weight
        beta = classifier weight
        gamma = margin penalty weight
        """

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    # -------------------------
    # MAIN DECISION LOGIC
    # -------------------------

    def decide(self,
               classifier_pred: dict,
               retrieval_pred: dict):

        reasons = []

        # -------------------------
        # 1. RETRIEVAL SCORE
        # -------------------------
        top_retrieval = retrieval_pred["candidates"][0]
        retrieval_score = top_retrieval["score"]
        code = top_retrieval["code"]

        # -------------------------
        # 2. CLASSIFIER SCORE
        # -------------------------
        clf_conf = classifier_pred["confidence"]
        clf_code = classifier_pred["code"]

        # -------------------------
        # 3. AGREEMENT
        # -------------------------
        agreement = (clf_code == code)

        if agreement:
            reasons.append("classifier == retrieval")

        # -------------------------
        # 4. COMBINED SCORE
        # -------------------------
        final_score = (
            self.alpha * retrieval_score +
            self.beta * clf_conf
        )

        # -------------------------
        # 5. DECISION RULES
        # -------------------------

        if final_score > 0.85 and agreement:
            mode = "AUTO"

        elif final_score > 0.65:
            mode = "REVIEW"

        else:
            mode = "MANUAL"

        # -------------------------
        # 6. OOD / UNCERTAINTY FLAGS
        # -------------------------

        if retrieval_score < 0.4:
            reasons.append("low retrieval similarity")

        if clf_conf < 0.5:
            reasons.append("low classifier confidence")

        if not agreement:
            reasons.append("model disagreement")

        return DecisionResult(
            code=code,
            mode=mode,
            confidence=final_score,
            reason=reasons
        )