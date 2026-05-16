class RiskEngine:

    @staticmethod
    def calculate_risk(
        retrieval_score: float,
        margin: float,
        entropy: float,
        agreement: bool,
        hierarchy_consistent: bool
    ):

        risk = 0
        reasons = []

        if retrieval_score < 0.75:
            risk += 2
            reasons.append("low_retrieval_score")

        if margin < 0.15:
            risk += 2
            reasons.append("low_margin")

        if entropy > 1.5:
            risk += 1
            reasons.append("high_entropy")

        if not agreement:
            risk += 3
            reasons.append("classifier_retrieval_disagreement")

        if not hierarchy_consistent:
            risk += 3
            reasons.append("hierarchy_inconsistent")

        if risk <= 2:
            level = "AUTO"

        elif risk <= 5:
            level = "REVIEW"

        else:
            level = "MANUAL"

        return {
            "risk_score": risk,
            "risk_level": level,
            "requires_review": level != "AUTO",
            "reasons": reasons
        }