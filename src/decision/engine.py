# src/decision/engine.py
from dataclasses import dataclass, field
from typing import List, Optional, Dict
from pathlib import Path
import pandas as pd
import numpy as np
from src.taxonomy.okpd_tree import is_same_branch

@dataclass
class DecisionResult:
    code: Optional[str]
    mode: str
    confidence: float
    reasons: List[str] = field(default_factory=list)
    risk_score: int = 0

class DecisionEngine:
    def __init__(
        self,
        auto_threshold: float = 0.85,
        review_threshold: float = 0.40,
        abbreviations_path: Optional[Path] = None,
        class_prior_path: Optional[Path] = None,
    ):
        self.auto_threshold = auto_threshold
        self.review_threshold = review_threshold

        # Prior Weighting
        self.class_priors: Dict[str, float] = {}
        if class_prior_path and Path(class_prior_path).exists():
            prior_df = pd.read_csv(class_prior_path)
            if 'code' in prior_df.columns and 'prior' in prior_df.columns:
                max_prior = prior_df['prior'].max()
                if max_prior > 0:
                    prior_df['weight'] = prior_df['prior'] / max_prior
                    self.class_priors = dict(zip(prior_df['code'], prior_df['weight']))

        # Domain Boosting
        self.domain_rules: Dict[str, List[str]] = {}
        if abbreviations_path and Path(abbreviations_path).exists():
            abbr_df = pd.read_excel(abbreviations_path, dtype=str)
            if 'abbr' in abbr_df.columns and 'okpd_codes' in abbr_df.columns:
                for _, row in abbr_df.iterrows():
                    abbr = str(row['abbr']).lower().strip()
                    codes_str = str(row['okpd_codes'])
                    if pd.notna(codes_str) and codes_str:
                        codes = [c.strip() for c in codes_str.split(';') if c.strip()]
                        if codes:
                            self.domain_rules[abbr] = codes

    def decide(
        self,
        classifier_pred: dict,
        retrieval_pred: dict,
        hierarchy_consistent: Optional[bool] = None,
        original_text: str = "",
    ) -> DecisionResult:
        reasons = []

        # --- 1. Извлечение сигналов ---
        top_retrieval = retrieval_pred["candidates"][0]
        retrieval_code = top_retrieval["code"]
        retrieval_score = top_retrieval["score"]

        retrieval_margin = 0.0
        if len(retrieval_pred["candidates"]) >= 2:
            retrieval_margin = (
                retrieval_pred["candidates"][0]["score"] - retrieval_pred["candidates"][1]["score"]
            )

        clf_code = classifier_pred.get("code")
        clf_conf = classifier_pred.get("confidence", 0.0)
        clf_entropy = classifier_pred.get("entropy", 0.0)

        # --- 2. Prior Weighting ---
        if clf_code and clf_code in self.class_priors:
            prior_weight = self.class_priors[clf_code]
            clf_conf = clf_conf * (0.5 + 0.5 * prior_weight)
            reasons.append(f"prior adjusted (weight={prior_weight:.2f})")

        # --- 3. Domain Boosting ---
        if original_text and clf_code and self.domain_rules:
            text_lower = original_text.lower()
            for abbr, allowed_codes in self.domain_rules.items():
                if abbr in text_lower:
                    if clf_code in allowed_codes:
                        clf_conf = min(clf_conf * 1.2, 0.99)
                        reasons.append(f"domain boost: {abbr} -> {clf_code}")
                    break

        # --- 4. Иерархия и согласие ---
        if hierarchy_consistent is None:
            hierarchy_consistent = is_same_branch(retrieval_code, clf_code, level=2)

        agreement = (clf_code == retrieval_code)

        # --- 5. Итоговая уверенность ---
        base_confidence = retrieval_score
        classifier_bonus = 0.0
        if clf_conf > 0.5 and clf_entropy < 2.5:
            if agreement:
                classifier_bonus = 0.15
                reasons.append("classifier agrees")
            elif hierarchy_consistent:
                classifier_bonus = 0.05
                reasons.append("classifier hierarchy ok")
            else:
                classifier_bonus = -0.05

        final_confidence = min(max(base_confidence + classifier_bonus, 0.0), 1.0)

        # --- 6. Маршрутизация ---
        if agreement and final_confidence >= self.auto_threshold:
            mode = "AUTO"
        elif retrieval_score >= self.review_threshold:
            mode = "REVIEW"
        else:
            mode = "MANUAL"

        return DecisionResult(
            code=retrieval_code,
            mode=mode,
            confidence=round(final_confidence, 4),
            reasons=reasons,
        )