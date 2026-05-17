from dataclasses import dataclass


@dataclass
class ClassifierOutput:
    code: str
    confidence: float
    top1_prob: float
    top2_prob: float
    probs: list