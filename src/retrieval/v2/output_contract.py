from dataclasses import dataclass


@dataclass
class RetrievalCandidate:
    code: str
    score: float


@dataclass
class RetrievalOutput:
    candidates: list[RetrievalCandidate]