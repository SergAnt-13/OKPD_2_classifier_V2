from dataclasses import dataclass
from typing import List


@dataclass
class ProcessedSample:
    raw_text: str
    classifier_text: str
    retrieval_text: str
    true_code: str