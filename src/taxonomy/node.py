from dataclasses import dataclass
from typing import Optional, List


@dataclass
class OKPDNode:
    code: str
    title: str

    parent_code: Optional[str] = None

    children: List[str] = None  # храним коды детей

    def __post_init__(self):
        if self.children is None:
            self.children = []