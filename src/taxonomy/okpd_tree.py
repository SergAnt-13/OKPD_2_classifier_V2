# src/taxonomy/okpd_tree.py
"""
Работа с иерархией классификатора ОКПД-2.
"""
from typing import List, Optional
import pandas as pd


def get_level(code: str) -> int:
    """
    Эвристика уровня детализации кода ОКПД-2.
    - 2 цифры (XX) → класс (уровень 1)
    - 4-5 цифр (XX.XX или XX.XX.X) → группа (уровень 2)
    - больше → вид/категория/подкатегория (уровень 3+)
    """
    digits = code.replace(".", "")
    length = len(digits)
    if length <= 2:
        return 1
    elif length <= 5:
        return 2
    else:
        return 3


def is_same_branch(code1: str, code2: str, level: int = 2) -> bool:
    """Проверяет, принадлежат ли два кода одной ветке."""
    if level == 1:
        return code1[:2] == code2[:2]
    elif level == 2:
        parts1 = code1.split(".")
        parts2 = code2.split(".")
        return ".".join(parts1[:2]) == ".".join(parts2[:2])
    else:
        return ".".join(code1.split(".")[:level]) == ".".join(code2.split(".")[:level])


def load_okpd_tree(path: str) -> pd.DataFrame:
    """Загружает эталонный справочник ОКПД-2."""
    if path.endswith(".csv"):
        df = pd.read_csv(path, dtype=str)
    else:
        df = pd.read_excel(path, dtype=str)
    required = {"code", "parent_code", "name"}
    if not required.issubset(set(df.columns)):
        raise ValueError(f"В файле должны быть колонки: {required}")
    return df


class OKPDTree:
    """Обёртка над деревом ОКПД-2."""

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.code_to_parent = dict(zip(df["code"], df["parent_code"]))
        self.code_to_name = dict(zip(df["code"], df["name"]))
        self.all_codes = set(df["code"])

    def get_parent(self, code: str) -> Optional[str]:
        return self.code_to_parent.get(code)

    def is_valid_code(self, code: str) -> bool:
        return code in self.all_codes

    def get_ancestors(self, code: str) -> List[str]:
        ancestors = []
        current = code
        while current:
            parent = self.get_parent(current)
            if parent:
                ancestors.append(parent)
                current = parent
            else:
                break
        return ancestors

    def is_child_of(self, code: str, parent_candidate: str) -> bool:
        return parent_candidate in self.get_ancestors(code)