# src/preprocessing/cleaner.py
import re
from pathlib import Path
from typing import Optional, Dict
import pandas as pd

from src.preprocessing.lemmatizer import Lemmatizer
from src.preprocessing.hybrid_lemmatizer import HybridLemmatizer


class TextCleaner:
    GOST_PATTERNS = [
        re.compile(r"\bгост\b\s*[\d\-]+", re.IGNORECASE),
        re.compile(r"\bту\b\s*[\d\-]+", re.IGNORECASE),
        re.compile(r"\bсто\b\s*[\d\-]+", re.IGNORECASE),
    ]

    def __init__(
        self,
        abbreviations_path: Optional[Path] = None,
        use_lemmatizer: bool = False,
        use_hybrid_lemmatizer=False
    ):
        self.abbreviations: Dict[str, str] = {}
        self.abbr_comments: Dict[str, str] = {}   # пока не используется
        if abbreviations_path and Path(abbreviations_path).exists():
            df = pd.read_excel(abbreviations_path, dtype=str)
            if "abbr" in df.columns and "expansion" in df.columns:
                df["abbr_clean"] = df["abbr"].str.lower().str.strip().str.rstrip(".")
                self.abbreviations = dict(
                    zip(
                        df["abbr_clean"],
                        df["expansion"].str.strip(),
                    )
                )
                # Сохраняем комментарии, если есть колонка 'comment'
                if "comment" in df.columns:
                    self.abbr_comments = dict(
                        zip(
                            df["abbr"].str.lower().str.strip(),
                            df["comment"].str.strip(),
                        )
                    )
        self.lemmatizer = None
        if use_hybrid_lemmatizer:
            self.lemmatizer = HybridLemmatizer()
        elif use_lemmatizer:
            self.lemmatizer = Lemmatizer()  # старый Mystem из lemmatizer.py
    @staticmethod
    def remove_gost(text: str) -> str:
        for pattern in TextCleaner.GOST_PATTERNS:
            text = pattern.sub(" ", text)
        return text

    @staticmethod
    def normalise_punctuation(text: str) -> str:
        text = re.sub(r"[^а-яёa-z0-9\s]", " ", text, flags=re.IGNORECASE)
        return text

    def apply_abbreviations(self, text: str) -> str:
        if not self.abbreviations:
            return text
        tokens = re.findall(r"\b\w+(?:\.\w+)+\b|\b\w+\b|[^\w\s]", text)
        result = []
        for token in tokens:
            token_lower = token.lower().rstrip(".")
            if token_lower in self.abbreviations:
                result.append(self.abbreviations[token_lower])
            else:
                result.append(token)
        return " ".join(result)

    @staticmethod
    def classifier_view(text: Optional[str]) -> str:
        if not text:
            return ""
        text = str(text).lower()
        text = re.sub(r"[^а-яёa-z0-9\s]", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def retrieval_view_normalized(self, text: Optional[str]) -> str:
        if not text:
            return ""
        text = str(text).strip().lower()
        if not text:
            return ""

        # 1. Расшифровка сокращений
        text = self.apply_abbreviations(text)

        # 2. Удаляем ГОСТы, ТУ, СТО
        text = self.remove_gost(text)

        # 3. Удаляем пунктуацию
        text = self.normalise_punctuation(text)

        # 4. Удаляем ведущие изолированные цифры
        text = re.sub(r"^\d+\s+", "", text)

        # 5. Нормализуем пробелы
        text = re.sub(r"\s+", " ", text).strip()

        # 6. Лемматизация (опционально)
        if self.lemmatizer:
            text = self.lemmatizer.lemmatize(text)

        return text