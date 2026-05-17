# src/preprocessing/cleaner.py
import re
from typing import Optional


class TextCleaner:
    """
    Два режима очистки текста:
    - classifier_view: агрессивная очистка для BERT
    - retrieval_view: минимальная очистка (сохраняет цифры, ГОСТ, CAPS)
    """

    @staticmethod
    def classifier_view(text: Optional[str]) -> str:
        """Чистый текст для BERT: lower, без спецсимволов, без лишних пробелов."""
        if not text:
            return ""
        text = str(text).lower()
        text = re.sub(r"[^а-яёa-z0-9\s]", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    @staticmethod
    def retrieval_view(text: Optional[str]) -> str:
        """Минимальная очистка: сохраняет цифры, ГОСТ, CAPS, артикулы."""
        if not text:
            return ""
        text = str(text).strip()
        text = re.sub(r"\s+", " ", text)
        return text

    @staticmethod
    def normalise_gost(text: str) -> str:
        """Приводит варианты написания ГОСТ к единому виду."""
        text = re.sub(r"ГОСТ\s*Р?\s*", "ГОСТ ", text, flags=re.IGNORECASE)
        text = re.sub(r"\bТУ\s*", "ТУ ", text, flags=re.IGNORECASE)
        return text