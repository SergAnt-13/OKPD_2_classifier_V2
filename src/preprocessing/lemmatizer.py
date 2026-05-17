# src/preprocessing/lemmatizer.py
"""
Лемматизация текста с помощью pymystem3 (Яндекс.Мystem).
Требуется установка: pip install pymystem3
"""
import re
from pymystem3 import Mystem


class Lemmatizer:
    def __init__(self):
        self.mystem = Mystem()

    def lemmatize(self, text: str) -> str:
        """
        Лемматизирует текст, сохраняя цифры, дефисы, точки в числах и т.п.
        """
        if not text or not text.strip():
            return ""
        # Mystem.lemmatize возвращает список лемм с переносами строк
        lemmas = self.mystem.lemmatize(text)
        result = []
        for token in lemmas:
            token = token.strip()
            if token and not token.isspace():
                result.append(token)
        # Склеиваем, убираем лишние пробелы
        result_text = " ".join(result)
        result_text = re.sub(r"\s+", " ", result_text).strip()
        return result_text