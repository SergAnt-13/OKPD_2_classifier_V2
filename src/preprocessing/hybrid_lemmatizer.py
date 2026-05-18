# src/preprocessing/hybrid_lemmatizer.py
import re
from pymorphy2 import MorphAnalyzer
from pymystem3 import Mystem


class HybridLemmatizer:
    """
    Быстрый гибридный лемматизатор:
    - pymorphy2 для основной массы слов (быстро)
    - pymystem3 для токенов, которые pymorphy2 не смог нормализовать (медленно, но точно)
    """
    def __init__(self):
        self.morph = MorphAnalyzer()
        self.mystem = Mystem()
        # Символы, при которых слово считается "техническим" и не подлежит лемматизации через pymorphy2
        self.technical_pattern = re.compile(r'[0-9/%\-\+\\.,()]')

    def _is_technical(self, token: str) -> bool:
        """Проверяет, содержит ли токен цифры или спецсимволы."""
        return bool(self.technical_pattern.search(token))

    def lemmatize(self, text: str) -> str:
        """Гибридная лемматизация строки."""
        if not text or not text.strip():
            return ""
        # Разбиваем на токены по пробелам
        tokens = text.split()
        lemmatized_tokens = []
        for token in tokens:
            token_clean = token.strip().lower()
            if not token_clean:
                continue
            # Если токен содержит цифры или спецсимволы – оставляем как есть
            if self._is_technical(token_clean):
                lemmatized_tokens.append(token_clean)
                continue
            # Пробуем лемматизировать через pymorphy2
            parsed = self.morph.parse(token_clean)
            if parsed:
                lemma = parsed[0].normal_form
                # Если лемма совпадает с исходным словом, и это словарное слово (вероятность > 0.5),
                # считаем, что лемматизация удалась
                if lemma != token_clean or parsed[0].score > 0.5:
                    lemmatized_tokens.append(lemma)
                    continue
            # Если pymorphy2 не справился (пустой разбор или низкая уверенность) – используем mystem
            try:
                lemma = self.mystem.lemmatize(token_clean)[0]
                if lemma and lemma != ' ':
                    lemmatized_tokens.append(lemma)
                else:
                    lemmatized_tokens.append(token_clean)
            except Exception:
                lemmatized_tokens.append(token_clean)
        return " ".join(lemmatized_tokens).strip()