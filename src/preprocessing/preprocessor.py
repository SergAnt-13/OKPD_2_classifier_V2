import re


class TextPreprocessor:

    @staticmethod
    def clean_for_classifier(text: str) -> str:
        text = text.lower().strip()

        text = re.sub(r"\s+", " ", text)

        return text

    @staticmethod
    def clean_for_embedding(text: str) -> str:
        """
        Минимальная очистка.
        ГОСТы, цифры, артикулы сохраняем.
        """

        text = text.strip()

        text = re.sub(r"\s+", " ", text)

        return text