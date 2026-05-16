import re


class TextViews:

    @staticmethod
    def classifier_view(text: str) -> str:
        """
        Более "чистый" текст для BERT
        """

        text = text.lower().strip()
        text = re.sub(r"\s+", " ", text)

        return text


    @staticmethod
    def retrieval_view(text: str) -> str:
        """
        Сохраняем максимум сигналов:
        - CAPS
        - граммовку
        - бренды
        """

        text = text.strip()

        # нормализуем пробелы, но НЕ убиваем структуру
        text = re.sub(r"\s+", " ", text)

        return text