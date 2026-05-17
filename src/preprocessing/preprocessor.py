from src.preprocessing.text_views import TextViews


class TextPreprocessor:

    def process(self, text: str) -> dict:
        """
        Возвращает 2 представления:
        - classifier_input
        - retrieval_input
        """

        return {
            "classifier_text": TextViews.classifier_view(text),
            "retrieval_text": TextViews.retrieval_view(text)
        }