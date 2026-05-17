class OKPDInferencePipeline:

    def __init__(self, bert_model, retriever, decision_engine):
        self.bert = bert_model
        self.retriever = retriever
        self.decision = decision_engine

    def predict(self, text: str):

        # -------------------------
        # 1. CLASSIFIER (BERT)
        # -------------------------
        clf_output = self.bert.predict(text)

        # ожидаемый формат:
        # {
        #   code,
        #   confidence,
        #   top1_prob,
        #   top2_prob,
        #   probs
        # }

        # -------------------------
        # 2. RETRIEVAL (FAISS / SBERT)
        # -------------------------
        retrieval_output = self.retriever.search(text)

        # {
        #   candidates: [{code, score}]
        # }

        # -------------------------
        # 3. DECISION LAYER
        # -------------------------
        decision = self.decision.decide(
            classifier_pred=clf_output,
            retrieval_pred=retrieval_output
        )

        return {
            "prediction": decision.code,
            "mode": decision.mode,
            "confidence": decision.confidence,
            "reasons": decision.reason
        }