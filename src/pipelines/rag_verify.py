import os
import time

from src.pipelines.rag_baseline import RAGPipeline


class RAGVerifyPipeline(RAGPipeline):
    """RAG with confidence and ambiguity checks before answering."""

    def __init__(self):
        super().__init__()
        self.verify_score = float(os.getenv("VERIFY_MIN_SCORE", "0.18"))
        self.verify_margin = float(os.getenv("VERIFY_MIN_MARGIN", "0.035"))

    def run(self, question: str) -> dict:
        start_time = time.time()
        docs = self.retriever.retrieve(question, top_k=3)
        top_score = docs[0].metadata.get("score", 0.0)
        second_score = docs[1].metadata.get("score", 0.0) if len(docs) > 1 else 0.0
        if top_score >= self.verify_score and top_score - second_score >= self.verify_margin:
            answer = docs[0].page_content
        else:
            answer = "【拒答】检索证据的相关性或唯一性不足，无法可靠回答。"
        return {
            "answer": answer,
            "latency": time.time() - start_time,
            "context": [doc.page_content for doc in docs],
        }
