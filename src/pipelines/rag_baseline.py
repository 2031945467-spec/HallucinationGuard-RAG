import os
import time

from src.retriever import DocumentRetriever


REFUSAL = "抱歉，提供的信息不足以回答该问题。"


class RAGPipeline:
    """Extractive RAG baseline for deterministic offline reproduction."""

    def __init__(self):
        self.retriever = DocumentRetriever()
        self.min_score = float(os.getenv("RAG_MIN_SCORE", "0.12"))

    def select_answer(self, docs: list) -> str:
        if not docs or docs[0].metadata.get("score", 0.0) < self.min_score:
            return REFUSAL
        return docs[0].page_content

    def run(self, question: str) -> dict:
        start_time = time.time()
        docs = self.retriever.retrieve(question, top_k=3)
        answer = self.select_answer(docs)
        return {
            "answer": answer,
            "latency": time.time() - start_time,
            "context": [doc.page_content for doc in docs],
        }
