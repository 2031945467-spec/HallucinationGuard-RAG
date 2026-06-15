import os
import re
import time

from src.pipelines.rag_baseline import RAGPipeline


class RAGCovePipeline(RAGPipeline):
    """Retrieval plus independent keyword checks and conservative refusal."""

    def __init__(self):
        super().__init__()
        self.cove_score = float(os.getenv("COVE_MIN_SCORE", "0.16"))
        self.keyword_coverage = float(os.getenv("COVE_KEYWORD_COVERAGE", "0.45"))

    @staticmethod
    def _keywords(question: str) -> list[str]:
        chunks = re.findall(r"[\u4e00-\u9fff]{2,}|[A-Za-z0-9.+/-]+", question)
        stop = {"什么", "哪个", "哪些", "多少", "是否", "通常", "主要", "为什么"}
        return [chunk for chunk in chunks if chunk not in stop]

    def run(self, question: str) -> dict:
        start_time = time.time()
        docs = self.retriever.retrieve(question, top_k=5)
        top = docs[0]
        evidence = f"{top.metadata.get('title', '')}{top.page_content}"
        keywords = self._keywords(question)
        coverage = (
            sum(keyword in evidence for keyword in keywords) / len(keywords)
            if keywords
            else 0.0
        )
        if (
            top.metadata.get("score", 0.0) >= self.cove_score
            and coverage >= self.keyword_coverage
        ):
            answer = top.page_content
        else:
            answer = "【拒答】独立核查未找到足够一致的证据，无法可靠回答。"
        return {
            "answer": answer,
            "latency": time.time() - start_time,
            "context": [doc.page_content for doc in docs],
        }
