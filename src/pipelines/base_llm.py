import time

from src.config import get_llm


class BasePipeline:
    def __init__(self):
        self.llm = get_llm()

    def run(self, question: str) -> dict:
        start_time = time.time()
        try:
            answer = self.llm.invoke(
                f"请直接、简洁地回答问题。不要编造不确定的信息。\n问题：{question}"
            ).content
        except Exception as error:
            answer = f"【降级回答】模型调用失败：{type(error).__name__}"
        return {
            "answer": answer,
            "latency": time.time() - start_time,
            "context": [],
        }
