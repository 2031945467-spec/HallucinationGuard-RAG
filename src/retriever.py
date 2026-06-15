import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class Document:
    page_content: str
    metadata: dict = field(default_factory=dict)


class DocumentRetriever:
    """Deterministic offline retriever using Chinese character n-grams."""

    _documents = []
    _matrix = None
    _vectorizer = None

    def __init__(self):
        default_kb = Path(__file__).resolve().parent.parent / "data" / "raw_docs" / "knowledge.txt"
        if not self.__class__._documents and default_kb.exists():
            self.ingest_document(str(default_kb))

    def ingest_document(self, file_path: str):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在：{file_path}")

        text = Path(file_path).read_text(encoding="utf-8")
        blocks = re.findall(
            r"===== DOC START =====\s*(.*?)\s*===== DOC END =====", text, re.S
        )
        if not blocks:
            blocks = [part for part in re.split(r"\n\s*\n", text) if part.strip()]

        documents = []
        for block in blocks:
            fields = {}
            for line in block.splitlines():
                if ":" in line:
                    key, value = line.split(":", 1)
                    fields[key.strip()] = value.strip()
            content_match = re.search(r"content:\s*\n(.*)", block, re.S)
            content = (
                content_match.group(1).strip()
                if content_match
                else fields.get("content", block.strip())
            )
            searchable = " ".join(
                [
                    fields.get("category", ""),
                    fields.get("title", ""),
                    fields.get("keywords", ""),
                    content,
                ]
            )
            documents.append(
                Document(
                    page_content=content,
                    metadata={
                        "doc_id": fields.get("doc_id", ""),
                        "title": fields.get("title", ""),
                        "searchable": searchable,
                    },
                )
            )

        vectorizer = TfidfVectorizer(
            analyzer="char", ngram_range=(1, 3), sublinear_tf=True
        )
        matrix = vectorizer.fit_transform(
            [doc.metadata["searchable"] for doc in documents]
        )
        self.__class__._documents = documents
        self.__class__._vectorizer = vectorizer
        self.__class__._matrix = matrix
        print(f"TF-IDF 知识库索引完成：{len(documents)} 个文档。")

    def retrieve(self, query: str, top_k: int = 2) -> list:
        if not self.__class__._documents:
            raise ValueError("知识库尚未建立索引。")

        query_vector = self.__class__._vectorizer.transform([query])
        scores = cosine_similarity(query_vector, self.__class__._matrix)[0]
        indices = scores.argsort()[::-1][:top_k]
        results = []
        for index in indices:
            source = self.__class__._documents[int(index)]
            metadata = dict(source.metadata)
            metadata["score"] = float(scores[index])
            results.append(
                Document(page_content=source.page_content, metadata=metadata)
            )
        return results
