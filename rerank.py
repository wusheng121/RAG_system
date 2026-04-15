from sentence_transformers import CrossEncoder
import os

class Reranker:
    def __init__(self):
        self.model = None
        self._initialized = False
        self._available = False
        self.local_files_only = os.getenv("HF_LOCAL_FILES_ONLY", "1") == "1"

    def _initialize(self):
        if not self._initialized:
            try:
                self.model = CrossEncoder("BAAI/bge-reranker-base", local_files_only=self.local_files_only)
                self._available = True
            except Exception:
                self.model = None
                self._available = False
            self._initialized = True

    def rerank(self, query, docs, top_k=2):
        if not docs:
            return []
        self._initialize()
        if not self._available or self.model is None:
            return docs[:top_k]
        pairs = [[query, doc] for doc in docs]
        scores = self.model.predict(pairs)
        sorted_docs = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in sorted_docs[:top_k]]
