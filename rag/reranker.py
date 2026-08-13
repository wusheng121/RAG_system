"""重排器：CrossEncoder 对 (query, doc) 打分重排序，未缓存时降级为取前 top_k。"""

from sentence_transformers import CrossEncoder

from core.config import config


class Reranker:
    def __init__(self):
        self.model = None
        self._initialized = False
        self._available = False

    def _initialize(self):
        if self._initialized:
            return
        try:
            self.model = CrossEncoder(config.RERANKER_MODEL, local_files_only=config.HF_LOCAL_FILES_ONLY)
            self._available = True
        except Exception:
            self.model = None
            self._available = False
        self._initialized = True

    def rerank(self, query, docs, top_k=3):
        if not docs:
            return []
        self._initialize()
        if not self._available or self.model is None:
            return docs[:top_k]
        pairs = [[query, doc] for doc in docs]
        scores = self.model.predict(pairs)
        sorted_docs = sorted(zip(docs, scores, strict=False), key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in sorted_docs[:top_k]]
