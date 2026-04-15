import faiss
import numpy as np
import jieba
import os
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

class Retriever:
    def __init__(self):
        with open("data/docs.txt", encoding="utf-8") as f:
            raw_text = f.read()
        self.docs = self._split_text(raw_text)
        self.model = None
        self.index = None
        self.bm25 = None
        self._initialized = False
        self._vector_available = False
        # 默认仅使用本地缓存模型，避免联网校验导致SSL问题。
        self.local_files_only = os.getenv("HF_LOCAL_FILES_ONLY", "1") == "1"

    def _initialize(self):
        if not self._initialized:
            tokenized_docs = [list(jieba.cut(doc)) for doc in self.docs]
            self.bm25 = BM25Okapi(tokenized_docs)
            try:
                self.model = SentenceTransformer("BAAI/bge-small-en", local_files_only=self.local_files_only)
                embeddings = self.model.encode(self.docs, normalize_embeddings=True)
                dim = embeddings.shape[1]
                self.index = faiss.IndexFlatIP(dim)
                self.index.add(np.array(embeddings))
                self._vector_available = True
            except Exception:
                # 网络/证书问题下保持服务可用，降级到BM25。
                self.model = None
                self.index = None
                self._vector_available = False
            self._initialized = True

    def _split_text(self, text, chunk_size=200, overlap=50):
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunks.append(text[start:end])
            start += chunk_size - overlap
        return chunks

    def vector_retrieve(self, query, top_k=3):
        self._initialize()
        if not self._vector_available or self.model is None or self.index is None:
            return []
        q_vec = self.model.encode([query], normalize_embeddings=True)
        D, I = self.index.search(np.array(q_vec), top_k)
        return [self.docs[i] for i in I[0]]

    def bm25_retrieve(self, query, top_k=3):
        self._initialize()
        if self.bm25 is None:
            return self.docs[:top_k]
        tokenized_query = list(jieba.cut(query))
        scores = self.bm25.get_scores(tokenized_query)
        ranked = sorted(zip(self.docs, scores), key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in ranked[:top_k]]

    def hybrid_retrieve(self, query):
        self._initialize()
        vec_docs = self.vector_retrieve(query, top_k=3)
        bm_docs = self.bm25_retrieve(query, top_k=3)
        docs = list(dict.fromkeys(vec_docs + bm_docs))
        return docs if docs else self.docs[:3]
