"""混合检索：BM25 + 中文向量（FAISS），结果用 RRF 倒数排名融合。

相比旧版的改进：
1. 向量模型 BAAI/bge-small-en → bge-small-zh-v1.5（支持中文语料）。
2. 分块：固定字符滑窗 → 按 【...】 段落语义切分。
3. 混合：拼接去重 → RRF（Reciprocal Rank Fusion）按排名融合。
4. embedding/索引持久化到 CACHE_DIR，避免每次重启重算。
5. 语料路径按工程根目录解析，不再依赖运行时 CWD。
"""

import hashlib
import json
import logging
from pathlib import Path

import faiss
import jieba
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from core.config import config

logger = logging.getLogger("rag_bookstore.retriever")

# RRF 常数：排名 r 的得分 = 1 / (k + r + 1)，k=60 是社区经验默认值
RRF_K = 60


class Retriever:
    def __init__(self):
        # 本模块位于 rag/，项目根需上溯一级到 parent.parent
        root = Path(__file__).resolve().parent.parent
        self.docs_path = root / "data" / "docs.txt"
        self.cache_dir = root / config.CACHE_DIR
        raw_text = self.docs_path.read_text(encoding="utf-8")
        self.docs = self._split_text(raw_text)
        self.source_hash = hashlib.md5(raw_text.encode("utf-8")).hexdigest()

        self.model: SentenceTransformer | None = None
        self.index = None
        self.bm25 = None
        self._initialized = False
        self._vector_available = False

    # ---- 初始化（惰性，首次检索时触发）----
    def _initialize(self):
        if self._initialized:
            return
        # BM25 始终可用（纯内存，无需模型）
        tokenized_docs = [list(jieba.cut(doc)) for doc in self.docs]
        self.bm25 = BM25Okapi(tokenized_docs)

        try:
            self.model = SentenceTransformer(
                config.EMBEDDING_MODEL, local_files_only=config.HF_LOCAL_FILES_ONLY
            )
            if self._load_cache():
                self._vector_available = True
            else:
                self._build_index()
                self._save_cache()
                self._vector_available = True
        except Exception as e:
            # 模型未缓存/网络不可用 → 降级到仅 BM25，保证服务可用
            logger.warning("向量模型 %s 加载失败，降级为仅 BM25 检索: %s", config.EMBEDDING_MODEL, e)
            self.model = None
            self.index = None
            self._vector_available = False
        self._initialized = True

    def _build_index(self):
        embeddings = self.model.encode(self.docs, normalize_embeddings=True)
        emb = np.array(embeddings, dtype="float32")
        dim = emb.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(emb)

    # ---- 持久化 ----
    def _cache_files(self):
        return (
            self.cache_dir / "docs.json",
            self.cache_dir / "faiss.index",
            self.cache_dir / "source.hash",
        )

    def _load_cache(self) -> bool:
        docs_file, index_file, hash_file = self._cache_files()
        if not (docs_file.exists() and index_file.exists() and hash_file.exists()):
            return False
        try:
            if hash_file.read_text(encoding="utf-8").strip() != self.source_hash:
                return False  # 语料已变更，缓存失效
            self.docs = json.loads(docs_file.read_text(encoding="utf-8"))
            self.index = faiss.read_index(str(index_file))
            return True
        except Exception:
            return False

    def _save_cache(self) -> None:
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            docs_file, index_file, hash_file = self._cache_files()
            docs_file.write_text(json.dumps(self.docs, ensure_ascii=False), encoding="utf-8")
            faiss.write_index(self.index, str(index_file))
            hash_file.write_text(self.source_hash, encoding="utf-8")
        except Exception:
            # 缓存写失败不影响检索，仅丢失持久化收益
            pass

    # ---- 分块：按 【...】 段落 ----
    def _split_text(self, text: str) -> list[str]:
        """按章节标题（【...】）切分，每章一块，语义连贯优于定长滑窗。"""
        chunks: list[str] = []
        current: list[str] = []
        for line in text.split("\n"):
            if line.startswith("【") and "】" in line:
                if current:
                    chunks.append("\n".join(current).strip())
                current = [line]
            else:
                current.append(line)
        if current:
            chunks.append("\n".join(current).strip())
        return [c for c in chunks if c]

    # ---- 单路检索 ----
    def vector_retrieve(self, query: str, top_k: int = 5) -> list[str]:
        self._initialize()
        if not self._vector_available or self.model is None or self.index is None:
            return []
        q_vec = self.model.encode([query], normalize_embeddings=True)
        _distances, indices = self.index.search(np.array(q_vec, dtype="float32"), top_k)
        return [self.docs[i] for i in indices[0] if 0 <= i < len(self.docs)]

    def bm25_retrieve(self, query: str, top_k: int = 5) -> list[str]:
        self._initialize()
        if self.bm25 is None:
            return self.docs[:top_k]
        tokenized_query = list(jieba.cut(query))
        scores = self.bm25.get_scores(tokenized_query)
        ranked = sorted(zip(self.docs, scores, strict=False), key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in ranked[:top_k]]

    # ---- 混合检索：RRF 融合 ----
    def hybrid_retrieve(self, query: str, top_k: int = 5) -> list[str]:
        """向量 + BM25 各取 top_k，按 RRF 融合排名后返回前 top_k 条。"""
        self._initialize()
        vec_docs = self.vector_retrieve(query, top_k=top_k)
        bm_docs = self.bm25_retrieve(query, top_k=top_k)

        scores: dict[str, float] = {}
        for rank, doc in enumerate(vec_docs):
            scores[doc] = scores.get(doc, 0.0) + 1.0 / (RRF_K + rank + 1)
        for rank, doc in enumerate(bm_docs):
            scores[doc] = scores.get(doc, 0.0) + 1.0 / (RRF_K + rank + 1)

        ranked = sorted(scores, key=scores.get, reverse=True)
        return ranked[:top_k] if ranked else self.docs[:top_k]
