import faiss
import numpy as np
import jieba
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

# ========== 文档处理 ==========
def split_text(text, chunk_size=200, overlap=50):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap

    return chunks


# ========== 加载数据 ==========
raw_text = open("data/docs.txt", encoding="utf-8").read()
docs = split_text(raw_text)

# ========== Embedding ==========
model = SentenceTransformer("BAAI/bge-small-en")
embeddings = model.encode(docs, normalize_embeddings=True)

dim = embeddings.shape[1]
index = faiss.IndexFlatIP(dim)
index.add(np.array(embeddings))


# ========== BM25 ==========
tokenized_docs = [list(jieba.cut(doc)) for doc in docs]
bm25 = BM25Okapi(tokenized_docs)


# ========== 向量检索 ==========
def vector_retrieve(query, top_k=3):
    q_vec = model.encode([query], normalize_embeddings=True)
    D, I = index.search(np.array(q_vec), top_k)

    return [docs[i] for i in I[0]]


# ========== BM25检索 ==========
def bm25_retrieve(query, top_k=3):
    tokenized_query = list(jieba.cut(query))
    scores = bm25.get_scores(tokenized_query)

    ranked = sorted(
        zip(docs, scores),
        key=lambda x: x[1],
        reverse=True
    )

    return [doc for doc, _ in ranked[:top_k]]


# ========== Hybrid ==========
def hybrid_retrieve(query):
    vec_docs = vector_retrieve(query, top_k=3)
    bm_docs = bm25_retrieve(query, top_k=3)

    return list(set(vec_docs + bm_docs))