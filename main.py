import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from rerank import rerank
from rank_bm25 import BM25Okapi
import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_TOKEN"] = "hf_sOqYzvXYMEjDIpXlVQLPBdBamelAEKmTEWhf_sOqYzvXYMEjDIpXlVQLPBdBamelAEKmTEW"

# ========== 1. 初始化 ==========
model = SentenceTransformer("BAAI/bge-small-en")

client = OpenAI(
    api_key="sk-c8f56d77ce3345458bdf55f3c3cd2f57",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)
# query扩展函数
def expand_query(query):
    prompt = f"""
请对下面的问题生成3个不同表达方式：

问题：{query}

输出：
"""
    result = generate(prompt)

    return result.split("\n")
# ========== 2. 读取文档 ==========
def load_docs(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read().split("\n")

docs = load_docs("data/docs.txt")

# 分词（简单版）
tokenized_docs = [doc.split() for doc in docs]
bm25 = BM25Okapi(tokenized_docs)

# BM25检索函数
def bm25_retrieve(query, top_k=3):
    tokenized_query = query.split()

    scores = bm25.get_scores(tokenized_query)

    ranked = sorted(
        zip(docs, scores),
        key=lambda x: x[1],
        reverse=True
    )

    return [doc for doc, _ in ranked[:top_k]]
# ========== 3. 向量化 ==========
embeddings = model.encode(docs, normalize_embeddings=True)

dim = embeddings.shape[1]
index = faiss.IndexFlatIP(dim)
index.add(np.array(embeddings))

# ========== 4. 检索 ==========
def retrieve(query, top_k=3):
    q_vec = model.encode([query], normalize_embeddings=True)
    D, I = index.search(np.array(q_vec), top_k)
    return [docs[i] for i in I[0]]

def hybrid_retrieve(query):
    # 向量检索
    vec_docs = retrieve(query, top_k=3)
    # BM25 检索
    bm_docs = bm25_retrieve(query, top_k=3)
    # 合并 + 去重
    combined = list(set(vec_docs + bm_docs))

    return combined

def multi_retrieve(query):
    queries = expand_query(query)

    all_docs = []

    for q in queries:
        docs = retrieve(q, top_k=3)
        all_docs.extend(docs)

    # 去重
    return list(set(all_docs))

# ========== 5. LLM ==========
def generate(prompt):
    response = client.chat.completions.create(
        model="qwen-plus",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content

# ========== 6. RAG ==========
def ask(query):
    # Multi-query
    queries = expand_query(query)

    all_docs = []

    for q in queries:
        docs = hybrid_retrieve(q)
        all_docs.extend(docs)

    # 去重
    all_docs = list(set(all_docs))

    # rerank
    reranked = rerank(query, all_docs, top_k=3)

    print("\n【Hybrid + MultiQuery + Rerank】")
    for d in reranked:
        print("-", d)

    context = "\n".join(reranked)

    prompt = f"""
基于以下信息回答问题：

{context}

问题：{query}
"""
    return generate(prompt)

# ========== 7. 测试 ==========
if __name__ == "__main__":
    while True:
        q = input("请输入问题：")
        print(ask(q))