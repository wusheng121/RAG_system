import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from rerank import rerank
from rank_bm25 import BM25Okapi
from fastapi import FastAPI
import os
import jieba
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_TOKEN"] = "hf_sOqYzvXYMEjDIpXlVQLPBdBamelAEKmTEWhf_sOqYzvXYMEjDIpXlVQLPBdBamelAEKmTEW"

# ========== 1. 初始化 ==========
app = FastAPI()
@app.get("/ask")
def ask_api(q: str):
    return {"answer": ask(q)}

model = SentenceTransformer("BAAI/bge-small-en")

client = OpenAI(
    # api_key="sk-c8f56d77ce3345458bdf55f3c3cd2f57",
    api_key=os.getenv("OPENAI_API_KEY"),
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

    queries = [q.strip() for q in result.split("\n") if q.strip()]
    return queries[:3]
# ========== 2. 读取文档 ==========
# def load_docs(path):
#     with open(path, "r", encoding="utf-8") as f:
#         return split_text(f.read())
# docs = load_docs("data/docs.txt")

def split_text(text, chunk_size=200, overlap=50):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap

    return chunks

raw_text = open("data/docs.txt", encoding="utf-8").read()
docs = split_text(raw_text)

# 分词（简单版）
tokenized_docs = [list(jieba.cut(doc)) for doc in docs]
bm25 = BM25Okapi(tokenized_docs)

# BM25检索函数
def bm25_retrieve(query, top_k=3):
    tokenized_query = list(jieba.cut(query))

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
faiss.write_index(index, "index.faiss")

# ========== 4. 检索 ==========
def retrieve(query, top_k=3):
    q_vec = model.encode([query], normalize_embeddings=True)
    D, I = index.search(np.array(q_vec), top_k)

    results = [docs[i] for i in I[0]]

    print("\n【检索到的内容】")
    for r in results:
        print("-", r)

    return results

def hybrid_retrieve(query):
    # 向量检索
    vec_docs = retrieve(query, top_k=3)
    # BM25 检索
    bm_docs = bm25_retrieve(query, top_k=3)
    # 合并 + 去重
    combined = list(set(vec_docs + bm_docs))

    return combined

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
你是一个严谨的问答系统，请严格根据提供的内容回答。

【已知信息】
{context}

【问题】
{query}

【回答要求】
1. 只能使用已知信息
2. 不要编造
3. 信息不足时说“无法确定”
"""
    try:
        return generate(prompt)
    except Exception as e:
        return f"系统错误: {str(e)}"

# ========== 7. 测试 ==========
# if __name__ == "__main__":
    # while True:
    #     q = input("请输入问题：")
    #     print(ask(q))

