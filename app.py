from fastapi import FastAPI
from llm import generate, expand_query
from retriever import hybrid_retrieve
from rerank import rerank

app = FastAPI()

def ask(query):
    try:
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

        return generate(prompt)

    except Exception as e:
        return f"系统错误: {str(e)}"


@app.get("/ask")
def ask_api(q: str):
    return {"answer": ask(q)}