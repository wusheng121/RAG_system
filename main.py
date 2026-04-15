# main.py
from llm import LLM
from retriever import Retriever
from rerank import Reranker

# Initialize components
llm = LLM()
retriever = Retriever()
reranker = Reranker()

def ask(query):
    """
    核心问答函数：
    - Multi-query 扩展
    - 向量 + BM25 检索
    - Rerank
    - 构造 Prompt 给 LLM
    """
    try:
        # 1️⃣ 扩展 query
        queries = llm.expand_query(query)
        all_docs = []

        for q in queries:
            docs = retriever.hybrid_retrieve(q)
            all_docs.extend(docs)

        # 去重
        all_docs = list(set(all_docs))

        # 2️⃣ rerank
        reranked = reranker.rerank(query, all_docs, top_k=3)

        # 3️⃣ 构造上下文
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

        # 4️⃣ LLM 生成
        return llm.generate(prompt)

    except Exception as e:
        return f"系统错误: {str(e)}"


if __name__ == "__main__":
    print("=== LLM + RAG 问答系统 ===")
    print("输入问题，按回车查看答案，输入 exit 退出")
    while True:
        query = input("\n请输入问题：")
        if query.lower() in ("exit", "quit"):
            break
        answer = ask(query)
        print("\n【回答】")
        print(answer)