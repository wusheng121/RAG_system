from sentence_transformers import CrossEncoder

# 加载 reranker 模型（非常稳定）
model = CrossEncoder("BAAI/bge-reranker-base")

def rerank(query, docs, top_k=2):
    pairs = [[query, doc] for doc in docs]

    scores = model.predict(pairs)

    # 排序
    sorted_docs = sorted(
        zip(docs, scores),
        key=lambda x: x[1],
        reverse=True
    )

    return [doc for doc, _ in sorted_docs[:top_k]]