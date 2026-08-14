"""RAG 检索单测：不依赖 LLM，验证 BM25+向量+RRF 的召回与分块逻辑。

CI 无向量模型缓存时，向量加载失败自动降级为仅 BM25，断言仍成立
（BM25 经 jieba 分词后对中文关键词召回正确）。
"""

import pytest

from rag.retriever import Retriever


@pytest.fixture(scope="module")
def retriever():
    """模块级共享：避免每个用例重复加载向量模型。"""
    return Retriever()


def test_split_text_produces_semantic_chunks(retriever):
    """分块应按 【...】 段落切分，每块以标题开头，数量充足。"""
    assert len(retriever.docs) >= 10, f"段落数偏少: {len(retriever.docs)}"
    assert all(c.startswith("【") for c in retriever.docs), "存在非段落开头的块"


def test_hybrid_retrieve_returns_relevant_payment(retriever):
    """支付类问题应召回含‘支付/支付宝’的段落。"""
    docs = retriever.hybrid_retrieve("支持哪些支付方式", top_k=3)
    assert docs, "应返回非空结果"
    joined = "\n".join(docs)
    assert "支付" in joined or "支付宝" in joined


def test_hybrid_retrieve_returns_relevant_delivery(retriever):
    """配送类问题应召回含‘配送/送达/工作日’的段落。"""
    docs = retriever.hybrid_retrieve("多久能送到", top_k=3)
    assert docs, "应返回非空结果"
    joined = "\n".join(docs)
    assert "送达" in joined or "工作日" in joined or "配送" in joined


def test_hybrid_retrieve_dedups(retriever):
    """多路召回后应去重（返回条数 <= top_k）。"""
    docs = retriever.hybrid_retrieve("退换货规则", top_k=3)
    assert len(docs) <= 3
    assert len(docs) == len(set(docs)), "存在重复段落"
