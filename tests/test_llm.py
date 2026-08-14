"""LLM 链路测试：需真实 ALI_API_KEY 且 LLM 实际可用。

设计：
- 无 ALI_API_KEY（如 CI）→ 整组 skip（pytestmark）。
- 有 key 但 LLM 不可用（额度耗尽 / 网络 / 限流）→ canary 探测后整组 skip，不误判为失败。
- LLM 真正可用时 → 跑 expand_query / generate / 完整 RAG+LLM 链路的真实断言。
"""

import pytest

from core.config import config
from rag.llm import LLM

# 无 key（CI）直接跳过
pytestmark = pytest.mark.skipif(not config.ALI_API_KEY, reason="需配置 ALI_API_KEY 才跑 LLM 测试")


def _is_real_answer(text: str) -> bool:
    """区分“真实生成”与“降级/失败”返回。"""
    return bool(text) and not text.startswith("LLM") and "调用失败" not in text


@pytest.fixture(scope="module", autouse=True)
def _require_working_llm():
    """canary：先试调一次，LLM 不可用则整组 skip（避免把额度/网络问题误判为测试失败）。"""
    out = LLM().generate("ping")
    if not _is_real_answer(out):
        pytest.skip(f"LLM 不可用（可能额度耗尽/网络/限流）: {out[:80]}")


def test_llm_generate_returns_text():
    """generate 应回到非空、非降级文本。"""
    out = LLM().generate("用一句话介绍智阅书店的客服")
    assert _is_real_answer(out), f"LLM 降级: {out}"


def test_llm_expand_query_returns_list():
    """expand_query 应回到非空查询列表。"""
    queries = LLM().expand_query("支持哪些支付方式")
    assert isinstance(queries, list) and len(queries) >= 1
    assert "支付" in "\n".join(queries)


def test_ask_hits_full_rag_llm_chain():
    """知识库型问题（无 db 跳过直答）应走完整 RAG+LLM，返回有意义回答。"""
    from services.rag_service import ask

    answer = ask("支持哪些支付方式")  # direct_db_answer 在 db=None 时返回 None → 走 RAG+LLM
    assert _is_real_answer(answer), f"未真正生成: {answer}"
    assert answer != "抱歉，我无法回答这个问题"
    assert any(k in answer for k in ["支付", "支付宝", "微信", "银行卡"]), f"回答未涉及支付: {answer}"


def test_ask_kb_delivery_question():
    """配送类问题同样走 RAG+LLM，回答应涉及配送。"""
    from services.rag_service import ask

    answer = ask("多久能送到")
    assert _is_real_answer(answer), f"未真正生成: {answer}"
    assert any(k in answer for k in ["工作日", "送达", "配送", "偏远"]), f"回答未涉及配送: {answer}"
