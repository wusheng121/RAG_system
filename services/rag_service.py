"""RAG 编排：直答优先，未命中走 检索 + 重排 + 生成，任一步失败都降级。

组件惰性构造：import 本模块不会读 docs.txt、不会加载模型。
"""

import logging

from models import Book, CartItem
from rag.llm import LLM
from rag.reranker import Reranker
from rag.retriever import Retriever
from services.direct_answer import direct_db_answer
from services.order_service import load_orders_for_user

logger = logging.getLogger("rag_bookstore.rag")


class _Lazy:
    """惰性单例：首次 .get() 才构造，避免 import 期副作用。"""

    def __init__(self, factory):
        self._factory = factory
        self._obj = None

    def get(self):
        if self._obj is None:
            self._obj = self._factory()
        return self._obj


_llm = _Lazy(LLM)
_retriever = _Lazy(Retriever)
_reranker = _Lazy(Reranker)


def get_llm() -> LLM:
    return _llm.get()


def get_retriever() -> Retriever:
    return _retriever.get()


def get_reranker() -> Reranker:
    return _reranker.get()


def ask(query, user=None, db=None) -> str:
    """核心问答：高频直答 -> 多路检索 -> 重排 -> LLM 生成，全链路降级。"""
    direct_answer = direct_db_answer(query, user=user, db=db)
    if direct_answer:
        return direct_answer

    # 1) Query 扩展（失败回退原 query）
    try:
        queries = get_llm().expand_query(query)
    except Exception as e:
        logger.warning("query 扩展失败，回退原 query: %s", e)
        queries = [query]
    if not queries:
        queries = [query]

    # 2) 多 query 混合检索（单条失败跳过）
    all_docs: list[str] = []
    for q in queries:
        try:
            docs = get_retriever().hybrid_retrieve(q)
            all_docs.extend(docs)
        except Exception as e:
            logger.warning("检索失败，跳过该 query: %s", e)
            continue
    all_docs = list(set(all_docs))

    # 3) 注入数据库动态信息：全量书籍 + 当前用户购物车/订单
    if db:
        try:
            books = db.query(Book).all()
            book_docs = [
                f"书籍信息: 书名={b.title}, 作者={b.author}, ISBN={b.isbn}, 价格={b.price}, 库存={b.stock}"
                for b in books
            ]
            all_docs.extend(book_docs)
        except Exception:
            pass

    if user and db:
        try:
            orders = load_orders_for_user(db, user.id)
            order_docs = []
            for o in orders:
                order_no = o.get("order_no", "未知")
                book_title = o.get("book_title", "未知")
                amount = o.get("amount", "未知")
                created_at = o.get("created_at", "未知")
                order_docs.append(
                    f"订单号: {order_no}, 书籍: {book_title}, 金额: {amount}, 时间: {created_at}"
                )
            all_docs.extend(order_docs)
        except Exception:
            # 兼容历史数据库结构，订单信息不可用时跳过。
            pass

        try:
            cart_rows = db.query(CartItem).filter(CartItem.user_id == user.id).all()
            cart_docs = [f"购物车: 书名={c.title}, 单价={c.price}, 数量={c.qty}" for c in cart_rows]
            all_docs.extend(cart_docs)
        except Exception:
            pass

    if not all_docs:
        all_docs = ["知识库暂时不可用，请稍后重试。"]

    # 4) 重排（失败取前 3）
    try:
        reranked = get_reranker().rerank(query, all_docs, top_k=3)
    except Exception as e:
        logger.warning("重排失败，取前 3 条: %s", e)
        reranked = all_docs[:3]
    context = "\n".join(reranked)
    prompt = (
        "你是一个智能客服，请基于提供的信息回答用户问题。\n"
        f"【知识库信息】\n{context}\n"
        f"【用户问题】\n{query}\n"
        "【回答要求】\n"
        "1. 只能使用知识库信息\n"
        "2. 不要编造\n"
        "3. 信息不足时说“抱歉，我无法回答这个问题”"
    )

    # 5) 生成（失败兜底）
    try:
        return get_llm().generate(prompt)
    except Exception as e:
        logger.warning("LLM 生成失败，返回兜底回答: %s", e)
        return "抱歉，我无法回答这个问题。"
