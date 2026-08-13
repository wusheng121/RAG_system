"""高频问题数据库直答策略。

对库存/购物车/订单/价格等事实型问题，直接查库返回确定性结果，
降低模型幻觉与接口失败率。未命中时返回 None，交给 RAG 兜底。
"""

import re

from sqlalchemy.orm import Session

from core.utils import format_money
from models import Book, CartItem
from services.order_service import load_orders_for_user


def _clean_text(text: str) -> str:
    return (text or "").strip().strip("  \t\n\r。？?！!；;：:《》\"'“”")


def _extract_book_title_hint(question: str) -> str:
    text = _clean_text(question)
    quoted = re.search(r"《([^》]+)》", text)
    if quoted:
        return quoted.group(1).strip()
    prefixes = [
        "你们店有没有",
        "你们有没有",
        "店里有没有",
        "请问有没有",
        "请问有卖",
        "有没有",
        "有卖",
        "有无",
        "是否有",
        "店里有",
        "书店有没有",
    ]
    for prefix in prefixes:
        if text.startswith(prefix):
            text = text[len(prefix) :].strip()
            break
    suffixes = [
        "有库存吗",
        "还有库存吗",
        "库存多少",
        "库存还有多少",
        "库存还剩多少",
        "有货吗",
        "还有货吗",
        "在售吗",
        "卖吗",
        "能买吗",
        "还有吗",
        "有吗",
        "在吗",
        "在不在",
        "吗",
        "库存",
    ]
    for suffix in suffixes:
        if text.endswith(suffix):
            return text[: -len(suffix)].strip("  \t\n\r。？?！!；;：:《》\"'“”")
    if "的库存" in text:
        return text.split("的库存", 1)[0].strip("  \t\n\r。？?！!；;：:《》\"'“”")
    return text


def _find_book_by_hint(db: Session, hint: str):
    hint = (hint or "").strip()
    if not hint:
        return None
    q = db.query(Book)
    exact = q.filter(Book.title == hint).first()
    if exact:
        return exact
    exact_isbn = q.filter(Book.isbn == hint).first()
    if exact_isbn:
        return exact_isbn
    return (
        q.filter(Book.title.contains(hint)).first()
        or q.filter(Book.author.contains(hint)).first()
        or q.filter(Book.isbn.contains(hint)).first()
    )


def direct_db_answer(question: str, user=None, db: Session = None):
    """命中高频问题则返回字符串答案，否则返回 None。"""
    if not db:
        return None

    q = _clean_text(question)
    if not q:
        return None

    cart_keywords = [
        "我购物车里有什么",
        "购物车里有什么",
        "购物车有什么",
        "我的购物车",
        "查看购物车",
        "购物车内容",
    ]
    if any(keyword in q for keyword in cart_keywords):
        if not user:
            return "请先登录后查看购物车。"
        rows = db.query(CartItem).filter(CartItem.user_id == user.id).order_by(CartItem.id.asc()).all()
        if not rows:
            return "你的购物车里还没有商品。"
        total = 0.0
        lines = [f"你的购物车里有 {len(rows)} 件商品："]
        for index, item in enumerate(rows, 1):
            subtotal = float(item.price or 0) * int(item.qty or 0)
            total += subtotal
            lines.append(
                f"{index}. {item.title} × {item.qty}，单价{format_money(item.price)}，小计{format_money(subtotal)}"
            )
        lines.append(f"购物车合计：{format_money(total)}")
        return "\n".join(lines)

    order_keywords = ["订单有哪些", "有哪些订单", "我的订单", "订单列表", "查看订单", "订单记录", "订单明细"]
    if any(keyword in q for keyword in order_keywords):
        if not user:
            return "请先登录后查看订单。"
        orders = load_orders_for_user(db, user.id)
        if not orders:
            return "你当前还没有订单记录。"
        lines = [f"你当前共有 {len(orders)} 笔订单："]
        for index, order in enumerate(orders, 1):
            created_at = order.get("created_at")
            created_text = (
                created_at.strftime("%Y-%m-%d %H:%M") if hasattr(created_at, "strftime") else str(created_at)
            )
            amount_text = (
                format_money(order.get("amount"))
                if order.get("amount") not in (None, "", "NULL")
                else "金额未知"
            )
            lines.append(
                f"{index}. 订单号：{order.get('order_no', '未知')}｜书名：{order.get('book_title', '未知')}"
                f"｜金额：{amount_text}｜时间：{created_text}"
            )
        return "\n".join(lines)

    price_keywords = ["最便宜", "最低价", "最低的书", "价格最低", "最贵", "最高价", "最高的书", "价格最高"]
    if any(keyword in q for keyword in price_keywords):
        books = db.query(Book).all()
        if not books:
            return "当前暂无在售书籍。"
        if any(keyword in q for keyword in ["最便宜", "最低价", "最低的书", "价格最低"]):
            target_price = min(float(book.price or 0) for book in books)
            matched = [book for book in books if float(book.price or 0) == target_price]
            prefix = "最便宜的书"
        else:
            target_price = max(float(book.price or 0) for book in books)
            matched = [book for book in books if float(book.price or 0) == target_price]
            prefix = "最贵的书"
        lines = [f"当前{prefix}价格为 {format_money(target_price)}，对应书籍如下："]
        for book in matched:
            lines.append(f"- {book.title}｜作者：{book.author}｜库存：{book.stock}")
        return "\n".join(lines)

    title_hint = _extract_book_title_hint(q)
    stock_keywords = [
        "库存",
        "有库存",
        "还有库存",
        "有货吗",
        "还有货吗",
        "在售吗",
        "卖吗",
        "有没有",
        "是否有",
        "在不在",
        "有卖",
        "这本书有吗",
    ]
    negative_stock_keywords = ["没有了", "没货", "缺货", "卖完", "断货", "下架", "不在", "未上架"]
    if any(keyword in q for keyword in negative_stock_keywords) or any(
        keyword in q for keyword in stock_keywords
    ):
        book = _find_book_by_hint(db, title_hint)
        if book:
            if int(book.stock or 0) > 0:
                return f"《{book.title}》当前有库存，剩余 {book.stock} 本。"
            return f"《{book.title}》当前暂无库存。"
        if title_hint:
            return f"目前书店未找到《{title_hint}》，如果你想查库存，可以告诉我更准确的书名或作者。"
        if any(keyword in q for keyword in negative_stock_keywords):
            return "目前书店里没有找到你提到的这本书。"
        return "请告诉我更准确的书名，我可以帮你查库存。"

    book_list_keywords = ["有哪些书", "有什么书", "在售书", "图书列表", "所有书", "书目", "全部书"]
    if any(keyword in q for keyword in book_list_keywords):
        books = db.query(Book).order_by(Book.id.asc()).all()
        if not books:
            return "当前暂无在售书籍。"
        lines = [f"当前在售书籍共 {len(books)} 本："]
        for book in books[:20]:
            lines.append(
                f"- {book.title}｜作者：{book.author}｜价格：{format_money(book.price)}｜库存：{book.stock}"
            )
        if len(books) > 20:
            lines.append(f"- 还有 {len(books) - 20} 本，请在搜索页继续查看。")
        return "\n".join(lines)

    return None
