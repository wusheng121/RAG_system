"""订单服务：运行时按表实际列做历史库兼容读写。

旧库可能含 product/status/order_date 等字段，新库用 order_no/book_title/
amount/created_at；此处统一探测列名后构造 SQL，保证两种结构都能用。
"""

import uuid
from datetime import datetime

from sqlalchemy import inspect, text
from sqlalchemy.orm import Session


def get_table_columns(db: Session, table_name: str) -> set:
    """运行时读取真实列名，兼容历史 SQLite 表结构。"""
    try:
        inspector = inspect(db.bind)
        names = inspector.get_table_names()
        if table_name not in names:
            return set()
        return {col.get("name") for col in inspector.get_columns(table_name)}
    except Exception:
        return set()


def load_orders_for_user(db: Session, user_id: int) -> list[dict]:
    """读取某用户全部订单，按实际列做兼容映射。"""
    cols = get_table_columns(db, "orders")
    if not cols:
        return []

    order_no_expr = "order_no" if "order_no" in cols else "CAST(id AS TEXT)"
    title_expr = "book_title" if "book_title" in cols else ("product" if "product" in cols else "''")
    amount_expr = "amount" if "amount" in cols else "NULL"
    time_expr = "created_at" if "created_at" in cols else ("order_date" if "order_date" in cols else "NULL")
    status_expr = "status" if "status" in cols else "''"
    order_by_expr = "created_at" if "created_at" in cols else ("order_date" if "order_date" in cols else "id")

    sql = text(
        f"""
        SELECT
            {order_no_expr} AS order_no,
            {title_expr} AS book_title,
            {amount_expr} AS amount,
            {time_expr} AS created_at,
            {status_expr} AS status
        FROM orders
        WHERE user_id = :user_id
        ORDER BY {order_by_expr} DESC, id DESC
        """
    )
    try:
        return [dict(row) for row in db.execute(sql, {"user_id": user_id}).mappings().all()]
    except Exception:
        return []


def create_orders_from_cart(db: Session, user_id: int, cart_rows) -> int:
    """按购物车逐项生成订单行（兼容新旧列），返回创建条数。"""
    cols = get_table_columns(db, "orders")
    if not cols:
        return 0

    created = 0
    now = datetime.utcnow()
    for row in cart_rows:
        item_title = getattr(row, "title", "未知商品")
        item_qty = int(getattr(row, "qty", 1) or 1)
        item_price = float(getattr(row, "price", 0) or 0)
        item_amount = round(item_qty * item_price, 2)

        payload = {"user_id": user_id}
        insert_cols = ["user_id"]
        insert_vals = [":user_id"]

        if "order_no" in cols:
            payload["order_no"] = f"ORD{uuid.uuid4().hex[:10].upper()}"
            insert_cols.append("order_no")
            insert_vals.append(":order_no")
        if "book_title" in cols:
            payload["book_title"] = item_title
            insert_cols.append("book_title")
            insert_vals.append(":book_title")
        if "product" in cols:
            payload["product"] = item_title
            insert_cols.append("product")
            insert_vals.append(":product")
        if "amount" in cols:
            payload["amount"] = item_amount
            insert_cols.append("amount")
            insert_vals.append(":amount")
        if "status" in cols:
            payload["status"] = "已下单"
            insert_cols.append("status")
            insert_vals.append(":status")
        if "created_at" in cols:
            payload["created_at"] = now
            insert_cols.append("created_at")
            insert_vals.append(":created_at")
        if "order_date" in cols:
            payload["order_date"] = now
            insert_cols.append("order_date")
            insert_vals.append(":order_date")

        sql = text(f"INSERT INTO orders ({', '.join(insert_cols)}) VALUES ({', '.join(insert_vals)})")
        db.execute(sql, payload)
        created += 1

    return created
