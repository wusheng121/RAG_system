"""pytest 全局夹具：测试间数据库隔离。

策略：每个用例结束后清空 users/cart_items/orders 并把示例书籍库存重置回种子值，
保证用例之间互不污染（前一个用例的结账扣库存、注册用户不会影响下一个）。

注：理想方案是“每用例一个事务最后回滚”，但 FastAPI 的 TestClient 会把同步端点
放到线程池执行，单一连接跨线程 + SQLite 易触发 database is locked；故采用更稳妥的
“重置到已知状态”方案，效果等价、可靠性更高。
"""

import pytest
from sqlalchemy import text

from core.database import SessionLocal
from core.seed import SAMPLE_BOOKS
from models import Book


@pytest.fixture(autouse=True)
def _reset_db():
    """每个用例结束后清理动态数据、恢复库存。"""
    yield
    db = SessionLocal()
    try:
        db.execute(text("DELETE FROM cart_items"))
        db.execute(text("DELETE FROM orders"))
        db.execute(text("DELETE FROM users"))
        for sample in SAMPLE_BOOKS:
            book = db.query(Book).filter(Book.title == sample.title).first()
            if book:
                book.stock = sample.stock
                book.price = sample.price
        db.commit()
    except Exception:
        db.rollback()
    finally:
        db.close()
