"""示例数据播种：首次启动时插入示例书籍，幂等。"""

from core.database import SessionLocal
from models import Book

# 纯字典而非 ORM 实例：避免模块级实例被会话绑定后变 detached
# （init_sample_books 添加并提交、会话关闭后，模块级实例再被读取会触发 DetachedInstanceError）。
SAMPLE_BOOKS = [
    {"title": "Python编程入门", "author": "张三", "isbn": "1234567890", "price": 39.99, "stock": 10},
    {"title": "机器学习基础", "author": "李四", "isbn": "0987654321", "price": 59.99, "stock": 5},
    {"title": "Web开发实战", "author": "王五", "isbn": "1122334455", "price": 49.99, "stock": 8},
    {"title": "老人与海", "author": "海明威", "isbn": "9787544774332", "price": 29.80, "stock": 12},
    {"title": "活着", "author": "余华", "isbn": "9787506365437", "price": 39.00, "stock": 7},
    {"title": "三体", "author": "刘慈欣", "isbn": "9787536692930", "price": 58.00, "stock": 6},
]


def init_sample_books() -> None:
    """仅插入库中尚不存在的示例书籍，重复执行安全。"""
    db = SessionLocal()
    try:
        existing_titles = {row[0] for row in db.query(Book.title).all()}
        added = False
        for data in SAMPLE_BOOKS:
            if data["title"] not in existing_titles:
                db.add(Book(**data))  # 每次构造新实例，避免共享 detached 对象
                added = True
        if added:
            db.commit()
    finally:
        db.close()
