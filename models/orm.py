"""ORM 数据模型。

字段与历史结构保持一致：Order 同时兼容新旧字段（order_no/book_title/amount/
created_at），由 order_service 在运行时按表实际列做兼容读写。
"""

from datetime import datetime

from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, String

from core.database import Base


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)


class Book(Base):
    __tablename__ = "books"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String)
    author = Column(String)
    isbn = Column(String)
    price = Column(Float)
    stock = Column(Integer)


class Order(Base):
    __tablename__ = "orders"

    id = Column(Integer, primary_key=True, index=True)
    order_no = Column(String, unique=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    book_title = Column(String)
    amount = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)


class CartItem(Base):
    __tablename__ = "cart_items"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), index=True)
    book_id = Column(Integer, ForeignKey("books.id"), index=True)
    title = Column(String)
    price = Column(Float)
    qty = Column(Integer, default=1)
    created_at = Column(DateTime, default=datetime.utcnow)
