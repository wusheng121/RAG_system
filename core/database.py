"""数据库引擎、会话工厂、声明式基类与依赖。

import 本模块不再有任何写库副作用（建表/播种统一移到 app 的 startup）。
"""

from collections.abc import Iterator

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from core.config import config

# SQLite 需要 check_same_thread=False 才能在 FastAPI 线程池中使用
_connect_args = {"check_same_thread": False} if config.DATABASE_URL.startswith("sqlite") else {}

engine = create_engine(config.DATABASE_URL, connect_args=_connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    """所有 ORM 模型的声明式基类（SQLAlchemy 2.0 风格）。"""


def get_db() -> Iterator[Session]:
    """FastAPI 依赖：每个请求一个会话，请求结束自动关闭。"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """幂等建表：仅在表不存在时创建。"""
    # 导入 models 以确保所有模型已注册到 Base.metadata
    import models  # noqa: F401  (避免循环导入：此处延迟导入)

    Base.metadata.create_all(bind=engine)
