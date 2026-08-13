"""Pydantic 请求/响应模型（接口契约）。"""

from pydantic import BaseModel


class UserCreate(BaseModel):
    username: str
    email: str
    password: str


class Token(BaseModel):
    access_token: str
    token_type: str
