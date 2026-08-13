"""认证与上下文：密码哈希、JWT、当前用户依赖、模板上下文工具。"""

from datetime import datetime, timedelta
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from fastapi import Depends, Request
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy.orm import Session

from core.config import config
from core.database import get_db
from models import User

pwd_context = CryptContext(schemes=["argon2"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token", auto_error=False)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)


def authenticate_user(db: Session, username: str, password: str):
    user = db.query(User).filter(User.username == username).first()
    if not user or not verify_password(password, user.hashed_password):
        return False
    return user


def create_access_token(data: dict, expires_delta: timedelta | None = None) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=config.ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, config.SECRET_KEY, algorithm=config.ALGORITHM)


def get_user_by_token(token: str | None, db: Session):
    if not token:
        return None
    try:
        payload = jwt.decode(token, config.SECRET_KEY, algorithms=[config.ALGORITHM])
        username = payload.get("sub")
        if not username:
            return None
        return db.query(User).filter(User.username == username).first()
    except JWTError:
        return None


def get_current_user(
    request: Request,
    token: str | None = Depends(oauth2_scheme),
    db: Session = Depends(get_db),
):
    """同时支持 Authorization 头与 HttpOnly Cookie 中的 access_token。"""
    cookie_token = request.cookies.get("access_token")
    final_token = token or cookie_token
    return get_user_by_token(final_token, db)


def set_access_token_cookie(response, token: str) -> None:
    """统一写入访问令牌 Cookie：HttpOnly + SameSite=Lax + 限时 + 生产 HTTPS-only。

    - httponly：防 JS 读取（XSS 偷取令牌）
    - samesite=lax：防跨站提交（CSRF 的浏览器层缓解）
    - max_age：与 JWT 过期一致，到期自动失效
    - secure：仅生产（HTTPS）发送，开发 http 下置 False 保证可用
    """
    response.set_cookie(
        key="access_token",
        value=token,
        httponly=True,
        samesite="lax",
        max_age=config.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        secure=config.IS_PRODUCTION,
    )


def build_context(request: Request, db: Session, **kwargs) -> dict:
    """构造模板上下文：注入当前用户与弹出消息。"""
    context = {
        "request": request,
        "current_user": get_user_by_token(request.cookies.get("access_token"), db),
    }
    msg = request.query_params.get("msg")
    if msg:
        context["popup_message"] = msg
    context.update(kwargs)
    return context


def append_msg_to_url(url: str, msg: str) -> str:
    """在 URL 上追加 msg 查询参数，用于重定向后展示一次性消息。"""
    parts = urlsplit(url)
    query = dict(parse_qsl(parts.query, keep_blank_values=True))
    query["msg"] = msg
    return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(query), parts.fragment))
