"""认证相关路由：注册、登录、登出、令牌。"""

from urllib.parse import quote

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import RedirectResponse
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session

from core.database import get_db
from models import Token, User, UserCreate
from security.auth import (
    append_msg_to_url,
    authenticate_user,
    build_context,
    create_access_token,
    get_password_hash,
    set_access_token_cookie,
    verify_password,
)
from security.csrf import verify_csrf_token
from web.templating import templates

router = APIRouter()


@router.post("/register", response_model=Token)
def register(user: UserCreate, db: Session = Depends(get_db)):
    """JSON 注册接口（返回 Token，不写 Cookie）。"""
    if db.query(User).filter(User.username == user.username).first():
        raise HTTPException(status_code=400, detail="用户名已被注册")
    hashed_password = get_password_hash(user.password)
    db_user = User(username=user.username, email=user.email, hashed_password=hashed_password)
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    access_token = create_access_token(data={"sub": user.username})
    return {"access_token": access_token, "token_type": "bearer"}


@router.post("/token", response_model=Token)
def login_token(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    """OAuth2 标准令牌端点（表单）。"""
    user = authenticate_user(db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(status_code=400, detail="用户名或密码错误")
    access_token = create_access_token(data={"sub": user.username})
    return {"access_token": access_token, "token_type": "bearer"}


@router.get("/login")
def login_page(request: Request, db: Session = Depends(get_db)):
    return templates.TemplateResponse(request, "login.html", build_context(request, db))


@router.post("/login")
async def login_page_post(
    request: Request,
    db: Session = Depends(get_db),
    _csrf=Depends(verify_csrf_token),
):
    """已注册即登录，未注册自动注册并登录。"""
    form_data = await request.form()
    username = (form_data.get("username") or "").strip()
    password = (form_data.get("password") or "").strip()
    email = (form_data.get("email") or "").strip()
    next_url = request.query_params.get("next") or "/"

    if not username or not password:
        return templates.TemplateResponse(
            request, "login.html", build_context(request, db, error="用户名和密码不能为空")
        )
    if len(username) < 3:
        return templates.TemplateResponse(
            request, "login.html", build_context(request, db, error="用户名至少3个字符")
        )
    if len(password) < 6:
        return templates.TemplateResponse(
            request, "login.html", build_context(request, db, error="密码至少6个字符")
        )

    user = db.query(User).filter(User.username == username).first()
    login_msg = "登录成功"
    if user:
        if not verify_password(password, user.hashed_password):
            return templates.TemplateResponse(
                request, "login.html", build_context(request, db, error="用户名或密码错误")
            )
    else:
        safe_email = email or f"{username}@local.dev"
        user = User(username=username, email=safe_email, hashed_password=get_password_hash(password))
        db.add(user)
        db.commit()
        db.refresh(user)
        login_msg = "注册成功，已自动登录"

    access_token = create_access_token(data={"sub": username})
    target_url = append_msg_to_url(next_url, login_msg)
    response = RedirectResponse(url=target_url, status_code=302)
    set_access_token_cookie(response, access_token)
    return response


@router.get("/register")
def register_page(request: Request, db: Session = Depends(get_db)):
    """渲染注册页（表单提交至 /login 复用自动注册逻辑）。"""
    return templates.TemplateResponse(request, "register.html", build_context(request, db))


@router.get("/logout")
def logout():
    response = RedirectResponse(url=f"/?msg={quote('已退出登录')}", status_code=302)
    response.delete_cookie(key="access_token")
    return response
