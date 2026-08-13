"""应用入口：创建 FastAPI 应用、中间件、路由挂载、启动期初始化。

本模块 import 时无任何写库副作用；建表与播种统一在 lifespan 中执行。
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from fastapi.staticfiles import StaticFiles

from core.config import config
from core.database import init_db
from core.seed import init_sample_books
from security.csrf import issue_csrf_token, set_csrf_cookie
from web.routes import auth, books, cart, chat, orders
from web.templating import templates

logger = logging.getLogger("rag_bookstore")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """启动期：幂等建表 + 播种示例数据。"""
    init_db()
    init_sample_books()
    yield


app = FastAPI(lifespan=lifespan)

# CORS：显式白名单 + credentials，安全且合法（不再用 *+credentials 的非法组合）。
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def csrf_middleware(request: Request, call_next):
    """CSRF 双重提交：注入令牌到 state，并在响应里回写 cookie（缺失时）。"""
    token = issue_csrf_token(request)
    response = await call_next(request)
    if not request.cookies.get("csrf_token"):
        set_csrf_cookie(response, token)
    return response


@app.exception_handler(Exception)
async def render_error_page(request: Request, exc: Exception):
    """未处理异常（500）渲染 error.html；HTTPException 仍走默认 JSON（401/403/404）。

    生产不向前端泄露异常细节，只记录到日志；模板渲染本身再失败则退化为纯文本。
    """
    logger.exception("未处理异常: %s", exc)
    try:
        return templates.TemplateResponse(
            request,
            "error.html",
            {"message": "服务器内部错误，请稍后重试"},
            status_code=500,
        )
    except Exception:
        return PlainTextResponse("服务器内部错误", status_code=500)


app.mount("/static", StaticFiles(directory="static"), name="static")

app.include_router(auth.router)
app.include_router(books.router)
app.include_router(cart.router)
app.include_router(orders.router)
app.include_router(chat.router)
