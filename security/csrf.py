"""CSRF 防护：双重提交令牌（double-submit cookie）模式。

原理：
1. 中间件在每个请求注入 `request.state.csrf_token`（来自 cookie 或新生成），
   并在响应里把令牌写回 cookie（HttpOnly，后端比对用）。
2. 模板把 `request.state.csrf_token` 渲染进表单隐藏域；JS 从 <meta> 读取
   后随 DELETE 等请求放在 `X-CSRF-Token` 头。
3. 状态变更的 POST/DELETE 依赖 `verify_csrf_token` 比对“提交的令牌”与
   “cookie 里的令牌”，不一致即 403。

cookie-auth 的应用天然怕 CSRF（浏览器自动带 cookie），SameSite=Lax 已挡掉
跨站 POST；再加双重提交令牌可防同站诱导提交，属于纵深防御。
"""

import secrets

from fastapi import HTTPException, Request

COOKIE_NAME = "csrf_token"


def _read_cookie(request: Request) -> str | None:
    return request.cookies.get(COOKIE_NAME)


def issue_csrf_token(request: Request) -> str:
    """读取或生成令牌并写入 request.state，供模板与中间件复用。"""
    token = _read_cookie(request) or secrets.token_urlsafe(32)
    request.state.csrf_token = token
    return token


def set_csrf_cookie(response, token: str) -> None:
    """令牌写回 cookie（仅当尚不存在时，避免反复覆盖）。"""
    # httponly=True：后端比对即可，JS 不需要直接读 cookie。
    response.set_cookie(
        key=COOKIE_NAME,
        value=token,
        httponly=True,
        samesite="lax",
        secure=False,  # 开发为 http；生产应按 config.IS_PRODUCTION 设置
    )


async def verify_csrf_token(request: Request) -> None:
    """状态变更端点依赖：提交令牌须与 cookie 令牌一致。"""
    cookie_token = _read_cookie(request)
    submitted = request.headers.get("X-CSRF-Token")
    if not submitted:
        # 表单字段兜底（普通表单 POST）
        try:
            form = await request.form()
            submitted = form.get("csrf_token")
        except Exception:
            submitted = None
    ok = bool(cookie_token) and bool(submitted) and secrets.compare_digest(str(cookie_token), str(submitted))
    if not ok:
        raise HTTPException(status_code=403, detail="CSRF 校验失败，请刷新页面后重试")
