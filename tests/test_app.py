import uuid

from fastapi.testclient import TestClient

from app import app


def _csrf_token(client: TestClient) -> str:
    """触发 CSRF 中间件写入令牌 cookie，并返回令牌用于表单提交。"""
    token = client.cookies.get("csrf_token")
    if not token:
        client.get("/")
        token = client.cookies.get("csrf_token")
    assert token, "未取到 CSRF 令牌"
    return token


def _login(client: TestClient):
    username = f"t_{uuid.uuid4().hex[:8]}"
    token = _csrf_token(client)
    resp = client.post(
        "/login?next=/",
        data={
            "username": username,
            "password": "123456",
            "email": f"{username}@local.dev",
            "csrf_token": token,
        },
    )
    assert resp.status_code == 200


def test_public_pages_ok():
    with TestClient(app) as client:
        assert client.get("/").status_code == 200
        assert client.get("/search").status_code == 200
        assert client.get("/book/1").status_code == 200


def test_auth_guard_for_chatbot():
    with TestClient(app) as client:
        resp = client.get("/ask", params={"q": "你好"})
        assert resp.status_code == 401


def test_cart_checkout_and_orders_flow():
    with TestClient(app) as client:
        _login(client)
        token = _csrf_token(client)

        assert client.post("/add/1", data={"csrf_token": token}, follow_redirects=False).status_code == 302
        assert client.get("/cart").status_code == 200
        # 结账成功走 PRG：302 -> /my_orders -> 200
        assert client.post("/checkout", data={"csrf_token": token}).status_code == 200
        assert client.get("/my_orders").status_code == 200


def test_chat_high_frequency_queries():
    with TestClient(app) as client:
        _login(client)

        queries = ["有哪些书", "最便宜的书", "我购物车里有什么", "订单有哪些"]
        for q in queries:
            resp = client.get("/ask", params={"q": q})
            assert resp.status_code == 200
            assert "answer" in resp.json()


def test_csrf_blocks_state_change_without_token():
    """无 CSRF 令牌的 POST 必须被 403 拦截。"""
    with TestClient(app) as client:
        _login(client)
        # 故意不附带 csrf_token
        resp = client.post("/add/1", data={}, follow_redirects=False)
        assert resp.status_code == 403


def test_unhandled_exception_renders_error_page():
    """未处理异常应渲染 error.html（500），而非裸 500/JSON。"""
    from core.database import get_db

    def _boom():
        raise RuntimeError("boom")

    app.dependency_overrides[get_db] = _boom
    try:
        # raise_server_exceptions=False：让异常处理器返回 500 响应，而非把异常抛给测试
        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.get("/")
            assert resp.status_code == 500
            assert "服务器内部错误" in resp.text
    finally:
        app.dependency_overrides.clear()
