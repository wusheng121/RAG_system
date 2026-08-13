"""端到端冒烟用例：覆盖 公共页→登录→购物车→结账→订单→客服问答 全链路。

由 smoke_test.py 薄封装调用，也可直接 pytest 运行。
"""

import uuid

from fastapi.testclient import TestClient

from app import app


def _csrf_token(client: TestClient) -> str:
    token = client.cookies.get("csrf_token")
    if not token:
        client.get("/")
        token = client.cookies.get("csrf_token")
    assert token, "未取到 CSRF 令牌"
    return token


def _login(client: TestClient):
    token = _csrf_token(client)
    username = f"demo_{uuid.uuid4().hex[:8]}"
    resp = client.post(
        "/login?next=/",
        data={
            "username": username,
            "password": "123456",
            "email": f"{username}@local.dev",
            "csrf_token": token,
        },
    )
    assert resp.status_code == 200, f"登录/注册失败: {resp.status_code}"


def test_smoke_end_to_end():
    with TestClient(app) as client:
        # 1) 公共页面
        assert client.get("/").status_code == 200
        assert client.get("/search").status_code == 200
        assert client.get("/book/1").status_code == 200

        # 2) 登录（不存在账号自动注册）
        _login(client)

        # 3) 购物流程（携带 CSRF 令牌）
        token = _csrf_token(client)
        assert client.post("/add/1", data={"csrf_token": token}, follow_redirects=False).status_code == 302
        assert client.get("/cart").status_code == 200
        token = _csrf_token(client)
        assert client.post("/checkout", data={"csrf_token": token}).status_code == 200
        assert client.get("/my_orders").status_code == 200

        # 4) 客服高频问题
        for q in ["有哪些书", "我购物车里有什么", "最便宜的书", "订单有哪些"]:
            resp = client.get("/ask", params={"q": q})
            assert resp.status_code == 200, f"客服问题失败: {q}"
            assert "answer" in resp.json(), f"客服返回格式错误: {q}"
