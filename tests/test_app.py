import uuid

from fastapi.testclient import TestClient

from app import app


def _login(client: TestClient):
    username = f"t_{uuid.uuid4().hex[:8]}"
    resp = client.post(
        "/login?next=/",
        data={"username": username, "password": "123456", "email": f"{username}@local.dev"},
    )
    assert resp.status_code == 200


def test_public_pages_ok():
    client = TestClient(app)
    assert client.get("/").status_code == 200
    assert client.get("/search").status_code == 200
    assert client.get("/book/1").status_code == 200


def test_auth_guard_for_chatbot():
    client = TestClient(app)
    resp = client.get("/ask", params={"q": "你好"})
    assert resp.status_code == 401


def test_cart_checkout_and_orders_flow():
    client = TestClient(app)
    _login(client)

    assert client.post("/add/1", follow_redirects=False).status_code == 302
    assert client.get("/cart").status_code == 200
    assert client.post("/checkout").status_code == 200
    assert client.get("/my_orders").status_code == 200


def test_chat_high_frequency_queries():
    client = TestClient(app)
    _login(client)

    queries = ["有哪些书", "最便宜的书", "我购物车里有什么", "订单有哪些"]
    for q in queries:
        resp = client.get("/ask", params={"q": q})
        assert resp.status_code == 200
        assert "answer" in resp.json()

