import uuid
from fastapi.testclient import TestClient

from app import app


def check(resp, expected, name):
    if resp.status_code != expected:
        raise RuntimeError(f"{name} 失败: 期望 {expected}, 实际 {resp.status_code}")


def main():
    client = TestClient(app)

    # 1) 公共页面
    check(client.get("/"), 200, "首页")
    check(client.get("/search"), 200, "搜索页")
    check(client.get("/book/1"), 200, "详情页")

    # 2) 登录（不存在账号自动注册）
    username = f"demo_{uuid.uuid4().hex[:8]}"
    resp = client.post(
        "/login?next=/",
        data={"username": username, "password": "123456", "email": f"{username}@local.dev"},
    )
    check(resp, 200, "登录/注册")

    # 3) 购物流程
    check(client.post("/add/1", follow_redirects=False), 302, "加入购物车")
    check(client.get("/cart"), 200, "购物车页")
    check(client.post("/checkout"), 200, "结账")
    check(client.get("/my_orders"), 200, "我的订单")

    # 4) 客服高频问题
    for q in ["有哪些书", "我购物车里有什么", "最便宜的书", "订单有哪些"]:
        ans = client.get("/ask", params={"q": q})
        check(ans, 200, f"客服问题: {q}")
        body = ans.json()
        if "answer" not in body:
            raise RuntimeError(f"客服返回格式错误: {q}")

    print("SMOKE TEST PASS")


if __name__ == "__main__":
    main()

