import uvicorn

from app import app  # noqa: F401  (import 触发 config.warn())

if __name__ == "__main__":
    try:
        uvicorn.run(app, host="127.0.0.1", port=8000)
    except Exception as e:
        print(f"启动失败: {e}")
        print("请检查环境变量和依赖。")
