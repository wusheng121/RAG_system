import uvicorn
import os
from app import app

# Check for required environment variables
if not os.getenv("ALI_API_KEY"):
    print("警告: 未设置环境变量 ALI_API_KEY。LLM功能将被禁用。")
if not os.getenv("SECRET_KEY") or os.getenv("SECRET_KEY") == "your-secret-key":
    print("警告: 使用默认SECRET_KEY，不安全。")

if __name__ == "__main__":
    try:
        uvicorn.run(app, host="127.0.0.1", port=8000)
    except Exception as e:
        print(f"启动失败: {e}")
        print("请检查环境变量和依赖。")
