import os


# 从环境变量读取
ALI_API_KEY = os.getenv("ALI_API_KEY")
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key")

# 检查（可选）
if not ALI_API_KEY:
    print("警告: 未设置环境变量 ALI_API_KEY。LLM功能将被禁用。")
if not SECRET_KEY or SECRET_KEY == "your-secret-key":
    print("警告: 使用默认SECRET_KEY，不安全。")
