import os


# 从环境变量读取
OPENAI_API_KEY = os.getenv("ALI_API_KEY")
# OPENAI_API_KEY = "sk-c8f56d77ce3345458bdf55f3c3cd2f57"
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

# 检查
if not OPENAI_API_KEY:
    raise ValueError("请先设置环境变量 OPENAI_API_KEY")