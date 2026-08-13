FROM python:3.10-slim

# faiss / torch(CPU) 运行时所需的系统库
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 先装 CPU 版 torch，避免 pip 默认拉取 CUDA 巨型包；再装其余依赖
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 预下载中文向量模型进镜像，运行时离线加载
# download_models.py 依赖 core/config.py，故先拷 core/ 与脚本
COPY download_models.py ./
COPY core ./core
RUN python download_models.py

# 拷贝应用代码
COPY . .

ENV HF_LOCAL_FILES_ONLY=1 \
    ENV=development \
    PYTHONUNBUFFERED=1

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
