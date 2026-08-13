"""预下载 HF 模型到本地缓存。

运行时设 HF_LOCAL_FILES_ONLY=1 即可离线加载，避免每次联网校验导致 SSL 问题。

用法：
  python download_models.py            # 仅下载向量模型（约95MB，默认）
  python download_models.py --reranker # 同时下载重排模型（约1GB）
"""

import sys

from sentence_transformers import CrossEncoder, SentenceTransformer

from core.config import config


def download_embedding() -> None:
    print(f"下载向量模型: {config.EMBEDDING_MODEL}")
    SentenceTransformer(config.EMBEDDING_MODEL, local_files_only=False)
    print("向量模型已缓存")


def download_reranker() -> None:
    print(f"下载重排模型: {config.RERANKER_MODEL}（约1GB，请耐心等待）")
    CrossEncoder(config.RERANKER_MODEL, local_files_only=False)
    print("重排模型已缓存")


if __name__ == "__main__":
    download_embedding()
    if "--reranker" in sys.argv:
        download_reranker()
    print("完成")
