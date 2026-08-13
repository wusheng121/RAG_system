"""全局配置：环境变量的唯一来源。

其它模块一律 `from config import config`，不要再直接 os.getenv，
避免配置散落、告警重复。
"""

import logging
import os

from dotenv import load_dotenv

# 自动加载项目根目录下的 .env（不存在时为空操作）
load_dotenv()

# 统一日志配置：替代散落各处的 print，便于排查检索降级/鉴权异常
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
)
logger = logging.getLogger("rag_bookstore")


class Config:
    # ---- LLM ----
    ALI_API_KEY: str | None = os.getenv("ALI_API_KEY")
    LLM_BASE_URL: str = os.getenv("LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "qwen-plus")

    # ---- 鉴权 ----
    SECRET_KEY: str = os.getenv("SECRET_KEY", "your-secret-key")
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))

    # ---- 运行环境 ----
    # development（默认，宽松） / production（强制安全配置）
    ENV: str = os.getenv("ENV", "development")
    IS_PRODUCTION: bool = ENV == "production"

    # ---- 数据库 ----
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./bookstore.db")

    # ---- CORS ----
    # 显式白名单，避免 allow_origins=["*"]+allow_credentials=True 的非法/不安全组合。
    # 默认仅允许本机开发来源；生产按需用逗号分隔设置 CORS_ORIGINS。
    CORS_ORIGINS: list = [
        o.strip()
        for o in os.getenv("CORS_ORIGINS", "http://127.0.0.1:8000,http://localhost:8000").split(",")
        if o.strip()
    ]

    # ---- 检索/模型 ----
    # 默认仅使用本地 HF 缓存，避免联网校验导致 SSL 问题；首次需手动下载模型。
    HF_LOCAL_FILES_ONLY: bool = os.getenv("HF_LOCAL_FILES_ONLY", "1") == "1"
    # 向量模型：bge-small-en 不支持中文，改用中文模型 bge-small-zh-v1.5
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")
    # 重排模型：CrossEncoder，未缓存时自动降级为取前 top_k
    RERANKER_MODEL: str = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-base")
    # 检索缓存目录（embedding/索引持久化，避免每次重启重算）
    CACHE_DIR: str = os.getenv("CACHE_DIR", "data/cache")

    @classmethod
    def warn(cls) -> None:
        """启动期一致性检查：仅在此处记录告警，消除历史三处重复 print。"""
        insecure_secret = (not cls.SECRET_KEY) or (cls.SECRET_KEY == "your-secret-key")
        if insecure_secret:
            # 生产环境用默认密钥 = 任何人都能伪造 JWT，必须失败退出。
            if cls.IS_PRODUCTION:
                raise RuntimeError("ENV=production 时必须设置 SECRET_KEY 环境变量")
            logger.warning("使用默认 SECRET_KEY，不安全，请在 .env 中设置。")
        if not cls.ALI_API_KEY:
            logger.warning("未设置环境变量 ALI_API_KEY。LLM 功能将被禁用。")


config = Config()
config.warn()
