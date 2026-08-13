# RAG 智能客服 + 线上书店（简历可展示版）

面向实习面试的全栈项目：将书店业务流程与 RAG 智能客服整合在一个站点中，重点体现"业务场景落地 + 工程稳定性 + 可维护分层"。

## 项目亮点（可直接写简历）

- 设计并实现 **FastAPI + SQLAlchemy + SQLite** 的书店业务闭环：浏览、搜索、详情、购物车、结账、订单。
- 搭建 **RAG 客服链路**：Query 扩展 → 混合检索（BM25 + 中文向量 + RRF 融合）→ 重排 → 大模型生成，全链路降级。
- 实现 **数据库直答策略**：高频问题不走大模型，直接查库，降低幻觉与接口耗时。
- 做实 **安全与稳定性**：CSRF 双重提交、原子扣库存防超卖、Cookie 加固、PRG 防重复提交、全局异常页。
- **分层架构**：根目录仅入口，按 `core/models/security/services/rag/web` 分层，import 时零写库副作用。
- **自动化验证**：`pytest` 7 用例（含 CSRF 拦截、异常页、DB 隔离）+ 端到端冒烟脚本。

## 你可以演示的功能

- 账号：登录页支持"已注册即登录，未注册自动注册并登录"。
- 书店：搜索图书、查看详情、加购、结账、查看我的订单。
- 客服：
  - 高频直答：`有哪些书`、`我购物车里有什么`、`最便宜/最贵的书`、`订单有哪些`、`某本书有库存吗`
  - RAG 兜底：命中知识库时给出解释型回答（支付/配送/退换/会员/发票等）。

## 技术架构

- 后端：`FastAPI`、`SQLAlchemy 2.0`、`Jinja2`
- 数据库：`SQLite`
- 检索：`BM25 + FAISS + SentenceTransformer(bge-small-zh-v1.5)`，结果用 RRF 融合
- 重排：`CrossEncoder(bge-reranker-base)`，未缓存自动降级
- 生成：阿里云 DashScope 兼容 OpenAI 接口（Qwen）
- 鉴权：`JWT + HttpOnly Cookie`
- 防护：`CSRF 双重提交`、`SameSite=Lax`、`原子扣库存`
- 部署：`Docker` / `docker-compose`

## 项目结构

```
RAG_system/
├── app.py            # 入口：FastAPI 实例 + lifespan + 中间件 + 异常处理 + 路由挂载
├── run.py            # uvicorn 启动器
├── main.py           # 命令行 RAG demo
├── smoke_test.py     # 冒烟 runner（调 pytest）
├── download_models.py
├── core/             # 基础设施：config / database / utils / seed
├── models/           # 数据模型：orm / schemas（__init__ 统一 re-export）
├── security/         # 鉴权与防护：auth / csrf
├── services/         # 业务服务：order_service / direct_answer / rag_service
├── rag/              # 检索/重排/LLM：llm / retriever / reranker
├── web/              # 表现层：templating + routes/(auth,books,cart,orders,chat)
├── templates/        # Jinja2 页面
├── static/
├── data/docs.txt     # 静态知识库语料；data/cache/ 为检索索引缓存（运行时生成）
├── tests/            # conftest + test_app + test_smoke
└── docs/resume_project.md
```

## 关键工程设计

- **高频问题数据库直答**：库存/价格/购物车/订单等事实型问题直查库，不走模型，结果确定可解释。
- **RAG 失败降级**：检索/重排/生成任一步失败都返回可用答案，避免 500；降级事件经 `logging` 可观测。
- **RRF 混合检索**：BM25 与向量各取 top-k，按倒数排名融合（k=60），优于拼接去重。
- **索引持久化**：embedding + FAISS 索引缓存到 `data/cache/`，语料变更（hash 不匹配）自动重建。
- **原子扣库存**：`UPDATE books SET stock=stock-qty WHERE id=? AND stock>=qty`，按 `rowcount` 判定，杜绝并发超卖。
- **CSRF 双重提交**：中间件种令牌 cookie + 表单隐藏域/请求头双重比对，cookie-auth 应用的纵深防御。
- **历史库兼容**：订单表运行时探测列名，兼容旧字段（`product/status/order_date`）与新字段（`order_no/book_title/amount/created_at`）。
- **零 import 副作用**：建表/播种移到 `lifespan`，RAG 组件惰性构造，`import app` 不写库、不加载模型。
- **测试 DB 隔离**：每用例后清理动态数据并重置库存（"重置到已知状态"，规避 TestClient 跨线程 SQLite 锁）。

## 环境变量

复制 `.env.example` 为 `.env` 按需填写（不填项用默认值）。关键项：

- `ALI_API_KEY`：大模型 API Key（未配置时 LLM 被禁用，客服退化为检索/直答）。
- `SECRET_KEY`：JWT 签名密钥；`ENV=production` 时用默认值会启动失败。
- `ENV`：`development`（默认）/ `production`（强制安全配置）。
- `HF_LOCAL_FILES_ONLY`：是否仅用本地 HF 缓存，默认 `1`；首次需 `python download_models.py`。
- `EMBEDDING_MODEL` / `RERANKER_MODEL`：向量与重排模型名（默认 `bge-small-zh-v1.5` / `bge-reranker-base`）。
- `CORS_ORIGINS`：允许跨域带凭证的来源，逗号分隔。
- `LOG_LEVEL`：日志级别，默认 `INFO`。

## 本地运行

> 需 Python 3.10 或 3.11（`sentence-transformers`/`faiss`/`torch` 在更高版本可能无 wheel）。

```powershell
# 1) 进入项目目录（任意路径即可）
cd <项目根目录>

# 2) 创建虚拟环境并安装依赖（CPU 版 torch 见 requirements.txt 注释）
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt

# 3) 配置环境变量
copy .env.example .env      # 填入 ALI_API_KEY / SECRET_KEY（不配也可跑，LLM 与密钥会降级）

# 4) 预下载中文向量模型（约 95MB，仅首次）
python download_models.py

# 5) 启动
$env:HF_LOCAL_FILES_ONLY='1'   # 仅用本地缓存，避免联网校验
python run.py
```

## 快速验证（推荐面试前执行）

```powershell
.\.venv\Scripts\Activate.ps1
python smoke_test.py    # 端到端冒烟
pytest -q               # 全部 7 个用例
```

## Docker 运行（可选）

```powershell
# 构建镜像（首次会下载 torch 与向量模型，耗时较长）
docker compose build
# 启动（SECRET_KEY 请在生产改为强随机值）
$env:ALI_API_KEY="your-key"; docker compose up
```

访问 http://127.0.0.1:8000/ 。

## 页面入口

- 首页：`http://127.0.0.1:8000/`
- 登录：`http://127.0.0.1:8000/login`
- 客服：`http://127.0.0.1:8000/chat`

## 对外接口（重点）

- `GET /ask?q=...`：智能客服问答（需登录）
- `POST /add/{book_id}`：加入购物车（需登录）
- `GET /cart`：查看购物车（需登录）
- `POST /checkout`：结账并生成订单（需登录）
- `GET /my_orders`：查看订单（需登录）

## 测试说明

- `tests/test_app.py`（6 单元用例）：公共页、鉴权守卫、购物-结账-订单链路、高频直答、CSRF 拦截（403）、异常页渲染（500）。
- `tests/test_smoke.py`（1 端到端用例）：把上述链路串成一个完整会话。
- `tests/conftest.py`：autouse 夹具，每用例后清理 users/cart_items/orders 并重置书籍库存，保证用例间互不污染。
- 未覆盖：RAG 全链路（向量检索+重排+LLM 生成）依赖模型/API Key，仅手动 sanity 检查，未纳入 pytest。

## 简历描述模板（可直接使用）

> 独立开发 RAG 智能客服与线上书店项目，使用 FastAPI + SQLAlchemy 构建业务闭环，并基于 BM25/中文向量/RRF 融合/重排实现客服问答；通过数据库直答策略解决高频问题稳定性，加入原子扣库存、CSRF 双重提交、模型降级与历史库兼容机制，并按 core/models/security/services/rag/web 分层组织代码，用 pytest + 端到端冒烟保障核心链路可回归。
