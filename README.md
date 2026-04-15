# RAG智能客服 + 线上书店（简历可展示版）

这是一个面向实习面试的全栈项目：将书店业务流程与 RAG 智能客服整合在一个站点中，重点体现“业务场景落地 + 工程稳定性”。

## 项目亮点（可直接写简历）

- 设计并实现 **FastAPI + SQLAlchemy + SQLite** 的书店业务闭环：浏览、搜索、详情、购物车、结账、订单。
- 搭建 **RAG 客服链路**：Query 扩展 -> 混合检索（BM25 + 向量）-> 重排 -> 大模型生成。
- 实现 **数据库直答策略**：高频问题不走大模型，直接查询数据库，提高稳定性与确定性。
- 完成 **故障降级与兼容**：模型不可用时自动降级；兼容历史订单表结构，避免线上 500。
- 增加 **自动化验证**：`smoke_test.py` 与 `pytest` 用例覆盖核心业务链路。

## 你可以演示的功能

- 账号：登录页支持“已注册即登录，未注册自动注册并登录”。
- 书店：搜索图书、查看详情、加购、结账、查看我的订单。
- 客服：
  - 高频直答：`有哪些书`、`我购物车里有什么`、`最便宜/最贵的书`、`订单有哪些`、`某本书有库存吗`
  - RAG 兜底：命中知识库时给出解释型回答。

## 技术架构

- 后端：`FastAPI`、`SQLAlchemy`
- 数据库：`SQLite`
- 检索：`BM25 + FAISS + SentenceTransformer`
- 重排：`CrossEncoder`
- 生成：阿里云兼容 OpenAI 接口（Qwen）
- 鉴权：`JWT + HttpOnly Cookie`

## 关键工程设计

- **高频问题数据库直答**：降低模型幻觉与接口耗时。
- **RAG 失败降级**：检索/重排/模型任一步失败都返回可用答案，避免 500。
- **历史库兼容**：订单表同时兼容旧字段（`product/status/order_date`）与新字段（`order_no/book_title/amount/created_at`）。
- **库存保护**：加购与结账时校验库存，避免超卖。

## 项目结构（核心文件）

- `app.py`：路由、数据库模型、认证、RAG 编排、直答策略。
- `retriever.py`：向量检索 + BM25 混合检索。
- `rerank.py`：重排器，失败自动降级。
- `llm.py`：大模型调用与 query 扩展。
- `data/docs.txt`：静态知识库语料。
- `templates/`：书店页面模板。
- `static/chat.html`：客服聊天页面。
- `smoke_test.py`：一键冒烟测试脚本。
- `tests/test_app.py`：核心接口自动化用例。

## 环境变量

- `ALI_API_KEY`：大模型 API Key（未配置时可运行，但回答能力受限）。
- `SECRET_KEY`：JWT 签名密钥。
- `HF_LOCAL_FILES_ONLY`：是否仅用本地模型缓存，默认 `1`。

## 本地运行

```powershell
cd D:\aWusheng\Python\pythonpro\rag_system
pip install -r requirements.txt
$env:HF_LOCAL_FILES_ONLY='1'
python run.py
```

## 快速验证（推荐面试前执行）

```powershell
cd D:\aWusheng\Python\pythonpro\rag_system
python smoke_test.py
pytest -q
```

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

## 简历描述模板（可直接使用）

> 独立开发 RAG 智能客服与线上书店项目，使用 FastAPI + SQLAlchemy 构建业务闭环，并基于 BM25/向量检索/重排实现客服问答；通过数据库直答策略解决高频问题稳定性，加入模型降级与历史数据库兼容机制，保障核心接口在异常场景下可用。
