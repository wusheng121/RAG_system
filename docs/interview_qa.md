# 面试追问与应答手册

> 对照本项目真实实现整理。每条含「面试官可能问」「怎么答」「代码位置」。
> 原则：**写进简历的点，原理必须能讲透；讲不透的宁可别写。**

---

## 1. RAG 整条链路是怎样的？

**追问**：你说"Query 扩展 + 混合检索 + 重排 + LLM 生成"，具体每一步怎么做？顺序为什么这样？

**答**：
1. 先走**数据库直答**——命中高频事实问题（库存/购物车/订单等）直接查库返回，不走 LLM。
2. 未命中才进 RAG：`expand_query` 用 LLM 把原问题生成 3 个不同表达，提升召回。
3. 对每个扩展 query 做**混合检索**（BM25 + 向量），合并。
4. 把合并结果 + 数据库动态信息（书籍/购物车/订单）一起喂给 **CrossEncoder 重排**，取 top-3。
5. 拼成 prompt 给 LLM 生成最终回答。

**代码**：`services/rag_service.py` `ask()`；直答 `services/direct_answer.py`。

---

## 2. 为什么"混合检索"还要 RRF？直接拼接去重不行吗？

**追问**：BM25 和向量两路结果你怎么合的？什么叫 RRF？

**答**：
- 拼接去重的**问题**：两路各返回自己的 top-k，但谁该排前没有依据——BM25 和向量分数量纲不同，不能直接比。
- **RRF（倒数排名融合）**：不看绝对分数，只看**排名**。第 r 名得分 `1/(k + r + 1)`，k=60 是社区经验值。两路对同一文档的得分相加，按总分重排。
- 好处：量纲无关、实现简单、对两路都公平。

**代码**：`rag/retriever.py` `hybrid_retrieve()`，`RRF_K = 60`。

---

## 3. 向量模型为什么用 `bge-small-zh-v1.5`？

**追问**：向量检索用什么模型？为什么不用英文的？

**答**：
- 语料 `data/docs.txt` 是**中文**，query 也是中文。
- 原来用的 `bge-small-en` 是英文模型，中文编码不到位，向量路基本失效，等于只剩 BM25——"挂着 RAG 的名实际没向量"。
- 换 `bge-small-zh-v1.5`（中文），中文语义检索才真正起作用。embedding 用 FAISS `IndexFlatIP`（内积，配合 normalize_embeddings 等价余弦）。

**代码**：`rag/retriever.py` `_initialize()` / `vector_retrieve()`。

---

## 4. 重排（rerank）和向量检索有什么区别？为什么两步？

**追问**：既然向量已经排过了，为什么还要 rerank？

**答**：
- 向量检索是**双塔**：query 和 doc 各自编码成向量算相似度，快但粗（没看 query-doc 的交互）。
- CrossEncoder 重排是**交叉**：把 (query, doc) 拼一起喂模型，能捕捉细粒度相关性，更准但慢。
- 所以**先用向量/BM25 粗召回到几十条，再用 CrossEncoder 精排到 top-3**——速度和质量的折中。
- 模型 `bge-reranker-base`；未缓存时自动降级为"取前 top_k"。

**代码**：`rag/reranker.py`。

---

## 5. 数据库直答策略——为什么不都问大模型？

**追问**：高频问题直答，低频走 RAG，你怎么划分？直答好处是啥？

**答**：
- 库存、价格、购物车内容、订单列表这类是**事实型**问题，不需要推理，直查库即可。
- 直答**好处**：①结果确定可解释（不幻觉订单号/库存数）；②不耗 LLM 接口，低延迟；③LLM 不可用时这些核心功能仍可用。
- 划分：按关键词命中（购物车/订单/价格/库存/书单），命中走直答，未命中走 RAG。

**代码**：`services/direct_answer.py` `direct_db_answer()`。

---

## 6. 原子扣库存怎么防超卖？

**追问**：你说防超卖，具体怎么做？并发下不会出问题？

**答**：
- 原来是**先查后扣两步**：`if stock < qty: 报错` 再 `stock -= qty`。两步之间有缝，并发下两个用户都查到"够"，都扣，库存变负数。
- 改成**一条原子语句**：`UPDATE books SET stock = stock - :qty WHERE id = :id AND stock >= :qty`。
- 数据库 UPDATE 对该行加锁，两个并发 UPDATE 排队：A 成功（stock 够→扣），B 再判 `stock>=qty` 不成立 → `rowcount=0` → 判定库存不足。
- 靠 `rowcount` 判成败，不用先查。

**代码**：`web/routes/cart.py` `process_checkout()`。

---

## 7. CSRF 怎么防的？为什么 cookie-auth 要特别防？

**追问**：什么是 CSRF？你用的是什么方案？

**答**：
- **CSRF**：你登录了书店（浏览器存了 cookie），访问恶意网站 `evil.com`，它诱导你浏览器发 POST 到书店——浏览器**自动带 cookie**，书店以为是你本人操作（借登录态伪造请求）。
- 我用**双重提交令牌**：服务端种一个随机 `csrf_token` cookie，同时把同值渲染进表单隐藏域；提交时比对"cookie 里的值 == 表单里的值"。攻击者能借 cookie 但**读不到书店页面**（同源策略），填不出隐藏域的正确值 → 403。
- 配 `SameSite=Lax` cookie，浏览器层再挡一道跨站 POST，纵深防御。

**代码**：`security/csrf.py`（中间件 `issue_csrf_token` + 依赖 `verify_csrf_token`，`secrets.compare_digest` 防时序攻击）。

---

## 8. 模型降级具体怎么做的？

**追问**：检索/重排/LLM 哪步失败了怎么办？

**答**：`ask()` 链路每一步都 `try/except`：
- `expand_query` 失败 → 回退原 query。
- 单条 query 检索失败 → 跳过该条。
- 重排失败 → 取 `all_docs[:3]`。
- LLM 生成失败 → 返回兜底文案"抱歉，我无法回答这个问题"。
- 向量模型加载失败（未缓存/断网）→ 降级为仅 BM25。
- **每步降级都打 `logger.warning`**，可观测，不是静默吞掉。

**代码**：`services/rag_service.py`；向量降级日志 `rag/retriever.py` `_initialize()`。

---

## 9. 测试怎么保证不互相污染？

**追问**：你说有 pytest 自动化测试，多个用例之间数据库状态怎么隔离？

**答**：
- 用 autouse 夹具，每用例**结束后清理** cart_items/orders/users 并把书籍库存重置回种子值——前一个用例的结账扣库存、注册用户不影响下一个。
- **为什么不用"事务回滚"**：FastAPI 的 TestClient 把同步端点丢线程池跑，单连接跨线程 + SQLite 易触发 `database is locked`；改用"重置到已知状态"更稳。
- LLM 测试用 **canary 探针**：先试调一次，LLM 不可用（无 key/额度/限流）就整组 `skip`，不误判失败；可用时才跑真实断言。

**代码**：`tests/conftest.py`（`_reset_db`）、`tests/test_llm.py`（canary）、`tests/test_retriever.py`（CI 无模型时降级 BM25 仍通过）。

---

## 10. CI 跑什么？为什么加 CI？

**追问**：GitHub Actions 里具体做了什么？

**答**：push/PR 触发，跑三件事：`ruff check`（lint）+ `ruff format --check`（格式）+ `pytest -q`。环境变量 `HF_LOCAL_FILES_ONLY=1` 让 CI 不下载向量模型，检索测试靠 BM25 降级路径照样过。
- CI 的价值：之前有个 bug 本地绿、CI 红——模块级 ORM 实例在全新库下变 detached，本地因历史数据掩盖而"假绿"，只有 CI 全新环境才暴露。

**代码**：`.github/workflows/ci.yml`；那个 bug 的修复 `core/seed.py`（种子数据改纯字典）。

---

## 11. 分层架构 + 没有 import 副作用，这俩是什么意思？

**追问**：你说分层，怎么分的？什么叫"import 时无写库副作用"？

**答**：
- 分层：根目录只放入口（`app.py`/`run.py` 等），内部按 `core`（配置/DB）/`models`（ORM+契约）/`security`（鉴权/CSRF）/`services`（业务）/`rag`（检索重排LLM）/`web`（路由+模板）分。
- **import 副作用**：Python 的 import 本该只加载模块，不该改外部状态。但很多教程在模块顶层直接 `Base.metadata.create_all()` + 播种 + 加载模型——`import app` 就建库插数据加载模型，污染测试、拖慢启动、被工具误触发。我改成：建表/播种移到 FastAPI 的 `lifespan`（启动时才跑），RAG 组件惰性构造（第一次用才加载模型）。

**代码**：`app.py` `lifespan`；惰性单例 `services/rag_service.py` `_Lazy`。

---

## 12. 历史库兼容是怎么回事？

**追问**：你说兼容历史订单表，什么场景？

**答**：订单表升级过，旧库可能有 `product/status/order_date` 字段，新库用 `order_no/book_title/amount/created_at`。读写订单时**运行时探测表的实际列名**（`inspect(db.bind).get_columns()`），按存在与否构造 SQL，保证新旧库都能读写，避免线上 500。

**代码**：`services/order_service.py` `get_table_columns` / `load_orders_for_user` / `create_orders_from_cart`。

---

## 13. JWT + HttpOnly Cookie 鉴权怎么工作的？

**追问**：登录态怎么维护？为什么不用 session？

**答**：
- 登录成功签发 JWT（HS256，载荷含 `sub=用户名` + `exp` 过期），写入 **HttpOnly Cookie**。之后请求浏览器自动带 cookie，服务端 `jwt.decode` 验签识别用户。
- **HttpOnly**：JS 读不到，防 XSS 偷令牌；`SameSite=Lax` 防 CSRF；`max_age` 与 JWT 过期一致；`secure` 生产仅 HTTPS 传输。
- 不用 session：无状态、易水平扩展，不需要服务端存 session。

**代码**：`security/auth.py` `create_access_token` / `set_access_token_cookie` / `get_current_user`。

---

## 14. 检索索引为什么要持久化？

**追问**：每次启动都重建索引不行吗？

**答**：embedding 全量编码语料要算一遍，语料大了慢。我把 docs + FAISS 索引 + 语料 hash 存到 `data/cache/`。下次启动：**比对 hash**，没变就直接 load，变了才重建。等于把"算 embedding"从每次启动降到"仅语料变更时"。

**代码**：`rag/retriever.py` `_load_cache` / `_save_cache` / `source_hash`。

---

## 别被问住·自查清单

| 点 | 能讲原理吗？ |
|---|---|
| RRF 公式与为什么 k=60 | ☐ |
| CrossEncoder vs 双塔向量 | ☐ |
| 原子 UPDATE 防超卖的行锁 | ☐ |
| CSRF 双重提交 + SameSite 区别 | ☐ |
| 为什么不用事务回滚做测试隔离 | ☐ |
| import 副作用为什么是坑 | ☐ |
| JWT 验签原理（HS256 对称） | ☐ |

> 任何一条讲不透，就从简历里删掉对应词。**宁可少写，别被问住。**
