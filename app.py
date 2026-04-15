from fastapi import FastAPI, Depends, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from fastapi.responses import RedirectResponse
from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, Float, create_engine, inspect, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from datetime import datetime, timedelta
from typing import Optional
from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from urllib.parse import quote, urlsplit, urlunsplit, parse_qsl, urlencode
import re
import os
import uuid

# 数据库配置
SQLALCHEMY_DATABASE_URL = "sqlite:///./exam_system.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# 密码哈希
pwd_context = CryptContext(schemes=["argon2"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token", auto_error=False)

# JWT配置
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key")
ALGORITHM = "HS256"

# 模型定义
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)

class Book(Base):
    __tablename__ = "books"
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String)
    author = Column(String)
    isbn = Column(String)
    price = Column(Float)
    stock = Column(Integer)

class Order(Base):
    __tablename__ = "orders"
    id = Column(Integer, primary_key=True, index=True)
    order_no = Column(String, unique=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    book_title = Column(String)
    amount = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)


class CartItem(Base):
    __tablename__ = "cart_items"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), index=True)
    book_id = Column(Integer, ForeignKey("books.id"), index=True)
    title = Column(String)
    price = Column(Float)
    qty = Column(Integer, default=1)
    created_at = Column(DateTime, default=datetime.utcnow)

# 数据库依赖
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# 认证函数
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def authenticate_user(db: Session, username: str, password: str):
    user = db.query(User).filter(User.username == username).first()
    if not user or not verify_password(password, user.hashed_password):
        return False
    return user

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=30))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def get_user_by_token(token: Optional[str], db: Session):
    if not token:
        return None
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        if not username:
            return None
        return db.query(User).filter(User.username == username).first()
    except JWTError:
        return None


def get_current_user(request: Request, token: Optional[str] = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    cookie_token = request.cookies.get("access_token")
    final_token = token or cookie_token
    return get_user_by_token(final_token, db)


def build_context(request: Request, db: Session, **kwargs):
    context = {"request": request, "current_user": get_user_by_token(request.cookies.get("access_token"), db)}
    msg = request.query_params.get("msg")
    if msg:
        context["popup_message"] = msg
    context.update(kwargs)
    return context


def append_msg_to_url(url: str, msg: str) -> str:
    parts = urlsplit(url)
    query = dict(parse_qsl(parts.query, keep_blank_values=True))
    query["msg"] = msg
    return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(query), parts.fragment))


def _format_money(value):
    try:
        return f"￥{float(value):.2f}"
    except Exception:
        return f"￥{value}"


def _clean_text(text: str) -> str:
    return (text or "").strip().strip("  \t\n\r。？?！!；;：:《》\"'“”")


def _extract_book_title_hint(question: str) -> str:
    text = _clean_text(question)
    quoted = re.search(r"《([^》]+)》", text)
    if quoted:
        return quoted.group(1).strip()
    prefixes = [
        "你们店有没有",
        "你们有没有",
        "店里有没有",
        "请问有没有",
        "请问有卖",
        "有没有",
        "有卖",
        "有无",
        "是否有",
        "店里有",
        "书店有没有",
    ]
    for prefix in prefixes:
        if text.startswith(prefix):
            text = text[len(prefix):].strip()
            break
    suffixes = [
        "有库存吗",
        "还有库存吗",
        "库存多少",
        "库存还有多少",
        "库存还剩多少",
        "有货吗",
        "还有货吗",
        "在售吗",
        "卖吗",
        "能买吗",
        "还有吗",
        "有吗",
        "在吗",
        "在不在",
        "吗",
        "库存",
    ]
    for suffix in suffixes:
        if text.endswith(suffix):
            return text[: -len(suffix)].strip("  \t\n\r。？?！!；;：:《》\"'“”")
    if "的库存" in text:
        return text.split("的库存", 1)[0].strip("  \t\n\r。？?！!；;：:《》\"'“”")
    return text


def _find_book_by_hint(db: Session, hint: str):
    hint = (hint or "").strip()
    if not hint:
        return None
    q = db.query(Book)
    exact = q.filter(Book.title == hint).first()
    if exact:
        return exact
    exact_isbn = q.filter(Book.isbn == hint).first()
    if exact_isbn:
        return exact_isbn
    return (
        q.filter(Book.title.contains(hint)).first()
        or q.filter(Book.author.contains(hint)).first()
        or q.filter(Book.isbn.contains(hint)).first()
    )


def _get_table_columns(db: Session, table_name: str):
    """Read real table columns at runtime to support historical SQLite schemas."""
    try:
        inspector = inspect(db.bind)
        names = inspector.get_table_names()
        if table_name not in names:
            return set()
        return {col.get("name") for col in inspector.get_columns(table_name)}
    except Exception:
        return set()


def _load_orders_for_user(db: Session, user_id: int):
    cols = _get_table_columns(db, "orders")
    if not cols:
        return []

    order_no_expr = "order_no" if "order_no" in cols else "CAST(id AS TEXT)"
    title_expr = "book_title" if "book_title" in cols else ("product" if "product" in cols else "''")
    amount_expr = "amount" if "amount" in cols else "NULL"
    time_expr = "created_at" if "created_at" in cols else ("order_date" if "order_date" in cols else "NULL")
    status_expr = "status" if "status" in cols else "''"
    order_by_expr = "created_at" if "created_at" in cols else ("order_date" if "order_date" in cols else "id")

    sql = text(
        f"""
        SELECT
            {order_no_expr} AS order_no,
            {title_expr} AS book_title,
            {amount_expr} AS amount,
            {time_expr} AS created_at,
            {status_expr} AS status
        FROM orders
        WHERE user_id = :user_id
        ORDER BY {order_by_expr} DESC, id DESC
        """
    )
    try:
        return [dict(row) for row in db.execute(sql, {"user_id": user_id}).mappings().all()]
    except Exception:
        return []


def _create_orders_from_cart(db: Session, user_id: int, cart_rows):
    cols = _get_table_columns(db, "orders")
    if not cols:
        return 0

    created = 0
    now = datetime.utcnow()
    for row in cart_rows:
        item_title = getattr(row, "title", "未知商品")
        item_qty = int(getattr(row, "qty", 1) or 1)
        item_price = float(getattr(row, "price", 0) or 0)
        item_amount = round(item_qty * item_price, 2)

        payload = {"user_id": user_id}
        insert_cols = ["user_id"]
        insert_vals = [":user_id"]

        if "order_no" in cols:
            payload["order_no"] = f"ORD{uuid.uuid4().hex[:10].upper()}"
            insert_cols.append("order_no")
            insert_vals.append(":order_no")
        if "book_title" in cols:
            payload["book_title"] = item_title
            insert_cols.append("book_title")
            insert_vals.append(":book_title")
        if "product" in cols:
            payload["product"] = item_title
            insert_cols.append("product")
            insert_vals.append(":product")
        if "amount" in cols:
            payload["amount"] = item_amount
            insert_cols.append("amount")
            insert_vals.append(":amount")
        if "status" in cols:
            payload["status"] = "已下单"
            insert_cols.append("status")
            insert_vals.append(":status")
        if "created_at" in cols:
            payload["created_at"] = now
            insert_cols.append("created_at")
            insert_vals.append(":created_at")
        if "order_date" in cols:
            payload["order_date"] = now
            insert_cols.append("order_date")
            insert_vals.append(":order_date")

        sql = text(f"INSERT INTO orders ({', '.join(insert_cols)}) VALUES ({', '.join(insert_vals)})")
        db.execute(sql, payload)
        created += 1

    return created


def _direct_db_answer(question: str, user=None, db: Session = None):
    if not db:
        return None

    q = _clean_text(question)
    if not q:
        return None

    cart_keywords = ["我购物车里有什么", "购物车里有什么", "购物车有什么", "我的购物车", "查看购物车", "购物车内容"]
    if any(keyword in q for keyword in cart_keywords):
        if not user:
            return "请先登录后查看购物车。"
        rows = db.query(CartItem).filter(CartItem.user_id == user.id).order_by(CartItem.id.asc()).all()
        if not rows:
            return "你的购物车里还没有商品。"
        total = 0.0
        lines = [f"你的购物车里有 {len(rows)} 件商品："]
        for index, item in enumerate(rows, 1):
            subtotal = float(item.price or 0) * int(item.qty or 0)
            total += subtotal
            lines.append(f"{index}. {item.title} × {item.qty}，单价{_format_money(item.price)}，小计{_format_money(subtotal)}")
        lines.append(f"购物车合计：{_format_money(total)}")
        return "\n".join(lines)

    order_keywords = ["订单有哪些", "有哪些订单", "我的订单", "订单列表", "查看订单", "订单记录", "订单明细"]
    if any(keyword in q for keyword in order_keywords):
        if not user:
            return "请先登录后查看订单。"
        orders = _load_orders_for_user(db, user.id)
        if not orders:
            return "你当前还没有订单记录。"
        lines = [f"你当前共有 {len(orders)} 笔订单："]
        for index, order in enumerate(orders, 1):
            created_at = order.get("created_at")
            created_text = created_at.strftime("%Y-%m-%d %H:%M") if hasattr(created_at, "strftime") else str(created_at)
            amount_text = _format_money(order.get("amount")) if order.get("amount") not in (None, "", "NULL") else "金额未知"
            lines.append(
                f"{index}. 订单号：{order.get('order_no', '未知')}｜书名：{order.get('book_title', '未知')}｜金额：{amount_text}｜时间：{created_text}"
            )
        return "\n".join(lines)

    price_keywords = ["最便宜", "最低价", "最低的书", "价格最低", "最贵", "最高价", "最高的书", "价格最高"]
    if any(keyword in q for keyword in price_keywords):
        books = db.query(Book).all()
        if not books:
            return "当前暂无在售书籍。"
        if any(keyword in q for keyword in ["最便宜", "最低价", "最低的书", "价格最低"]):
            target_price = min(float(book.price or 0) for book in books)
            matched = [book for book in books if float(book.price or 0) == target_price]
            prefix = "最便宜的书"
        else:
            target_price = max(float(book.price or 0) for book in books)
            matched = [book for book in books if float(book.price or 0) == target_price]
            prefix = "最贵的书"
        lines = [f"当前{prefix}价格为 {_format_money(target_price)}，对应书籍如下："]
        for book in matched:
            lines.append(f"- {book.title}｜作者：{book.author}｜库存：{book.stock}")
        return "\n".join(lines)

    title_hint = _extract_book_title_hint(q)
    stock_keywords = ["库存", "有库存", "还有库存", "有货吗", "还有货吗", "在售吗", "卖吗", "有没有", "是否有", "在不在", "有卖", "这本书有吗"]
    negative_stock_keywords = ["没有了", "没货", "缺货", "卖完", "断货", "下架", "不在", "未上架"]
    if any(keyword in q for keyword in negative_stock_keywords) or any(keyword in q for keyword in stock_keywords):
        book = _find_book_by_hint(db, title_hint)
        if book:
            if int(book.stock or 0) > 0:
                return f"《{book.title}》当前有库存，剩余 {book.stock} 本。"
            return f"《{book.title}》当前暂无库存。"
        if title_hint:
            return f"目前书店未找到《{title_hint}》，如果你想查库存，可以告诉我更准确的书名或作者。"
        if any(keyword in q for keyword in negative_stock_keywords):
            return "目前书店里没有找到你提到的这本书。"
        return "请告诉我更准确的书名，我可以帮你查库存。"

    book_list_keywords = ["有哪些书", "有什么书", "在售书", "图书列表", "所有书", "书目", "全部书"]
    if any(keyword in q for keyword in book_list_keywords):
        books = db.query(Book).order_by(Book.id.asc()).all()
        if not books:
            return "当前暂无在售书籍。"
        lines = [f"当前在售书籍共 {len(books)} 本："]
        for book in books[:20]:
            lines.append(f"- {book.title}｜作者：{book.author}｜价格：{_format_money(book.price)}｜库存：{book.stock}")
        if len(books) > 20:
            lines.append(f"- 还有 {len(books) - 20} 本，请在搜索页继续查看。")
        return "\n".join(lines)

    return None

# 初始化数据库
Base.metadata.create_all(bind=engine)

# 添加示例书籍
def init_sample_books():
    db = SessionLocal()
    sample_books = [
        Book(title="Python编程入门", author="张三", isbn="1234567890", price=39.99, stock=10),
        Book(title="机器学习基础", author="李四", isbn="0987654321", price=59.99, stock=5),
        Book(title="Web开发实战", author="王五", isbn="1122334455", price=49.99, stock=8),
        Book(title="老人与海", author="海明威", isbn="9787544774332", price=29.80, stock=12),
        Book(title="活着", author="余华", isbn="9787506365437", price=39.00, stock=7),
        Book(title="三体", author="刘慈欣", isbn="9787536692930", price=58.00, stock=6),
    ]
    existing_titles = {row[0] for row in db.query(Book.title).all()}
    added = False
    for book in sample_books:
        if book.title not in existing_titles:
            db.add(book)
            added = True
    if added:
        db.commit()
    db.close()

init_sample_books()

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
app.mount("/static", StaticFiles(directory="static"), name="static")

# 模板
templates = Jinja2Templates(directory="templates")

# Pydantic模型
from pydantic import BaseModel

class UserCreate(BaseModel):
    username: str
    email: str
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str

# 导入RAG组件
from llm import LLM
from retriever import Retriever
from rerank import Reranker

llm = LLM()
retriever = Retriever()
reranker = Reranker()

def ask(query, user=None, db=None):
    direct_answer = _direct_db_answer(query, user=user, db=db)
    if direct_answer:
        return direct_answer

    # 检索静态知识库（任何一步失败都降级，避免接口500）
    try:
        queries = llm.expand_query(query)
    except Exception:
        queries = [query]
    if not queries:
        queries = [query]

    all_docs = []
    for q in queries:
        try:
            docs = retriever.hybrid_retrieve(q)
            all_docs.extend(docs)
        except Exception:
            continue
    all_docs = list(set(all_docs))

    # 注入数据库动态信息：全量书籍 + 当前用户购物车/订单
    if db:
        try:
            books = db.query(Book).all()
            book_docs = [
                f"书籍信息: 书名={b.title}, 作者={b.author}, ISBN={b.isbn}, 价格={b.price}, 库存={b.stock}"
                for b in books
            ]
            all_docs.extend(book_docs)
        except Exception:
            pass

    # 如果有用户，添加订单信息
    if user and db:
        try:
            orders = _load_orders_for_user(db, user.id)
            order_docs = []
            for o in orders:
                order_no = o.get("order_no", "未知")
                book_title = o.get("book_title", "未知")
                amount = o.get("amount", "未知")
                created_at = o.get("created_at", "未知")
                order_docs.append(f"订单号: {order_no}, 书籍: {book_title}, 金额: {amount}, 时间: {created_at}")
            all_docs.extend(order_docs)
        except Exception:
            # 兼容历史数据库结构，订单信息不可用时跳过。
            pass

        try:
            cart_rows = db.query(CartItem).filter(CartItem.user_id == user.id).all()
            cart_docs = [f"购物车: 书名={c.title}, 单价={c.price}, 数量={c.qty}" for c in cart_rows]
            all_docs.extend(cart_docs)
        except Exception:
            pass

    if not all_docs:
        all_docs = ["知识库暂时不可用，请稍后重试。"]

    try:
        reranked = reranker.rerank(query, all_docs, top_k=3)
    except Exception:
        reranked = all_docs[:3]
    context = "\n".join(reranked)
    prompt = f"你是一个智能客服，请基于提供的信息回答用户问题。\n【知识库信息】\n{context}\n【用户问题】\n{query}\n【回答要求】\n1. 只能使用知识库信息\n2. 不要编造\n3. 信息不足时说“抱歉，我无法回答这个问题”"
    try:
        return llm.generate(prompt)
    except Exception:
        return "抱歉，我无法回答这个问题。"

# API端点
@app.post("/register", response_model=Token)
def register(user: UserCreate, db: Session = Depends(get_db)):
    if db.query(User).filter(User.username == user.username).first():
        raise HTTPException(status_code=400, detail="用户名已被注册")
    hashed_password = get_password_hash(user.password)
    db_user = User(username=user.username, email=user.email, hashed_password=hashed_password)
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    access_token = create_access_token(data={"sub": user.username})
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/token", response_model=Token)
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = authenticate_user(db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(status_code=400, detail="用户名或密码错误")
    access_token = create_access_token(data={"sub": user.username})
    return {"access_token": access_token, "token_type": "bearer"}

@app.get("/ask")
def ask_api(q: str, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    if not current_user:
        raise HTTPException(status_code=401, detail="请先登录后再使用客服")
    if not q or not q.strip():
        raise HTTPException(status_code=400, detail="问题不能为空")
    try:
        return {"answer": ask(q.strip(), current_user, db)}
    except Exception:
        # 防止任何未预期错误导致前端显示500。
        return {"answer": "抱歉，客服暂时不可用，请稍后重试。"}

# 书店路由
@app.get("/")
def home(request: Request, db: Session = Depends(get_db)):
    books = db.query(Book).all()
    return templates.TemplateResponse("home.html", build_context(request, db, books=books))


@app.get("/chat")
def chat_page(current_user: User = Depends(get_current_user)):
    if not current_user:
        login_url = f"/login?next=/chat&msg={quote('请先登录后再使用客服聊天')}"
        return RedirectResponse(url=login_url, status_code=302)
    return RedirectResponse(url="/static/chat.html", status_code=302)

@app.get("/book/{book_id}")
def book_detail(request: Request, book_id: int, db: Session = Depends(get_db)):
    book = db.query(Book).filter(Book.id == book_id).first()
    if not book:
        raise HTTPException(status_code=404, detail="书籍未找到")
    return templates.TemplateResponse("book_detail.html", build_context(request, db, book=book))

@app.get("/search")
def search_books(request: Request, q: str = "", db: Session = Depends(get_db)):
    if q:
        books = db.query(Book).filter(Book.title.contains(q) | Book.author.contains(q)).all()
    else:
        books = db.query(Book).all()
    return templates.TemplateResponse("search.html", build_context(request, db, books=books, query=q))

@app.post("/add/{book_id}")
def add_to_cart(book_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    if not current_user:
        login_url = f"/login?next=/book/{book_id}&msg={quote('请先登录后再加入购物车')}"
        return RedirectResponse(url=login_url, status_code=302)
    book = db.query(Book).filter(Book.id == book_id).first()
    if not book:
        raise HTTPException(status_code=404, detail="书籍未找到")
    if int(book.stock or 0) <= 0:
        return RedirectResponse(url=f"/book/{book_id}?msg={quote('该书当前无库存，暂不可加入购物车')}", status_code=302)

    existing = db.query(CartItem).filter(CartItem.user_id == current_user.id, CartItem.book_id == book.id).first()
    if existing:
        if int(existing.qty or 0) + 1 > int(book.stock or 0):
            return RedirectResponse(url=f"/cart?msg={quote('加入失败：库存不足')}", status_code=302)
        existing.qty += 1
    else:
        db.add(CartItem(user_id=current_user.id, book_id=book.id, title=book.title, price=book.price, qty=1))
    db.commit()
    return RedirectResponse(url=f"/cart?msg={quote('已加入购物车')}", status_code=302)

@app.get("/cart")
def cart(request: Request, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    if not current_user:
        login_url = f"/login?next=/cart&msg={quote('请先登录后查看购物车')}"
        return RedirectResponse(url=login_url, status_code=302)
    rows = db.query(CartItem).filter(CartItem.user_id == current_user.id).order_by(CartItem.id.asc()).all()
    user_cart = [{"id": r.id, "title": r.title, "price": r.price, "qty": r.qty} for r in rows]
    return templates.TemplateResponse("cart.html", build_context(request, db, cart_items=user_cart))

@app.delete("/cart/{index}")
def remove_from_cart(index: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    if not current_user:
        raise HTTPException(status_code=401, detail="请先登录")
    rows = db.query(CartItem).filter(CartItem.user_id == current_user.id).order_by(CartItem.id.asc()).all()
    if 0 <= index < len(rows):
        db.delete(rows[index])
        db.commit()
        return {"message": "已移除"}
    raise HTTPException(status_code=404, detail="项未找到")

@app.get("/checkout")
def checkout(request: Request, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    if not current_user:
        login_url = f"/login?next=/checkout&msg={quote('请先登录后再结账')}"
        return RedirectResponse(url=login_url, status_code=302)
    return templates.TemplateResponse("checkout.html", build_context(request, db, success=False))

@app.post("/checkout")
def process_checkout(request: Request, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    if not current_user:
        login_url = f"/login?next=/checkout&msg={quote('请先登录后再结账')}"
        return RedirectResponse(url=login_url, status_code=302)
    cart_rows = db.query(CartItem).filter(CartItem.user_id == current_user.id).all()
    if not cart_rows:
        return templates.TemplateResponse("checkout.html", build_context(request, db, success=False, error="购物车为空，无法结账"))

    try:
        # 同步扣减库存并按数据库实际结构写入订单。
        for row in cart_rows:
            book = db.query(Book).filter(Book.id == row.book_id).first()
            if not book:
                return templates.TemplateResponse("checkout.html", build_context(request, db, success=False, error=f"商品不存在：{row.title}"))
            if int(book.stock or 0) < int(row.qty or 0):
                return templates.TemplateResponse("checkout.html", build_context(request, db, success=False, error=f"库存不足：{row.title}"))

        for row in cart_rows:
            book = db.query(Book).filter(Book.id == row.book_id).first()
            book.stock = int(book.stock or 0) - int(row.qty or 0)
        _create_orders_from_cart(db, current_user.id, cart_rows)
        db.query(CartItem).filter(CartItem.user_id == current_user.id).delete()
        db.commit()
    except Exception:
        db.rollback()
        return templates.TemplateResponse("checkout.html", build_context(request, db, success=False, error="订单提交失败，请稍后重试"))

    return templates.TemplateResponse("checkout.html", build_context(request, db, success=True))


@app.get("/my_orders")
def my_orders(request: Request, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    if not current_user:
        login_url = f"/login?next=/my_orders&msg={quote('请先登录后查看订单')}"
        return RedirectResponse(url=login_url, status_code=302)
    orders = _load_orders_for_user(db, current_user.id)
    return templates.TemplateResponse("my_orders.html", build_context(request, db, orders=orders))

@app.get("/login")
def login_page(request: Request, db: Session = Depends(get_db)):
    return templates.TemplateResponse("login.html", build_context(request, db))

@app.post("/login")
async def login_page_post(request: Request, db: Session = Depends(get_db)):
    form_data = await request.form()
    username = (form_data.get("username") or "").strip()
    password = (form_data.get("password") or "").strip()
    email = (form_data.get("email") or "").strip()
    next_url = request.query_params.get("next") or "/"

    if not username or not password:
        return templates.TemplateResponse("login.html", build_context(request, db, error="用户名和密码不能为空"))
    if len(username) < 3:
        return templates.TemplateResponse("login.html", build_context(request, db, error="用户名至少3个字符"))
    if len(password) < 6:
        return templates.TemplateResponse("login.html", build_context(request, db, error="密码至少6个字符"))

    user = db.query(User).filter(User.username == username).first()
    login_msg = "登录成功"
    if user:
        if not verify_password(password, user.hashed_password):
            return templates.TemplateResponse("login.html", build_context(request, db, error="用户名或密码错误"))
    else:
        safe_email = email or f"{username}@local.dev"
        user = User(username=username, email=safe_email, hashed_password=get_password_hash(password))
        db.add(user)
        db.commit()
        db.refresh(user)
        login_msg = "注册成功，已自动登录"

    access_token = create_access_token(data={"sub": username})
    target_url = append_msg_to_url(next_url, login_msg)
    response = RedirectResponse(url=target_url, status_code=302)
    response.set_cookie(key="access_token", value=access_token, httponly=True)
    return response

@app.get("/logout")
def logout():
    response = RedirectResponse(url=f"/?msg={quote('已退出登录')}", status_code=302)
    response.delete_cookie(key="access_token")
    return response
