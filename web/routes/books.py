"""书店浏览路由：首页、书籍详情、搜索。"""

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session

from core.database import get_db
from models import Book
from security.auth import build_context
from web.templating import templates

router = APIRouter()


@router.get("/")
def home(request: Request, db: Session = Depends(get_db)):
    books = db.query(Book).all()
    return templates.TemplateResponse(request, "home.html", build_context(request, db, books=books))


@router.get("/book/{book_id}")
def book_detail(request: Request, book_id: int, db: Session = Depends(get_db)):
    book = db.query(Book).filter(Book.id == book_id).first()
    if not book:
        raise HTTPException(status_code=404, detail="书籍未找到")
    return templates.TemplateResponse(request, "book_detail.html", build_context(request, db, book=book))


@router.get("/search")
def search_books(request: Request, q: str = "", db: Session = Depends(get_db)):
    if q:
        books = db.query(Book).filter(Book.title.contains(q) | Book.author.contains(q)).all()
    else:
        books = db.query(Book).all()
    return templates.TemplateResponse(
        request, "search.html", build_context(request, db, books=books, query=q)
    )
