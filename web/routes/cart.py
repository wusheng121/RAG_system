"""购物车与结账路由。"""

from urllib.parse import quote

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import RedirectResponse
from sqlalchemy import update
from sqlalchemy.orm import Session

from core.database import get_db
from models import Book, CartItem
from security.auth import build_context, get_current_user
from security.csrf import verify_csrf_token
from services.order_service import create_orders_from_cart
from web.templating import templates

router = APIRouter()


@router.post("/add/{book_id}")
def add_to_cart(
    book_id: int,
    current_user=Depends(get_current_user),
    db: Session = Depends(get_db),
    _csrf=Depends(verify_csrf_token),
):
    if not current_user:
        login_url = f"/login?next=/book/{book_id}&msg={quote('请先登录后再加入购物车')}"
        return RedirectResponse(url=login_url, status_code=302)
    book = db.query(Book).filter(Book.id == book_id).first()
    if not book:
        raise HTTPException(status_code=404, detail="书籍未找到")
    if int(book.stock or 0) <= 0:
        return RedirectResponse(
            url=f"/book/{book_id}?msg={quote('该书当前无库存，暂不可加入购物车')}", status_code=302
        )

    existing = (
        db.query(CartItem).filter(CartItem.user_id == current_user.id, CartItem.book_id == book.id).first()
    )
    if existing:
        if int(existing.qty or 0) + 1 > int(book.stock or 0):
            return RedirectResponse(url=f"/cart?msg={quote('加入失败：库存不足')}", status_code=302)
        existing.qty += 1
    else:
        db.add(CartItem(user_id=current_user.id, book_id=book.id, title=book.title, price=book.price, qty=1))
    db.commit()
    return RedirectResponse(url=f"/cart?msg={quote('已加入购物车')}", status_code=302)


@router.get("/cart")
def cart(request: Request, db: Session = Depends(get_db), current_user=Depends(get_current_user)):
    if not current_user:
        login_url = f"/login?next=/cart&msg={quote('请先登录后查看购物车')}"
        return RedirectResponse(url=login_url, status_code=302)
    rows = db.query(CartItem).filter(CartItem.user_id == current_user.id).order_by(CartItem.id.asc()).all()
    user_cart = [{"id": r.id, "title": r.title, "price": r.price, "qty": r.qty} for r in rows]
    return templates.TemplateResponse(request, "cart.html", build_context(request, db, cart_items=user_cart))


@router.delete("/cart/{item_id}")
def remove_from_cart(
    item_id: int,
    current_user=Depends(get_current_user),
    db: Session = Depends(get_db),
    _csrf=Depends(verify_csrf_token),
):
    """按购物车项 ID 删除（不再按列表位置，避免并发删错项）。"""
    if not current_user:
        raise HTTPException(status_code=401, detail="请先登录")
    item = db.query(CartItem).filter(CartItem.id == item_id, CartItem.user_id == current_user.id).first()
    if item:
        db.delete(item)
        db.commit()
        return {"message": "已移除"}
    raise HTTPException(status_code=404, detail="项未找到")


@router.get("/checkout")
def checkout(request: Request, db: Session = Depends(get_db), current_user=Depends(get_current_user)):
    if not current_user:
        login_url = f"/login?next=/checkout&msg={quote('请先登录后再结账')}"
        return RedirectResponse(url=login_url, status_code=302)
    return templates.TemplateResponse(request, "checkout.html", build_context(request, db, success=False))


@router.post("/checkout")
def process_checkout(
    request: Request,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
    _csrf=Depends(verify_csrf_token),
):
    if not current_user:
        login_url = f"/login?next=/checkout&msg={quote('请先登录后再结账')}"
        return RedirectResponse(url=login_url, status_code=302)
    cart_rows = db.query(CartItem).filter(CartItem.user_id == current_user.id).all()
    if not cart_rows:
        return templates.TemplateResponse(
            request, "checkout.html", build_context(request, db, success=False, error="购物车为空，无法结账")
        )

    try:
        # 原子扣减库存：UPDATE ... SET stock=stock-qty WHERE id=? AND stock>=qty
        # 依据 rowcount 判定是否成功，杜绝“先查后扣”在并发下超卖。
        for row in cart_rows:
            book = db.query(Book).filter(Book.id == row.book_id).first()
            if not book:
                return templates.TemplateResponse(
                    request,
                    "checkout.html",
                    build_context(request, db, success=False, error=f"商品不存在：{row.title}"),
                )
            qty = int(row.qty or 0)
            result = db.execute(
                update(Book).where(Book.id == row.book_id, Book.stock >= qty).values(stock=Book.stock - qty)
            )
            if result.rowcount == 0:
                db.rollback()
                return templates.TemplateResponse(
                    request,
                    "checkout.html",
                    build_context(request, db, success=False, error=f"库存不足：{row.title}"),
                )

        create_orders_from_cart(db, current_user.id, cart_rows)
        db.query(CartItem).filter(CartItem.user_id == current_user.id).delete()
        db.commit()
    except Exception:
        db.rollback()
        return templates.TemplateResponse(
            request,
            "checkout.html",
            build_context(request, db, success=False, error="订单提交失败，请稍后重试"),
        )

    # PRG：成功后重定向到订单页，避免刷新重复提交订单。
    return RedirectResponse(url=f"/my_orders?msg={quote('订单提交成功')}", status_code=302)
