"""订单查看路由。"""

from urllib.parse import quote

from fastapi import APIRouter, Depends, Request
from fastapi.responses import RedirectResponse
from sqlalchemy.orm import Session

from core.database import get_db
from security.auth import build_context, get_current_user
from services.order_service import load_orders_for_user
from web.templating import templates

router = APIRouter()


@router.get("/my_orders")
def my_orders(request: Request, db: Session = Depends(get_db), current_user=Depends(get_current_user)):
    if not current_user:
        login_url = f"/login?next=/my_orders&msg={quote('请先登录后查看订单')}"
        return RedirectResponse(url=login_url, status_code=302)
    orders = load_orders_for_user(db, current_user.id)
    return templates.TemplateResponse(request, "my_orders.html", build_context(request, db, orders=orders))
