"""智能客服路由：聊天页面与问答接口。"""

from urllib.parse import quote

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import RedirectResponse
from sqlalchemy.orm import Session

from core.database import get_db
from security.auth import build_context, get_current_user
from services.rag_service import ask
from web.templating import templates

router = APIRouter()


@router.get("/chat")
def chat_page(request: Request, current_user=Depends(get_current_user), db: Session = Depends(get_db)):
    if not current_user:
        login_url = f"/login?next=/chat&msg={quote('请先登录后再使用客服聊天')}"
        return RedirectResponse(url=login_url, status_code=302)
    return templates.TemplateResponse(request, "chat.html", build_context(request, db))


@router.get("/ask")
def ask_api(q: str, current_user=Depends(get_current_user), db: Session = Depends(get_db)):
    if not current_user:
        raise HTTPException(status_code=401, detail="请先登录后再使用客服")
    if not q or not q.strip():
        raise HTTPException(status_code=400, detail="问题不能为空")
    try:
        return {"answer": ask(q.strip(), current_user, db)}
    except Exception:
        # 防止任何未预期错误导致前端显示 500。
        return {"answer": "抱歉，客服暂时不可用，请稍后重试。"}
