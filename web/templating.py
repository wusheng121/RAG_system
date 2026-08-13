"""共享模板实例与 Jinja 过滤器。"""

from fastapi.templating import Jinja2Templates

from core.utils import format_money

templates = Jinja2Templates(directory="templates")
templates.env.filters["money"] = format_money
