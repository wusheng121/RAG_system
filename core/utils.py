"""通用工具：金额格式化等纯函数。"""

from __future__ import annotations


def format_money(value) -> str:
    """统一金额格式为 ￥xx.xx，供后端拼接与 Jinja 过滤器共用。"""
    try:
        return f"￥{float(value):.2f}"
    except Exception:
        return f"￥{value}"
