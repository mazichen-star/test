"""运行时配置。

配置优先级：环境变量 > Streamlit Secrets > 默认值。

这样做的目的：
1. 本地没有 ``.streamlit/secrets.toml`` 时不会因为访问 ``st.secrets`` 而崩溃；
2. 同一套代码可以跑在 Streamlit Cloud、SealOS、本地 SQLite 或 Supabase；
3. 单元测试不需要启动 Streamlit。
"""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LOCAL_DB = PROJECT_ROOT / "data" / "app.db"


def get_secret(key: str, default: str = "") -> str:
    """读取配置，优先环境变量，其次 Streamlit Secrets。

    注意：这里故意做了宽泛的异常捕获。不同 Streamlit 版本在 secrets 文件
    缺失时可能抛 ``FileNotFoundError``、``KeyError`` 或自定义异常。
    """

    value = os.environ.get(key)
    if value is not None and str(value).strip() != "":
        return str(value)

    try:
        import streamlit as st

        try:
            value = st.secrets.get(key, default)  # type: ignore[union-attr]
        except Exception:
            value = default
    except Exception:
        value = default

    return str(value) if value is not None else default


def supabase_url() -> str:
    return get_secret("SUPABASE_URL").strip().rstrip("/")


def supabase_key() -> str:
    return get_secret("SUPABASE_KEY").strip()


def use_supabase() -> bool:
    return bool(supabase_url() and supabase_key())


def local_db_path() -> Path:
    raw = get_secret("LOCAL_DB_PATH").strip()
    return Path(raw).expanduser() if raw else DEFAULT_LOCAL_DB


def smtp_configured() -> bool:
    return bool(
        get_secret("SMTP_HOST").strip()
        and get_secret("SMTP_USER").strip()
        and get_secret("SMTP_PASSWORD").strip()
    )


def ai_config() -> tuple[str, str, str]:
    """返回 ``(api_key, base_url, model)``。"""

    deepseek_key = get_secret("DEEPSEEK_API_KEY").strip()
    ark_key = get_secret("ARK_API_KEY").strip()
    if deepseek_key:
        return (
            deepseek_key,
            get_secret("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1/chat/completions"),
            get_secret("DEEPSEEK_MODEL", "deepseek-chat"),
        )
    if ark_key:
        return (
            ark_key,
            get_secret(
                "ARK_BASE_URL",
                "https://ark.cn-beijing.volces.com/api/v3/chat/completions",
            ),
            get_secret("ARK_MODEL"),
        )
    return "", "", ""
