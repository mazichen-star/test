# app.py
# 材料力学性能预测系统 - Streamlit
# 运行: pip install -r requirements.txt ; streamlit run app.py

from __future__ import annotations

import io
import json
import os
import re
import secrets
import smtplib
import time
from datetime import datetime
from email.header import Header
from email.mime.text import MIMEText
from email.utils import formataddr

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import streamlit as st
from matplotlib import font_manager
from sklearn.gaussian_process import GaussianProcessRegressor

from core import adhesion as adhesion_core
from core.config import ai_config, get_secret, smtp_configured
from core.data import (
    DataError,
    dataset_signature,
    default_feature_columns,
    default_target_column,
    frame_signature,
    load_table_bytes,
    training_signature,
)
from core.modeling import (
    GPR_KERNEL_NAMES,
    MODEL_NAMES,
    ModelingError,
    predict_one,
    train_models,
)
from core.plotting import configure_plot_style, style_figure
from core.reporting import excel_bytes, make_word_report
from core.security import valid_password, valid_username
from core.storage import StorageError, get_storage, storage_label
from core.supabase_auth import (
    SupabaseAuth,
    SupabaseAuthError,
    friendly_auth_error,
)

try:
    from skopt import gp_minimize
    from skopt.space import Real

    _HAS_SKOPT = True
except Exception:
    _HAS_SKOPT = False

# ============ 页面与字体 ============
st.set_page_config(
    page_title="Materia · 力学性能工作台",
    page_icon="🧪",
    layout="wide",
)

_FONT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "fonts", "NotoSansSC-Regular.ttf"
)
if os.path.exists(_FONT_PATH):
    font_manager.fontManager.addfont(_FONT_PATH)
    _CN_FONT = font_manager.FontProperties(fname=_FONT_PATH).get_name()
else:
    _CN_FONT = "Microsoft YaHei"

matplotlib.rcParams["font.sans-serif"] = [
    _CN_FONT,
    "Noto Sans CJK SC",
    "WenQuanYi Zen Hei",
    "Microsoft YaHei",
    "SimHei",
    "DejaVu Sans",
]
matplotlib.rcParams["axes.unicode_minus"] = False
configure_plot_style(_CN_FONT)


# ============ 多语言与主题 ============
st.sidebar.markdown(
    """
    <div class="brand">
      <svg class="brand-mark" viewBox="0 0 32 32" fill="none" aria-hidden="true">
        <rect x="1" y="1" width="30" height="30" rx="4" stroke="currentColor" stroke-width="1.4"/>
        <path d="M8.5 22.5 16 8.5l7.5 14M11.5 17.5h9" stroke="currentColor" stroke-width="1.4"/>
      </svg>
      <div>
        <div class="brand-name">Materia</div>
        <div class="brand-sub">材料研发工作台</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)
LANGS = ["中文", "English"]
if "ui_lang" not in st.session_state:
    st.session_state["ui_lang"] = "中文"
st.session_state["ui_lang"] = st.sidebar.selectbox(
    "Language / 语言",
    LANGS,
    index=LANGS.index(st.session_state["ui_lang"]),
    key="lang_select",
)
EN = st.session_state["ui_lang"] == "English"


def tr(zh: str, en: str) -> str:
    return en if EN else zh


BASE_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Serif:wght@500;600&display=swap');

:root {
    --ground: #edf1ef;
    --paper: #fbfcfb;
    --ink: #17201e;
    --muted: #5d6a66;
    --line: #d5ddda;
    --line-strong: #b6c1bd;
    --tone-1: #f3f6f4;
    --tone-2: #e5ebe8;
    --accent: #0f5c5e;
    --accent-soft: #d9e7e4;
    --danger: #9d3c32;
    --radius: 4px;
    --font-sans: "IBM Plex Sans", "Noto Sans SC", "Microsoft YaHei", "PingFang SC", sans-serif;
    --font-display: "IBM Plex Serif", "Noto Serif SC", "Songti SC", "SimSun", Georgia, serif;
}

html, body, [data-testid="stAppViewContainer"], .stApp {
    background: var(--ground);
    color: var(--ink);
    font-family: var(--font-sans) !important;
    color-scheme: light;
}
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li,
input, textarea, button, select {
    font-family: var(--font-sans) !important;
}
[data-testid="stAppViewContainer"] > .main,
[data-testid="stMain"] { background: var(--paper); }
[data-testid="stHeader"] { background: transparent; }
[data-testid="stDecoration"] { display: none; }
[data-testid="stMainBlockContainer"] {
    max-width: 1660px;
    padding: 2.2rem clamp(1.2rem, 2.8vw, 2.8rem) 4rem;
}
[data-testid="stSidebar"] {
    background: var(--ground);
    min-width: 300px;
    max-width: 330px;
}
[data-testid="stSidebar"] > div:first-child { padding-top: 1.35rem; }
[data-testid="stSidebar"] [data-testid="stVerticalBlock"] { gap: .55rem; }

h1, h2, h3, h4 {
    font-family: var(--font-display) !important;
    color: var(--ink);
    letter-spacing: -.015em;
    font-weight: 500;
}
h2, h3 { font-weight: 600; }
h1 { font-size: clamp(2rem, 2.4vw, 2.7rem); line-height: 1.08; margin: 0 0 .35rem; }
h2 { font-size: clamp(1.35rem, 1.6vw, 1.75rem); line-height: 1.18; margin: 1.6rem 0 .5rem; }
h3 { font-size: 1.15rem; margin: 1.1rem 0 .35rem; }
p, li, [data-testid="stMarkdownContainer"] p { color: var(--ink); line-height: 1.55; }
[data-testid="stCaptionContainer"] p,
[data-testid="stMarkdownContainer"] small { color: var(--muted); font-size: 11px; }
label, [data-testid="stWidgetLabel"] p { color: var(--muted); font-size: 12px; font-weight: 500; }

/* Shell */
.masthead {
    display: flex;
    justify-content: space-between;
    align-items: flex-end;
    gap: 2rem;
    padding: .35rem 0 1.7rem;
}
.masthead h1 { max-width: 15ch; }
.masthead p {
    max-width: 58ch;
    margin: .55rem 0 0;
    color: var(--muted);
    font-size: 15px;
}
.status-row {
    display: flex;
    flex-wrap: wrap;
    justify-content: flex-end;
    gap: .45rem;
    padding-bottom: .25rem;
}
.status-chip {
    background: var(--tone-2);
    color: var(--ink);
    padding: 5px 9px;
    font-size: 11px;
    line-height: 1;
    border-radius: 3px;
    white-space: nowrap;
}
.status-chip.quiet { background: var(--tone-1); color: var(--muted); }

.brand { display: flex; align-items: center; gap: 10px; padding: .1rem 0 1.2rem; }
.brand-mark { width: 30px; height: 30px; flex: 0 0 30px; color: var(--accent); }
.brand-name { font-family: var(--font-display); font-size: 20px; line-height: 1; letter-spacing: -.01em; }
.brand-sub { color: var(--muted); font-size: 11px; margin-top: 4px; }
.side-section {
    color: var(--muted);
    font-size: 11px;
    font-weight: 500;
    margin: 1.35rem 0 .35rem;
}
.side-space { height: .8rem; }
.side-user { display: flex; align-items: center; gap: 10px; padding: .55rem 0 .3rem; }
.avatar {
    width: 30px; height: 30px; border-radius: 50%;
    display: grid; place-items: center;
    background: var(--accent-soft); color: var(--accent);
    font-size: 12px; font-weight: 600;
}
.side-user strong { display: block; font-size: 13px; line-height: 1.2; }
.side-user span { display: block; color: var(--muted); font-size: 11px; margin-top: 2px; }

.data-line {
    display: flex;
    align-items: baseline;
    gap: 8px;
    margin: .2rem 0 .45rem;
    color: var(--muted);
    font-size: 11px;
}
.data-line strong { color: var(--ink); font-weight: 500; font-size: 12px; }
.empty-state {
    max-width: 820px;
    margin-top: 1.4rem;
    padding: 2rem 2.2rem;
    background: var(--tone-1);
    border-radius: var(--radius);
}
.empty-state h2 { margin: 0 0 .55rem; font-size: 1.6rem; }
.empty-state p { margin: 0; max-width: 62ch; color: var(--muted); font-size: 14px; }
.empty-points {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 1.4rem;
    margin-top: 1.6rem;
}
.empty-points div { color: var(--muted); font-size: 12px; line-height: 1.5; }
.empty-points strong {
    display: block;
    color: var(--ink);
    font-size: 12px;
    font-weight: 500;
    margin-bottom: .25rem;
}

.auth-aside {
    background: var(--tone-1);
    border-radius: var(--radius);
    padding: 2rem 2rem 1.6rem;
    position: sticky;
    top: 1.5rem;
}
.auth-aside h2 {
    margin: 0 0 .7rem;
    font-size: 1.55rem;
    line-height: 1.2;
    max-width: 16ch;
}
.auth-aside p {
    margin: 0 0 1.7rem;
    color: var(--muted);
    font-size: 13px;
    max-width: 38ch;
}
.aside-item { margin: 1.25rem 0; }
.aside-item strong {
    display: block;
    font-size: 12px;
    font-weight: 500;
    color: var(--ink);
    margin-bottom: .2rem;
}
.aside-item span {
    display: block;
    font-size: 12px;
    line-height: 1.5;
    color: var(--muted);
}
.aside-note {
    margin-top: 1.8rem;
    font-size: 11px;
    color: var(--muted);
}

/* Navigation */
[data-testid="stTabs"] > div > div[role="tablist"] {
    gap: 2px;
    background: transparent;
    border-bottom: 0;
    padding: 0;
    margin-bottom: 1.15rem;
}
[data-testid="stTabs"] [data-testid="stTab"] {
    height: auto;
    padding: 9px 14px;
    border-radius: var(--radius);
    color: var(--muted);
    font-family: var(--font-sans);
    font-size: 13px;
    font-weight: 500;
    cursor: pointer;
}
[data-testid="stTabs"] [data-testid="stTab"] p {
    color: inherit;
    font-family: var(--font-sans) !important;
}
[data-testid="stTabs"] [data-testid="stTab"]:hover {
    background: var(--tone-2);
    color: var(--ink);
}
[data-testid="stTabs"] [data-testid="stTab"][aria-selected="true"] {
    background: var(--ink);
    color: var(--paper);
}
[data-testid="stTabs"] .react-aria-SelectionIndicator { display: none; }
[data-testid="stTabs"] [data-testid="stTabs"] [data-testid="stTab"] {
    font-size: 12px;
    padding: 7px 11px;
    color: var(--muted);
}
[data-testid="stTabs"] [data-testid="stTabs"] [data-testid="stTab"][aria-selected="true"] {
    background: var(--tone-2);
    color: var(--ink);
}

/* Controls */
.stButton > button,
.stDownloadButton > button,
.stFormSubmitButton > button {
    min-height: 2.15rem;
    border-radius: var(--radius);
    border: 1px solid var(--line-strong);
    background: transparent;
    color: var(--ink);
    font-family: var(--font-sans);
    font-weight: 500;
    box-shadow: none;
}
.stButton > button:hover,
.stDownloadButton > button:hover,
.stFormSubmitButton > button:hover {
    border-color: var(--ink);
    background: var(--tone-2);
    color: var(--ink);
}
.stButton > button[kind="primary"],
.stFormSubmitButton > button[kind="primary"],
.stDownloadButton > button[kind="primary"] {
    background: var(--ink);
    border-color: var(--ink);
    color: var(--paper);
}
.stButton > button[kind="primary"]:hover,
.stFormSubmitButton > button[kind="primary"]:hover {
    background: var(--accent);
    border-color: var(--accent);
    color: #fff;
}
[data-testid="stSidebar"] .stButton > button { width: 100%; text-align: left; }

input, textarea,
[data-baseweb="input"],
[data-baseweb="select"] > div,
[data-baseweb="base-input"] {
    background: var(--paper) !important;
    color: var(--ink) !important;
    border-color: var(--line) !important;
    border-radius: var(--radius) !important;
    box-shadow: none !important;
}
input:focus, textarea:focus,
[data-baseweb="input"]:focus-within,
[data-baseweb="select"] > div:focus-within {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 1px var(--accent) !important;
}
[data-baseweb="select"] [data-testid="stMarkdownContainer"] p { color: var(--ink) !important; }
[data-testid="stNumberInput"] button { background: var(--tone-1); color: var(--ink); border-color: var(--line); }

[data-testid="stMetric"] {
    background: transparent;
    padding: 0 1.2rem 0 0;
    border-radius: 0;
}
[data-testid="stMetricLabel"] p { color: var(--muted); font-size: 11px; }
[data-testid="stMetricValue"] {
    font-family: var(--font-display);
    color: var(--ink);
    font-size: 30px;
    line-height: 1.2;
}
[data-testid="stMetricDelta"] { font-size: 11px; }

[data-testid="stExpander"] {
    border: 0;
    border-radius: var(--radius);
    background: var(--tone-1);
    overflow: hidden;
}
[data-testid="stExpander"] details { border: 0; }
[data-testid="stExpander"] summary {
    padding: .65rem .8rem;
    font-weight: 500;
    color: var(--ink);
}
[data-testid="stExpander"] [data-testid="stExpanderDetails"] { padding: 0 .8rem .8rem; }

[data-testid="stForm"] {
    background: var(--tone-1);
    padding: 1rem 1.1rem;
    border-radius: var(--radius);
    border: 0;
}

[data-testid="stFileUploaderDropzone"] {
    background: var(--tone-1) !important;
    border: 1px solid var(--line-strong) !important;
    border-radius: var(--radius) !important;
    padding: 1.1rem !important;
}
[data-testid="stFileUploaderDropzone"] * { color: var(--ink) !important; }
[data-testid="stFileUploaderDropzone"] small { color: var(--muted) !important; }

[data-testid="stDataFrame"] {
    border: 1px solid var(--line);
    border-radius: 3px;
    overflow: hidden;
}
[data-testid="stAlert"] {
    border: 0 !important;
    border-radius: var(--radius) !important;
    background: var(--tone-1) !important;
    color: var(--ink) !important;
}
[data-testid="stAlert"] p { color: var(--ink) !important; }
[data-testid="stAlert"] svg { color: var(--accent); }
[data-testid="stButton"] button p,
[data-testid="stButton"] button span,
[data-testid="stFormSubmitButton"] button p,
[data-testid="stFormSubmitButton"] button span,
[data-testid="stDownloadButton"] button p,
[data-testid="stDownloadButton"] button span,
[data-testid="stFileUploaderDropzone"] button p {
    color: inherit !important;
}

hr { border: 0; height: 18px; background: transparent; margin: 0; }
[data-testid="stSlider"] [data-baseweb="slider"] { padding-top: .2rem; }
[data-testid="stCheckbox"] label p, [data-testid="stRadio"] label p { color: var(--ink); }
[data-testid="stCode"] { border-radius: var(--radius); }

.stPlot, [data-testid="stImage"] { display: flex; justify-content: center; }
.stPlot img, [data-testid="stImage"] img { margin: auto; max-width: 100%; height: auto; }

@media (max-width: 900px) {
    [data-testid="stMainBlockContainer"] { padding-top: 1.4rem; }
    .masthead { display: block; }
    .masthead h1 { max-width: none; }
    .status-row { justify-content: flex-start; padding-top: .9rem; }
    .masthead p { font-size: 14px; }
}
@media (max-width: 768px) {
    h2 { font-size: 1.25rem; }
    [data-testid="stHorizontalBlock"] { flex-direction: column; }
    [data-testid="stHorizontalBlock"] > div { width: 100% !important; }
    [data-testid="stTabs"] [data-testid="stTab"] { padding: 8px 10px; font-size: 12px; }
}
</style>
"""

st.markdown(BASE_CSS, unsafe_allow_html=True)

# ============ 存储初始化 ============
try:
    STORAGE = get_storage()
except Exception as exc:
    st.error(f"存储初始化失败：{exc}")
    st.stop()

SUPABASE_AUTH: SupabaseAuth | None = None
if STORAGE.mode == "supabase":
    try:
        SUPABASE_AUTH = SupabaseAuth()
    except Exception as exc:
        st.error(f"Supabase Auth 初始化失败：{exc}")
        st.stop()
    _token = st.session_state.get("sb_access_token")
    if _token:
        STORAGE.set_access_token(_token)

_ai_key, _, _ai_model = ai_config()
_ai_status = tr("AI 已接入", "AI connected") if _ai_key else tr("AI 未配置", "AI not configured")
st.markdown(
    f"""
    <div class="masthead">
      <div>
        <h1>{tr("力学性能工作台", "Mechanical property workspace")}</h1>
        <p>{tr(
            "上传材料数据，比较模型，导出可复现的预测结果。",
            "Upload material data, compare models, and export reproducible predictions.",
        )}</p>
      </div>
      <div class="status-row">
        <span class="status-chip">{storage_label()}</span>
        <span class="status-chip{' quiet' if not _ai_key else ''}">{_ai_status}</span>
        <span class="status-chip quiet">{tr("10 个回归模型", "10 regressors")}</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)


def set_flash(message: str, kind: str = "success") -> None:
    st.session_state["_flash"] = (kind, message)


def show_flash() -> None:
    item = st.session_state.pop("_flash", None)
    if not item:
        return
    kind, message = item
    getattr(st, kind, st.info)(message)


def set_adhesion_flash(message: str, kind: str = "success") -> None:
    st.session_state["_adhesion_flash"] = (kind, message)


def show_adhesion_flash() -> None:
    item = st.session_state.pop("_adhesion_flash", None)
    if not item:
        return
    kind, message = item
    getattr(st, kind, st.info)(message)


def clear_user_state() -> None:
    """退出登录时清除模型、文件和页面状态，避免下一个账号看到上一个账号的数据。"""

    keep = {"ui_lang", "dark_mode", "lang_select", "dark_toggle"}
    for key in list(st.session_state.keys()):
        if key in keep:
            continue
        st.session_state.pop(key, None)


def fmt(value, digits: int = 4) -> str:
    if value is None:
        return "—"
    try:
        if isinstance(value, float) and np.isnan(value):
            return "—"
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def send_email(to_addr: str, subject: str, body: str) -> tuple[bool, str]:
    """通过 SMTP 发送邮件；支持 SSL(465) 与 STARTTLS(587)。"""

    if not smtp_configured():
        return False, "未配置 SMTP"

    host = get_secret("SMTP_HOST").strip()
    try:
        port = int(get_secret("SMTP_PORT", "465"))
    except ValueError:
        port = 465
    user = get_secret("SMTP_USER").strip()
    password = get_secret("SMTP_PASSWORD")
    use_ssl = get_secret("SMTP_USE_SSL", "true").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )

    message = MIMEText(body, "plain", "utf-8")
    message["From"] = formataddr((str(Header("力学预测系统", "utf-8")), user))
    message["To"] = to_addr
    message["Subject"] = Header(subject, "utf-8")
    try:
        if use_ssl:
            server = smtplib.SMTP_SSL(host, port, timeout=20)
        else:
            server = smtplib.SMTP(host, port, timeout=20)
        with server:
            if not use_ssl:
                server.starttls()
            server.login(user, password)
            server.sendmail(user, [to_addr], message.as_string())
        return True, "ok"
    except Exception as exc:
        return False, str(exc)


def _render_auth_forms() -> None:
    """登录 / 注册 / 找回密码表单。"""

    tab_login, tab_reg, tab_forgot = st.tabs(
        [tr("登录", "Login"), tr("注册", "Register"), tr("忘记密码", "Forgot password")]
    )

    with tab_login:
        with st.form("login_form"):
            username = st.text_input(tr("用户名", "Username"), key="login_u")
            password = st.text_input(
                tr("密码", "Password"), type="password", key="login_p"
            )
            submitted = st.form_submit_button(
                tr("登录", "Login"), type="primary", use_container_width=True
            )
        if submitted:
            if not username or not password:
                st.warning(tr("请输入用户名和密码", "Please enter username and password"))
            else:
                try:
                    if STORAGE.verify_user(username, password):
                        clear_user_state()
                        st.session_state["username"] = username
                        st.rerun()
                    st.error(tr("用户名或密码错误", "Wrong username or password"))
                except StorageError as exc:
                    st.error(f"{tr('登录失败', 'Login failed')}：{exc}")

    with tab_reg:
        st.caption(
            tr(
                "用户名 5-20 位，只能包含数字、字母、下划线；密码至少 8 位，须包含数字和字母。",
                "Username: 5-20 letters/digits/underscore. Password: 8+ chars with letters and digits.",
            )
        )
        with st.form("register_form"):
            username2 = st.text_input(tr("用户名", "Username"), key="reg_u")
            email2 = st.text_input(
                tr("绑定邮箱（用于找回密码）", "Email (for password recovery)"),
                key="reg_email",
            )
            password2 = st.text_input(
                tr("设置密码", "Password"), type="password", key="reg_p"
            )
            password3 = st.text_input(
                tr("确认密码", "Confirm password"), type="password", key="reg_p3"
            )
            submitted2 = st.form_submit_button(
                tr("注册", "Register"), use_container_width=True
            )
        if submitted2:
            if not username2 or not password2:
                st.warning(tr("用户名和密码不能为空", "Username and password are required"))
            elif not valid_username(username2):
                st.warning(
                    tr(
                        "用户名需 5-20 位，只能使用数字、字母、下划线",
                        "Username must be 5-20 letters/digits/underscore",
                    )
                )
            elif not valid_password(password2):
                st.warning(
                    tr(
                        "密码至少 8 位，须同时包含数字和字母，不能含空格",
                        "Password must be 8+ chars and contain letters and digits",
                    )
                )
            elif not re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", email2 or ""):
                st.warning(tr("请输入有效邮箱", "Please enter a valid email"))
            elif password2 != password3:
                st.warning(tr("两次密码不一致", "Passwords do not match"))
            else:
                try:
                    if STORAGE.user_exists(username2):
                        st.warning(
                            tr(
                                "该用户名已被注册，请换一个",
                                "Username already registered",
                            )
                        )
                    else:
                        STORAGE.add_user(username2, password2, email2)
                        st.success(tr("注册成功，请登录", "Registered. Please log in."))
                except StorageError as exc:
                    st.error(f"{tr('注册失败', 'Registration failed')}：{exc}")

    with tab_forgot:
        st.caption(
            tr(
                "输入注册时绑定的邮箱，系统会生成临时密码并发送到该邮箱。"
                "出于安全考虑，页面不会显示临时密码。",
                "Enter your bound email. A temporary password will be emailed to you; "
                "it is never displayed on this page.",
            )
        )
        if not smtp_configured():
            st.warning(
                tr(
                    "未配置 SMTP，自助找回密码不可用。请联系管理员重置。",
                    "SMTP is not configured; self-service recovery is unavailable.",
                )
            )
        with st.form("forgot_form"):
            email_f = st.text_input(
                tr("绑定邮箱", "Bound email"), key="forgot_email"
            )
            submitted3 = st.form_submit_button(
                tr("找回密码", "Recover"),
                disabled=not smtp_configured(),
                use_container_width=True,
            )
        if submitted3:
            generic = tr(
                "如果该邮箱已注册，重置邮件已发送，请检查收件箱。",
                "If the email is registered, a reset message has been sent.",
            )
            try:
                user = STORAGE.find_user_by_email(email_f.strip())
            except StorageError as exc:
                st.error(f"{tr('查询失败', 'Lookup failed')}：{exc}")
                user = None
            if user:
                old_hash = user.get("password_hash")
                temp_password = secrets.token_urlsafe(9)
                try:
                    STORAGE.change_password(user["username"], temp_password)
                    ok, detail = send_email(
                        email_f.strip(),
                        "Mechanical Prediction - Password Recovery",
                        (
                            f"Hello,\n\n"
                            f"Your temporary password for {user['username']} is: "
                            f"{temp_password}\n\n"
                            "Please log in and change it in Account settings.\n"
                            "This message is generated automatically; do not reply."
                        ),
                    )
                    if not ok:
                        # 邮件失败必须回滚，绝不能在页面上显示临时密码
                        if old_hash:
                            STORAGE.set_password_hash(user["username"], old_hash)
                        st.error(
                            tr(
                                f"邮件发送失败，密码未修改。请稍后重试或联系管理员。（{detail}）",
                                f"Email failed and the password was not changed. ({detail})",
                            )
                        )
                    else:
                        st.success(generic)
                except StorageError as exc:
                    st.error(f"{tr('重置失败', 'Reset failed')}：{exc}")
            else:
                # 不暴露邮箱是否注册
                st.success(generic)


def _render_auth_aside() -> None:
    st.markdown(
        f"""
        <aside class="auth-aside">
          <h2>{tr("把实验数据变成可检查的预测。", "Turn lab data into predictions you can inspect.")}</h2>
          <p>{tr(
              "一个面向材料研发的工作台：数据、模型和文献证据放在同一处，结果可导出、可复核。",
              "A workspace for materials R&D: data, models and literature evidence in one place.",
          )}</p>
          <div class="aside-item">
            <strong>{tr("你的数据", "Your data")}</strong>
            <span>{tr("上传 CSV / Excel，或从历史文件继续。", "Upload CSV / Excel or continue from a saved file.")}</span>
          </div>
          <div class="aside-item">
            <strong>{tr("你的模型", "Your models")}</strong>
            <span>{tr("从线性回归到 GPR，比较后选最优，并给出预测区间。", "From linear models to GPR, with prediction intervals.")}</span>
          </div>
          <div class="aside-item">
            <strong>{tr("你的证据", "Your evidence")}</strong>
            <span>{tr("文献提取、材料库、Excel 与 Word 报告都可追溯。", "Literature extraction, material library and exports.")}</span>
          </div>
          <div class="aside-note">{tr("当前存储", "Storage")} · {storage_label()}</div>
        </aside>
        """,
        unsafe_allow_html=True,
    )


def _set_cloud_session(session: dict, email: str) -> None:
    token = session["access_token"]
    user = session.get("user") or SUPABASE_AUTH.get_user(token)
    user_id = user.get("id")
    if not user_id:
        raise SupabaseAuthError("登录成功但未返回用户 ID")
    profile = SUPABASE_AUTH.get_profile(token, user_id)
    if not profile:
        metadata = user.get("user_metadata") or {}
        username = str(metadata.get("username") or email.split("@")[0])
        profile = SUPABASE_AUTH.ensure_profile(token, user_id, username, email)
    clear_user_state()
    st.session_state.update(
        {
            "username": profile["username"],
            "sb_access_token": token,
            "sb_refresh_token": session.get("refresh_token", ""),
            "sb_expires_at": session.get("expires_at", time.time() + 3600),
            "sb_email": email,
            "sb_user_id": user_id,
        }
    )
    STORAGE.set_access_token(token)


def _ensure_cloud_session() -> bool:
    if not SUPABASE_AUTH:
        return True
    token = st.session_state.get("sb_access_token")
    if not token:
        return False
    expires_at = float(st.session_state.get("sb_expires_at") or 0)
    if expires_at and time.time() > expires_at - 60:
        refresh_token = st.session_state.get("sb_refresh_token", "")
        if not refresh_token:
            return False
        try:
            session = SUPABASE_AUTH.refresh(refresh_token)
        except SupabaseAuthError:
            return False
        st.session_state.update(
            {
                "sb_access_token": session["access_token"],
                "sb_refresh_token": session.get("refresh_token", refresh_token),
                "sb_expires_at": session.get("expires_at", time.time() + 3600),
            }
        )
        token = session["access_token"]
    STORAGE.set_access_token(token)
    return bool(st.session_state.get("username"))


def _cloud_auth_notice(kind: str, text: str) -> None:
    st.session_state["cloud_auth_notice"] = (kind, text)


def _render_cloud_auth_forms() -> None:
    st.caption(
        tr(
            "云账号使用邮箱登录，密码和会话由 Supabase Auth 管理；数据访问通过 RLS 绑定到当前用户。",
            "Cloud accounts use email; Supabase Auth and RLS bind data access to the current user.",
        )
    )
    login_label = tr("登录", "Login")
    register_label = tr("注册", "Register")
    forgot_label = tr("忘记密码", "Forgot password")
    choice = st.segmented_control(
        tr("账号操作", "Account action"),
        [login_label, register_label, forgot_label],
        default=login_label,
        key="cloud_auth_mode",
        label_visibility="collapsed",
    )
    choice = choice or login_label

    if choice == login_label:
        with st.form("cloud_login_form"):
            email = st.text_input(tr("邮箱", "Email"), key="cloud_login_email")
            password = st.text_input(
                tr("密码", "Password"), type="password", key="cloud_login_pw"
            )
            submitted = st.form_submit_button(
                tr("登录", "Login"), type="primary", use_container_width=True
            )
        if submitted:
            if not email.strip() or not password:
                _cloud_auth_notice(
                    "warning", tr("请输入邮箱和密码", "Please enter email and password")
                )
            elif not re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", email or ""):
                _cloud_auth_notice(
                    "warning", tr("请输入有效邮箱", "Please enter a valid email")
                )
            else:
                try:
                    with st.spinner(tr("正在登录…", "Signing in…")):
                        session = SUPABASE_AUTH.sign_in(email.strip(), password)
                        _set_cloud_session(session, email.strip())
                    st.rerun()
                except SupabaseAuthError as exc:
                    _cloud_auth_notice(
                        "error",
                        f"{tr('登录失败', 'Login failed')}：{friendly_auth_error(exc)}",
                    )

    elif choice == register_label:
        st.caption(
            tr(
                "用户名 5-20 位，只能包含数字、字母、下划线；密码至少 8 位，须包含数字和字母。",
                "Username: 5-20 letters/digits/underscore. Password: 8+ chars with letters and digits.",
            )
        )
        with st.form("cloud_register_form"):
            username = st.text_input(tr("用户名", "Username"), key="cloud_reg_u")
            email = st.text_input(tr("邮箱", "Email"), key="cloud_reg_email")
            password = st.text_input(
                tr("设置密码", "Password"), type="password", key="cloud_reg_pw"
            )
            confirm = st.text_input(
                tr("确认密码", "Confirm password"), type="password", key="cloud_reg_pw2"
            )
            submitted = st.form_submit_button(
                tr("注册", "Register"), use_container_width=True
            )
        if submitted:
            if not valid_username(username):
                _cloud_auth_notice(
                    "warning",
                    tr(
                        "用户名需 5-20 位，只能使用数字、字母、下划线",
                        "Username must be 5-20 letters/digits/underscore",
                    ),
                )
            elif not valid_password(password):
                _cloud_auth_notice(
                    "warning",
                    tr(
                        "密码至少 8 位，须同时包含数字和字母，不能含空格",
                        "Password must be 8+ chars with letters and digits",
                    ),
                )
            elif not re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", email or ""):
                _cloud_auth_notice(
                    "warning", tr("请输入有效邮箱", "Please enter a valid email")
                )
            elif password != confirm:
                _cloud_auth_notice(
                    "warning", tr("两次密码不一致", "Passwords do not match")
                )
            else:
                try:
                    with st.spinner(tr("正在创建账号…", "Creating account…")):
                        session = SUPABASE_AUTH.sign_up(
                            email.strip(), password, username.strip()
                        )
                        if session:
                            _set_cloud_session(session, email.strip())
                            st.rerun()
                    _cloud_auth_notice(
                        "success",
                        tr(
                            "注册成功。请查收确认邮件，完成验证后再登录。",
                            "Registered. Confirm your email, then log in.",
                        ),
                    )
                except SupabaseAuthError as exc:
                    _cloud_auth_notice(
                        "error",
                        f"{tr('注册失败', 'Registration failed')}：{friendly_auth_error(exc, 'register')}",
                    )

    elif choice == forgot_label:
        st.caption(
            tr(
                "输入绑定邮箱，Supabase Auth 会发送重置密码邮件。",
                "Enter your email; Supabase Auth will send a password reset message.",
            )
        )
        with st.form("cloud_forgot_form"):
            email = st.text_input(tr("邮箱", "Email"), key="cloud_forgot_email")
            submitted = st.form_submit_button(
                tr("发送重置邮件", "Send reset email"), use_container_width=True
            )
        if submitted:
            try:
                with st.spinner(tr("正在发送…", "Sending…")):
                    SUPABASE_AUTH.recover(email.strip())
                _cloud_auth_notice(
                    "success",
                    tr(
                        "如果该邮箱已注册，重置邮件已发送，请检查收件箱。",
                        "If the email is registered, a reset message has been sent.",
                    ),
                )
            except SupabaseAuthError as exc:
                _cloud_auth_notice(
                    "error",
                    f"{tr('发送失败', 'Send failed')}：{friendly_auth_error(exc, 'recover')}",
                )

    notice = st.session_state.pop("cloud_auth_notice", None)
    if notice:
        kind, message = notice
        if kind == "success":
            st.success(message)
        elif kind == "warning":
            st.warning(message)
        else:
            st.error(message)


def render_auth() -> None:
    """登录页：左侧表单，右侧说明工作台能做什么。"""

    left, right = st.columns([0.62, 0.38], gap="large")
    with left:
        if SUPABASE_AUTH:
            _render_cloud_auth_forms()
        else:
            _render_auth_forms()
    with right:
        _render_auth_aside()
    st.stop()


def render_sidebar():
    with st.sidebar:
        username = st.session_state["username"]
        initial = (username or "?")[0].upper()
        st.markdown(
            f"""
            <div class="side-user">
              <div class="avatar">{initial}</div>
              <div>
                <strong>{username}</strong>
                <span>{storage_label()}</span>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button(tr("退出登录", "Log out"), use_container_width=True):
            if SUPABASE_AUTH:
                token = st.session_state.get("sb_access_token")
                if token:
                    SUPABASE_AUTH.sign_out(token)
                STORAGE.set_access_token(None)
            clear_user_state()
            st.rerun()

        with st.expander(tr("账号与存储", "Account & storage"), expanded=False):
            if storage_label() == "本地 SQLite":
                st.caption(
                    tr(
                        "当前为本地演示模式，数据保存在 data/app.db。",
                        "Local demo mode; data is stored in data/app.db.",
                    )
                )
            current_pw = st.text_input(
                tr("当前密码", "Current password"),
                type="password",
                key="cur_pw",
            )
            new_pw = st.text_input(
                tr("新密码", "New password"), type="password", key="new_pw"
            )
            confirm_pw = st.text_input(
                tr("确认新密码", "Confirm new password"),
                type="password",
                key="confirm_pw",
            )
            if st.button(tr("修改密码", "Change password"), use_container_width=True):
                if not valid_password(new_pw):
                    st.warning(
                        tr(
                            "新密码至少 8 位，须同时包含数字和字母，不能含空格",
                            "New password must be 8+ chars with letters and digits",
                        )
                    )
                elif new_pw != confirm_pw:
                    st.warning(tr("两次新密码不一致", "Passwords do not match"))
                else:
                    try:
                        if SUPABASE_AUTH:
                            email = st.session_state.get("sb_email", "")
                            if not email:
                                st.error(tr("当前会话缺少邮箱，请重新登录", "Missing email; log in again"))
                            else:
                                # 用旧密码换一个新 session，确认当前密码正确
                                session = SUPABASE_AUTH.sign_in(email, current_pw)
                                token = session["access_token"]
                                SUPABASE_AUTH.update_password(token, new_pw)
                                st.session_state.update(
                                    {
                                        "sb_access_token": token,
                                        "sb_refresh_token": session.get("refresh_token", ""),
                                        "sb_expires_at": session.get(
                                            "expires_at", time.time() + 3600
                                        ),
                                    }
                                )
                                STORAGE.set_access_token(token)
                                st.success(tr("密码已修改", "Password changed"))
                        elif not STORAGE.verify_user(
                            st.session_state["username"], current_pw
                        ):
                            st.error(tr("当前密码错误", "Current password is wrong"))
                        else:
                            STORAGE.change_password(
                                st.session_state["username"], new_pw
                            )
                            st.success(tr("密码已修改", "Password changed"))
                    except (StorageError, SupabaseAuthError) as exc:
                        st.error(f"{tr('修改失败', 'Change failed')}：{exc}")

            st.markdown('<div class="side-space"></div>', unsafe_allow_html=True)
            confirm_delete = st.checkbox(
                tr("我确认删除账号及云端数据", "I confirm deleting my account"),
                key="confirm_delete_account",
            )
            if st.button(
                tr("删除账号", "Delete account"),
                type="secondary",
                disabled=not confirm_delete,
                use_container_width=True,
            ):
                try:
                    if SUPABASE_AUTH:
                        token = st.session_state.get("sb_access_token")
                        if not token:
                            raise SupabaseAuthError("登录已过期，请重新登录")
                        SUPABASE_AUTH.delete_account(token)
                        SUPABASE_AUTH.sign_out(token)
                        STORAGE.set_access_token(None)
                    else:
                        STORAGE.delete_account(st.session_state["username"])
                    clear_user_state()
                    st.rerun()
                except (StorageError, SupabaseAuthError) as exc:
                    st.error(f"{tr('删除失败', 'Delete failed')}：{exc}")

        st.markdown(
            f'<div class="side-section">{tr("数据文件", "Data file")}</div>',
            unsafe_allow_html=True,
        )
        uploaded_file = st.file_uploader(
            tr("上传数据文件", "Upload data file"),
            type=["csv", "xlsx", "xls"],
            key="data_upload",
        )

        st.markdown(
            f'<div class="side-section">{tr("历史文件", "Saved files")}</div>',
            unsafe_allow_html=True,
        )
        none_label = tr("（不加载）", "(none)")
        try:
            saved_files = STORAGE.list_user_files(st.session_state["username"])
        except StorageError as exc:
            st.warning(f"{tr('读取文件列表失败', 'Failed to load file list')}：{exc}")
            saved_files = []
        saved_names = [item["filename"] for item in saved_files]
        if saved_names:
            chosen = st.selectbox(
                tr("选择已保存的文件", "Choose a saved file"),
                [none_label] + saved_names,
                key=f"saved_file_{st.session_state['username']}",
            )
            with st.expander(tr("管理文件", "Manage files"), expanded=False):
                for item in saved_files:
                    left, right = st.columns([4, 1])
                    with left:
                        st.caption(item["filename"])
                    with right:
                        if st.button(
                            tr("删", "Del"),
                            key=f"delete_file_{item['id']}",
                            use_container_width=True,
                        ):
                            try:
                                STORAGE.delete_user_file(
                                    st.session_state["username"], item["filename"]
                                )
                                set_flash(tr("文件已删除", "File deleted"))
                                st.rerun()
                            except StorageError as exc:
                                st.error(f"{tr('删除失败', 'Delete failed')}：{exc}")
        else:
            chosen = none_label
            st.caption(tr("暂无历史文件", "No saved files yet"))

        return uploaded_file, chosen


@st.cache_data(show_spinner=False, max_entries=8)
def load_table_cached(content: bytes, name: str) -> pd.DataFrame:
    return load_table_bytes(content, name)


@st.cache_data(ttl=30, show_spinner=False)
def cached_materials(keyword: str = "") -> list[dict]:
    return STORAGE.list_materials(keyword)


# ============ 材料库 ============
def render_library_tab() -> None:
    st.subheader(tr("共享材料库", "Shared material library"))
    st.caption(
        tr(
            "用户共建、按化学式去重；删除只能删除自己录入的记录。",
            "Community-built, de-duplicated by formula; you can delete only your own records.",
        )
    )
    show_flash()
    tab_browse, tab_add, tab_batch, tab_nlp = st.tabs(
        [
            tr("浏览/搜索", "Browse/Search"),
            tr("录入新材料", "Add material"),
            tr("批量导入", "Batch import"),
            tr("文献提取", "Literature extraction"),
        ]
    )

    with tab_browse:
        keyword = st.text_input(
            tr("按化学式搜索（如 Fe、Al2O3）", "Search by formula (e.g. Fe, Al2O3)"),
            key="mat_keyword",
        )
        try:
            materials = cached_materials(keyword.strip())
        except StorageError as exc:
            st.error(f"{tr('读取材料库失败', 'Failed to load materials')}：{exc}")
            materials = []
        if not materials:
            st.info(tr("库中暂无材料，去录入页添加第一条", "No materials yet."))
        else:
            rows = []
            def dash(value):
                return "—" if value is None or value == "" else value

            for material in materials:
                props = material.get("properties") or {}
                rows.append(
                    {
                        "ID": material.get("id"),
                        tr("化学式/成分", "Formula"): dash(material.get("formula")),
                        tr("结构", "Structure"): dash(material.get("structure")),
                        tr("抗拉强度(MPa)", "Tensile (MPa)"): dash(props.get(
                            "tensile_strength"
                        )),
                        tr("屈服强度(MPa)", "Yield (MPa)"): dash(props.get(
                            "yield_strength"
                        )),
                        tr("杨氏模量(GPa)", "Modulus (GPa)"): dash(props.get(
                            "youngs_modulus"
                        )),
                        tr("电导率(S/m)", "Conductivity (S/m)"): dash(props.get(
                            "conductivity"
                        )),
                        tr("热导率(W/mK)", "Thermal cond. (W/mK)"): dash(props.get(
                            "thermal_conductivity"
                        )),
                        tr("数据来源", "Source"): dash(material.get("source")),
                        tr("参考文献", "Reference"): dash(material.get("reference")),
                        tr("录入者", "Contributor"): dash(material.get("contributor")),
                    }
                )
            st.dataframe(
                rows,
                use_container_width=True,
                height=max(120, min(380, 42 + 38 * len(rows))),
            )

            mine = [
                material
                for material in materials
                if material.get("contributor") == st.session_state["username"]
            ]
            if mine:
                st.caption(tr("我录入的记录（可删除）", "My records"))
                for material in mine:
                    left, right = st.columns([5, 1])
                    with left:
                        st.write(
                            f"#{material['id']}  {material['formula']}  "
                            f"({material.get('source', '')})"
                        )
                    with right:
                        if st.button(
                            tr("删除", "Delete"),
                            key=f"delete_material_{material['id']}",
                            use_container_width=True,
                        ):
                            try:
                                if STORAGE.delete_material(
                                    material["id"], st.session_state["username"]
                                ):
                                    cached_materials.clear()
                                    set_flash(tr("已删除", "Deleted"))
                                    st.rerun()
                                st.error(tr("删除失败", "Delete failed"))
                            except StorageError as exc:
                                st.error(f"{tr('删除失败', 'Delete failed')}：{exc}")

    with tab_add:
        left, right = st.columns(2)
        with left:
            formula = st.text_input(
                tr("化学式/成分（如 Fe-0.2C、Al2O3、PLA）", "Formula"),
                key="new_material_formula",
            )
            structure = st.text_input(
                tr("结构描述（如 奥氏体 FCC、马氏体、非晶）", "Structure"),
                key="new_material_structure",
            )
            source = st.selectbox(
                tr("数据来源", "Source"),
                [tr("实验测量", "Experiment"), tr("文献报道", "Literature"),
                 tr("手册数据", "Handbook"), tr("估算/预测", "Estimated")],
                key="new_material_source",
            )
            reference = st.text_input(
                tr("参考文献（标题/DOI，选填）", "Reference (optional)"),
                key="new_material_reference",
            )
        with right:
            tensile = st.number_input(
                tr("抗拉强度 MPa（0 表示不填）", "Tensile strength MPa (0 = skip)"),
                value=0.0,
                step=10.0,
                key="new_material_tensile",
            )
            yield_strength = st.number_input(
                tr("屈服强度 MPa（0 表示不填）", "Yield strength MPa (0 = skip)"),
                value=0.0,
                step=10.0,
                key="new_material_yield",
            )
            modulus = st.number_input(
                tr("杨氏模量 GPa（0 表示不填）", "Young's modulus GPa (0 = skip)"),
                value=0.0,
                step=1.0,
                key="new_material_modulus",
            )
            conductivity = st.number_input(
                tr("电导率 S/m（0 表示不填）", "Conductivity S/m (0 = skip)"),
                value=0.0,
                step=1e6,
                format="%.2e",
                key="new_material_conductivity",
            )
            thermal = st.number_input(
                tr("热导率 W/(m·K)（0 表示不填）", "Thermal conductivity W/(m·K)"),
                value=0.0,
                step=1.0,
                key="new_material_thermal",
            )
        if st.button(
            tr("提交录入", "Submit"), type="primary", key="submit_material"
        ):
            if not formula.strip():
                st.warning(tr("请填写化学式/成分", "Please enter a formula"))
            else:
                try:
                    duplicate = STORAGE.find_material(formula)
                    if duplicate:
                        st.warning(
                            tr(
                                f"该材料已在库中（ID={duplicate['id']}）",
                                f"Material already exists (ID={duplicate['id']})",
                            )
                        )
                    else:
                        properties = {}
                        if tensile:
                            properties["tensile_strength"] = float(tensile)
                        if yield_strength:
                            properties["yield_strength"] = float(yield_strength)
                        if modulus:
                            properties["youngs_modulus"] = float(modulus)
                        if conductivity:
                            properties["conductivity"] = float(conductivity)
                        if thermal:
                            properties["thermal_conductivity"] = float(thermal)
                        STORAGE.add_material(
                            formula,
                            structure,
                            properties,
                            source,
                            st.session_state["username"],
                            reference,
                        )
                        cached_materials.clear()
                        set_flash(tr("录入成功", "Material added"))
                        st.rerun()
                except StorageError as exc:
                    st.error(f"{tr('录入失败', 'Add failed')}：{exc}")

    with tab_batch:
        st.caption(
            tr(
                "支持 CSV / Excel。列名可用 formula、structure、tensile_strength、"
                "yield_strength、youngs_modulus、conductivity、thermal_conductivity、"
                "source、reference，也支持常见中文别名。",
                "CSV / Excel with formula, structure and property columns.",
            )
        )
        batch_file = st.file_uploader(
            tr("批量导入文件", "Batch file"),
            type=["csv", "xlsx", "xls"],
            key="batch_material_file",
        )
        if batch_file is not None:
            try:
                batch_df = load_table_bytes(batch_file.getvalue(), batch_file.name)
            except DataError as exc:
                st.error(f"{tr('读取失败', 'Read failed')}：{exc}")
                batch_df = None
            if batch_df is not None:
                aliases = {
                    "formula": ["formula", "chemical_formula", "compound", "material",
                                "化学式", "成分", "材料", "材料名称", "成分体系", "牌号"],
                    "structure": ["structure", "microstructure", "phase", "结构",
                                  "显微组织", "物相", "组织"],
                    "tensile_strength": ["tensile_strength", "uts", "tensile",
                                         "抗拉强度", "拉伸强度", "抗拉"],
                    "yield_strength": ["yield_strength", "ys", "yield",
                                       "屈服强度", "屈服"],
                    "youngs_modulus": ["youngs_modulus", "young_modulus",
                                       "elastic_modulus", "e_modulus", "杨氏模量",
                                       "弹性模量", "模量"],
                    "conductivity": ["conductivity", "electrical_conductivity",
                                     "电导率", "导电率"],
                    "thermal_conductivity": ["thermal_conductivity", "tc",
                                             "热导率", "导热系数"],
                    "source": ["source", "data_source", "来源", "数据来源", "文献"],
                    "reference": ["reference", "ref", "doi", "citation",
                                  "参考文献", "文献标题", "引用"],
                }
                lower_map = {
                    str(column).strip().lower(): column for column in batch_df.columns
                }
                norm: dict[str, str] = {}
                for standard, names in aliases.items():
                    for name in names:
                        if name.lower() in lower_map:
                            norm[standard] = lower_map[name.lower()]
                            break
                st.caption(f"{tr('识别到列', 'Detected columns')}：{norm}")
                if "formula" not in norm:
                    st.error(
                        tr(
                            "找不到材料名列（formula/化学式/成分/材料 等）",
                            "No formula column found",
                        )
                    )
                else:
                    st.dataframe(batch_df, use_container_width=True, height=240)
                    if st.button(
                        tr("确认批量导入", "Import"),
                        type="primary",
                        key="batch_import_btn",
                    ):
                        try:
                            existing = {
                                str(item.get("formula") or "").strip().upper()
                                for item in STORAGE.list_materials()
                            }
                        except StorageError:
                            existing = set()
                        rows_to_insert = []
                        skipped = 0
                        for _, row in batch_df.iterrows():
                            formula_value = row.get(norm["formula"])
                            formula_text = (
                                "" if pd.isna(formula_value) else str(formula_value)
                            ).strip()
                            if not formula_text or formula_text.lower() == "nan":
                                continue
                            if formula_text.upper() in existing:
                                skipped += 1
                                continue

                            properties = {}
                            for key in [
                                "tensile_strength",
                                "yield_strength",
                                "youngs_modulus",
                                "conductivity",
                                "thermal_conductivity",
                            ]:
                                column = norm.get(key)
                                if not column:
                                    continue
                                value = row.get(column)
                                if value is None or pd.isna(value):
                                    continue
                                try:
                                    properties[key] = float(value)
                                except (TypeError, ValueError):
                                    pass

                            def cell(key: str, default: str = "") -> str:
                                column = norm.get(key)
                                if not column:
                                    return default
                                value = row.get(column)
                                if value is None or pd.isna(value):
                                    return default
                                text = str(value).strip()
                                return "" if text.lower() in ("nan", "none") else text

                            rows_to_insert.append(
                                {
                                    "formula": formula_text,
                                    "structure": cell("structure"),
                                    "properties": properties,
                                    "source": cell("source") or "批量导入",
                                    "reference": cell("reference"),
                                    "contributor": st.session_state["username"],
                                }
                            )
                            existing.add(formula_text.upper())

                        if not rows_to_insert:
                            st.warning(
                                tr(
                                    "没有可导入的新记录（全部重复或为空）",
                                    "No new records to import",
                                )
                            )
                        else:
                            try:
                                inserted = STORAGE.add_materials(rows_to_insert)
                                cached_materials.clear()
                                set_flash(
                                    tr(
                                        f"成功导入 {inserted} 条，跳过重复 {skipped} 条",
                                        f"Imported {inserted}, skipped {skipped} duplicates",
                                    )
                                )
                                st.rerun()
                            except StorageError as exc:
                                st.error(f"{tr('导入失败', 'Import failed')}：{exc}")

    with tab_nlp:
        render_nlp_tab()


def _ai_extract_material(
    text: str, api_key: str, base_url: str, model: str
) -> dict:
    """用大模型从文献文本中提取单个材料记录。"""

    prompt = f"""你是材料科学文献数据抽取助手。请从下面文本中提取一个材料性能记录。
只输出 JSON 对象，不要解释，不要 markdown 代码块。

字段：
- formula: 化学式/成分/牌号（如 Al2O3、Ti6Al4V、304 stainless steel）
- material_name: 文献中使用的材料名称
- tensile: 抗拉强度 MPa，没有则 null
- yield: 屈服强度 MPa，没有则 null
- youngs: 杨氏模量 GPa，没有则 null
- cond: 电导率 S/m，没有则 null
- therm: 热导率 W/(m·K)，没有则 null
- doi: DOI，没有则空字符串
- page: 页码，没有则 null
- quote: 支撑上述数值的原文短句（最多 100 字）
- confidence: 0 到 1 的置信度

规则：
- 只提取文本中明确写出的数值，不要猜测。
- 找不到的字段填 null。
- 不要把不同材料/不同实验的数据合并。
- 数值单位必须按上面字段统一。

文献文本：
{text[:8000]}"""

    response = requests.post(
        base_url,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": "你是材料学文献数据抽取助手，只输出合法 JSON。",
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.1,
        },
        timeout=90,
    )
    if response.status_code != 200:
        raise RuntimeError(f"AI HTTP {response.status_code}: {response.text[:200]}")
    content = response.json()["choices"][0]["message"]["content"].strip()
    content = re.sub(r"^```(?:json)?", "", content).strip()
    content = re.sub(r"```$", "", content).strip()
    start = content.find("{")
    end = content.rfind("}")
    if start >= 0 and end > start:
        content = content[start : end + 1]
    data = json.loads(content)
    if isinstance(data, dict) and isinstance(data.get("materials"), list) and data["materials"]:
        data = data["materials"][0]
    if isinstance(data, list) and data:
        data = data[0]
    if not isinstance(data, dict):
        raise ValueError("AI 返回的不是 JSON 对象")

    def number(key: str):
        value = data.get(key)
        if value in (None, "", "null"):
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    page = data.get("page")
    try:
        page = int(float(page)) if page not in (None, "") else None
    except (TypeError, ValueError):
        page = None

    return {
        "formula": str(data.get("formula") or "").strip(),
        "material_name": str(data.get("material_name") or "").strip(),
        "tensile": number("tensile"),
        "yield": number("yield"),
        "youngs": number("youngs"),
        "cond": number("cond"),
        "therm": number("therm"),
        "doi": str(data.get("doi") or "").strip(),
        "page": page,
        "quote": str(data.get("quote") or "")[:200],
        "confidence": number("confidence"),
    }


def render_nlp_tab() -> None:
    st.caption(
        tr(
            "上传 PDF 或粘贴文献文本，优先用 AI 提取材料性能字段，未配置 AI 时自动使用规则提取；结果需要人工确认。",
            "Upload a PDF or paste text. AI extraction is used when configured, with a rule-based fallback.",
        )
    )
    _ai_key, _ai_base, _ai_model = ai_config()
    if _ai_key:
        st.caption(
            tr(
                f"AI 已接入：{_ai_model or '未填写模型名'}；提取结果会保留原文引用，仍需人工确认。",
                f"AI enabled: {_ai_model or 'model missing'}.",
            )
        )
    else:
        st.info(
            tr(
                "未配置 AI Key，当前使用规则提取。配置 DEEPSEEK_API_KEY / ARK_API_KEY 后会自动优先使用 AI。",
                "No AI key; rule-based extraction is used. Configure DEEPSEEK_API_KEY / ARK_API_KEY to enable AI.",
            )
        )

    text = st.text_area(tr("文献文本", "Literature text"), height=180, key="nlp_text")
    pdf_file = st.file_uploader(
        tr("或上传 PDF", "Or upload a PDF"),
        type=["pdf"],
        key="nlp_pdf",
    )
    if pdf_file is not None:
        try:
            import pdfplumber

            with pdfplumber.open(io.BytesIO(pdf_file.getvalue())) as pdf:
                text = "\n".join((page.extract_text() or "") for page in pdf.pages)
            st.caption(f"{tr('PDF 提取字符数', 'PDF characters')}：{len(text)}")
        except Exception as exc:
            st.warning(f"{tr('PDF 读取失败', 'PDF read failed')}：{exc}")

    if st.button(
        tr("提取材料数据", "Extract"), type="primary", key="nlp_extract_btn"
    ):
        if not text or len(text) < 20:
            st.warning(tr("文本太短", "Text is too short"))
        else:
            result = None
            source = "规则提取"
            if _ai_key:
                with st.spinner(tr("AI 正在提取文献数据...", "AI extracting...")):
                    try:
                        result = _ai_extract_material(text, _ai_key, _ai_base, _ai_model)
                        source = f"AI 提取（{_ai_model or '模型'}）"
                    except Exception as exc:
                        st.warning(f"AI 提取失败，改用规则提取：{exc}")

            formula_pattern = re.compile(
                r"\b((?:[A-Z][a-z]?\d*){2,}(?:[-–]\d+(?:\.\d+)?[A-Za-z]*)?)\b"
            )
            stop = {
                "THE", "AND", "FOR", "ARE", "WITH", "FROM", "THIS", "THAT",
                "FIG", "TABLE", "ALLOY", "STEEL", "SAMPLE",
            }
            formula_candidates = [
                c
                for c in formula_pattern.findall(text)
                if len(c) >= 2 and c.upper() not in stop
            ]

            if result is not None and not result.get("formula") and formula_candidates:
                result["formula"] = formula_candidates[0]

            if result is None:
                # 正则 fallback
                def grab(keywords, unit_pattern):
                    keyword_pattern = re.compile("|".join(keywords), re.I)
                    value_pattern = re.compile(
                        r"(-?\d+(?:\.\d+)?)\s*(" + unit_pattern + r")", re.I
                    )
                    for match in keyword_pattern.finditer(text):
                        window = text[match.end(): match.end() + 100]
                        value = value_pattern.search(window)
                        if value:
                            return float(value.group(1))
                    return None
                tensile = grab(["tensile strength","ultimate tensile","UTS","抗拉强度","拉伸强度"], r"MPa")
                yield_strength = grab(["yield strength","yield stress","YS","屈服强度"], r"MPa")
                modulus = grab(["Young'?s modulus","elastic modulus","杨氏模量","弹性模量"], r"GPa")
                conductivity = grab(["conductivity","电导率","导电率"], r"S\s*/\s*m")
                thermal = grab(["thermal conductivity","热导率","导热"], r"W\s*/\s*(?:\(\s*m\s*[·*]?\s*K\s*\)|m\s*[·*]?\s*K)")
                doi_match = re.search(r"10\.\d{4,9}/[-._;()/:A-Z0-9]+", text, re.I)
                result = {
                    "formula": formula_candidates[0] if formula_candidates else "",
                    "material_name": "",
                    "tensile": tensile,
                    "yield": yield_strength,
                    "youngs": modulus,
                    "cond": conductivity,
                    "therm": thermal,
                    "doi": doi_match.group(0) if doi_match else "",
                    "page": None,
                    "quote": "",
                    "confidence": None,
                }
                source = "规则提取"

            result["_source"] = source
            st.session_state["nlp_result"] = result
            st.session_state["nlp_extract_source"] = source

    if "nlp_result" in st.session_state:
        result = st.session_state["nlp_result"]
        source = st.session_state.get("nlp_extract_source") or result.get("_source", "")
        if source:
            st.caption(tr(f"提取方式：{source}", f"Extraction: {source}"))
        st.write(tr("提取结果", "Extracted"))
        st.json(result)
        formula_edit = st.text_input(
            tr("确认/修改化学式", "Confirm/edit formula"),
            value=result.get("formula") or "",
            key="nlp_formula_edit",
        )
        if st.button(tr("确认录入材料库", "Save to library"), key="nlp_save_btn"):
            if not (formula_edit or "").strip():
                st.warning(tr("化学式为空", "Formula is empty"))
            else:
                try:
                    duplicate = STORAGE.find_material(formula_edit)
                    if duplicate:
                        st.warning(
                            tr(
                                f"该材料已在库中（ID={duplicate['id']}）",
                                f"Material already exists (ID={duplicate['id']})",
                            )
                        )
                    else:
                        properties = {}
                        if result.get("tensile"):
                            properties["tensile_strength"] = result["tensile"]
                        if result.get("yield"):
                            properties["yield_strength"] = result["yield"]
                        if result.get("youngs"):
                            properties["youngs_modulus"] = result["youngs"]
                        if result.get("cond"):
                            properties["conductivity"] = result["cond"]
                        if result.get("therm"):
                            properties["thermal_conductivity"] = result["therm"]
                        STORAGE.add_material(
                            formula_edit,
                            "",
                            properties,
                            tr("文献报道", "Literature"),
                            st.session_state["username"],
                            result.get("doi", ""),
                        )
                        cached_materials.clear()
                        st.session_state.pop("nlp_result", None)
                        set_flash(tr("已录入材料库", "Saved to library"))
                        st.rerun()
                except StorageError as exc:
                    st.error(f"{tr('录入失败', 'Save failed')}：{exc}")


def render_compare_tab() -> None:
    st.subheader(tr("材料性能对比", "Material comparison"))
    try:
        materials = cached_materials()
    except StorageError as exc:
        st.error(f"{tr('读取材料库失败', 'Failed to load materials')}：{exc}")
        return
    if len(materials) < 2:
        st.info(tr("库中至少需要 2 条材料才能对比", "Need at least 2 materials"))
        return

    labels = [f"#{item['id']}  {item['formula']}" for item in materials]
    left, right = st.columns(2)
    with left:
        index_a = st.selectbox(
            tr("材料 A", "Material A"), range(len(labels)), format_func=lambda i: labels[i],
            key="compare_a",
        )
    with right:
        index_b = st.selectbox(
            tr("材料 B", "Material B"), range(len(labels)),
            index=1 if len(labels) > 1 else 0,
            format_func=lambda i: labels[i],
            key="compare_b",
        )
    if index_a == index_b:
        st.info(tr("请选择两种不同的材料", "Choose two different materials"))
        return

    material_a, material_b = materials[index_a], materials[index_b]
    props_a = material_a.get("properties") or {}
    props_b = material_b.get("properties") or {}
    metrics = [
        (tr("抗拉强度", "Tensile strength"), "tensile_strength", "MPa"),
        (tr("屈服强度", "Yield strength"), "yield_strength", "MPa"),
        (tr("杨氏模量", "Young's modulus"), "youngs_modulus", "GPa"),
        (tr("电导率", "Conductivity"), "conductivity", "S/m"),
        (tr("热导率", "Thermal conductivity"), "thermal_conductivity", "W/(m·K)"),
    ]
    rows = []
    chart_data = []
    for label, key, unit in metrics:
        value_a, value_b = props_a.get(key), props_b.get(key)
        if value_a is None and value_b is None:
            continue
        rows.append(
            {
                tr("指标", "Metric"): f"{label} ({unit})",
                f"A: {material_a['formula']}": value_a if value_a is not None else "—",
                f"B: {material_b['formula']}": value_b if value_b is not None else "—",
                tr("差值(B-A)", "Difference (B-A)"): (
                    (value_b - value_a)
                    if value_a is not None and value_b is not None
                    else "—"
                ),
            }
        )
        if value_a is not None and value_b is not None:
            chart_data.append((label, unit, float(value_a), float(value_b)))
    st.dataframe(rows, use_container_width=True)

    # 不同指标量纲差异很大，拆成独立子图，避免 GPa 把 MPa 压扁
    if chart_data:
        columns = st.columns(2)
        for index, (label, unit, value_a, value_b) in enumerate(chart_data):
            with columns[index % 2]:
                fig, ax = plt.subplots(figsize=(4.2, 2.8))
                ax.bar(
                    [f"A: {material_a['formula']}", f"B: {material_b['formula']}"],
                    [value_a, value_b],
                    color=["#0f5c5e", "#6f8f6a"],
                )
                ax.set_title(f"{label} ({unit})", fontsize=10)
                ax.tick_params(labelsize=8)
                style_figure(fig)
                st.pyplot(fig)
                plt.close(fig)

    if material_a.get("reference"):
        st.caption(f"A {tr('参考文献', 'Reference')}：{material_a['reference']}")
    if material_b.get("reference"):
        st.caption(f"B {tr('参考文献', 'Reference')}：{material_b['reference']}")


# ============ 数据加载与 AI 清洗 ============
def apply_ai_cleaning(df: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
    api_key, base_url, model = ai_config()
    if not api_key:
        st.caption(
            tr(
                "配置 DEEPSEEK_API_KEY 或 ARK_API_KEY 后可启用 AI 清洗。",
                "Configure DEEPSEEK_API_KEY or ARK_API_KEY to enable AI cleaning.",
            )
        )
        return df, False

    enabled = st.checkbox(
        tr("AI 智能清洗表格结构", "AI table cleaning"),
        value=True,
        key="ai_clean_toggle",
        help=tr(
            "会抽取表格前 50 行发送到配置的大模型服务；不上传整个文件。",
            "Sends the first 50 rows to the configured LLM; not the whole file.",
        ),
    )
    if not enabled:
        return df, False

    signature = f"{frame_signature(df)}|{model}"
    if st.session_state.get("_clean_sig") == signature and "df_cleaned" in st.session_state:
        return st.session_state["df_cleaned"], True

    with st.spinner(tr("AI 正在识别表头与数据区...", "AI is cleaning the table...")):
        try:
            prompt = (
                "你是材料实验数据清洗助手。下面是一个表格前 50 行导出的 CSV 文本，"
                "可能包含标题行、参数行、真正的表头行、单位行、空行和签名行。\n"
                "请返回干净的 CSV（带表头，不要解释，不要 markdown 代码块）。\n\n"
                + df.head(50).to_csv(index=False)
            )
            response = requests.post(
                base_url,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.1,
                },
                timeout=60,
            )
            if response.status_code != 200:
                st.caption(
                    tr(
                        f"AI 清洗跳过（HTTP {response.status_code}），使用原始数据。",
                        f"AI cleaning skipped (HTTP {response.status_code}).",
                    )
                )
                return df, False
            content = response.json()["choices"][0]["message"]["content"].strip()
            if content.startswith("```"):
                content = content.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            cleaned = pd.read_csv(io.StringIO(content))
            if cleaned.empty or cleaned.columns.duplicated().any():
                raise ValueError("AI 返回的表格为空或列名重复")
            cleaned.columns = [str(column) for column in cleaned.columns]
            st.session_state["df_cleaned"] = cleaned
            st.session_state["_clean_sig"] = signature
            return cleaned, True
        except Exception as exc:
            st.caption(
                tr(
                    f"AI 清洗失败，使用原始数据：{exc}",
                    f"AI cleaning failed; using raw data: {exc}",
                )
            )
            return df, False


# ============ 结果渲染 ============
def plot_feature_curves(result: dict) -> None:
    model = result["model"]
    scaler = result["scaler"]
    feature_cols = result["feature_cols"]
    encoders = result["encoders"]
    fill_values = result["fill_values"]
    target_col = result["target_col"]
    X, y = result["X"], result["y"]

    max_features = 6
    shown = feature_cols[:max_features]
    if len(feature_cols) > max_features:
        st.caption(
            tr(
                f"特征较多，仅展示前 {max_features} 个；其余特征请用重要性图判断。",
                f"Showing the first {max_features} features.",
            )
        )

    rows = len(shown)
    fig, axes = plt.subplots(rows, 1, figsize=(7.2, 3.1 * rows), squeeze=False)
    base = np.array([[fill_values[column] for column in feature_cols]], dtype=float)
    for row_index, column in enumerate(shown):
        ax = axes[row_index][0]
        feature_index = feature_cols.index(column)
        ax.scatter(X[:, feature_index], y, alpha=0.45, color="#0f5c5e", s=18, label=tr("样本", "Samples"))

        if column in encoders:
            codes = np.unique(X[:, feature_index].astype(int))
            grid = np.tile(base, (len(codes), 1))
            grid[:, feature_index] = codes
            grid_scaled = scaler.transform(grid)
            if result["is_gpr"]:
                prediction, std = model.predict(grid_scaled, return_std=True)
                ax.fill_between(
                    codes,
                    prediction - 1.96 * std,
                    prediction + 1.96 * std,
                    alpha=0.18,
                    color="#0f5c5e",
                )
            else:
                prediction = model.predict(grid_scaled)
            ax.plot(codes, prediction, "o-", color="#9d3c32", label=tr("拟合", "Fit"))
            labels = list(encoders[column].classes_)
            ax.set_xticks(codes)
            ax.set_xticklabels(
                [labels[code] if 0 <= code < len(labels) else str(code) for code in codes],
                rotation=25,
                ha="right",
                fontsize=7,
            )
            x_plot = codes.astype(float)
        else:
            low, high = result["feature_ranges"][column]
            xs = np.linspace(low, high, 100)
            grid = np.tile(base, (len(xs), 1))
            grid[:, feature_index] = xs
            grid_scaled = scaler.transform(grid)
            if result["is_gpr"]:
                prediction, std = model.predict(grid_scaled, return_std=True)
                ax.fill_between(
                    xs,
                    prediction - 1.96 * std,
                    prediction + 1.96 * std,
                    alpha=0.18,
                    color="#0f5c5e",
                )
            else:
                prediction = model.predict(grid_scaled)
            ax.plot(xs, prediction, "-", color="#9d3c32", lw=1.8, label=tr("拟合", "Fit"))
            x_plot = xs

        if len(x_plot):
            maximum = int(np.argmax(prediction))
            minimum = int(np.argmin(prediction))
            for index, color, text, offset in [
                (maximum, "#4f6b57", tr("最大", "Max"), 10),
                (minimum, "#9d3c32", tr("最小", "Min"), -16),
            ]:
                ax.scatter(
                    x_plot[index],
                    prediction[index],
                    s=60,
                    color=color,
                    edgecolors="white",
                    zorder=5,
                )
                ax.annotate(
                    f"{text}: {x_plot[index]:.3g} → {prediction[index]:.3g}",
                    xy=(x_plot[index], prediction[index]),
                    xytext=(8, offset),
                    textcoords="offset points",
                    fontsize=7,
                    color=color,
                    arrowprops=dict(arrowstyle="->", color=color, lw=0.8),
                )
        ax.set_xlabel(column, fontsize=8)
        ax.set_ylabel(target_col, fontsize=8)
        ax.tick_params(labelsize=7)
        ax.set_title(f"{column} → {target_col}", fontsize=9)
        ax.legend(fontsize=7, loc="best")
    style_figure(fig)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def render_results(result: dict) -> None:
    metrics = result["metrics"]
    st.subheader(tr("训练结果", "Training results"))
    st.caption(
        tr(
            f"数据：{result.get('data_source', '')}｜有效样本 {result['stats']['valid_rows']} 条"
            f"｜目标列：{result['target_col']}",
            f"Data: {result.get('data_source', '')} | valid rows: {result['stats']['valid_rows']}"
            f" | target: {result['target_col']}",
        )
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("R²", fmt(metrics.get("r2")))
    col2.metric("MAE", fmt(metrics.get("mae")))
    col3.metric("RMSE", fmt(metrics.get("rmse")))
    cv_r2 = metrics.get("cv_r2")
    col4.metric(
        tr("交叉验证 R²", "CV R²"),
        fmt(cv_r2) if cv_r2 is not None else tr("样本不足", "N/A"),
    )

    comparisons = result.get("cmp_df")
    if comparisons is not None and len(comparisons) > 1:
        st.subheader(tr("模型对比", "Model comparison"))
        st.dataframe(comparisons, use_container_width=True)
    errors = result.get("errors") or {}
    if errors:
        with st.expander(tr("部分模型训练失败", "Some models failed"), expanded=False):
            for name, message in errors.items():
                st.caption(f"{name}: {message}")

    importance = result.get("importance")
    if importance is not None and len(importance):
        st.subheader(tr("特征重要性", "Feature importance"))
        fig, ax = plt.subplots(figsize=(8, max(2.2, 0.35 * len(importance))))
        importance.plot.barh(ax=ax, color="#0f5c5e")
        ax.set_xlabel(tr("重要性", "Importance"), fontsize=8)
        ax.tick_params(labelsize=8)
        style_figure(fig)
        st.pyplot(fig)
        plt.close(fig)

    left, right = st.columns(2)
    with left:
        st.subheader(tr("预测值 vs 真实值", "Predicted vs true"))
        fig, ax = plt.subplots(figsize=(5.2, 3.8))
        y_test, y_pred = result["y_test"], result["y_pred"]
        ax.scatter(y_test, y_pred, alpha=0.7, color="#6f8f6a", s=22)
        low = float(min(np.min(y_test), np.min(y_pred)))
        high = float(max(np.max(y_test), np.max(y_pred)))
        ax.plot([low, high], [low, high], "r--", lw=1.5)
        ax.set_xlabel(tr("真实值", "True"), fontsize=8)
        ax.set_ylabel(tr("预测值", "Predicted"), fontsize=8)
        ax.set_title(f"{result['model_name']}", fontsize=9)
        ax.tick_params(labelsize=7)
        style_figure(fig)
        st.pyplot(fig)
        plt.close(fig)

    with right:
        if result["is_gpr"] and result.get("y_std") is not None:
            st.subheader(tr("95% 预测区间", "95% prediction interval"))
            fig, ax = plt.subplots(figsize=(5.2, 3.8))
            order = np.argsort(y_test)
            y_std = result["y_std"]
            ax.plot(
                range(len(y_test)),
                y_test[order],
                "k.",
                label=tr("真实值", "True"),
                markersize=4,
            )
            ax.plot(
                range(len(y_test)),
                y_pred[order],
                "-",
                color="#0f5c5e",
                label=tr("预测均值", "Mean"),
                lw=1.3,
            )
            ax.fill_between(
                range(len(y_test)),
                y_pred[order] - 1.96 * y_std[order],
                y_pred[order] + 1.96 * y_std[order],
                alpha=0.22,
                color="#0f5c5e",
                label=tr("95% 区间", "95% interval"),
            )
            ax.set_xlabel(tr("样本序号", "Sample"), fontsize=8)
            ax.set_ylabel(result["target_col"], fontsize=8)
            ax.tick_params(labelsize=7)
            ax.legend(fontsize=7)
            style_figure(fig)
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.subheader(tr("残差", "Residuals"))
            fig, ax = plt.subplots(figsize=(5.2, 3.8))
            residual = y_test - y_pred
            ax.scatter(y_pred, residual, alpha=0.7, color="#a8773e", s=22)
            ax.axhline(0, color="r", linestyle="--", lw=1)
            ax.set_xlabel(tr("预测值", "Predicted"), fontsize=8)
            ax.set_ylabel(tr("残差", "Residual"), fontsize=8)
            ax.tick_params(labelsize=7)
            style_figure(fig)
            st.pyplot(fig)
            plt.close(fig)

    st.subheader(tr("特征拟合曲线", "Feature response curves"))
    st.caption(
        tr(
            "其他特征固定为训练集均值/众数；曲线反映模型局部行为，不代表因果关系。",
            "Other features fixed at training mean/mode; this shows model behavior, not causality.",
        )
    )
    plot_feature_curves(result)

    st.subheader(tr("预测结果明细", "Prediction details"))
    st.dataframe(result["result_df"], use_container_width=True, height=300)

    download_left, download_right = st.columns(2)
    with download_left:
        st.download_button(
            tr("下载含预测列的 Excel", "Download Excel"),
            data=result["excel_bytes"],
            file_name="prediction_result.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )
    with download_right:
        if result.get("docx_bytes"):
            st.download_button(
                tr("下载 Word 训练报告", "Download Word report"),
                data=result["docx_bytes"],
                file_name="training_report.docx",
                mime=(
                    "application/vnd.openxmlformats-officedocument."
                    "wordprocessingml.document"
                ),
                use_container_width=True,
            )
        else:
            st.caption(
                tr("安装 python-docx 后可导出 Word 报告", "Install python-docx for Word export")
            )


def render_new_sample(result: dict) -> None:
    st.divider()
    st.subheader(tr("用训练好的模型预测新样本", "Predict a new sample"))
    st.caption(
        tr(
            "数值特征留空则用训练集均值填充；类别特征选“未知/留空”则用众数填充。",
            "Blank numeric fields use training means; unknown categories use the mode.",
        )
    )
    feature_cols = result["feature_cols"]
    encoders = result["encoders"]
    fill_values = result["fill_values"]
    signature = result["train_signature"]

    values: list[float] = []
    columns = st.columns(min(len(feature_cols), 4))
    for index, column in enumerate(feature_cols):
        with columns[index % 4]:
            if column in encoders:
                options = [tr("（未知/留空）", "(unknown)")] + list(
                    encoders[column].classes_
                )
                selected = st.selectbox(
                    column,
                    options,
                    key=f"new_value_{signature}_{column}",
                )
                if selected == options[0]:
                    values.append(float(fill_values[column]))
                else:
                    values.append(float(encoders[column].transform([selected])[0]))
            else:
                low, high = result["feature_ranges"][column]
                raw = st.text_input(
                    column,
                    value="",
                    key=f"new_value_{signature}_{column}",
                    help=(
                        f"{tr('训练范围', 'Training range')}: {low:.4g} ~ {high:.4g}; "
                        f"{tr('留空用均值', 'blank = mean')} {fill_values[column]:.4g}"
                    ),
                )
                if raw.strip() == "":
                    values.append(float(fill_values[column]))
                else:
                    try:
                        values.append(float(raw))
                    except ValueError:
                        st.warning(
                            tr(
                                f"“{column}”不是数字，已用均值填充",
                                f"'{column}' is not numeric; using the mean",
                            )
                        )
                        values.append(float(fill_values[column]))

    if st.button(
        tr("预测", "Predict"), type="secondary", key=f"predict_btn_{signature}"
    ):
        try:
            prediction, std = predict_one(
                result["model"],
                result["scaler"],
                feature_cols,
                values,
            )
            st.session_state["last_prediction"] = {
                "signature": signature,
                "target": result["target_col"],
                "prediction": prediction,
                "std": std,
            }
            st.rerun()
        except Exception as exc:
            st.error(f"{tr('预测失败', 'Prediction failed')}：{exc}")

    last = st.session_state.get("last_prediction")
    if last and last.get("signature") == signature:
        if last.get("std") is not None:
            st.success(
                tr(
                    f"预测的 **{last['target']}** = **{last['prediction']:.4f} "
                    f"± {1.96 * last['std']:.4f}**（95% 预测区间）",
                    f"Predicted **{last['target']}** = **{last['prediction']:.4f} "
                    f"± {1.96 * last['std']:.4f}** (95% interval)",
                )
            )
        else:
            st.success(
                tr(
                    f"预测的 **{last['target']}** = **{last['prediction']:.4f}**",
                    f"Predicted **{last['target']}** = **{last['prediction']:.4f}**",
                )
            )


# ============ 贝叶斯优化 ============
def render_optimization(result: dict) -> None:
    st.divider()
    st.subheader(tr("贝叶斯优化寻优", "Bayesian optimization"))
    if not _HAS_SKOPT:
        st.info(
            tr(
                "未安装 scikit-optimize，无法使用贝叶斯优化。",
                "Install scikit-optimize to enable Bayesian optimization.",
            )
        )
        return

    feature_cols = result["feature_cols"]
    encoders = result["encoders"]
    numeric_features = [column for column in feature_cols if column not in encoders]
    if not numeric_features:
        st.info(tr("没有数值特征可优化", "No numeric features to optimize"))
        return

    st.caption(
        tr(
            f"当前模型：{result['model_name']}；类别特征固定为训练集众数。",
            f"Model: {result['model_name']}; categorical features fixed at the mode.",
        )
    )
    direction = st.radio(
        tr("优化目标", "Objective"),
        [tr("最大化", "Maximize"), tr("最小化", "Minimize")],
        horizontal=True,
        key="bo_direction",
    )
    col1, col2 = st.columns(2)
    with col1:
        n_iter = st.number_input(
            tr("迭代次数", "Iterations"),
            min_value=10,
            max_value=100,
            value=30,
            step=5,
            key="bo_iterations",
        )
    with col2:
        top_k = st.number_input(
            tr("输出前几个候选", "Top candidates"),
            min_value=5,
            max_value=20,
            value=10,
            key="bo_top_k",
        )

    st.caption(tr("调整搜索范围（默认使用训练数据真实 min/max）", "Search bounds"))
    bounds: dict[str, tuple[float, float]] = {}
    bound_columns = st.columns(2)
    for index, column in enumerate(numeric_features):
        low, high = result["feature_ranges"][column]
        with bound_columns[index % 2]:
            lower = st.number_input(
                f"{column} {tr('下限', 'min')}",
                value=float(low),
                format="%.6g",
                key=f"bo_low_{column}",
            )
            upper = st.number_input(
                f"{column} {tr('上限', 'max')}",
                value=float(high),
                format="%.6g",
                key=f"bo_high_{column}",
            )
            bounds[column] = (float(lower), float(upper))

    if st.button(tr("开始寻优", "Optimize"), type="primary", key="bo_run"):
        invalid = [column for column, (low, high) in bounds.items() if low >= high]
        if invalid:
            st.error(
                tr(
                    f"以下特征的下限必须小于上限：{', '.join(invalid)}",
                    f"Lower bound must be less than upper: {', '.join(invalid)}",
                )
            )
            return

        target_col = result["target_col"]
        direction_sign = -1.0 if direction == tr("最大化", "Maximize") else 1.0
        scaler = result["scaler"]
        model = result["model"]
        fill_values = result["fill_values"]

        def objective(x):
            full = [0.0] * len(feature_cols)
            for index, column in enumerate(numeric_features):
                full[feature_cols.index(column)] = float(x[index])
            for column in feature_cols:
                if column in encoders:
                    full[feature_cols.index(column)] = fill_values[column]
            scaled = scaler.transform(np.asarray(full, dtype=float).reshape(1, -1))
            return direction_sign * float(model.predict(scaled)[0])

        space = [Real(low, high, name=column) for column, (low, high) in bounds.items()]
        with st.spinner(tr("贝叶斯优化中...", "Optimizing...")):
            try:
                optimization = gp_minimize(
                    objective,
                    space,
                    n_calls=int(n_iter),
                    n_initial_points=min(5, max(2, int(n_iter) - 1)),
                    random_state=42,
                )
            except Exception as exc:
                st.error(f"{tr('寻优失败', 'Optimization failed')}：{exc}")
                return

        trials = []
        for x, value in zip(optimization.x_iters, optimization.func_vals):
            record = {
                column: float(x[index])
                for index, column in enumerate(numeric_features)
            }
            record[target_col] = float(value) * direction_sign
            trials.append(record)

        best_so_far = []
        current = float("-inf") if direction == tr("最大化", "Maximize") else float("inf")
        for record in trials:
            value = record[target_col]
            current = (
                max(current, value)
                if direction == tr("最大化", "Maximize")
                else min(current, value)
            )
            best_so_far.append(current)

        display = sorted(
            trials,
            key=lambda record: record[target_col],
            reverse=direction == tr("最大化", "Maximize"),
        )[: int(top_k)]
        st.session_state["bo_result"] = {
            "signature": result["train_signature"],
            "target": target_col,
            "direction": direction,
            "bounds": bounds,
            "n_iter": int(n_iter),
            "best": display[0] if display else None,
            "display": display,
            "best_so_far": best_so_far,
        }
        st.rerun()

    saved = st.session_state.get("bo_result")
    if (
        saved
        and saved.get("signature") == result["train_signature"]
        and saved.get("direction") == direction
        and saved.get("bounds") == bounds
    ):
        best = saved.get("best")
        if best:
            st.success(
                tr(
                    f"最优 {saved['target']} = {best[saved['target']]:.4f}",
                    f"Best {saved['target']} = {best[saved['target']]:.4f}",
                )
            )
        st.dataframe(pd.DataFrame(saved["display"]), use_container_width=True, height=320)
        fig, ax = plt.subplots(figsize=(8, 3.8))
        ax.plot(
            range(1, len(saved["best_so_far"]) + 1),
            saved["best_so_far"],
            marker="o",
            markersize=3,
        )
        ax.set_xlabel(tr("迭代次数", "Iteration"), fontsize=8)
        ax.set_ylabel(f"Best {saved['target']}", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.set_title(tr("收敛曲线", "Convergence"), fontsize=9)
        style_figure(fig)
        st.pyplot(fig)
        plt.close(fig)
        st.download_button(
            tr("下载寻优结果 CSV", "Download CSV"),
            data=pd.DataFrame(saved["display"]).to_csv(index=False).encode("utf-8-sig"),
            file_name="optimization_candidates.csv",
            mime="text/csv",
        )


# ============ 建模主流程 ============
def render_modeling_tab(uploaded_file, chosen_file: str) -> None:
    content: bytes | None = None
    data_source = ""
    none_label = tr("（不加载）", "(none)")

    if uploaded_file is not None:
        content = uploaded_file.getvalue()
        data_source = uploaded_file.name
        signature = dataset_signature(content, data_source)
        if st.session_state.get("saved_upload_signature") != signature:
            try:
                STORAGE.save_user_file(
                    st.session_state["username"], data_source, content
                )
                st.session_state["saved_upload_signature"] = signature
            except StorageError as exc:
                st.warning(f"{tr('保存到云端失败', 'Cloud save failed')}：{exc}")
    elif chosen_file and chosen_file != none_label:
        try:
            content = STORAGE.load_user_file(
                st.session_state["username"], chosen_file
            )
        except StorageError as exc:
            st.error(f"{tr('读取历史文件失败', 'Load failed')}：{exc}")
            return
        data_source = chosen_file
        if content is None:
            st.error(tr("读取历史文件失败", "Failed to load saved file"))
            return

    if content is None:
        st.markdown(
            f"""
            <div class="empty-state">
              <h2>{tr("从一份数据表开始", "Start with a data table")}</h2>
              <p>{tr(
                  "上传记录成分、工艺与性能的 CSV / Excel，或从左侧加载历史文件。"
                  "训练后会在这里看到指标、图表和新样本预测。",
                  "Upload a CSV / Excel with composition, process and properties, "
                  "or load a saved file. Metrics, charts and predictions appear here.",
              )}</p>
              <div class="empty-points">
                <div><strong>{tr("目标列", "Target")}</strong>{tr("要预测的力学性能，如抗拉强度。", "The property to predict.")}</div>
                <div><strong>{tr("特征列", "Features")}</strong>{tr("成分、工艺参数、组织结构等输入。", "Composition, process, structure.")}</div>
                <div><strong>{tr("导出", "Export")}</strong>{tr("预测明细 Excel 与 Word 报告。", "Prediction Excel and Word report.")}</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.caption(
            tr(
                "支持 CSV / XLSX / XLS；首行为列名，每行一个样本。",
                "CSV / XLSX / XLS supported; first row is the header.",
            )
        )
        return

    try:
        df = load_table_cached(content, data_source)
    except DataError as exc:
        st.error(f"{tr('文件读取失败', 'Read failed')}：{exc}")
        return

    raw_signature = dataset_signature(content, data_source)
    old_result = st.session_state.get("train_result")
    if old_result and old_result.get("raw_data_signature") not in (
        None,
        raw_signature,
    ):
        # 数据源已经变化，旧模型不能继续用
        st.session_state.pop("train_result", None)
        st.session_state.pop("last_prediction", None)
        st.session_state.pop("bo_result", None)

    st.markdown(
        f"""
        <div class="data-line">
          <span>{tr("数据集", "Dataset")}</span>
          <strong>{data_source}</strong>
          <span>{df.shape[0]} {tr("行", "rows")} · {df.shape[1]} {tr("列", "columns")}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    with st.expander(
        tr("查看数据表（前 20 行）", "View data table (first 20 rows)"),
        expanded=False,
    ):
        st.dataframe(df.head(20), use_container_width=True, height=260)

    df, cleaned = apply_ai_cleaning(df)
    df = df.copy()
    data_signature = raw_signature + ("|ai=" + frame_signature(df) if cleaned else "")
    if cleaned:
        st.caption(tr("已使用 AI 清洗后的表格", "Using AI-cleaned table"))
        with st.expander(tr("查看清洗后数据（前 20 行）", "View cleaned data")):
            st.dataframe(df.head(20), use_container_width=True, height=240)

    columns = [str(column) for column in df.columns]
    df.columns = columns
    default_target = default_target_column(df) or columns[0]
    target_index = columns.index(default_target) if default_target in columns else 0
    target_col = st.selectbox(
        tr("目标列（要预测的力学性能）", "Target column"),
        columns,
        index=target_index,
        key=f"target_column_{data_signature[:10]}",
    )

    numeric_target = pd.to_numeric(df[target_col], errors="coerce")
    if numeric_target.notna().sum() == 0:
        st.error(
            tr(
                "目标列不是数值列。请先把抗拉强度、屈服强度等转成数字。",
                "The target column is not numeric.",
            )
        )
        return

    default_features = [
        column
        for column in default_feature_columns(df, target_col)
        if column != target_col
    ]
    feature_cols = st.multiselect(
        tr("特征列（输入参数）", "Feature columns"),
        [column for column in columns if column != target_col],
        default=default_features,
        key=f"feature_columns_{data_signature[:10]}_{target_col}",
    )
    feature_cols = [column for column in feature_cols if column != target_col]
    if not feature_cols:
        st.warning(tr("请至少选择一个特征列。", "Select at least one feature column."))
        return

    st.subheader(tr("模型设置", "Model settings"))
    n_estimators = st.slider(
        tr("训练轮数（树模型）", "Training rounds (trees)"),
        50,
        1000,
        300,
        step=50,
        key="n_estimators_slider",
    )
    compare_mode = st.checkbox(
        tr("多模型对比（自动选最优）", "Compare models (auto-select best)"),
        key="compare_mode_checkbox",
    )
    gpr_kernel = "RBF + 白噪声"
    if compare_mode:
        selected_models = st.multiselect(
            tr("选择要对比的模型", "Models to compare"),
            MODEL_NAMES,
            default=["线性回归", "随机森林", "梯度提升树", "MLP 神经网络"],
            key="compare_models_select",
        )
        model_name = selected_models[0] if selected_models else ""
        if "高斯过程回归 (GPR)" in selected_models:
            gpr_kernel = st.selectbox(
                tr("GPR 核函数", "GPR kernel"),
                GPR_KERNEL_NAMES,
                key="gpr_kernel_compare",
            )
    else:
        model_name = st.selectbox(
            tr("选择模型", "Model"), MODEL_NAMES, key="single_model_select"
        )
        selected_models = [model_name]
        if model_name == "高斯过程回归 (GPR)":
            gpr_kernel = st.selectbox(
                tr("GPR 核函数", "GPR kernel"),
                GPR_KERNEL_NAMES,
                key="gpr_kernel_single",
            )
            st.caption(
                tr(
                    "GPR 复杂度较高，样本建议少于 2000；会同时输出预测区间。",
                    "GPR is expensive; recommended for fewer than 2000 samples.",
                )
            )

    test_size = st.slider(
        tr("测试集比例", "Test size"), 0.1, 0.4, 0.2, 0.05, key="test_size_slider"
    )
    if not selected_models:
        st.warning(tr("请至少选择一个模型", "Select at least one model"))
        return

    signature = training_signature(
        data_signature=data_signature,
        target_col=target_col,
        feature_cols=feature_cols,
        model_names=selected_models,
        test_size=float(test_size),
        n_estimators=int(n_estimators),
        gpr_kernel=gpr_kernel,
    )

    if st.button(
        tr("开始训练与预测", "Train and predict"),
        type="primary",
        key="train_button",
    ):
        try:
            with st.spinner(tr("正在训练模型...", "Training models...")):
                result = train_models(
                    df,
                    feature_cols,
                    target_col,
                    selected_models,
                    test_size=float(test_size),
                    n_estimators=int(n_estimators),
                    gpr_kernel=gpr_kernel,
                    data_signature=data_signature,
                    train_signature=signature,
                )
            result["data_source"] = data_source
            result["raw_data_signature"] = raw_signature
            result["trained_at"] = datetime.now().strftime("%Y-%m-%d %H:%M")
            result["excel_bytes"] = excel_bytes(
                result["data_out"],
                metrics=result["metrics"],
                model_name=result["model_name"],
            )
            result["docx_bytes"] = make_word_report(
                target_col=target_col,
                data_source=data_source,
                stats=result["stats"],
                feature_cols=feature_cols,
                model_name=result["model_name"],
                metrics=result["metrics"],
                cmp_df=result["cmp_df"],
                y_test=result["y_test"],
                y_pred=result["y_pred"],
                result_df=result["result_df"],
                importance=result["importance"],
                generated_at=result["trained_at"],
                font_name=_CN_FONT,
            )
            st.session_state["train_result"] = result
            st.session_state.pop("last_prediction", None)
            st.session_state.pop("bo_result", None)
            st.rerun()
        except (ModelingError, DataError) as exc:
            st.error(f"{tr('训练失败', 'Training failed')}：{exc}")
        except Exception as exc:
            st.error(f"{tr('训练出现异常', 'Unexpected training error')}：{exc}")

    result = st.session_state.get("train_result")
    if not result:
        st.info(
            tr(
                "设置好目标列、特征列和模型后，点击“开始训练与预测”。",
                "Configure the target, features and model, then click Train.",
            )
        )
        return
    if result.get("train_signature") != signature:
        st.warning(
            tr(
                "数据、目标列、特征列或模型设置已更改，请重新训练。",
                "Data or settings changed; please retrain.",
            )
        )
        return

    render_results(result)
    render_new_sample(result)
    render_optimization(result)


# ============ PAA-DA 黏附：正式版集成 ============
def _native_records(frame: pd.DataFrame) -> list[dict]:
    records = []
    for record in frame.to_dict("records"):
        clean = {}
        for key, value in record.items():
            if value is None:
                clean[key] = None
            elif isinstance(value, (np.integer,)):
                clean[key] = int(value)
            elif isinstance(value, (np.floating,)):
                clean[key] = None if np.isnan(value) else float(value)
            elif isinstance(value, float) and np.isnan(value):
                clean[key] = None
            else:
                try:
                    clean[key] = None if pd.isna(value) else value
                except (TypeError, ValueError):
                    clean[key] = value
        records.append(clean)
    return records


def _adhesion_dataframes() -> tuple[pd.DataFrame, pd.DataFrame]:
    username = st.session_state["username"]
    try:
        literature_rows = STORAGE.list_adhesion_literature(username)
        experiment_rows = STORAGE.list_adhesion_experiments(username)
    except StorageError as exc:
        st.error(f"{tr('读取黏附数据失败', 'Failed to load adhesion data')}：{exc}")
        return pd.DataFrame(), pd.DataFrame()
    literature = pd.DataFrame(literature_rows)
    experiments = pd.DataFrame(experiment_rows)
    return literature, experiments


def _adhesion_similar(bundle: dict, literature: pd.DataFrame) -> pd.DataFrame:
    if literature is None or literature.empty:
        return pd.DataFrame()
    fields = [field for field in bundle["numeric_features"] if field in literature.columns]
    if not fields:
        return pd.DataFrame()
    base = literature.copy()
    # Supabase/JSON 里的数值列可能是 object dtype，先强制转成 float
    for field in fields:
        base[field] = pd.to_numeric(base[field], errors="coerce")
    base = base.dropna(subset=fields)
    if base.empty:
        return pd.DataFrame()
    frame = bundle["frame"]
    median = frame[fields].apply(pd.to_numeric, errors="coerce").median()
    scale = (
        frame[fields]
        .apply(pd.to_numeric, errors="coerce")
        .std()
        .replace(0, 1)
        .fillna(1)
    )
    distance = (
        ((base[fields] - median) / scale).pow(2).sum(axis=1).pow(0.5)
    )
    base["_distance"] = pd.to_numeric(distance, errors="coerce")
    base = base.dropna(subset=["_distance"])
    return base.nsmallest(5, "_distance")


def _plot_adhesion_importance(bundle: dict) -> None:
    model = bundle["pipeline"].named_steps["model"]
    preprocessor = bundle["pipeline"].named_steps["preprocessor"]
    try:
        names = [
            name.split("__", 1)[-1]
            for name in preprocessor.get_feature_names_out()
        ]
        if hasattr(model, "feature_importances_"):
            values = model.feature_importances_
        elif hasattr(model, "coef_"):
            values = np.abs(np.asarray(model.coef_)).ravel()
        else:
            return
        importance = pd.Series(values, index=names).groupby(level=0).sum()
        importance = importance.sort_values(ascending=True)
        fig, ax = plt.subplots(figsize=(7, max(2.2, 0.32 * len(importance))))
        importance.plot.barh(ax=ax, color="#6f8f6a")
        ax.set_title(tr("特征重要性", "Feature importance"))
        ax.tick_params(labelsize=8)
        style_figure(fig)
        st.pyplot(fig)
        plt.close(fig)
    except Exception:
        return


def render_adhesion_literature(target_unit: str) -> None:
    st.subheader(tr("文献数据", "Literature"))
    st.caption(
        tr(
            "上传 PDF 后由 AI 抽取 PAA-DA 黏附数据，保存到 Supabase；所有数值都要人工核对。",
            "Upload PDFs, extract PAA-DA adhesion data with AI, and save to Supabase.",
        )
    )
    uploaded = st.file_uploader(
        tr("上传文献 PDF（可多选）", "Upload PDFs"),
        type=["pdf"],
        accept_multiple_files=True,
        key="adhesion_pdf_upload",
    )
    pages_by_file: dict[str, list[dict]] = {}
    if uploaded:
        for pdf_file in uploaded:
            try:
                pages = adhesion_core.parse_pdf_bytes(pdf_file.getvalue())
                pages_by_file[pdf_file.name] = pages
                st.caption(f"{pdf_file.name}: {len(pages)} {tr('页', 'pages')}")
            except Exception as exc:
                st.error(f"{pdf_file.name}: {exc}")

    ai_key, ai_base, ai_model = ai_config()
    if not ai_key:
        st.caption(
            tr(
                "未配置 AI Key，可手动粘贴 JSON；配置 DEEPSEEK_API_KEY / ARK_API_KEY 后启用 AI 抽取。",
                "No AI key; use manual JSON or configure DEEPSEEK_API_KEY / ARK_API_KEY.",
            )
        )
    else:
        st.caption(f"AI: {ai_model or tr('未填写模型', 'model missing')}")
        if st.button(
            tr("用 AI 抽取全部 PDF", "Extract all PDFs with AI"),
            type="primary",
            disabled=not pages_by_file,
            key="adhesion_ai_extract",
        ):
            records: list[dict] = []
            progress = st.progress(0.0, text=tr("准备抽取...", "Preparing..."))

            def callback(index: int, total: int, page: int) -> None:
                progress.progress(
                    index / max(total, 1),
                    text=tr(
                        f"抽取第 {index}/{total} 段（第 {page} 页）...",
                        f"Extracting {index}/{total} (page {page})...",
                    ),
                )

            with st.spinner(tr("AI 正在阅读文献...", "AI reading...")):
                for filename, pages in pages_by_file.items():
                    try:
                        records.extend(
                            adhesion_core.extract_records_with_llm(
                                pages,
                                title=filename,
                                doi="",
                                target_unit=target_unit,
                                api_key=ai_key,
                                base_url=ai_base,
                                model=ai_model,
                                progress_cb=callback,
                            )
                        )
                    except Exception as exc:
                        st.error(f"{filename}: {exc}")
            progress.empty()
            if records:
                try:
                    count = STORAGE.save_adhesion_literature(
                        records, st.session_state["username"]
                    )
                    set_adhesion_flash(
                        tr(
                            f"已保存 {count} 条文献记录到 Supabase，请核对",
                            f"Saved {count} literature records to Supabase",
                        )
                    )
                    st.rerun()
                except StorageError as exc:
                    st.error(f"{tr('保存失败', 'Save failed')}：{exc}")
            else:
                st.warning(tr("没有抽取到有效记录", "No valid records extracted"))

    with st.expander(tr("手动粘贴 JSON", "Manual JSON"), expanded=False):
        st.code(
            json.dumps(
                [
                    {
                        "title": "paper title",
                        "doi": "10.xxxx/xxxx",
                        "page": 3,
                        "quote": "lap shear strength 42.5 kPa",
                        "dopamine_content": 4.0,
                        "polymer_concentration": 25.0,
                        "ph": 6.0,
                        "curing_temp_c": 60.0,
                        "curing_time_min": 60.0,
                        "crosslinker": "EDC/NHS",
                        "substrate": "glass",
                        "test_type": "lap_shear",
                        "adhesion_value": 42.5,
                        "adhesion_unit": "kPa",
                        "test_temp_c": 25.0,
                        "humidity": 50.0,
                        "confidence": 0.9,
                    }
                ],
                ensure_ascii=False,
                indent=2,
            ),
            language="json",
        )
        manual_json = st.text_area(
            tr("粘贴记录数组", "Paste records"), height=220, key="adhesion_manual_json"
        )
        if st.button(tr("导入 JSON", "Import JSON"), key="adhesion_import_json"):
            try:
                records = adhesion_core.parse_records_json(manual_json, target_unit)
                count = STORAGE.save_adhesion_literature(
                    records, st.session_state["username"]
                )
                set_adhesion_flash(tr(f"导入 {count} 条", f"Imported {count}"))
                st.rerun()
            except Exception as exc:
                st.error(f"{tr('JSON 解析失败', 'JSON failed')}：{exc}")

    literature, _ = _adhesion_dataframes()
    st.subheader(tr("已保存的文献记录", "Saved literature"))
    if literature.empty:
        st.info(tr("还没有文献记录", "No literature records yet"))
        return
    display_columns = [
        column
        for column in [
            "id", "title", "doi", "page", "test_type", "dopamine_content",
            "polymer_concentration", "ph", "curing_temp_c", "curing_time_min",
            "substrate", "adhesion_value", "adhesion_unit", "adhesion_target",
            "target_unit", "confidence", "quote",
        ]
        if column in literature.columns
    ]
    st.dataframe(
        literature[display_columns],
        use_container_width=True,
        height=max(160, min(420, 42 + 36 * len(literature))),
    )
    st.download_button(
        tr("下载文献数据 CSV", "Download CSV"),
        data=literature.to_csv(index=False).encode("utf-8-sig"),
        file_name="adhesion_literature.csv",
        mime="text/csv",
    )
    delete_id = st.selectbox(
        tr("删除一条记录", "Delete one"),
        ["(不删除)"] + literature["id"].astype(int).tolist(),
        key="adhesion_delete_record",
    )
    if delete_id != "(不删除)" and st.button(
        tr("确认删除", "Delete"), key="adhesion_confirm_delete"
    ):
        try:
            STORAGE.delete_adhesion_literature(
                int(delete_id), st.session_state["username"]
            )
            set_adhesion_flash(tr("已删除", "Deleted"))
            st.rerun()
        except StorageError as exc:
            st.error(f"{tr('删除失败', 'Delete failed')}：{exc}")
    confirm_clear = st.checkbox(
        tr("我确认清空全部文献记录", "Confirm clearing all literature"),
        key="adhesion_clear_literature_confirm",
    )
    if st.button(
        tr("清空文献记录", "Clear literature"),
        disabled=not confirm_clear,
        key="adhesion_clear_literature",
    ):
        try:
            STORAGE.clear_adhesion_literature(st.session_state["username"])
            set_adhesion_flash(tr("已清空", "Cleared"))
            st.rerun()
        except StorageError as exc:
            st.error(f"{tr('清空失败', 'Clear failed')}：{exc}")


def render_adhesion_experiments(target_unit: str) -> None:
    st.subheader(tr("实验数据", "Experiments"))
    st.caption(
        tr(
            "上传 30 条左右的实验数据，映射到标准字段后保存到 Supabase。",
            "Upload ~30 experiments, map columns and save to Supabase.",
        )
    )
    uploaded = st.file_uploader(
        tr("上传实验数据 CSV / Excel", "Upload CSV / Excel"),
        type=["csv", "xlsx", "xls"],
        key="adhesion_experiment_upload",
    )
    frame = None
    if uploaded is not None:
        try:
            frame = adhesion_core.load_experiment_table(
                uploaded.getvalue(), uploaded.name
            )
            st.success(f"{len(frame)} 行 × {len(frame.columns)} 列")
            st.dataframe(frame, use_container_width=True, height=240)
        except Exception as exc:
            st.error(f"{tr('读取失败', 'Read failed')}：{exc}")

    if frame is not None:
        columns = [str(column) for column in frame.columns]
        default_target = adhesion_core.guess_target(columns)
        target_index = columns.index(default_target) if default_target in columns else 0
        target_column = st.selectbox(
            tr("目标列（黏附强度）", "Target column"),
            columns,
            index=target_index,
            key="adhesion_target_column",
        )
        target_unit_in = st.selectbox(
            tr("目标列原始单位", "Original unit"),
            ["kPa", "MPa", "N/cm", "N/25mm", "gf/25mm", "N/mm"],
            key="adhesion_target_unit_in",
        )
        mapping: dict[str, str | None] = {}
        left, right = st.columns(2)
        for index, field in enumerate(adhesion_core.ALL_FIELDS):
            with (left if index % 2 == 0 else right):
                default = adhesion_core.find_column(
                    columns, adhesion_core.FIELD_ALIASES.get(field, [])
                )
                options = ["(不使用)"] + columns
                value = st.selectbox(
                    adhesion_core.FIELD_LABELS.get(field, field),
                    options,
                    index=options.index(default) if default in options else 0,
                    key=f"adhesion_map_{field}",
                )
                mapping[field] = None if value == "(不使用)" else value

        if st.button(
            tr("保存实验数据到 Supabase", "Save experiments to Supabase"),
            type="primary",
            key="adhesion_save_experiments",
        ):
            try:
                standardized = adhesion_core.standardize_experiments(
                    frame, mapping, target_column, target_unit, target_unit_in
                )
                standardized = standardized[
                    standardized[adhesion_core.TARGET_FIELD].notna()
                ]
                if not len(standardized):
                    st.error(tr("没有可转换的目标值", "No convertible target values"))
                else:
                    rows = _native_records(standardized)
                    count = STORAGE.replace_adhesion_experiments(
                        st.session_state["username"], rows
                    )
                    set_adhesion_flash(
                        tr(f"已保存 {count} 条实验数据", f"Saved {count} rows")
                    )
                    st.rerun()
            except Exception as exc:
                st.error(f"{tr('保存失败', 'Save failed')}：{exc}")

    _, experiments = _adhesion_dataframes()
    st.subheader(tr("已保存的实验数据", "Saved experiments"))
    if experiments.empty:
        st.info(tr("还没有实验数据", "No experiments yet"))
        return
    display_columns = [
        column
        for column in [
            "sample_id", "dopamine_content", "polymer_concentration", "ph",
            "curing_temp_c", "curing_time_min", "test_temp_c", "humidity",
            "crosslinker", "substrate", "test_type", "adhesion_target",
            "target_unit",
        ]
        if column in experiments.columns
    ]
    st.dataframe(
        experiments[display_columns],
        use_container_width=True,
        height=max(160, min(420, 42 + 36 * len(experiments))),
    )
    if "adhesion_target" in experiments.columns:
        st.caption(
            tr(
                f"目标均值：{experiments['adhesion_target'].mean():.3f} {target_unit}",
                f"Target mean: {experiments['adhesion_target'].mean():.3f} {target_unit}",
            )
        )
    if st.button(
        tr("删除我的全部实验数据", "Delete my experiments"),
        key="adhesion_delete_experiments",
    ):
        try:
            STORAGE.delete_adhesion_experiments(st.session_state["username"])
            set_adhesion_flash(tr("已删除实验数据", "Experiments deleted"))
            st.rerun()
        except StorageError as exc:
            st.error(f"{tr('删除失败', 'Delete failed')}：{exc}")


def render_adhesion_modeling(target_unit: str) -> None:
    st.subheader(tr("训练与预测", "Train and predict"))
    literature, experiments = _adhesion_dataframes()
    if experiments.empty:
        st.info(tr("请先保存实验数据", "Save experiments first"))
        return
    if "target_unit" in experiments.columns:
        experiments = experiments[
            experiments["target_unit"].fillna("") == target_unit
        ]
    if "target_unit" in literature.columns:
        literature = literature[
            literature["target_unit"].fillna("") == target_unit
        ]
    if experiments.empty:
        st.warning(
            tr(
                f"没有 {target_unit} 单位的实验数据，请回实验页重新保存。",
                f"No experiments in {target_unit}; re-save them.",
            )
        )
        return

    test_types = ["全部"]
    for frame in (experiments, literature):
        if "test_type" in frame.columns:
            test_types.extend(
                sorted(
                    {
                        str(value)
                        for value in frame["test_type"].dropna().tolist()
                        if str(value) not in ("", "unknown")
                    }
                )
            )
    test_types = list(dict.fromkeys(test_types))
    default_type = "lap_shear" if "lap_shear" in test_types else test_types[0]
    test_type = st.selectbox(
        tr("测试类型", "Test type"),
        test_types,
        index=test_types.index(default_type),
        key="adhesion_test_type",
    )
    if test_type == "全部":
        st.warning(
            tr(
                "不同测试方法不要混在一起训练。",
                "Do not mix different test methods.",
            )
        )
    use_literature = st.checkbox(
        tr("把文献数据加入训练集", "Use literature data"),
        value=not literature.empty,
        disabled=literature.empty,
        key="adhesion_use_literature",
    )
    frame, numeric_features, categorical_features = adhesion_core.build_training_frame(
        experiments, literature, target_unit, test_type, use_literature
    )
    if not len(frame):
        st.warning(tr("没有可训练的数据", "No training data"))
        return
    st.caption(
        tr(
            f"训练样本：{len(frame)} 条（实验 "
            f"{(frame['source_type'] == 'experiment').sum()}，文献 "
            f"{(frame['source_type'] == 'literature').sum()}）",
            f"Samples: {len(frame)}",
        )
    )
    show_columns = list(
        dict.fromkeys(
            column
            for column in [
                "sample_id", "source_type", "test_type", "adhesion_target",
                *numeric_features, *categorical_features,
            ]
            if column in frame.columns
        )
    )
    st.dataframe(frame[show_columns], use_container_width=True, height=220)

    model_name = st.selectbox(
        tr("模型", "Model"),
        ["高斯过程回归", "随机森林", "梯度提升树", "Ridge 回归"],
        key="adhesion_model",
    )
    if st.button(
        tr("开始训练", "Train"), type="primary", key="adhesion_train"
    ):
        try:
            bundle = adhesion_core.train_adhesion_model(
                frame,
                numeric_features,
                categorical_features,
                model_name,
                target_unit,
                test_type,
            )
            st.session_state["adhesion_bundle"] = bundle
            st.session_state.pop("adhesion_last_prediction", None)
            st.session_state.pop("adhesion_ai_explanation", None)
            st.success(tr("训练完成", "Trained"))
        except Exception as exc:
            st.error(f"{tr('训练失败', 'Training failed')}：{exc}")

    bundle = st.session_state.get("adhesion_bundle")
    if not bundle:
        st.info(tr("点击“开始训练”后查看结果", "Train to see results"))
        return
    metrics = bundle["metrics"]
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("R²", fmt(metrics["r2"]))
    col2.metric("MAE", fmt(metrics["mae"]))
    col3.metric("RMSE", fmt(metrics["rmse"]))
    col4.metric(tr("交叉验证 R²", "CV R²"), fmt(metrics["cv_r2"]))

    left, right = st.columns(2)
    with left:
        fig, ax = plt.subplots(figsize=(5.2, 3.8))
        ax.scatter(bundle["y_test"], bundle["y_pred"], alpha=0.75, color="#0f5c5e")
        low = float(min(np.min(bundle["y_test"]), np.min(bundle["y_pred"])))
        high = float(max(np.max(bundle["y_test"]), np.max(bundle["y_pred"])))
        ax.plot([low, high], [low, high], "r--", lw=1.3)
        ax.set_xlabel(tr("真实值", "True"))
        ax.set_ylabel(tr("预测值", "Predicted"))
        ax.set_title(tr("预测 vs 真实", "Predicted vs true"))
        style_figure(fig)
        st.pyplot(fig)
        plt.close(fig)
    with right:
        _plot_adhesion_importance(bundle)

    st.subheader(tr("预测新配方", "Predict a new formulation"))
    input_data = {}
    columns = st.columns(min(len(bundle["features"]), 4))
    frame = bundle["frame"]
    for index, field in enumerate(bundle["features"]):
        with columns[index % 4]:
            if field in bundle["numeric_features"]:
                default = (
                    float(frame[field].median())
                    if frame[field].notna().any()
                    else 0.0
                )
                input_data[field] = st.number_input(
                    adhesion_core.FIELD_LABELS.get(field, field),
                    value=default,
                    key=f"adhesion_predict_{field}",
                )
            else:
                options = sorted(
                    {str(value) for value in frame[field].dropna().tolist()}
                ) or ["unknown"]
                input_data[field] = st.selectbox(
                    adhesion_core.FIELD_LABELS.get(field, field),
                    options,
                    key=f"adhesion_predict_{field}",
                )
    if st.button(tr("预测", "Predict"), key="adhesion_predict_btn"):
        try:
            prediction, std = adhesion_core.predict_bundle(
                bundle, pd.DataFrame([input_data])
            )
            st.session_state["adhesion_last_prediction"] = {
                "prediction": prediction,
                "std": std,
                "input": input_data,
                "unit": bundle["target_unit"],
            }
            if std is not None:
                st.success(
                    tr(
                        f"预测黏附强度：**{prediction:.3f} ± {1.96 * std:.3f} "
                        f"{bundle['target_unit']}**（95% 预测区间）",
                        f"Prediction: **{prediction:.3f} ± {1.96 * std:.3f} "
                        f"{bundle['target_unit']}**",
                    )
                )
            else:
                st.success(
                    tr(
                        f"预测黏附强度：**{prediction:.3f} {bundle['target_unit']}**",
                        f"Prediction: **{prediction:.3f} {bundle['target_unit']}**",
                    )
                )
        except Exception as exc:
            st.error(f"{tr('预测失败', 'Prediction failed')}：{exc}")

    last = st.session_state.get("adhesion_last_prediction")
    if last:
        st.caption(
            tr(
                f"最近预测：{last['prediction']:.3f} {last['unit']}"
                + (
                    f" ± {1.96 * last['std']:.3f}"
                    if last.get("std") is not None
                    else ""
                ),
                f"Last prediction: {last['prediction']:.3f} {last['unit']}"
                + (
                    f" ± {1.96 * last['std']:.3f}"
                    if last.get("std") is not None
                    else ""
                ),
            )
        )

    similar = _adhesion_similar(bundle, literature)
    if not similar.empty:
        st.subheader(tr("最相似的文献记录", "Most similar literature"))
        show = [
            column
            for column in [
                "title", "doi", "page", "test_type", "adhesion_target", "quote",
            ]
            if column in similar.columns
        ]
        st.dataframe(similar[show], use_container_width=True, height=220)

    ai_key, ai_base, ai_model = ai_config()
    if ai_key and last:
        if st.button(
            tr("生成 AI 解释（带文献引用）", "AI explanation with citations"),
            key="adhesion_ai_explain",
        ):
            context = ""
            if not similar.empty:
                for _, row in similar.iterrows():
                    context += (
                        f"- {row.get('title', '')} p.{row.get('page', '')}: "
                        f"{row.get('quote', '')} | adhesion={row.get('adhesion_target', '')} "
                        f"{row.get('target_unit', '')}\n"
                    )
            prompt = f"""你是材料学助手。只根据下面给出的预测结果和文献证据回答，
不要编造文献，不要给出没有证据的精确数值。

预测结果：{last['prediction']:.3f} {last['unit']}
模型：{bundle['model_name']}
测试类型：{bundle['test_type']}

文献证据：
{context or '（没有检索到相似文献）'}

请用中文给出：
1. 这个预测意味着什么；
2. 哪些文献/因素支持它；
3. 可信度和需要补充的实验。
"""
            try:
                with st.spinner(tr("AI 解释中...", "Generating explanation...")):
                    explanation = adhesion_core.call_llm(
                        prompt, ai_key, ai_base, ai_model
                    )
                st.session_state["adhesion_ai_explanation"] = explanation
            except Exception as exc:
                st.error(f"{tr('AI 解释失败', 'AI explanation failed')}：{exc}")
    explanation = st.session_state.get("adhesion_ai_explanation")
    if explanation:
        st.markdown(explanation)


def render_adhesion_tab() -> None:
    # Streamlit Cloud 热更新有时只重跑 app.py，不会重新导入 core.storage。
    # 这时会出现 'SupabaseStorage' object has no attribute 'list_adhesion_literature'。
    if not hasattr(STORAGE, "list_adhesion_literature"):
        st.error(
            tr(
                "当前运行进程仍在加载旧版 core/storage.py（缺少黏附数据库方法）。"
                "请在 Streamlit Cloud 的 Manage app 中点击 Reboot app，"
                "或重新部署最新 main 分支。",
                "The running process still has an old core/storage.py. "
                "Use Manage app → Reboot app, or redeploy the latest main branch.",
            )
        )
        return

    st.subheader(tr("PAA-DA 黏附性能：文献 + 小样本预测", "PAA-DA adhesion: literature + prediction"))
    st.caption(
        tr(
            "文献和实验数据保存在 Supabase；AI 负责抽取和解释，模型负责数值预测。"
            "不同测试方法（lap shear / peel / tack）请分开训练。",
            "Literature and experiments are stored in Supabase. AI extracts/explanations; ML predicts.",
        )
    )
    show_adhesion_flash()
    target_unit = st.selectbox(
        tr("黏附目标单位", "Adhesion target unit"),
        ["kPa", "N/cm"],
        key="adhesion_target_unit",
        help=tr(
            "lap shear 通常用 kPa；peel 通常用 N/cm。切换后请重新保存/抽取对应单位的数据。",
            "lap shear usually kPa; peel usually N/cm.",
        ),
    )
    if STORAGE.mode != "supabase":
        st.warning(
            tr(
                "当前为本地 SQLite 调试模式；线上请配置 Supabase，否则 Streamlit Cloud 重启后数据会丢失。",
                "Local SQLite debug mode; configure Supabase for production.",
            )
        )
    tab_lit, tab_exp, tab_model = st.tabs(
        [
            tr("文献数据", "Literature"),
            tr("实验数据", "Experiments"),
            tr("训练与预测", "Train & predict"),
        ]
    )
    with tab_lit:
        render_adhesion_literature(target_unit)
    with tab_exp:
        render_adhesion_experiments(target_unit)
    with tab_model:
        render_adhesion_modeling(target_unit)


# ============ 入口 ============
def main() -> None:
    if SUPABASE_AUTH:
        if not _ensure_cloud_session():
            render_auth()
    elif not st.session_state.get("username"):
        render_auth()

    uploaded_file, chosen_file = render_sidebar()

    tab_model, tab_library, tab_compare, tab_adhesion = st.tabs(
        [
            tr("建模工作台", "Workspace"),
            tr("材料库", "Library"),
            tr("性能对比", "Compare"),
            tr("PAA-DA 研究", "PAA-DA study"),
        ]
    )
    with tab_model:
        render_modeling_tab(uploaded_file, chosen_file)
    with tab_library:
        render_library_tab()
    with tab_compare:
        render_compare_tab()
    with tab_adhesion:
        render_adhesion_tab()


main()
