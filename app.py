# app.py
# 材料力学性能预测系统 - Streamlit
# 运行: pip install -r requirements.txt ; streamlit run app.py

from __future__ import annotations

import io
import os
import re
import secrets
import smtplib
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
from core.reporting import excel_bytes, make_word_report
from core.security import valid_password, valid_username
from core.storage import StorageError, get_storage, storage_label

try:
    from skopt import gp_minimize
    from skopt.space import Real

    _HAS_SKOPT = True
except Exception:
    _HAS_SKOPT = False

# ============ 页面与字体 ============
st.set_page_config(
    page_title="材料力学性能预测系统",
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


# ============ 多语言与主题 ============
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


if "dark_mode" not in st.session_state:
    st.session_state["dark_mode"] = False
st.session_state["dark_mode"] = st.sidebar.toggle(
    tr("暗色模式", "Dark mode"),
    value=st.session_state["dark_mode"],
    key="dark_toggle",
)

DARK_CSS = """
<style>
.stApp { background-color: #0e1117; color: #e6e6e6; }
[data-testid="stSidebar"] { background-color: #1a1d24; }
h1, h2, h3, h4, p, label { color: #e6e6e6; }
input, textarea, [data-baseweb="select"] > div {
    background-color: #262932 !important;
    color: #e6e6e6 !important;
    border-color: #4a4d55 !important;
}
.stButton > button {
    background-color: #262932;
    color: #e6e6e6;
    border-color: #4a4d55;
}
[data-testid="stFileUploaderDropzone"] {
    background-color: #262932 !important;
    border-color: #4a4d55 !important;
}
[data-testid="stFileUploaderDropzone"] * { color: #e6e6e6 !important; }
hr { border-color: #4a4d55; }
</style>
"""

LIGHT_CSS = """
<style>
.stPlot, [data-testid="stImage"] { display: flex; justify-content: center; }
.stPlot img, [data-testid="stImage"] img { margin: auto; max-width: 100%; height: auto; }
@media (max-width: 768px) {
    h1 { font-size: 1.6rem !important; }
    h2, h3 { font-size: 1.2rem !important; }
    [data-testid="stHorizontalBlock"] { flex-direction: column; }
    [data-testid="stHorizontalBlock"] > div { width: 100% !important; }
}
</style>
"""

st.markdown(DARK_CSS if st.session_state["dark_mode"] else LIGHT_CSS, unsafe_allow_html=True)

st.title(tr("材料力学性能预测系统", "Mechanical Property Prediction"))
st.markdown(
    tr(
        "上传 CSV / Excel 数据，训练回归模型，预测材料力学性能"
        "（抗拉强度、屈服强度、弹性模量、延伸率等）。",
        "Upload CSV/Excel, train regression models and predict mechanical "
        "properties such as tensile strength, yield strength, modulus and elongation.",
    )
)

# ============ 存储初始化 ============
try:
    STORAGE = get_storage()
except Exception as exc:
    st.error(f"存储初始化失败：{exc}")
    st.stop()


def set_flash(message: str, kind: str = "success") -> None:
    st.session_state["_flash"] = (kind, message)


def show_flash() -> None:
    item = st.session_state.pop("_flash", None)
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


def render_auth() -> None:
    """登录 / 注册 / 找回密码。未登录时主页面不会继续执行。"""

    st.info(
        tr(
            "首次使用请先注册。数据默认保存在本地 SQLite；配置 Supabase 后自动切换为云数据库。",
            "Register first. Data is stored in local SQLite by default; configure Supabase to use the cloud.",
        )
    )
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

    st.stop()


def render_sidebar():
    with st.sidebar:
        st.caption(
            f"{tr('当前用户', 'User')}: **{st.session_state['username']}**"
        )
        st.caption(f"{tr('存储', 'Storage')}: {storage_label()}")
        if storage_label() == "本地 SQLite":
            st.caption(
                tr(
                    "本地演示模式，数据保存在 data/app.db",
                    "Local demo mode; data is stored in data/app.db",
                )
            )
        if st.button(tr("退出登录", "Log out"), use_container_width=True):
            clear_user_state()
            st.rerun()

        with st.expander(tr("账号设置", "Account settings"), expanded=False):
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
                        if not STORAGE.verify_user(
                            st.session_state["username"], current_pw
                        ):
                            st.error(tr("当前密码错误", "Current password is wrong"))
                        else:
                            STORAGE.change_password(
                                st.session_state["username"], new_pw
                            )
                            st.success(tr("密码已修改", "Password changed"))
                    except StorageError as exc:
                        st.error(f"{tr('修改失败', 'Change failed')}：{exc}")

            st.divider()
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
                    STORAGE.delete_account(st.session_state["username"])
                    clear_user_state()
                    st.rerun()
                except StorageError as exc:
                    st.error(f"{tr('删除失败', 'Delete failed')}：{exc}")

        st.divider()
        uploaded_file = st.file_uploader(
            tr("上传数据文件", "Upload data file"),
            type=["csv", "xlsx", "xls"],
            key="data_upload",
        )

        st.subheader(tr("我的数据文件", "My data files"))
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
            tr("文献NLP提取", "Literature extraction"),
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
            for material in materials:
                props = material.get("properties") or {}
                rows.append(
                    {
                        "ID": material.get("id"),
                        tr("化学式/成分", "Formula"): material.get("formula"),
                        tr("结构", "Structure"): material.get("structure"),
                        tr("抗拉强度(MPa)", "Tensile (MPa)"): props.get(
                            "tensile_strength"
                        ),
                        tr("屈服强度(MPa)", "Yield (MPa)"): props.get(
                            "yield_strength"
                        ),
                        tr("杨氏模量(GPa)", "Modulus (GPa)"): props.get(
                            "youngs_modulus"
                        ),
                        tr("电导率(S/m)", "Conductivity (S/m)"): props.get(
                            "conductivity"
                        ),
                        tr("热导率(W/mK)", "Thermal cond. (W/mK)"): props.get(
                            "thermal_conductivity"
                        ),
                        tr("数据来源", "Source"): material.get("source"),
                        tr("参考文献", "Reference"): material.get("reference"),
                        tr("录入者", "Contributor"): material.get("contributor"),
                    }
                )
            st.dataframe(rows, use_container_width=True, height=320)

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


def render_nlp_tab() -> None:
    st.caption(
        tr(
            "粘贴文献文本或上传 PDF，按规则提取材料性能字段；结果需要人工确认。",
            "Paste text or upload a PDF; fields are extracted by rules and must be reviewed.",
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
            _ai_key, _ai_base, _ai_model = ai_config()
            result = None
            if _ai_key:
                with st.spinner(tr("AI 正在提取文献数据...", "AI extracting...")):
                    try:
                        prompt = f"""从下面材料科学文献文本中提取材料性能数据，返回JSON格式，不要解释。
字段：
- formula: 材料化学式/成分（如 Al2O3, Ti6Al4V, 304 stainless steel）
- tensile: 抗拉强度(MPa)，没有则null
- yield: 屈服强度(MPa)，没有则null
- youngs: 杨氏模量(GPa)，没有则null
- cond: 电导率(S/m)，没有则null
- therm: 热导率(W/(m·K))，没有则null
- doi: DOI号，没有则空字符串

文献文本：
{text[:4000]}"""
                        r = requests.post(
                            _ai_base,
                            headers={"Authorization": f"Bearer {_ai_key}", "Content-Type": "application/json"},
                            json={"model": _ai_model, "messages": [{"role": "user", "content": prompt}], "temperature": 0.1},
                            timeout=60,
                        )
                        if r.status_code == 200:
                            import json as _json
                            raw = r.json()["choices"][0]["message"]["content"].strip()
                            if raw.startswith("```"):
                                raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
                            result = _json.loads(raw)
                    except Exception as exc:
                        st.caption(f"AI 提取失败，改用规则提取: {exc}")
            if result is None:
                # 正则 fallback
                formula_pattern = re.compile(
                    r"\b((?:[A-Z][a-z]?\d*){2,}(?:[-–]\d+(?:\.\d+)?[A-Za-z]*)?)\b"
                )
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
                candidates = formula_pattern.findall(text)
                stop = {"THE","AND","FOR","ARE","WITH","FROM","THIS","THAT","FIG","TABLE","ALLOY","STEEL","SAMPLE"}
                candidates = [c for c in candidates if len(c) >= 2 and c.upper() not in stop]
                result = {
                    "formula": candidates[0] if candidates else "",
                    "tensile": tensile,
                    "yield": yield_strength,
                    "youngs": modulus,
                    "cond": conductivity,
                    "therm": thermal,
                    "doi": doi_match.group(0) if doi_match else "",
                }
            st.session_state["nlp_result"] = result

    if "nlp_result" in st.session_state:
        result = st.session_state["nlp_result"]
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
                    color=["#4C9AFF", "#36B37E"],
                )
                ax.set_title(f"{label} ({unit})", fontsize=10)
                ax.tick_params(labelsize=8)
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
        ax.scatter(X[:, feature_index], y, alpha=0.45, color="#4C9AFF", s=18, label=tr("样本", "Samples"))

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
                    color="#4C9AFF",
                )
            else:
                prediction = model.predict(grid_scaled)
            ax.plot(codes, prediction, "o-", color="#B22222", label=tr("拟合", "Fit"))
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
                    color="#4C9AFF",
                )
            else:
                prediction = model.predict(grid_scaled)
            ax.plot(xs, prediction, "-", color="#B22222", lw=1.8, label=tr("拟合", "Fit"))
            x_plot = xs

        if len(x_plot):
            maximum = int(np.argmax(prediction))
            minimum = int(np.argmin(prediction))
            for index, color, text, offset in [
                (maximum, "#228B22", tr("最大", "Max"), 10),
                (minimum, "#B22222", tr("最小", "Min"), -16),
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
        importance.plot.barh(ax=ax, color="#4C9AFF")
        ax.set_xlabel(tr("重要性", "Importance"), fontsize=8)
        ax.tick_params(labelsize=8)
        st.pyplot(fig)
        plt.close(fig)

    left, right = st.columns(2)
    with left:
        st.subheader(tr("预测值 vs 真实值", "Predicted vs true"))
        fig, ax = plt.subplots(figsize=(5.2, 3.8))
        y_test, y_pred = result["y_test"], result["y_pred"]
        ax.scatter(y_test, y_pred, alpha=0.7, color="#36B37E", s=22)
        low = float(min(np.min(y_test), np.min(y_pred)))
        high = float(max(np.max(y_test), np.max(y_pred)))
        ax.plot([low, high], [low, high], "r--", lw=1.5)
        ax.set_xlabel(tr("真实值", "True"), fontsize=8)
        ax.set_ylabel(tr("预测值", "Predicted"), fontsize=8)
        ax.set_title(f"{result['model_name']}", fontsize=9)
        ax.tick_params(labelsize=7)
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
                color="#4C9AFF",
                label=tr("预测均值", "Mean"),
                lw=1.3,
            )
            ax.fill_between(
                range(len(y_test)),
                y_pred[order] - 1.96 * y_std[order],
                y_pred[order] + 1.96 * y_std[order],
                alpha=0.22,
                color="#4C9AFF",
                label=tr("95% 区间", "95% interval"),
            )
            ax.set_xlabel(tr("样本序号", "Sample"), fontsize=8)
            ax.set_ylabel(result["target_col"], fontsize=8)
            ax.tick_params(labelsize=7)
            ax.legend(fontsize=7)
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.subheader(tr("残差", "Residuals"))
            fig, ax = plt.subplots(figsize=(5.2, 3.8))
            residual = y_test - y_pred
            ax.scatter(y_pred, residual, alpha=0.7, color="#FFAB00", s=22)
            ax.axhline(0, color="r", linestyle="--", lw=1)
            ax.set_xlabel(tr("预测值", "Predicted"), fontsize=8)
            ax.set_ylabel(tr("残差", "Residual"), fontsize=8)
            ax.tick_params(labelsize=7)
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
        st.info(
            tr(
                "请在左侧上传数据文件，或选择已保存的文件开始使用。",
                "Upload a data file or choose a saved file in the sidebar.",
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

    st.caption(f"{tr('当前数据文件', 'Current file')}: **{data_source}**")
    st.subheader(tr("数据预览", "Data preview"))
    st.dataframe(df, use_container_width=True, height=380)
    st.caption(
        tr(
            f"数据形状：{df.shape[0]} 行 × {df.shape[1]} 列",
            f"Shape: {df.shape[0]} rows × {df.shape[1]} columns",
        )
    )

    df, cleaned = apply_ai_cleaning(df)
    df = df.copy()
    data_signature = raw_signature + ("|ai=" + frame_signature(df) if cleaned else "")
    if cleaned:
        st.caption(tr("已使用 AI 清洗后的表格", "Using AI-cleaned table"))
        st.dataframe(df.head(30), use_container_width=True, height=240)

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


# ============ 入口 ============
def main() -> None:
    if not st.session_state.get("username"):
        render_auth()

    uploaded_file, chosen_file = render_sidebar()

    tab_model, tab_library, tab_compare = st.tabs(
        [
            tr("数据建模", "Modeling"),
            tr("共享材料库", "Material library"),
            tr("材料对比", "Material comparison"),
        ]
    )
    with tab_model:
        render_modeling_tab(uploaded_file, chosen_file)
    with tab_library:
        render_library_tab()
    with tab_compare:
        render_compare_tab()


main()
