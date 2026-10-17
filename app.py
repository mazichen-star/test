# app.py
# 材料力学性能预测系统 - Streamlit
# 运行: pip install -r requirements.txt ; streamlit run app.py

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
try:
    from skopt import gp_minimize
    from skopt.space import Real
    _HAS_SKOPT = True
except Exception:
    _HAS_SKOPT = False
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel, ConstantKernel as C
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import font_manager
import os
import io
import hashlib
import base64
import requests
import smtplib
from email.mime.text import MIMEText
from email.header import Header
from email.utils import formataddr

# ============ Supabase 云数据库配置 ============
# 请在 Streamlit Secrets 中配置 SUPABASE_URL 和 SUPABASE_KEY
# 本地运行时可新建 .streamlit/secrets.toml
SUPABASE_URL = st.secrets.get("SUPABASE_URL", "")
SUPABASE_KEY = st.secrets.get("SUPABASE_KEY", "")
if not SUPABASE_URL or not SUPABASE_KEY:
    st.error("未配置 Supabase。请在 Streamlit Secrets 中设置 SUPABASE_URL 和 SUPABASE_KEY。")
    st.stop()
REST = f"{SUPABASE_URL}/rest/v1"
HEADERS = {
    "apikey": SUPABASE_KEY,
    "Authorization": f"Bearer {SUPABASE_KEY}",
    "Content-Type": "application/json",
    "Prefer": "return=representation",
}


def hash_pw(pw):
    return hashlib.sha256(pw.encode("utf-8")).hexdigest()


def user_exists(username):
    r = requests.get(f"{REST}/users?username=eq.{username}&select=username", headers=HEADERS, timeout=15)
    if r.status_code != 200:
        st.error(f"数据库错误({r.status_code}): {r.text[:200]}")
        return False
    return len(r.json()) > 0


def add_user(username, pw, email):
    r = requests.post(f"{REST}/users", headers=HEADERS,
                      json={"username": username, "password_hash": hash_pw(pw), "email": email},
                      timeout=15)
    if r.status_code not in (200, 201):
        st.error(f"注册失败({r.status_code}): {r.text[:200]}")


def find_user_by_email(email):
    r = requests.get(f"{REST}/users?email=eq.{email}&select=username,password_hash",
                     headers=HEADERS, timeout=15)
    if r.status_code != 200:
        return None
    data = r.json()
    return data[0] if data else None


def verify_user(username, pw):
    r = requests.get(f"{REST}/users?username=eq.{username}&select=password_hash", headers=HEADERS, timeout=15)
    if r.status_code != 200:
        st.error(f"登录失败，数据库错误({r.status_code}): {r.text[:200]}")
        return False
    data = r.json()
    if not isinstance(data, list) or not data:
        return False
    return data[0].get("password_hash") == hash_pw(pw)


def list_user_files(username):
    r = requests.get(f"{REST}/user_files?username=eq.{username}&select=id,filename,uploaded_at&order=uploaded_at.desc",
                     headers=HEADERS, timeout=15)
    if r.status_code != 200:
        return []
    return r.json()


def save_user_file(username, filename, content_bytes):
    b64 = base64.b64encode(content_bytes).decode("ascii")
    r = requests.post(f"{REST}/user_files", headers=HEADERS,
                      json={"username": username, "filename": filename, "content": b64}, timeout=30)
    if r.status_code not in (200, 201):
        raise RuntimeError(f"{r.status_code}: {r.text[:200]}")


def load_user_file(username, filename):
    r = requests.get(f"{REST}/user_files?username=eq.{username}&filename=eq.{filename}&select=content",
                     headers=HEADERS, timeout=30)
    if r.status_code != 200:
        return None
    data = r.json()
    if not isinstance(data, list) or not data:
        return None
    raw = data[0]["content"]
    if isinstance(raw, str) and raw.startswith("\\x"):
        return bytes.fromhex(raw[2:])
    try:
        return base64.b64decode(raw)
    except Exception:
        return None




# ============ 共享材料库 ============
def list_materials(keyword=""):
    url = f"{REST}/material_library?select=id,formula,structure,properties,source,contributor,created_at&order=created_at.desc"
    if keyword:
        url += "&formula=ilike.*" + keyword + "*"
    r = requests.get(url, headers=HEADERS, timeout=15)
    if r.status_code != 200:
        return []
    return r.json()

def find_material(formula):
    f = formula.strip().upper()
    r = requests.get(f"{REST}/material_library?select=id,formula,structure", headers=HEADERS, timeout=15)
    if r.status_code != 200:
        return None
    for m in r.json():
        if m.get("formula","").strip().upper() == f:
            return m
    return None

def add_material(formula, structure, properties, source, contributor):
    payload = {"formula": formula.strip(), "structure": structure.strip(),
               "properties": properties, "source": source.strip(), "contributor": contributor}
    r = requests.post(f"{REST}/material_library", headers=HEADERS, json=payload, timeout=15)
    return r.status_code in (200,201), r.text[:300]

def change_password(username, new_pw):
    requests.patch(f"{REST}/users?username=eq.{username}", headers=HEADERS,
                   json={"password_hash": hash_pw(new_pw)}, timeout=15)


def delete_account(username):
    requests.delete(f"{REST}/user_files?username=eq.{username}", headers=HEADERS, timeout=15)
    requests.delete(f"{REST}/users?username=eq.{username}", headers=HEADERS, timeout=15)


def send_email(to_addr, subject, body):
    """通过 st.secrets 配置的 SMTP 发送邮件。未配置时静默失败。"""
    try:
        host = st.secrets["SMTP_HOST"]
        port = int(st.secrets.get("SMTP_PORT", 465))
        user = st.secrets["SMTP_USER"]
        pw = st.secrets["SMTP_PASSWORD"]
    except KeyError:
        return False, "未配置 SMTP"

    msg = MIMEText(body, "plain", "utf-8")
    msg["From"] = formataddr((str(Header("力学预测系统", "utf-8")), user))
    msg["To"] = to_addr
    msg["Subject"] = Header(subject, "utf-8")
    try:
        with smtplib.SMTP_SSL(host, port, timeout=20) as s:
            s.login(user, pw)
            s.sendmail(user, [to_addr], msg.as_string())
        return True, "ok"
    except Exception as e:
        return False, str(e)

# 解决 matplotlib 中文显示方块问题
# 优先加载仓库自带的开源中文字体（Noto Sans SC），兼容 Windows / Linux / Streamlit Cloud
_FONT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts", "NotoSansSC-Regular.ttf")
if os.path.exists(_FONT_PATH):
    font_manager.fontManager.addfont(_FONT_PATH)
    _CN_FONT = font_manager.FontProperties(fname=_FONT_PATH).get_name()
else:
    _CN_FONT = "Microsoft YaHei"

matplotlib.rcParams["font.sans-serif"] = [
    _CN_FONT,
    "Noto Sans CJK SC", "WenQuanYi Zen Hei",
    "Microsoft YaHei", "SimHei", "DejaVu Sans"
]
matplotlib.rcParams["axes.unicode_minus"] = False

st.set_page_config(page_title="力学性能预测系统", layout="wide")

# 暗色模式切换
if "dark_mode" not in st.session_state:
    st.session_state["dark_mode"] = False

dark_css = """
<style>
.stApp { background-color: #0e1117; color: #e6e6e6; }
[data-testid="stSidebar"] { background-color: #1a1d24; }
.stDataFrame, table { color: #e6e6e6; }
h1, h2, h3, h4, p, label, span, div { color: #e6e6e6 !important; }
input, textarea {
    background-color: #262932 !important;
    color: #e6e6e6 !important;
    border-color: #4a4d55 !important;
}
[data-baseweb="select"] > div {
    background-color: #262932 !important;
    color: #e6e6e6 !important;
    border-color: #4a4d55 !important;
}
[data-baseweb="popover"] {
    background-color: #262932 !important;
    color: #e6e6e6 !important;
}
.stButton > button {
    background-color: #262932;
    color: #e6e6e6;
    border-color: #4a4d55;
}
.stSlider > div { background-color: #262932; }
hr { border-color: #4a4d55; }
[data-testid="stFileUploaderDropzone"] {
    background-color: #262932 !important;
    border-color: #4a4d55 !important;
}
[data-testid="stFileUploaderDropzone"] * { color: #e6e6e6 !important; }
[data-testid="stFileUploaderDropzone"] button {
    background-color: #3a3d45;
    color: #e6e6e6;
}
</style>
"""

light_css = """
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

# ============ 多语言 ============
LANGS = ["中文", "English"]
if "ui_lang" not in st.session_state:
    st.session_state["ui_lang"] = "中文"
st.session_state["ui_lang"] = st.sidebar.selectbox("Language / 语言", LANGS,
                                                  index=LANGS.index(st.session_state["ui_lang"]),
                                                  key="lang_select")
EN = st.session_state["ui_lang"] == "English"
T = {
    "app_title": "Mechanical Property Prediction" if EN else "材料力学性能预测系统",
    "app_desc": ("Upload CSV/Excel, train regression models to predict mechanical properties "
                 "(tensile strength, yield strength, Young's modulus, elongation, etc.).")
                if EN else "上传 CSV / Excel 数据，训练回归模型，预测材料力学性能（抗拉强度、屈服强度、弹性模量、延伸率等）。",
    "dark": "Dark mode" if EN else "暗色模式",
    "login": "Login" if EN else "登录",
    "register": "Register" if EN else "注册",
    "forgot": "Forgot password" if EN else "忘记密码",
    "username": "Username" if EN else "用户名",
    "password": "Password" if EN else "密码",
    "login_btn": "Login" if EN else "登录",
    "reg_u_rule": ("Username: 5-20 chars, letters/digits/underscore only, no spaces.")
                  if EN else "用户名：5-20 位，只能用数字、字母、下划线，不能含空格",
    "reg_email": "Bound email (for password recovery)" if EN else "绑定邮箱（用于找回密码）",
    "reg_p_rule": "Password: at least 8 chars, must contain digits and letters, no spaces." if EN else "密码：至少 8 位，须同时包含数字和字母，不能含空格",
    "set_pw": "Set password" if EN else "设置密码",
    "confirm_pw": "Confirm password" if EN else "确认密码",
    "reg_btn": "Register" if EN else "注册",
    "forgot_caption": "Enter your bound email to get a temporary password." if EN else "输入注册时绑定的邮箱，系统会生成临时密码。",
    "bound_email": "Bound email" if EN else "绑定邮箱",
    "recover_btn": "Recover" if EN else "找回密码",
    "upload": "Upload data file" if EN else "上传数据文件",
    "my_files": "My data files" if EN else "我的数据文件",
    "choose_saved": "Choose a saved file" if EN else "选择已保存的文件",
    "no_files": "No files yet" if EN else "暂无历史文件",
    "logout": "Log out" if EN else "退出登录",
    "account_settings": "Account settings" if EN else "账号设置",
    "new_pw": "New password" if EN else "新密码",
    "confirm_new_pw": "Confirm new password" if EN else "确认新密码",
    "change_pw": "Change password" if EN else "修改密码",
    "delete_acc": "Delete account" if EN else "删除账号",
    "data_preview": "Data preview" if EN else "数据预览",
    "target_col": "Target column (property to predict)" if EN else "目标列（要预测的力学性能）",
    "feature_cols": "Feature columns (inputs)" if EN else "特征列（输入参数）",
    "model_settings": "Model settings" if EN else "模型设置",
    "n_est": "Training rounds (n_estimators)" if EN else "训练轮数",
    "test_size": "Test set ratio" if EN else "测试集比例",
    "train_btn": "Train & predict" if EN else "开始训练与预测",
    "compare_mode": "Compare multiple models (auto-pick best)" if EN else "多模型对比（自动选最优）",
    "compare_models": "Select models to compare" if EN else "选择要对比的模型",
    "model_compare_result": "Model comparison" if EN else "模型对比结果",
    "best_model": "Best model" if EN else "最优模型",
    "metrics": "Metrics" if EN else "指标",
    "feature_imp": "Feature importance" if EN else "特征重要性",
    "pred_vs_true": "Predicted vs True" if EN else "预测值 vs 真实值",
    "fit_curves": "Feature fit curves (others fixed at mean)" if EN else "特征拟合曲线（其他特征固定为均值）",
    "pred_detail": "Prediction details" if EN else "预测结果明细",
    "download": "Download Excel with predictions" if EN else "下载含预测列的 Excel",
    "predict_new": "Predict a new sample" if EN else "用训练好的模型预测新样本",
    "fill_hint": "Leave blank to use training mean" if EN else "留空则用训练集均值填充",
    "predict_btn": "Predict" if EN else "预测",
    "need_train": "Please train first." if EN else "请先点击「开始训练与预测」。",
    "pw_reminder": ("First login (or password recovery), please change your password in Account settings.")
                   if EN else "首次登录（或通过找回密码登录）后，请尽快在左侧「账号设置」中修改为自己的密码。",
    "dismiss": "Dismiss" if EN else "我知道了，关闭提示",
}

st.session_state["dark_mode"] = st.sidebar.toggle(T["dark"], value=st.session_state["dark_mode"], key="dark_toggle")
if st.session_state["dark_mode"]:
    st.markdown(dark_css + light_css, unsafe_allow_html=True)
else:
    st.markdown(light_css, unsafe_allow_html=True)
st.title(T["app_title"])
st.markdown(T["app_desc"])

# ============ 登录 / 注册 / 找回密码 ============
import secrets as _secrets
import re as _re
_USER_RE = _re.compile(r"^[a-zA-Z0-9_]{5,20}$")
_PW_RE = _re.compile(r"^(?=.*[0-9])(?=.*[a-zA-Z])\S{8,}$")
if "username" not in st.session_state:
    tab_login, tab_reg, tab_forgot = st.tabs([T["login"], T["register"], T["forgot"]])
    with tab_login:
        u = st.text_input(T["username"], key="login_u")
        p = st.text_input(T["password"], type="password", key="login_p")
        if st.button(T["login_btn"], type="primary"):
            if not u or not p:
                st.warning("请输入用户名和密码" if not EN else "Please enter username and password")
            elif verify_user(u, p):
                st.session_state["username"] = u
                st.session_state["show_pw_reminder"] = True
                st.rerun()
            else:
                st.error("用户名或密码错误" if not EN else "Wrong username or password")
    with tab_reg:
        st.caption(T["reg_u_rule"])
        u2 = st.text_input(T["username"], key="reg_u")
        email2 = st.text_input(T["reg_email"], key="reg_email")
        st.caption(T["reg_p_rule"])
        p2 = st.text_input(T["set_pw"], type="password", key="reg_p")
        p3 = st.text_input(T["confirm_pw"], type="password", key="reg_p3")
        if st.button(T["reg_btn"]):
            if not u2 or not p2:
                st.warning("用户名和密码不能为空" if not EN else "Username and password cannot be empty")
            elif " " in u2 or not _USER_RE.match(u2):
                st.warning(T["reg_u_rule"])
            elif " " in p2 or not _PW_RE.match(p2):
                st.warning(T["reg_p_rule"])
            elif not email2 or "@" not in email2:
                st.warning("请输入有效邮箱" if not EN else "Please enter a valid email")
            elif p2 != p3:
                st.warning("两次密码不一致" if not EN else "Passwords do not match")
            elif user_exists(u2):
                st.warning("该用户名已存在" if not EN else "Username already exists")
            else:
                add_user(u2, p2, email2)
                st.success("注册成功，请登录" if not EN else "Registered, please log in.")
    with tab_forgot:
        st.caption(T["forgot_caption"])
        email_f = st.text_input(T["bound_email"], key="forgot_email")
        if st.button(T["recover_btn"]):
            if not email_f:
                st.warning("请输入邮箱" if not EN else "Please enter email")
            else:
                rec = find_user_by_email(email_f)
                if rec is None:
                    st.error("该邮箱未绑定任何账号" if not EN else "No account bound to this email")
                else:
                    tmp_pw = _secrets.token_urlsafe(8)
                    change_password(rec["username"], tmp_pw)
                    body = (f"Hello,\n\nYour temporary password for {rec['username']} is: {tmp_pw}\n\n"
                            f"Please log in and change it in Account settings.\n\nDo not reply.")
                    ok, msg = send_email(email_f, "Mechanical Prediction - Password Recovery", body)
                    if ok:
                        st.success(f"Temporary password sent to {email_f}." if EN else f"临时密码已发送到 {email_f}")
                    else:
                        st.warning(f"邮件发送失败（{msg}），临时密码: `{tmp_pw}`" if not EN else f"Email failed ({msg}), temp password: `{tmp_pw}`")
    st.stop()

# 已登录
if st.session_state.get("show_pw_reminder"):
    st.warning(T["pw_reminder"])
    if st.button(T["dismiss"]):
        st.session_state["show_pw_reminder"] = False
        st.rerun()

with st.sidebar:
    st.caption(f"当前用户: **{st.session_state['username']}**" if not EN else f"User: **{st.session_state['username']}**")
    if st.button(T["logout"]):
        del st.session_state["username"]
        st.rerun()
    with st.expander(T["account_settings"]):
        new_pw = st.text_input(T["new_pw"], type="password", key="new_pw")
        confirm_pw = st.text_input(T["confirm_new_pw"], type="password", key="confirm_pw")
        if st.button(T["change_pw"]):
            if not new_pw or new_pw != confirm_pw:
                st.warning("两次新密码不一致" if not EN else "Passwords do not match")
            else:
                change_password(st.session_state["username"], new_pw)
                st.success("密码已修改" if not EN else "Password changed")
        st.divider()
        if st.button(T["delete_acc"], type="secondary"):
            delete_account(st.session_state["username"])
            del st.session_state["username"]
            st.rerun()
    st.divider()

    st.header(T["model_settings"])
    uploaded_file = st.file_uploader(T["upload"], type=["csv", "xlsx", "xls"])

    st.subheader(T["my_files"])
    try:
        saved = list_user_files(st.session_state["username"])
        saved_names = [f["filename"] for f in saved]
    except Exception:
        saved_names = []
    if saved_names:
        none_label = "(none)" if EN else "（不加载）"
        chosen = st.selectbox(T["choose_saved"], [none_label] + saved_names)
    else:
        none_label = "(none)" if EN else "（不加载）"
        chosen = none_label
        st.caption(T["no_files"])


@st.cache_data
def load_data(file):
    if file.name.endswith(".csv"):
        return pd.read_csv(file)
    else:
        return pd.read_excel(file)


@st.cache_data
def load_data_bytes(b, name):
    bio = io.BytesIO(b)
    if name.endswith(".csv"):
        return pd.read_csv(bio)
    else:
        return pd.read_excel(bio)


# 确定数据源：优先新上传，其次历史文件
data_source = None
if uploaded_file is not None:
    content = uploaded_file.getvalue()
    try:
        save_user_file(st.session_state["username"], uploaded_file.name, content)
    except Exception as e:
        st.warning(f"文件保存到云端失败: {e}")
    df = load_data(uploaded_file)
    data_source = uploaded_file.name
elif chosen not in ("（不加载）", "(none)"):
    b = load_user_file(st.session_state["username"], chosen)
    if b is None:
        st.error("读取历史文件失败" if not EN else "Failed to load saved file")
        st.stop()
    df = load_data_bytes(b, chosen)
    data_source = chosen



# ============ 材料库：浏览 + 录入 ============
with st.expander("共享材料库（用户共建，去重录入）", expanded=False):
    tab_browse, tab_add = st.tabs(["浏览/搜索", "录入新材料"])
    with tab_browse:
        kw = st.text_input("按化学式搜索（如 Fe、Al2O3）", value="", key="mat_kw")
        mats = list_materials(kw.strip())
        if not mats:
            st.info("库中暂无材料，去录入页添加第一条")
        else:
            rows = []
            for m in mats:
                p = m.get("properties") or {}
                rows.append({
                    "ID": m.get("id"),
                    "化学式/成分": m.get("formula"),
                    "结构": m.get("structure"),
                    "抗拉强度(MPa)": p.get("tensile_strength"),
                    "屈服强度(MPa)": p.get("yield_strength"),
                    "杨氏模量(GPa)": p.get("youngs_modulus"),
                    "电导率(S/m)": p.get("conductivity"),
                    "热导率(W/mK)": p.get("thermal_conductivity"),
                    "数据来源": m.get("source"),
                    "录入者": m.get("contributor"),
                })
            st.dataframe(rows, use_container_width=True, height=300)
    with tab_add:
        c1, c2 = st.columns(2)
        with c1:
            f_new = st.text_input("化学式/成分（如 Fe-0.2C、Al2O3、PLA）", key="mat_formula")
            s_new = st.text_input("结构描述（如 奥氏体FCC、马氏体、非晶）", key="mat_struct")
            src_new = st.selectbox("数据来源", ["实验测量", "文献报道", "手册数据", "估算/预测"], key="mat_source")
        with c2:
            ts = st.number_input("抗拉强度 MPa（可选）", value=0.0, step=10.0, key="mat_ts")
            ys = st.number_input("屈服强度 MPa（可选）", value=0.0, step=10.0, key="mat_ys")
            ym = st.number_input("杨氏模量 GPa（可选）", value=0.0, step=1.0, key="mat_ym")
            ec = st.number_input("电导率 S/m（可选）", value=0.0, step=1e6, format="%.2e", key="mat_ec")
            tc = st.number_input("热导率 W/(m·K)（可选）", value=0.0, step=1.0, key="mat_tc")
        if st.button("提交录入", type="primary", key="mat_submit"):
            if not f_new.strip():
                st.warning("请填写化学式/成分")
            else:
                dup = find_material(f_new)
                if dup:
                    st.warning(f"该材料已在库中（ID={dup['id']}，结构：{dup.get('structure','')}），不重复录入")
                else:
                    props = {}
                    if ts: props["tensile_strength"] = ts
                    if ys: props["yield_strength"] = ys
                    if ym: props["youngs_modulus"] = ym
                    if ec: props["conductivity"] = ec
                    if tc: props["thermal_conductivity"] = tc
                    ok, msg = add_material(f_new, s_new, props, src_new, st.session_state["username"])
                    if ok:
                        st.success("录入成功，已加入共享材料库")
                        st.rerun()
                    else:
                        st.error(f"录入失败: {msg}")

if data_source is None:
    st.info("请在左侧上传数据文件，或选择已保存的文件开始使用。")
    st.stop()

st.caption(f"当前数据文件: **{data_source}**")
st.subheader("数据预览")
st.dataframe(df, use_container_width=True, height=400)
st.write(f"数据形状: **{df.shape[0]} 行 × {df.shape[1]} 列**（表格可上下左右滚动查看全部）")

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = [c for c in df.columns if c not in numeric_cols]

col1, col2 = st.columns(2)
with col1:
    target_col = st.selectbox("目标列（要预测的力学性能）", df.columns, key="target_col_select")
with col2:
    candidate_features = [c for c in df.columns if c != target_col]
    feature_cols = st.multiselect(
        "特征列（输入参数）",
        candidate_features, default=[], key="feature_cols_multi"
    )
    # 自动剔除意外混入的目标列
    feature_cols = [c for c in feature_cols if c != target_col]

if not feature_cols:
    st.warning("请至少选择一个特征列。")
    st.stop()


def preprocess(df_raw, feature_cols, target_col, le_dict=None, fit=True):
    data = df_raw[feature_cols + [target_col]].copy()
    data[target_col] = pd.to_numeric(data[target_col], errors="coerce")
    data = data.dropna(subset=[target_col])
    if fit:
        le_dict = {}
        for c in feature_cols:
            if not pd.api.types.is_numeric_dtype(data[c]):
                le = LabelEncoder()
                data[c] = le.fit_transform(data[c].astype(str))
                le_dict[c] = le
            else:
                data[c] = pd.to_numeric(data[c], errors="coerce")
        data = data.dropna()
    else:
        for c in feature_cols:
            if c in le_dict:
                le = le_dict[c]
                data[c] = data[c].astype(str).map(
                    lambda x: le.transform([x])[0] if x in le.classes_ else 0)
            else:
                data[c] = pd.to_numeric(data[c], errors="coerce")
        data = data.dropna()
    X = data[feature_cols].values
    y = data[target_col].values
    return X, y, data, le_dict


st.subheader("模型设置")

gpr_kernel_options = {
    "RBF + 白噪声": C(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
                    + WhiteKernel(noise_level=1e-2, noise_level_bounds=(1e-10, 1e1)),
    "RBF 单尺度": C(1.0) * RBF(length_scale=1.0),
    "Matern 3/2": C(1.0) * Matern(length_scale=1.0, nu=1.5),
}

# 树模型/迭代模型的轮数（n_estimators），用户可调
n_estimators = st.slider("训练轮数（树模型迭代次数，越大越拟合，过大会过拟合）",
                          50, 1000, 300, step=50, key="n_est_slider")

model_options = {
    "线性回归": LinearRegression(),
    "Ridge 回归": Ridge(alpha=1.0),
    "Lasso 回归": Lasso(alpha=0.1, max_iter=5000),
    "随机森林": RandomForestRegressor(n_estimators=n_estimators, max_depth=None,
                                      min_samples_leaf=1, random_state=42, n_jobs=-1),
    "梯度提升树": GradientBoostingRegressor(n_estimators=n_estimators, learning_rate=0.05,
                                            max_depth=4, subsample=0.8, random_state=42),
    "极端随机树": ExtraTreesRegressor(n_estimators=n_estimators, max_depth=None,
                                       random_state=42, n_jobs=-1),
    "SVR 支持向量机": SVR(kernel="rbf", C=10, epsilon=0.05),
    "核岭回归 (RBF)": KernelRidge(kernel="rbf", alpha=0.1, gamma=0.1),
    "MLP 神经网络": MLPRegressor(hidden_layer_sizes=(128, 64, 32), activation="relu",
                                  alpha=1e-3, max_iter=2000, early_stopping=True,
                                  learning_rate_init=0.001, random_state=42),
}

model_names = list(model_options.keys()) + ["高斯过程回归 (GPR)"]
compare_mode = st.checkbox("多模型对比（同时训练多个模型，自动选最优）", key="compare_mode")

if compare_mode:
    selected_models = st.multiselect("选择要对比的模型", model_names,
                                     default=["线性回归", "随机森林", "梯度提升树", "MLP 神经网络"],
                                     key="compare_multi")
    model_name = None
    gpr_kernel_name = st.selectbox("GPR 核函数（如选了 GPR）", list(gpr_kernel_options.keys()),
                                   key="gpr_kernel_select")
else:
    model_name = st.selectbox("选择模型", model_names, key="model_select")
    gpr_kernel_name = None
    if model_name == "高斯过程回归 (GPR)":
        gpr_kernel_name = st.selectbox("GPR 核函数", list(gpr_kernel_options.keys()), key="gpr_kernel_select")
        st.caption("注意：GPR 复杂度 O(n^3)，样本数建议 < 2000；它还能输出预测不确定性（置信区间）。")

test_size = st.slider("测试集比例", 0.1, 0.4, 0.2, 0.05, key="test_size_slider")
run_btn = st.button("开始训练与预测", type="primary")


def build_model(name, n_est, gpr_kernel):
    if name == "高斯过程回归 (GPR)":
        return GaussianProcessRegressor(kernel=gpr_kernel_options[gpr_kernel],
                                         n_restarts_optimizer=5, normalize_y=True, random_state=42)
    opts = {
        "线性回归": LinearRegression(),
        "Ridge 回归": Ridge(alpha=1.0),
        "Lasso 回归": Lasso(alpha=0.1, max_iter=5000),
        "随机森林": RandomForestRegressor(n_estimators=n_est, random_state=42, n_jobs=-1),
        "梯度提升树": GradientBoostingRegressor(n_estimators=n_est, learning_rate=0.05,
                                                max_depth=4, subsample=0.8, random_state=42),
        "极端随机树": ExtraTreesRegressor(n_estimators=n_est, random_state=42, n_jobs=-1),
        "SVR 支持向量机": SVR(kernel="rbf", C=10, epsilon=0.05),
        "核岭回归 (RBF)": KernelRidge(kernel="rbf", alpha=0.1, gamma=0.1),
        "MLP 神经网络": MLPRegressor(hidden_layer_sizes=(128, 64, 32), activation="relu",
                                      alpha=1e-3, max_iter=2000, early_stopping=True,
                                      learning_rate_init=0.001, random_state=42),
    }
    return opts[name]

if run_btn:
    X, y, data, le_dict = preprocess(df, feature_cols, target_col, fit=True)

    if len(X) < 3:
        st.error("有效样本太少，请检查数据。")
        st.stop()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # ===== 多模型对比 =====
    if compare_mode:
        if not selected_models:
            st.error("请至少选择一个模型进行对比")
            st.stop()
        rows = []
        trained = {}
        with st.spinner("正在训练并对比多个模型，请稍候..."):
            for mname in selected_models:
                try:
                    m = build_model(mname, n_estimators, gpr_kernel_name)
                    m.fit(X_train_s, y_train)
                    yp = m.predict(X_test_s)
                    r2v = r2_score(y_test, yp)
                    maev = mean_absolute_error(y_test, yp)
                    rmsev = np.sqrt(mean_squared_error(y_test, yp))
                    trained[mname] = m
                    rows.append({"模型": mname, "R2": r2v, "MAE": maev, "RMSE": rmsev})
                except Exception as e:
                    rows.append({"模型": mname, "R2": np.nan, "MAE": np.nan, "RMSE": np.nan})
        cmp_df = pd.DataFrame(rows).sort_values("R2", ascending=False).reset_index(drop=True)
        cmp_df.index = cmp_df.index + 1
        cmp_df.index.name = "排名"
        st.subheader("模型对比结果")
        st.dataframe(cmp_df, use_container_width=True)
        best_name = cmp_df["模型"].iloc[0]
        st.success(f"最优模型: **{best_name}** (R2={cmp_df['R2'].iloc[0]:.4f})")
        model = trained[best_name]
        model_name = best_name
    else:
        with st.spinner(f"正在训练 {model_name} ..."):
            model = build_model(model_name, n_estimators, gpr_kernel_name)
            model.fit(X_train_s, y_train)

    is_gpr = isinstance(model, GaussianProcessRegressor)
    if is_gpr:
        y_pred, y_std = model.predict(X_test_s, return_std=True)
    else:
        y_pred = model.predict(X_test_s)
        y_std = None

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    with st.spinner("正在计算 5 折交叉验证..."):
        cv_r2 = cross_val_score(model, scaler.fit_transform(X), y, cv=5, scoring="r2").mean()

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("测试集 R²", f"{r2:.4f}")
    m2.metric("MAE", f"{mae:.4f}")
    m3.metric("RMSE", f"{rmse:.4f}")
    m4.metric("5 折交叉验证 R²", f"{cv_r2:.4f}")

    if hasattr(model, "feature_importances_"):
        st.subheader("特征重要性")
        importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=True)
        fig, ax = plt.subplots(figsize=(8, 0.35 * len(feature_cols)))
        importances.plot.barh(ax=ax, color="#4C9AFF")
        ax.set_xlabel("重要性得分")
        st.pyplot(fig)

    y_axis_label = target_col

    left_col, _ = st.columns(2)
    with left_col:
        st.subheader("预测值 vs 真实值")
        fig2, ax2 = plt.subplots(figsize=(3, 1.8))
        ax2.scatter(y_test, y_pred, alpha=0.7, color="#36B37E", s=20)
        lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
        ax2.plot(lims, lims, "r--", lw=2)
        ax2.set_xlabel(f"真实 {y_axis_label}", fontsize=7)
        ax2.set_ylabel(f"预测 {y_axis_label}", fontsize=7)
        ax2.tick_params(labelsize=6)
        ax2.set_title(f"{model_name}  R2={r2:.3f}", fontsize=8)
        if is_gpr:
            fig2.text(0.5, -0.15, f"核: {model.kernel_}", ha="center", fontsize=5, color="gray")
        st.pyplot(fig2)

    left_col, _ = st.columns(2)
    with left_col:
        if is_gpr:
            st.subheader("测试集预测带 +/-95% 置信区间")
            order = np.argsort(y_test)
            fig3, ax3 = plt.subplots(figsize=(7, 4))
            ax3.plot(range(len(y_test)), y_test[order], "k.", label="真实值", markersize=4)
            ax3.plot(range(len(y_test)), y_pred[order], "-", color="#4C9AFF", label="预测均值", lw=1.5)
            ax3.fill_between(range(len(y_test)),
                             y_pred[order] - 1.96 * y_std[order],
                             y_pred[order] + 1.96 * y_std[order],
                             alpha=0.3, color="#4C9AFF", label="95% 置信区间")
            ax3.set_xlabel("样本序号", fontsize=9)
            ax3.set_ylabel(y_axis_label, fontsize=9)
            ax3.tick_params(labelsize=8)
            ax3.legend(fontsize=8)
            st.pyplot(fig3)

    # ===== 每个特征的拟合曲线图（其他特征固定为均值/众数）=====
    left_col2, _ = st.columns(2)
    with left_col2:
        st.subheader("特征拟合曲线（其他特征固定为均值）")
        fill_vals = {}
        for i, c in enumerate(feature_cols):
            if c in le_dict:
                fill_vals[c] = int(pd.Series(X[:, i]).mode().iloc[0])
            else:
                fill_vals[c] = float(np.nanmean(X[:, i]))

        grid_n = 100
        n_feat = len(feature_cols)
        ncols = 1
        nrows = n_feat
        fig4, axes = plt.subplots(nrows, ncols, figsize=(7, 4 * nrows), squeeze=False)
        for idx, c in enumerate(feature_cols):
            ax = axes[idx][0]
            i = feature_cols.index(c)
            ax.scatter(X[:, i], y, alpha=0.5, color="#4C9AFF", s=20, label="样本")
            if c in le_dict:
                cats = sorted(set(X[:, i].astype(int)))
                X_grid = np.tile(np.array([[fill_vals[f] for f in feature_cols]], dtype=float), (len(cats), 1))
                X_grid[:, i] = np.array(cats)
                y_grid = model.predict(scaler.transform(X_grid))
                ax.plot(cats, y_grid, "ro-", label="拟合")
                ax.set_xticks(cats)
                xs_plot = np.array(cats, dtype=float)
            else:
                x_min, x_max = X[:, i].min(), X[:, i].max()
                xs = np.linspace(x_min, x_max, grid_n)
                X_grid = np.tile(np.array([[fill_vals[f] for f in feature_cols]], dtype=float), (grid_n, 1))
                X_grid[:, i] = xs
                y_grid = model.predict(scaler.transform(X_grid))
                ax.plot(xs, y_grid, "r-", lw=2, label="拟合曲线")
                xs_plot = xs

            imax = int(np.argmax(y_grid))
            imin = int(np.argmin(y_grid))
            for idx_ext, color, txt, dy in [(imax, "#228B22", "最大", 10), (imin, "#B22222", "最小", -16)]:
                ax.scatter(xs_plot[idx_ext], y_grid[idx_ext], s=90, color=color, zorder=5, edgecolors="white")
                ax.annotate(f"{txt}: x={xs_plot[idx_ext]:.3g}, y={y_grid[idx_ext]:.3g}",
                            xy=(xs_plot[idx_ext], y_grid[idx_ext]),
                            xytext=(8, dy),
                            textcoords="offset points", fontsize=8, color=color,
                            arrowprops=dict(arrowstyle="->", color=color, lw=0.8))
            ax.set_xlabel(c, fontsize=9)
            ax.set_ylabel(target_col, fontsize=9)
            ax.tick_params(labelsize=8)
            ax.set_title(f"{c} → {target_col}", fontsize=10)
            ax.legend(fontsize=8, loc="best")
        fig4.tight_layout()
        st.pyplot(fig4)

    result_df = pd.DataFrame({"真实值": y_test, "预测值": y_pred, "误差": y_test - y_pred})
    if y_std is not None:
        result_df["预测标准差"] = y_std
    st.subheader("预测结果明细")
    st.dataframe(result_df, use_container_width=True, height=300)

    X_all_s = scaler.transform(X)
    data_out = data.copy()
    if is_gpr:
        pred_all, std_all = model.predict(X_all_s, return_std=True)
        data_out["预测_" + target_col] = pred_all
        data_out["预测标准差"] = std_all
    else:
        data_out["预测_" + target_col] = model.predict(X_all_s)

    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        data_out.to_excel(writer, index=False, sheet_name="预测结果")
    buf.seek(0)
    st.download_button("下载含预测列的 Excel", buf, file_name="prediction_result.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    st.session_state["model"] = model
    st.session_state["scaler"] = scaler
    st.session_state["le_dict"] = le_dict
    st.session_state["feature_cols"] = feature_cols
    st.session_state["target_col"] = target_col
    st.session_state["is_gpr"] = is_gpr
    st.session_state["fill_values"] = fill_vals

st.divider()
st.subheader("用训练好的模型预测新样本（留空的特征自动用训练集均值/众数填充）")

if "model" in st.session_state:
    fcols = st.session_state["feature_cols"]
    led = st.session_state["le_dict"]
    fill_vals = st.session_state["fill_values"]
    input_vals = []
    cols = st.columns(min(len(fcols), 4))
    for i, f in enumerate(fcols):
        with cols[i % 4]:
            if f in led:
                options = ["（未知/留空）"] + list(led[f].classes_)
                v = st.selectbox(f, options, key=f"new_{f}")
                if v == "（未知/留空）":
                    input_vals.append(fill_vals[f])
                else:
                    input_vals.append(led[f].transform([v])[0])
            else:
                raw = st.text_input(f, value="", key=f"new_{f}",
                                    help="留空则用训练集均值 {:.4g} 填充".format(fill_vals[f]))
                if raw.strip() == "":
                    input_vals.append(fill_vals[f])
                else:
                    try:
                        input_vals.append(float(raw))
                    except ValueError:
                        st.warning(f"「{f}」输入不是数字，已用均值填充")
                        input_vals.append(fill_vals[f])
    if st.button("预测", type="secondary"):
        x_new = st.session_state["scaler"].transform(np.array(input_vals).reshape(1, -1))
        t_label = st.session_state["target_col"]
        if st.session_state.get("is_gpr"):
            pred, std = st.session_state["model"].predict(x_new, return_std=True)
            st.success(f"预测的 **{t_label}** = **{pred[0]:.4f} +/- {1.96*std[0]:.4f}** (95% CI)")
        else:
            pred = st.session_state["model"].predict(x_new)
            st.success(f"预测的 **{t_label}** = **{pred[0]:.4f}**")
else:
    st.info("请先点击「开始训练与预测」。")


# ============ 贝叶斯优化寻优 ============
st.divider()
st.subheader("贝叶斯优化寻优：自动搜索最优配方")

if "model" not in st.session_state:
    st.info("请先训练模型后再使用贝叶斯优化。")
elif not _HAS_SKOPT:
    st.warning("未安装 scikit-optimize，无法使用贝叶斯优化。请在 requirements.txt 中添加后重新部署。")
else:
    fcols = st.session_state["feature_cols"]
    led = st.session_state["le_dict"]
    numeric_features = [f for f in fcols if f not in led]
    if not numeric_features:
        st.info("没有数值特征可优化（全是类别特征）。")
    else:
        scaler = st.session_state["scaler"]
        model = st.session_state["model"]
        is_gpr = st.session_state.get("is_gpr")
        # 用训练数据统计范围
        mins = {}
        maxs = {}
        for f in numeric_features:
            col_idx = fcols.index(f)
            orig = scaler.inverse_transform(np.zeros((1, len(fcols))))[0]  # 均值
            # 用 X 反推范围；这里简化为：取训练集时没存，用 scaler.mean_ 和 scale_ 估
            mean = scaler.mean_[col_idx]
            std = scaler.scale_[col_idx]
            mins[f] = float(mean - 2*std)
            maxs[f] = float(mean + 2*std)

        c1, c2, c3 = st.columns(3)
        with c1:
            direction = st.radio("优化目标", ["最大化", "最小化"], horizontal=True)
        with c2:
            n_iter = st.number_input("迭代次数", min_value=10, max_value=100, value=30, step=5)
        with c3:
            top_k = st.number_input("输出前几个候选", min_value=5, max_value=20, value=10)

        with st.expander("调整各特征搜索范围（默认训练数据均值±2σ）"):
            bounds = {}
            bcols = st.columns(2)
            for i, f in enumerate(numeric_features):
                with bcols[i % 2]:
                    lo = st.number_input(f"{f} 下限", value=mins[f], format="%.4g", key=f"bo_lo_{f}")
                    hi = st.number_input(f"{f} 上限", value=maxs[f], format="%.4g", key=f"bo_hi_{f}")
                    bounds[f] = (lo, hi)

        if st.button("开始寻优", type="primary", key="bo_run"):
            target_col = st.session_state["target_col"]
            direction_sign = -1.0 if direction == "最大化" else 1.0

            def objective(x):
                full = [0.0]*len(fcols)
                for i, f in enumerate(numeric_features):
                    full[fcols.index(f)] = x[i]
                for f in fcols:
                    if f in led:
                        full[fcols.index(f)] = st.session_state["fill_values"][f]
                xs = scaler.transform(np.array(full).reshape(1, -1))
                if is_gpr:
                    p = model.predict(xs, return_std=False)
                else:
                    p = model.predict(xs)
                return direction_sign * float(p[0])

            space = [Real(bounds[f][0], bounds[f][1], name=f) for f in numeric_features]
            with st.spinner(f"贝叶斯优化迭代 {int(n_iter)} 次中..."):
                res = gp_minimize(objective, space, n_calls=int(n_iter), random_state=42, n_initial_points=5)

            # 收集所有评估点
            trials = []
            for x, fun in zip(res.x_iters, res.func_vals):
                rec = {f: x[i] for i, f in enumerate(numeric_features)}
                rec[target_col] = fun * direction_sign
                trials.append(rec)
            trials.sort(key=lambda r: r[target_col], reverse=(direction == "最大化"))

            st.success(f"寻优完成，最优 {target_col} = {trials[0][target_col]:.4f}")
            st.dataframe(trials[:int(top_k)], use_container_width=True, height=350)

            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(8, 4))
            best_so_far = []
            cur = trials[0][target_col]
            for t in trials:
                if direction == "最大化":
                    cur = max(cur, t[target_col])
                else:
                    cur = min(cur, t[target_col])
                best_so_far.append(cur)
            ax.plot(range(1, len(best_so_far)+1), best_so_far, marker="o")
            ax.set_xlabel("Iteration")
            ax.set_ylabel(f"Best {target_col}")
            ax.set_title("Convergence")
            st.pyplot(fig)

