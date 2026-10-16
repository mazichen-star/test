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
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel, ConstantKernel as C
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import font_manager
import os
import io

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
st.title("材料力学性能预测系统")
st.markdown("上传 CSV / Excel 数据，训练回归模型，预测材料力学性能（抗拉强度、屈服强度、弹性模量、延伸率等）。")

with st.sidebar:
    st.header("配置")
    uploaded_file = st.file_uploader("上传数据文件", type=["csv", "xlsx", "xls"])


@st.cache_data
def load_data(file):
    if file.name.endswith(".csv"):
        return pd.read_csv(file)
    else:
        return pd.read_excel(file)


if uploaded_file is None:
    st.info("请在左侧上传数据文件开始使用。")
    st.stop()

df = load_data(uploaded_file)
st.subheader("数据预览")
st.dataframe(df, use_container_width=True, height=400)
st.write(f"数据形状: **{df.shape[0]} 行 × {df.shape[1]} 列**（表格可上下左右滚动查看全部）")

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = [c for c in df.columns if c not in numeric_cols]

col1, col2 = st.columns(2)
with col1:
    target_col = st.selectbox("目标列（要预测的力学性能）", df.columns)
with col2:
    candidate_features = [c for c in df.columns if c != target_col]
    default_features = [c for c in numeric_cols if c != target_col]
    feature_cols = st.multiselect(
        "特征列（输入参数；数值列已预选，类别列可选）",
        candidate_features, default=default_features
    )

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
                          50, 1000, 300, step=50)

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

model_name = st.selectbox("选择模型", list(model_options.keys()) + ["高斯过程回归 (GPR)"])

gpr_kernel_name = None
if model_name == "高斯过程回归 (GPR)":
    gpr_kernel_name = st.selectbox("GPR 核函数", list(gpr_kernel_options.keys()))
    st.caption("注意：GPR 复杂度 O(n^3)，样本数建议 < 2000；它还能输出预测不确定性（置信区间）。")

test_size = st.slider("测试集比例", 0.1, 0.4, 0.2, 0.05)
run_btn = st.button("开始训练与预测", type="primary")

if run_btn:
    X, y, data, le_dict = preprocess(df, feature_cols, target_col, fit=True)

    if len(X) < 3:
        st.error("有效样本太少，请检查数据。")
        st.stop()
    if model_name == "高斯过程回归 (GPR)" and len(X) > 3000:
        st.warning(f"样本量 {len(X)} 较大，GPR 可能较慢（已继续执行，耐心等待）。")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    if model_name == "高斯过程回归 (GPR)":
        kernel = gpr_kernel_options[gpr_kernel_name]
        model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5,
                                         normalize_y=True, random_state=42)
    else:
        model = model_options[model_name]

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

    st.subheader("预测值 vs 真实值")
    fig2, ax2 = plt.subplots(figsize=(5.5, 3.5))
    ax2.scatter(y_test, y_pred, alpha=0.7, color="#36B37E")
    lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    ax2.plot(lims, lims, "r--", lw=2)
    ax2.set_xlabel(f"真实 {y_axis_label}")
    ax2.set_ylabel(f"预测 {y_axis_label}")
    title = f"{model_name}  (R2={r2:.3f})"
    if is_gpr:
        title += f"\n最优核: {model.kernel_}"
    ax2.set_title(title)
    st.pyplot(fig2)

    if is_gpr:
        st.subheader("测试集预测带 +/-95% 置信区间")
        order = np.argsort(y_test)
        fig3, ax3 = plt.subplots(figsize=(5.5, 3.2))
        ax3.plot(range(len(y_test)), y_test[order], "k.", label="真实值")
        ax3.plot(range(len(y_test)), y_pred[order], "-", color="#4C9AFF", label="预测均值")
        ax3.fill_between(range(len(y_test)),
                         y_pred[order] - 1.96 * y_std[order],
                         y_pred[order] + 1.96 * y_std[order],
                         alpha=0.3, color="#4C9AFF", label="95% 置信区间")
        ax3.set_xlabel("样本序号")
        ax3.set_ylabel(y_axis_label)
        ax3.legend()
        st.pyplot(fig3)

    # ===== 每个特征的拟合曲线图（其他特征固定为均值/众数）=====
    st.subheader("特征拟合曲线（其他特征固定为均值）")
    fill_vals = {}
    for i, c in enumerate(feature_cols):
        if c in le_dict:
            fill_vals[c] = int(pd.Series(X[:, i]).mode().iloc[0])
        else:
            fill_vals[c] = float(np.nanmean(X[:, i]))

    grid_n = 100
    n_feat = len(feature_cols)
    ncols = 2
    nrows = int(np.ceil(n_feat / ncols))
    fig4, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 4 * nrows), squeeze=False)
    for idx, c in enumerate(feature_cols):
        ax = axes[idx // ncols][idx % ncols]
        i = feature_cols.index(c)
        # 原始散点
        ax.scatter(X[:, i], y, alpha=0.5, color="#4C9AFF", s=20, label="样本")
        # 生成该特征的网格
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

        # 标出预测最大值和最小值（同时显示横坐标和纵坐标）
        imax = int(np.argmax(y_grid))
        imin = int(np.argmin(y_grid))
        for idx_ext, color, txt, dy in [(imax, "#228B22", "最大", 10), (imin, "#B22222", "最小", -16)]:
            ax.scatter(xs_plot[idx_ext], y_grid[idx_ext], s=90, color=color, zorder=5, edgecolors="white")
            ax.annotate(f"{txt}: x={xs_plot[idx_ext]:.3g}, y={y_grid[idx_ext]:.3g}",
                        xy=(xs_plot[idx_ext], y_grid[idx_ext]),
                        xytext=(8, dy),
                        textcoords="offset points", fontsize=8, color=color,
                        arrowprops=dict(arrowstyle="->", color=color, lw=0.8))
        ax.set_xlabel(c)
        ax.set_ylabel(target_col)
        ax.set_title(f"{c} → {target_col}")
        ax.legend(fontsize=8, loc="best")
    # 隐藏多余子图
    for j in range(n_feat, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")
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
