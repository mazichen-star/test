# app.py
# 材料力学性能预测系统 - Streamlit
# 运行: pip install -r requirements.txt ; streamlit run app.py

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel, ConstantKernel as C
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import io

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
st.dataframe(df.head(10), use_container_width=True)
st.write(f"数据形状: **{df.shape[0]} 行 × {df.shape[1]} 列**")

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = [c for c in df.columns if c not in numeric_cols]
if cat_cols:
    st.info(f"检测到 {len(cat_cols)} 个非数值列（将自动做标签编码）: {', '.join(cat_cols)}")

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

model_options = {
    "线性回归": LinearRegression(),
    "Ridge 回归": Ridge(alpha=1.0),
    "Lasso 回归": Lasso(alpha=0.1, max_iter=5000),
    "随机森林": RandomForestRegressor(n_estimators=200, random_state=42),
    "梯度提升树": GradientBoostingRegressor(n_estimators=200, random_state=42),
    "SVR 支持向量机": SVR(kernel="rbf", C=10),
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

    st.subheader("预测值 vs 真实值")
    fig2, ax2 = plt.subplots(figsize=(6, 6))
    ax2.scatter(y_test, y_pred, alpha=0.7, color="#36B37E")
    lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    ax2.plot(lims, lims, "r--", lw=2)
    ax2.set_xlabel("真实值"); ax2.set_ylabel("预测值")
    title = f"{model_name}  (R2={r2:.3f})"
    if is_gpr:
        title += f"\n最优核: {model.kernel_}"
    ax2.set_title(title)
    st.pyplot(fig2)

    if is_gpr:
        st.subheader("测试集预测带 +/-95% 置信区间")
        order = np.argsort(y_test)
        fig3, ax3 = plt.subplots(figsize=(8, 4))
        ax3.plot(range(len(y_test)), y_test[order], "k.", label="真实值")
        ax3.plot(range(len(y_test)), y_pred[order], "-", color="#4C9AFF", label="预测均值")
        ax3.fill_between(range(len(y_test)),
                         y_pred[order] - 1.96 * y_std[order],
                         y_pred[order] + 1.96 * y_std[order],
                         alpha=0.3, color="#4C9AFF", label="95% 置信区间")
        ax3.set_xlabel("样本序号"); ax3.set_ylabel(target_col)
        ax3.legend()
        st.pyplot(fig3)

    result_df = pd.DataFrame({"真实值": y_test, "预测值": y_pred, "误差": y_test - y_pred})
    if y_std is not None:
        result_df["预测标准差"] = y_std
    st.subheader("预测结果明细")
    st.dataframe(result_df, use_container_width=True)

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

st.divider()
st.subheader("用训练好的模型预测新样本")

if "model" in st.session_state:
    fcols = st.session_state["feature_cols"]
    led = st.session_state["le_dict"]
    input_vals = []
    cols = st.columns(min(len(fcols), 4))
    for i, f in enumerate(fcols):
        with cols[i % 4]:
            if f in led:
                options = list(led[f].classes_)
                v = st.selectbox(f, options, key=f"new_{f}")
                input_vals.append(led[f].transform([v])[0])
            else:
                v = st.number_input(f, value=0.0, key=f"new_{f}")
                input_vals.append(v)
    if st.button("预测", type="secondary"):
        x_new = st.session_state["scaler"].transform(np.array(input_vals).reshape(1, -1))
        if st.session_state.get("is_gpr"):
            pred, std = st.session_state["model"].predict(x_new, return_std=True)
            st.success(f"预测的 **{st.session_state['target_col']}** = **{pred[0]:.4f} +/- {1.96*std[0]:.4f}** (95% CI)")
        else:
            pred = st.session_state["model"].predict(x_new)
            st.success(f"预测的 **{st.session_state['target_col']}** = **{pred[0]:.4f}**")
else:
    st.info("请先点击「开始训练与预测」。")
