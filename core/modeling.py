"""模型注册、训练、评估与交叉验证。"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, Matern, WhiteKernel
from sklearn.inspection import permutation_importance
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from .data import (
    DataError,
    make_feature_ranges,
    make_fill_values,
    preprocess,
)

MODEL_NAMES = [
    "线性回归",
    "Ridge 回归",
    "Lasso 回归",
    "随机森林",
    "梯度提升树",
    "极端随机树",
    "SVR 支持向量机",
    "核岭回归 (RBF)",
    "MLP 神经网络",
    "高斯过程回归 (GPR)",
]

GPR_KERNEL_NAMES = ["RBF + 白噪声", "RBF 单尺度", "Matern 3/2"]


class ModelingError(RuntimeError):
    """训练或预测阶段可预期的错误。"""


def kernel_options() -> dict[str, Any]:
    """每次调用都创建新的 kernel，避免多个模型共享可变对象。"""

    return {
        "RBF + 白噪声": C(1.0, (1e-3, 1e3))
        * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
        + WhiteKernel(noise_level=1e-2, noise_level_bounds=(1e-10, 1e1)),
        "RBF 单尺度": C(1.0) * RBF(length_scale=1.0),
        "Matern 3/2": C(1.0) * Matern(length_scale=1.0, nu=1.5),
    }


def build_model(
    name: str,
    n_estimators: int = 300,
    gpr_kernel: str = "RBF + 白噪声",
    random_state: int = 42,
):
    if name not in MODEL_NAMES:
        raise ModelingError(f"未知模型：{name}")

    if name == "高斯过程回归 (GPR)":
        kernel = kernel_options().get(gpr_kernel)
        if kernel is None:
            raise ModelingError(f"未知 GPR 核函数：{gpr_kernel}")
        return GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=5,
            normalize_y=True,
            random_state=random_state,
        )

    mapping = {
        "线性回归": lambda: LinearRegression(),
        "Ridge 回归": lambda: Ridge(alpha=1.0),
        "Lasso 回归": lambda: Lasso(alpha=0.1, max_iter=5000),
        "随机森林": lambda: RandomForestRegressor(
            n_estimators=n_estimators, random_state=random_state, n_jobs=-1
        ),
        "梯度提升树": lambda: GradientBoostingRegressor(
            n_estimators=n_estimators,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            random_state=random_state,
        ),
        "极端随机树": lambda: ExtraTreesRegressor(
            n_estimators=n_estimators, random_state=random_state, n_jobs=-1
        ),
        "SVR 支持向量机": lambda: SVR(kernel="rbf", C=10, epsilon=0.05),
        "核岭回归 (RBF)": lambda: KernelRidge(kernel="rbf", alpha=0.1, gamma=0.1),
        "MLP 神经网络": lambda: MLPRegressor(
            hidden_layer_sizes=(128, 64, 32),
            activation="relu",
            alpha=1e-3,
            max_iter=2000,
            early_stopping=True,
            learning_rate_init=0.001,
            random_state=random_state,
        ),
    }
    return mapping[name]()


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    if len(y_true) >= 2:
        r2 = float(r2_score(y_true, y_pred))
    else:
        r2 = float("nan")
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {"r2": r2, "mae": mae, "rmse": rmse}


def cross_validate_r2(
    model, X: np.ndarray, y: np.ndarray
) -> tuple[float | None, int]:
    """用 Pipeline 做交叉验证，避免在 CV 之外先缩放造成数据泄漏。"""

    n_samples = len(X)
    if n_samples < 4:
        return None, 0
    folds = min(5, max(2, n_samples // 2))
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", clone(model)),
        ]
    )
    try:
        scores = cross_val_score(
            pipeline, X, y, cv=folds, scoring="r2", error_score=np.nan
        )
    except Exception:
        return None, folds
    valid = scores[~np.isnan(scores)]
    if not len(valid):
        return None, folds
    return float(np.mean(valid)), folds


def feature_importance(
    model,
    feature_cols: list[str],
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> pd.Series | None:
    if hasattr(model, "feature_importances_"):
        values = np.asarray(model.feature_importances_, dtype=float)
        if len(values) == len(feature_cols):
            return pd.Series(values, index=feature_cols).sort_values(ascending=True)
    if hasattr(model, "coef_"):
        values = np.asarray(model.coef_, dtype=float)
        if values.ndim > 1:
            values = np.abs(values).mean(axis=0)
        values = np.abs(values).ravel()
        if len(values) == len(feature_cols):
            return pd.Series(values, index=feature_cols).sort_values(ascending=True)
    if len(X_test) >= 2 and len(feature_cols):
        try:
            result = permutation_importance(
                model,
                X_test,
                y_test,
                n_repeats=5,
                random_state=42,
                scoring="r2",
            )
            return pd.Series(
                result.importances_mean, index=feature_cols
            ).sort_values(ascending=True)
        except Exception:
            return None
    return None


def train_models(
    frame: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    model_names: list[str],
    *,
    test_size: float = 0.2,
    n_estimators: int = 300,
    gpr_kernel: str = "RBF + 白噪声",
    data_signature: str = "",
    train_signature: str = "",
    random_state: int = 42,
) -> dict[str, Any]:
    """训练一个或多个模型，返回可直接渲染/下载的结果字典。"""

    model_names = [name for name in model_names if name in MODEL_NAMES]
    if not model_names:
        raise ModelingError("请至少选择一个模型")

    try:
        X, y, data, encoders, stats = preprocess(
            frame, feature_cols, target_col, fit=True
        )
    except DataError as exc:
        raise ModelingError(str(exc)) from exc

    if len(X) < 3:
        raise ModelingError(
            f"有效样本只有 {len(X)} 条，至少需要 3 条才能训练。"
            "请检查目标列是否为数值、特征列是否缺失过多。"
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    trained: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    errors: dict[str, str] = {}

    for name in model_names:
        try:
            model = build_model(name, n_estimators, gpr_kernel, random_state)
            model.fit(X_train_scaled, y_train)
            prediction = model.predict(X_test_scaled)
            metrics = regression_metrics(y_test, prediction)
            trained[name] = model
            rows.append({"模型": name, "R2": metrics["r2"], "MAE": metrics["mae"], "RMSE": metrics["rmse"]})
        except Exception as exc:  # 单个模型失败不应拖垮整个对比
            errors[name] = str(exc)

    if not trained:
        detail = "；".join(f"{name}: {message}" for name, message in errors.items())
        raise ModelingError(f"所有模型都训练失败。{detail}")

    comparisons = pd.DataFrame(rows)
    comparisons = comparisons.sort_values(
        ["R2", "RMSE"], ascending=[False, True], na_position="last"
    ).reset_index(drop=True)
    comparisons.index = comparisons.index + 1
    comparisons.index.name = "排名"

    best_name = str(comparisons.iloc[0]["模型"])
    model = trained[best_name]
    is_gpr = isinstance(model, GaussianProcessRegressor)

    if is_gpr:
        y_pred, y_std = model.predict(X_test_scaled, return_std=True)
        y_std = np.asarray(y_std, dtype=float)
    else:
        y_pred = np.asarray(model.predict(X_test_scaled), dtype=float)
        y_std = None

    metrics = regression_metrics(y_test, y_pred)
    cv_r2, cv_folds = cross_validate_r2(model, X, y)

    result_df = pd.DataFrame(
        {"真实值": y_test, "预测值": y_pred, "误差": y_test - y_pred}
    )
    if y_std is not None:
        result_df["预测标准差"] = y_std

    X_all_scaled = scaler.transform(X)
    data_out = data.copy()
    if is_gpr:
        all_pred, all_std = model.predict(X_all_scaled, return_std=True)
        data_out["预测_" + target_col] = all_pred
        data_out["预测标准差"] = all_std
    else:
        data_out["预测_" + target_col] = model.predict(X_all_scaled)

    return {
        "model": model,
        "model_name": best_name,
        "trained": trained,
        "cmp_df": comparisons,
        "errors": errors,
        "is_gpr": is_gpr,
        "scaler": scaler,
        "encoders": encoders,
        "feature_cols": feature_cols,
        "target_col": target_col,
        "fill_values": make_fill_values(X, feature_cols, encoders),
        "feature_ranges": make_feature_ranges(X, feature_cols, encoders),
        "X": X,
        "y": y,
        "data": data,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "y_pred": y_pred,
        "y_std": y_std,
        "metrics": {**metrics, "cv_r2": cv_r2, "cv_folds": cv_folds},
        "result_df": result_df,
        "data_out": data_out,
        "importance": feature_importance(model, feature_cols, X_test_scaled, y_test),
        "stats": stats,
        "data_signature": data_signature,
        "train_signature": train_signature,
    }


def predict_one(
    model,
    scaler: StandardScaler,
    feature_cols: list[str],
    values: list[float],
) -> tuple[float, float | None]:
    x = np.asarray(values, dtype=float).reshape(1, -1)
    x_scaled = scaler.transform(x)
    if isinstance(model, GaussianProcessRegressor):
        prediction, std = model.predict(x_scaled, return_std=True)
        return float(prediction[0]), float(std[0])
    return float(model.predict(x_scaled)[0]), None
