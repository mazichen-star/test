"""PAA-DA 黏附材料：文献抽取、实验数据标准化与预测。

界面层只负责展示，所有可测试逻辑放在这里。
"""

from __future__ import annotations

import io
import json
import re
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import requests
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data import load_table_bytes

NUMERIC_FIELDS = [
    "dopamine_content",
    "polymer_concentration",
    "ph",
    "curing_temp_c",
    "curing_time_min",
    "test_temp_c",
    "humidity",
]
CATEGORICAL_FIELDS = ["crosslinker", "substrate", "test_type"]
ALL_FIELDS = NUMERIC_FIELDS + CATEGORICAL_FIELDS
TARGET_UNITS = ["kPa", "N/cm"]
TARGET_FIELD = "adhesion_target"

FIELD_LABELS = {
    "dopamine_content": "多巴胺含量 (wt%)",
    "polymer_concentration": "聚合物浓度 (wt%)",
    "ph": "pH",
    "curing_temp_c": "固化温度 (°C)",
    "curing_time_min": "固化时间 (min)",
    "test_temp_c": "测试温度 (°C)",
    "humidity": "湿度 (%)",
    "crosslinker": "交联剂",
    "substrate": "基材",
    "test_type": "测试类型",
    "adhesion_target": "黏附强度（目标）",
}

FIELD_ALIASES = {
    "dopamine_content": [
        "多巴胺", "dopamine", "dopamine_content", "da含量", "da content", "da",
    ],
    "polymer_concentration": [
        "浓度", "固含", "concentration", "polymer_concentration",
        "solid content", "聚合物浓度",
    ],
    "ph": ["ph", "酸碱度"],
    "curing_temp_c": [
        "固化温度", "curing_temp", "curing_temp_c", "curing temp",
        "curing temperature", "加热温度",
    ],
    "curing_time_min": [
        "固化时间", "curing_time", "curing_time_min", "curing time", "加热时间",
    ],
    "test_temp_c": [
        "测试温度", "test_temp", "test_temp_c", "test temp", "test temperature",
    ],
    "humidity": ["湿度", "humidity", "rh"],
    "crosslinker": ["交联剂", "crosslink", "crosslinker", "edc", "nhs", "京尼平"],
    "substrate": ["基材", "substrate", "基底", "材料", "adherend"],
    "test_type": [
        "测试类型", "test_type", "test type", "testtype", "测试方法", "method",
        "剪切", "剥离",
    ],
}

TARGET_ALIASES = [
    "黏附", "粘附", "adhesion", "lap", "shear", "peel", "tack",
    "强度", "kpa", "mpa", "n/cm",
]


class AdhesionError(RuntimeError):
    pass


def to_kpa(value: Any, unit: str) -> float | None:
    if value is None or not unit:
        return None
    text = str(unit).strip().lower().replace(" ", "")
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if text == "kpa":
        return number
    if text in ("mpa", "n/mm2", "n/mm²"):
        return number * 1_000.0
    if text == "gpa":
        return number * 1_000_000.0
    if text == "pa":
        return number / 1_000.0
    if text in ("n/cm2", "n/cm²"):
        return number * 10.0
    return None


def to_n_cm(value: Any, unit: str) -> float | None:
    if value is None or not unit:
        return None
    text = str(unit).strip().lower().replace(" ", "")
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if text == "n/cm":
        return number
    if text == "n/mm":
        return number * 10.0
    if text == "n/25mm":
        return number / 2.5
    if text == "gf/25mm":
        return number * 0.00980665 / 2.5
    if text == "gf/cm":
        return number * 0.00980665
    return None


def to_target(value: Any, unit: str, target_unit: str) -> float | None:
    if target_unit == "kPa":
        return to_kpa(value, unit)
    if target_unit == "N/cm":
        return to_n_cm(value, unit)
    return None


def normalize_test_type(value: Any) -> str:
    text = str(value or "").lower()
    if any(word in text for word in ("lap", "shear", "剪切", "搭接")):
        return "lap_shear"
    if any(word in text for word in ("peel", "剥离")):
        return "peel"
    if any(word in text for word in ("tack", "初粘", "初黏")):
        return "tack"
    if text in ("", "nan", "none", "unknown", "未知"):
        return "unknown"
    return "other"


def normalize_record(raw: dict, target_unit: str) -> dict:
    def number(key: str) -> float | None:
        value = raw.get(key)
        if value is None or value == "":
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    value = number("adhesion_value")
    unit = str(raw.get("adhesion_unit") or "").strip()
    target = number("adhesion_target")
    if target is None and value is not None:
        target = to_target(value, unit, target_unit)

    page = raw.get("page")
    try:
        page = int(float(page)) if page not in (None, "") else None
    except (TypeError, ValueError):
        page = None

    return {
        "source_file": str(raw.get("source_file") or ""),
        "title": str(raw.get("title") or ""),
        "doi": str(raw.get("doi") or ""),
        "page": page,
        "quote": str(raw.get("quote") or ""),
        "material_system": str(raw.get("material_system") or "PAA-DA"),
        "dopamine_content": number("dopamine_content"),
        "polymer_concentration": number("polymer_concentration"),
        "ph": number("ph"),
        "curing_temp_c": number("curing_temp_c"),
        "curing_time_min": number("curing_time_min"),
        "crosslinker": str(raw.get("crosslinker") or ""),
        "substrate": str(raw.get("substrate") or ""),
        "test_type": normalize_test_type(raw.get("test_type")),
        "adhesion_value": value,
        "adhesion_unit": unit,
        "adhesion_target": target,
        "target_unit": target_unit,
        "test_temp_c": number("test_temp_c"),
        "humidity": number("humidity"),
        "confidence": number("confidence"),
    }


def parse_pdf_bytes(pdf_bytes: bytes) -> list[dict]:
    try:
        import pdfplumber
    except ImportError as exc:
        raise AdhesionError("请先安装 pdfplumber：pip install pdfplumber") from exc
    pages: list[dict] = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for index, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            if text.strip():
                pages.append({"page": index, "text": text})
    return pages


def chunk_pages(pages: list[dict], size: int = 1_800) -> list[dict]:
    chunks: list[dict] = []
    for page in pages:
        text = re.sub(r"\n{2,}", "\n", page["text"]).strip()
        for start in range(0, len(text), size):
            piece = text[start : start + size].strip()
            if len(piece) >= 80:
                chunks.append({"page": page["page"], "text": piece})
    return chunks


def call_llm(prompt: str, api_key: str, base_url: str, model: str) -> str:
    if not api_key:
        raise AdhesionError("未配置 AI Key")
    if not model:
        raise AdhesionError("未配置模型名（DEEPSEEK_MODEL / ARK_MODEL）")
    response = requests.post(
        base_url,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": "你是材料学文献数据抽取助手，只输出合法 JSON。"},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.1,
        },
        timeout=120,
    )
    if response.status_code != 200:
        raise AdhesionError(f"LLM HTTP {response.status_code}: {response.text[:200]}")
    return response.json()["choices"][0]["message"]["content"].strip()


def extract_records_with_llm(
    pages: list[dict],
    *,
    title: str,
    doi: str,
    target_unit: str,
    api_key: str,
    base_url: str,
    model: str,
    progress_cb: Callable[[int, int, int], None] | None = None,
) -> list[dict]:
    chunks = chunk_pages(pages)
    records: list[dict] = []
    for index, chunk in enumerate(chunks, start=1):
        if progress_cb:
            progress_cb(index, len(chunks), chunk["page"])
        prompt = f"""从下面的材料论文片段中抽取“聚丙烯酸-多巴胺（PAA-DA）黏附性能”数据。

只输出 JSON 数组，不要解释，不要 markdown。每个元素包含：
material_system, dopamine_content, polymer_concentration, ph,
curing_temp_c, curing_time_min, crosslinker, substrate, test_type,
adhesion_value, adhesion_unit, test_temp_c, humidity, page, quote, confidence, title, doi

规则：
- 只提取片段中明确写出的数值；没有就填 null。
- test_type 只能是 lap_shear / peel / tack / other / unknown。
- adhesion_value 保留原文数值，adhesion_unit 保留原文单位。
- quote 必须是原文中的短句，page 使用给定页码。
- 不要跨片段猜测，不要合并不同实验。
- 材料体系默认 PAA-DA。

来源文件：{title}
DOI：{doi}
页码：{chunk['page']}
文本：
{chunk['text']}
"""
        try:
            content = call_llm(prompt, api_key, base_url, model)
            content = re.sub(r"^```(?:json)?", "", content).strip()
            content = re.sub(r"```$", "", content).strip()
            start = content.find("[")
            end = content.rfind("]")
            if start >= 0 and end > start:
                content = content[start : end + 1]
            parsed = json.loads(content)
            if isinstance(parsed, dict):
                parsed = parsed.get("records", [parsed])
            for item in parsed:
                if not isinstance(item, dict):
                    continue
                item.setdefault("title", title)
                item.setdefault("doi", doi)
                item.setdefault("page", chunk["page"])
                item.setdefault("source_file", title)
                record = normalize_record(item, target_unit)
                if record["adhesion_value"] is not None or record["adhesion_target"] is not None:
                    records.append(record)
        except Exception:
            # 单段失败不阻断整篇文献；调用方可以显示重试提示
            continue
    return records


def parse_records_json(text: str, target_unit: str) -> list[dict]:
    parsed = json.loads(text)
    if isinstance(parsed, dict):
        parsed = [parsed]
    if not isinstance(parsed, list):
        raise AdhesionError("JSON 必须是对象或数组")
    return [normalize_record(item, target_unit) for item in parsed if isinstance(item, dict)]


def find_column(columns: list[str], aliases: list[str]) -> str | None:
    lowered = {str(column).strip().lower(): str(column) for column in columns}
    for alias in aliases:
        if alias.lower() in lowered:
            return lowered[alias.lower()]
    for column in columns:
        text = str(column).lower()
        if any(alias.lower() in text for alias in aliases):
            return str(column)
    return None


def guess_target(columns: list[str]) -> str | None:
    return find_column(columns, TARGET_ALIASES)


def load_experiment_table(content: bytes, filename: str) -> pd.DataFrame:
    try:
        return load_table_bytes(content, filename)
    except Exception as exc:
        raise AdhesionError(f"实验数据读取失败：{exc}") from exc


def standardize_experiments(
    frame: pd.DataFrame,
    mapping: dict[str, str | None],
    target_column: str,
    target_unit: str,
    target_unit_in: str,
) -> pd.DataFrame:
    result = pd.DataFrame(index=frame.index)
    result["sample_id"] = [f"E{index + 1:03d}" for index in range(len(frame))]
    for field in ALL_FIELDS:
        column = mapping.get(field)
        if column and column in frame.columns:
            if field in NUMERIC_FIELDS:
                result[field] = pd.to_numeric(frame[column], errors="coerce")
            else:
                result[field] = frame[column].astype("string").fillna("unknown")
        else:
            result[field] = np.nan if field in NUMERIC_FIELDS else "unknown"

    values = pd.to_numeric(frame[target_column], errors="coerce")
    result[TARGET_FIELD] = values.apply(
        lambda value: to_target(value, target_unit_in, target_unit)
    )
    result["target_unit"] = target_unit
    result["source_type"] = "experiment"
    return result


def build_training_frame(
    experiments: pd.DataFrame,
    literature: pd.DataFrame,
    target_unit: str,
    test_type: str,
    use_literature: bool,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    frames = []
    if experiments is not None and len(experiments):
        exp = experiments.copy()
        exp["source_type"] = "experiment"
        frames.append(exp)
    if use_literature and literature is not None and len(literature):
        lit = literature.copy()
        if "target_unit" in lit.columns:
            lit = lit[lit["target_unit"].fillna("") == target_unit].copy()
        if TARGET_FIELD not in lit.columns:
            lit[TARGET_FIELD] = np.nan
        lit[TARGET_FIELD] = lit.apply(
            lambda row: (
                row[TARGET_FIELD]
                if pd.notna(row.get(TARGET_FIELD))
                else to_target(row.get("adhesion_value"), row.get("adhesion_unit"), target_unit)
            ),
            axis=1,
        )
        lit["source_type"] = "literature"
        frames.append(lit)
    if not frames:
        return pd.DataFrame(), [], []

    frame = pd.concat(frames, ignore_index=True)
    frame = frame[frame[TARGET_FIELD].notna()].copy()
    if test_type and test_type != "全部":
        frame = frame[frame["test_type"].fillna("unknown") == test_type].copy()
    if not len(frame):
        return frame, [], []

    numeric_features = [field for field in NUMERIC_FIELDS if field in frame.columns]
    numeric_features = [
        field
        for field in numeric_features
        if pd.to_numeric(frame[field], errors="coerce").notna().any()
    ]
    categorical_features = [field for field in ["crosslinker", "substrate"] if field in frame.columns]
    if test_type == "全部":
        categorical_features.append("test_type")
    categorical_features = [field for field in categorical_features if frame[field].notna().any()]
    return frame, numeric_features, categorical_features


def make_pipeline(
    numeric_features: list[str],
    categorical_features: list[str],
    model_name: str,
):
    numeric_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        [
            (
                "imputer",
                SimpleImputer(strategy="most_frequent", keep_empty_features=True),
            ),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    preprocessor = ColumnTransformer(
        [("num", numeric_pipe, numeric_features), ("cat", categorical_pipe, categorical_features)],
        remainder="drop",
    )
    if model_name == "高斯过程回归":
        model = GaussianProcessRegressor(
            kernel=C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2))
            + WhiteKernel(1e-2, (1e-10, 1e1)),
            n_restarts_optimizer=3,
            normalize_y=True,
            random_state=42,
        )
    elif model_name == "随机森林":
        model = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
    elif model_name == "梯度提升树":
        model = GradientBoostingRegressor(
            n_estimators=200, learning_rate=0.03, max_depth=2, random_state=42
        )
    else:
        model = Ridge(alpha=10.0)
    return Pipeline([("preprocessor", preprocessor), ("model", model)])


def train_adhesion_model(
    frame: pd.DataFrame,
    numeric_features: list[str],
    categorical_features: list[str],
    model_name: str,
    target_unit: str,
    test_type: str,
) -> dict:
    if len(frame) < 6:
        raise AdhesionError("有效训练样本少于 6 条，建议先补充实验数据")
    features = numeric_features + categorical_features
    X = frame[features]
    y = frame[TARGET_FIELD].astype(float)
    pipeline = make_pipeline(numeric_features, categorical_features, model_name)
    # 小样本时避免某一折只有 1 条测试样本导致 R² 无定义
    folds = min(5, max(2, len(frame) // 2))
    cv = KFold(n_splits=folds, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipeline, X, y, cv=cv, scoring="r2")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    metrics = {
        "r2": float(r2_score(y_test, y_pred)) if len(y_test) >= 2 else float("nan"),
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "cv_r2": float(np.nanmean(cv_scores)),
    }
    return {
        "pipeline": pipeline,
        "features": features,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "model_name": model_name,
        "target_unit": target_unit,
        "test_type": test_type,
        "metrics": metrics,
        "y_test": y_test.to_numpy(),
        "y_pred": y_pred,
        "frame": frame,
    }


def predict_bundle(bundle: dict, row: pd.DataFrame) -> tuple[float, float | None]:
    preprocessor = bundle["pipeline"].named_steps["preprocessor"]
    model = bundle["pipeline"].named_steps["model"]
    transformed = preprocessor.transform(row[bundle["features"]])
    if isinstance(model, GaussianProcessRegressor):
        prediction, std = model.predict(transformed, return_std=True)
        return float(prediction[0]), float(std[0])
    return float(model.predict(transformed)[0]), None


def demo_experiments() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    rows = 30
    frame = pd.DataFrame(
        {
            "sample_id": [f"DEMO{index + 1:03d}" for index in range(rows)],
            "dopamine_content": rng.uniform(1.0, 10.0, rows).round(2),
            "polymer_concentration": rng.uniform(10.0, 40.0, rows).round(1),
            "ph": rng.uniform(3.0, 9.0, rows).round(1),
            "curing_temp_c": rng.uniform(25.0, 80.0, rows).round(0),
            "curing_time_min": rng.uniform(10.0, 120.0, rows).round(0),
            "test_temp_c": rng.choice([25.0, 37.0], rows),
            "humidity": rng.uniform(30.0, 70.0, rows).round(0),
            "crosslinker": rng.choice(["EDC/NHS", "genipin", "none"], rows),
            "substrate": rng.choice(["glass", "steel", "PTFE"], rows),
            "test_type": "lap_shear",
        }
    )
    frame[TARGET_FIELD] = (
        18
        + 4.2 * frame["dopamine_content"]
        + 0.7 * frame["polymer_concentration"]
        - 1.1 * (frame["ph"] - 6).abs()
        + 0.08 * frame["curing_time_min"]
        + rng.normal(0, 2.2, rows)
    ).round(2)
    frame["target_unit"] = "kPa"
    frame["source_type"] = "experiment"
    return frame
