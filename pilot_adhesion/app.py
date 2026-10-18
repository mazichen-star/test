"""PAA-DA 黏附性能：文献数据 + 小样本预测（独立试用版）。

这个文件刻意不依赖主应用的 Supabase / core 模块，方便单独试用和回退：
    streamlit run pilot_adhesion/app.py

数据默认保存在 pilot_adhesion/data/ 下，不会污染主应用。
"""

from __future__ import annotations

import io
import json
import os
import re
import sqlite3
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import streamlit as st
from matplotlib import font_manager
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = DATA_DIR / "pilot.db"
EXPERIMENT_PATH = DATA_DIR / "experiments.csv"

_FONT_PATH = BASE_DIR.parent / "fonts" / "NotoSansSC-Regular.ttf"
if _FONT_PATH.exists():
    font_manager.fontManager.addfont(str(_FONT_PATH))
    _CN_FONT = font_manager.FontProperties(fname=str(_FONT_PATH)).get_name()
else:
    _CN_FONT = "Microsoft YaHei"
matplotlib.rcParams["font.sans-serif"] = [
    _CN_FONT,
    "Noto Sans CJK SC",
    "Microsoft YaHei",
    "DejaVu Sans",
]
matplotlib.rcParams["axes.unicode_minus"] = False

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
    "dopamine_content": ["多巴胺", "dopamine", "da含量", "da content", "da"],
    "polymer_concentration": ["浓度", "固含", "concentration", "solid content", "聚合物浓度"],
    "ph": ["ph", "酸碱度"],
    "curing_temp_c": ["固化温度", "curing temp", "curing temperature", "加热温度"],
    "curing_time_min": ["固化时间", "curing time", "加热时间"],
    "test_temp_c": ["测试温度", "test temp", "test temperature"],
    "humidity": ["湿度", "humidity", "rh"],
    "crosslinker": ["交联剂", "crosslink", "crosslinker", "edc", "nhs", "京尼平"],
    "substrate": ["基材", "substrate", "基底", "材料", "adherend"],
    "test_type": ["测试类型", "test type", "测试方法", "method", "剪切", "剥离"],
}

TARGET_ALIASES = [
    "黏附", "粘附", "adhesion", "lap", "shear", "peel", "tack",
    "强度", "kpa", "mpa", "n/cm",
]

TARGET_UNITS = ["kPa", "N/cm"]


def get_secret(key: str, default: str = "") -> str:
    value = os.environ.get(key)
    if value:
        return value
    # 试用版允许把 secrets 放在 pilot_adhesion/.streamlit/secrets.toml，
    # 这样从仓库根目录启动也能读到。
    local_path = BASE_DIR / ".streamlit" / "secrets.toml"
    if local_path.exists():
        try:
            import tomllib

            with local_path.open("rb") as handle:
                local = tomllib.load(handle)
            if key in local and str(local[key]).strip() != "":
                return str(local[key])
        except Exception:
            pass
    try:
        value = st.secrets.get(key, default)
    except Exception:
        value = default
    return str(value) if value is not None else default


def set_flash(message: str, kind: str = "success") -> None:
    st.session_state["_pilot_flash"] = (kind, message)


def show_flash() -> None:
    item = st.session_state.pop("_pilot_flash", None)
    if item:
        kind, message = item
        getattr(st, kind, st.info)(message)


def ai_settings() -> tuple[str, str, str]:
    deepseek = get_secret("DEEPSEEK_API_KEY").strip()
    if deepseek:
        return (
            deepseek,
            get_secret("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1/chat/completions"),
            get_secret("DEEPSEEK_MODEL", "deepseek-chat"),
        )
    ark = get_secret("ARK_API_KEY").strip()
    if ark:
        return (
            ark,
            get_secret(
                "ARK_BASE_URL",
                "https://ark.cn-beijing.volces.com/api/v3/chat/completions",
            ),
            get_secret("ARK_MODEL"),
        )
    return "", "", ""


def init_db() -> None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS literature_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_file TEXT NOT NULL DEFAULT '',
                title TEXT NOT NULL DEFAULT '',
                doi TEXT NOT NULL DEFAULT '',
                page INTEGER,
                quote TEXT NOT NULL DEFAULT '',
                material_system TEXT NOT NULL DEFAULT 'PAA-DA',
                dopamine_content REAL,
                polymer_concentration REAL,
                ph REAL,
                curing_temp_c REAL,
                curing_time_min REAL,
                crosslinker TEXT NOT NULL DEFAULT '',
                substrate TEXT NOT NULL DEFAULT '',
                test_type TEXT NOT NULL DEFAULT 'unknown',
                adhesion_value REAL,
                adhesion_unit TEXT NOT NULL DEFAULT '',
                adhesion_target REAL,
                target_unit TEXT NOT NULL DEFAULT '',
                test_temp_c REAL,
                humidity REAL,
                confidence REAL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            """
        )
        # 兼容已经创建过的试用版数据库
        try:
            conn.execute(
                "ALTER TABLE literature_records "
                "ADD COLUMN target_unit TEXT NOT NULL DEFAULT ''"
            )
        except sqlite3.OperationalError:
            pass


init_db()


def load_literature() -> pd.DataFrame:
    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql_query(
            "SELECT * FROM literature_records ORDER BY id DESC", conn
        )


def delete_literature(record_id: int) -> None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("DELETE FROM literature_records WHERE id = ?", (record_id,))


def clear_literature() -> None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("DELETE FROM literature_records")


def save_records(records: list[dict]) -> int:
    if not records:
        return 0
    columns = [
        "source_file",
        "title",
        "doi",
        "page",
        "quote",
        "material_system",
        "dopamine_content",
        "polymer_concentration",
        "ph",
        "curing_temp_c",
        "curing_time_min",
        "crosslinker",
        "substrate",
        "test_type",
        "adhesion_value",
        "adhesion_unit",
        "adhesion_target",
        "target_unit",
        "test_temp_c",
        "humidity",
        "confidence",
    ]
    inserted = 0
    with sqlite3.connect(DB_PATH) as conn:
        for record in records:
            values = [record.get(column) for column in columns]
            conn.execute(
                "INSERT INTO literature_records("
                + ",".join(columns)
                + ") VALUES ("
                + ",".join(["?"] * len(columns))
                + ")",
                values,
            )
            inserted += 1
    return inserted


def to_kpa(value: float, unit: str) -> float | None:
    if value is None or not unit:
        return None
    text = str(unit).strip().lower().replace(" ", "")
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if text in ("kpa",):
        return number
    if text in ("mpa", "n/mm2", "n/mm²"):
        return number * (1_000.0 if "mpa" in text else 1_000.0)
    if text in ("gpa",):
        return number * 1_000_000.0
    if text in ("pa",):
        return number / 1_000.0
    if text in ("n/cm2", "n/cm²"):
        return number * 10.0
    return None


def to_n_cm(value: float, unit: str) -> float | None:
    if value is None or not unit:
        return None
    text = str(unit).strip().lower().replace(" ", "")
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if text in ("n/cm",):
        return number
    if text in ("n/mm",):
        return number * 10.0
    if text in ("n/25mm",):
        return number / 2.5
    if text in ("gf/25mm",):
        return number * 0.00980665 / 2.5
    if text in ("gf/cm",):
        return number * 0.00980665
    return None


def to_target(value, unit: str, target_unit: str) -> float | None:
    if target_unit == "kPa":
        return to_kpa(value, unit)
    if target_unit == "N/cm":
        return to_n_cm(value, unit)
    return None


def normalize_test_type(value: str) -> str:
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
    def number(key):
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

    return {
        "source_file": str(raw.get("source_file") or ""),
        "title": str(raw.get("title") or ""),
        "doi": str(raw.get("doi") or ""),
        "page": int(raw["page"]) if str(raw.get("page", "")).isdigit() else None,
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


def parse_pdf(pdf_bytes: bytes) -> list[dict]:
    try:
        import pdfplumber
    except ImportError as exc:
        raise RuntimeError("请先安装 pdfplumber：pip install pdfplumber") from exc
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


def call_llm(prompt: str) -> str:
    key, base_url, model = ai_settings()
    if not key:
        raise RuntimeError("未配置 DEEPSEEK_API_KEY 或 ARK_API_KEY")
    if not model:
        raise RuntimeError("未配置模型名（DEEPSEEK_MODEL / ARK_MODEL）")
    response = requests.post(
        base_url,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
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
        timeout=120,
    )
    if response.status_code != 200:
        raise RuntimeError(f"LLM HTTP {response.status_code}: {response.text[:200]}")
    return response.json()["choices"][0]["message"]["content"].strip()


def extract_records_with_llm(
    pages: list[dict], title: str, doi: str, target_unit: str
) -> list[dict]:
    chunks = chunk_pages(pages)
    records: list[dict] = []
    progress = st.progress(0.0, text="准备抽取...")
    for index, chunk in enumerate(chunks, start=1):
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
        progress.progress(index / max(len(chunks), 1), text=f"抽取第 {index}/{len(chunks)} 段...")
        try:
            content = call_llm(prompt)
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
        except Exception as exc:
            st.warning(f"第 {chunk['page']} 页有一段抽取失败：{exc}")
    progress.empty()
    return records


def load_experiment_table(content: bytes, filename: str) -> pd.DataFrame:
    suffix = Path(filename).suffix.lower()
    if suffix in (".csv", ".txt"):
        last_error = None
        for encoding in ("utf-8-sig", "utf-8", "gb18030", "gbk"):
            try:
                return pd.read_csv(io.BytesIO(content), encoding=encoding)
            except UnicodeDecodeError as exc:
                last_error = exc
        raise RuntimeError(f"CSV 编码读取失败：{last_error}")
    if suffix in (".xlsx", ".xls"):
        return pd.read_excel(io.BytesIO(content))
    raise RuntimeError("只支持 CSV / XLSX / XLS")


def find_column(columns: list[str], aliases: list[str]) -> str | None:
    lowered = {str(column).strip().lower(): str(column) for column in columns}
    for alias in aliases:
        key = alias.lower()
        if key in lowered:
            return lowered[key]
    for column in columns:
        text = str(column).lower()
        if any(alias.lower() in text for alias in aliases):
            return str(column)
    return None


def guess_target(columns: list[str]) -> str | None:
    return find_column(columns, TARGET_ALIASES)


def standardize_experiments(
    frame: pd.DataFrame,
    mapping: dict[str, str | None],
    target_column: str,
    target_unit: str,
    target_unit_in: str,
    target_name: str = "adhesion_target",
) -> pd.DataFrame:
    result = pd.DataFrame(index=frame.index)
    result["sample_id"] = [f"E{i + 1:03d}" for i in range(len(frame))]
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
    converted = values.apply(lambda value: to_target(value, target_unit_in, target_unit))
    result[target_name] = converted
    result["target_unit"] = target_unit
    result["source_type"] = "experiment"
    return result


def demo_experiments() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    rows = 30
    frame = pd.DataFrame(
        {
            "sample_id": [f"DEMO{i + 1:03d}" for i in range(rows)],
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
    frame["adhesion_target"] = (
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


def demo_literature(target_unit: str = "kPa") -> list[dict]:
    rows = [
        ("10.1000/demo1", "PAA-DA adhesive on glass", 3, "lap shear strength 42.5 kPa", 4.0, 25.0, 6.0, 60.0, 60.0, "EDC/NHS", "glass", 42.5, "kPa", "lap_shear"),
        ("10.1000/demo1", "PAA-DA adhesive on glass", 4, "lap shear strength 55.1 kPa", 7.0, 30.0, 7.0, 70.0, 60.0, "EDC/NHS", "glass", 55.1, "kPa", "lap_shear"),
        ("10.1000/demo2", "Dopamine-modified PAA on steel", 5, "adhesion strength 0.081 MPa", 5.0, 20.0, 5.5, 50.0, 45.0, "genipin", "steel", 0.081, "MPa", "lap_shear"),
        ("10.1000/demo2", "Dopamine-modified PAA on steel", 6, "peel strength 1.8 N/cm", 5.0, 20.0, 5.5, 50.0, 45.0, "genipin", "steel", 1.8, "N/cm", "peel"),
        ("10.1000/demo3", "PAA-DA tissue adhesive", 2, "lap shear strength 28.7 kPa", 2.0, 15.0, 6.5, 37.0, 30.0, "none", "tissue", 28.7, "kPa", "lap_shear"),
        ("10.1000/demo3", "PAA-DA tissue adhesive", 3, "lap shear strength 33.9 kPa", 3.0, 18.0, 6.5, 37.0, 30.0, "none", "tissue", 33.9, "kPa", "lap_shear"),
    ]
    records = []
    for doi, title, page, quote, da, conc, ph, temp, time, cross, substrate, value, unit, test_type in rows:
        records.append(
            normalize_record(
                {
                    "source_file": "demo",
                    "title": title,
                    "doi": doi,
                    "page": page,
                    "quote": quote,
                    "material_system": "PAA-DA",
                    "dopamine_content": da,
                    "polymer_concentration": conc,
                    "ph": ph,
                    "curing_temp_c": temp,
                    "curing_time_min": time,
                    "crosslinker": cross,
                    "substrate": substrate,
                    "test_type": test_type,
                    "adhesion_value": value,
                    "adhesion_unit": unit,
                    "test_temp_c": 25.0,
                    "humidity": 50.0,
                    "confidence": 0.99,
                },
                target_unit,
            )
        )
    return records


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
        lit["adhesion_target"] = lit.apply(
            lambda row: (
                row["adhesion_target"]
                if pd.notna(row.get("adhesion_target"))
                else to_target(row.get("adhesion_value"), row.get("adhesion_unit"), target_unit)
            ),
            axis=1,
        )
        lit["source_type"] = "literature"
        frames.append(lit)
    if not frames:
        return pd.DataFrame(), [], []

    frame = pd.concat(frames, ignore_index=True)
    frame = frame[frame["adhesion_target"].notna()].copy()
    if test_type and test_type != "全部":
        frame = frame[frame["test_type"].fillna("unknown") == test_type].copy()
    if not len(frame):
        return frame, [], []

    numeric_features = [field for field in NUMERIC_FIELDS if field in frame.columns]
    categorical_features = [
        field for field in ["crosslinker", "substrate"]
        if field in frame.columns
    ]
    if test_type == "全部":
        categorical_features.append("test_type")
    numeric_features = [
        field for field in numeric_features if frame[field].notna().any()
    ]
    categorical_features = [
        field for field in categorical_features if frame[field].notna().any()
    ]
    return frame, numeric_features, categorical_features


def make_pipeline(
    numeric_features: list[str],
    categorical_features: list[str],
    model_name: str,
):
    numeric_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    preprocessor = ColumnTransformer(
        [
            ("num", numeric_pipe, numeric_features),
            ("cat", categorical_pipe, categorical_features),
        ],
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
        model = RandomForestRegressor(
            n_estimators=300, random_state=42, n_jobs=-1
        )
    elif model_name == "梯度提升树":
        model = GradientBoostingRegressor(
            n_estimators=200, learning_rate=0.03, max_depth=2, random_state=42
        )
    else:
        model = Ridge(alpha=10.0)

    return Pipeline([("preprocessor", preprocessor), ("model", model)])


def plot_scatter(y_true, y_pred) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    ax.scatter(y_true, y_pred, alpha=0.75, color="#4C9AFF")
    low = float(min(np.min(y_true), np.min(y_pred)))
    high = float(max(np.max(y_true), np.max(y_pred)))
    ax.plot([low, high], [low, high], "r--", lw=1.3)
    ax.set_xlabel("真实值")
    ax.set_ylabel("预测值")
    ax.set_title("预测 vs 真实")
    st.pyplot(fig)
    plt.close(fig)


def predict_bundle(bundle: dict, row: pd.DataFrame) -> tuple[float, float | None]:
    preprocessor = bundle["pipeline"].named_steps["preprocessor"]
    model = bundle["pipeline"].named_steps["model"]
    transformed = preprocessor.transform(row[bundle["features"]])
    if isinstance(model, GaussianProcessRegressor):
        prediction, std = model.predict(transformed, return_std=True)
        return float(prediction[0]), float(std[0])
    return float(model.predict(transformed)[0]), None


def render_literature_tab(target_unit: str) -> None:
    st.subheader("1. 文献数据")
    st.caption(
        "上传 3 篇 PDF，用大模型抽取 PAA-DA 的黏附数据；也可以直接粘贴 JSON。"
        "所有结果都需要人工检查后再用于训练。"
    )

    uploaded = st.file_uploader(
        "上传文献 PDF（可多选）",
        type=["pdf"],
        accept_multiple_files=True,
        key="pilot_pdf_upload",
    )
    pages_by_file: dict[str, list[dict]] = {}
    if uploaded:
        for pdf_file in uploaded:
            try:
                pages = parse_pdf(pdf_file.getvalue())
                pages_by_file[pdf_file.name] = pages
                st.caption(f"{pdf_file.name}: {len(pages)} 页")
                with st.expander(f"预览：{pdf_file.name}", expanded=False):
                    st.text_area(
                        "文本预览",
                        value="\n\n".join(
                            f"[第 {page['page']} 页]\n{page['text'][:1200]}"
                            for page in pages[:3]
                        ),
                        height=220,
                        key=f"preview_{pdf_file.name}",
                    )
            except Exception as exc:
                st.error(f"{pdf_file.name} 解析失败：{exc}")

    key, _, model = ai_settings()
    if not key:
        st.info(
            "未配置 AI Key。你可以：1）把 DEEPSEEK_API_KEY / ARK_API_KEY 写入 "
            "pilot_adhesion/.streamlit/secrets.toml；2）在下面手动粘贴 JSON。"
        )
    else:
        st.caption(f"AI 已配置：{model or '未填写模型名'}")
        if st.button(
            "用 AI 抽取全部 PDF",
            type="primary",
            disabled=not pages_by_file,
            key="pilot_ai_extract",
        ):
            if not model:
                st.error("请先配置 DEEPSEEK_MODEL 或 ARK_MODEL")
            else:
                all_records: list[dict] = []
                for filename, pages in pages_by_file.items():
                    st.write(f"正在处理：{filename}")
                    try:
                        all_records.extend(
                            extract_records_with_llm(
                                pages,
                                title=filename,
                                doi="",
                                target_unit=target_unit,
                            )
                        )
                    except Exception as exc:
                        st.error(f"{filename} 抽取失败：{exc}")
                if all_records:
                    count = save_records(all_records)
                    set_flash(f"已抽取并保存 {count} 条记录，请检查下方表格")
                    st.rerun()
                else:
                    st.warning("没有抽取到有效记录")

    with st.expander("手动粘贴 JSON", expanded=False):
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
            "粘贴记录数组", height=220, key="pilot_manual_json"
        )
        if st.button("导入 JSON", key="pilot_import_json"):
            try:
                parsed = json.loads(manual_json)
                if isinstance(parsed, dict):
                    parsed = [parsed]
                records = [normalize_record(item, target_unit) for item in parsed]
                count = save_records(records)
                set_flash(f"导入 {count} 条")
                st.rerun()
            except Exception as exc:
                st.error(f"JSON 解析失败：{exc}")

    if st.button("载入 6 条演示文献", key="pilot_demo_literature"):
        count = save_records(demo_literature(target_unit))
        set_flash(f"已载入 {count} 条演示文献")
        st.rerun()

    literature = load_literature()
    st.subheader("已保存的文献记录")
    if literature.empty:
        st.info("还没有文献记录")
    else:
        display_columns = [
            "id", "title", "doi", "page", "test_type",
            "dopamine_content", "polymer_concentration", "ph",
            "curing_temp_c", "curing_time_min",
            "substrate", "adhesion_value", "adhesion_unit",
            "adhesion_target", "target_unit", "confidence", "quote",
        ]
        display_columns = [column for column in display_columns if column in literature.columns]
        st.dataframe(literature[display_columns], use_container_width=True, height=320)
        st.download_button(
            "下载文献数据 CSV",
            data=literature.to_csv(index=False).encode("utf-8-sig"),
            file_name="literature_records.csv",
            mime="text/csv",
        )
        delete_id = st.selectbox(
            "删除一条记录",
            ["(不删除)"] + literature["id"].astype(int).tolist(),
            key="pilot_delete_literature",
        )
        if delete_id != "(不删除)" and st.button(
            "确认删除", key="pilot_confirm_delete_literature"
        ):
            delete_literature(int(delete_id))
            st.rerun()
        confirm_clear = st.checkbox(
            "我确认清空文献数据", key="pilot_clear_literature_confirm"
        )
        if st.button(
            "清空全部文献数据",
            disabled=not confirm_clear,
            key="pilot_clear_literature",
        ):
            clear_literature()
            st.rerun()


def render_experiment_tab(target_unit: str) -> None:
    st.subheader("2. 实验数据")
    st.caption("上传 30 条左右的实验数据，或先生成一份演示数据跑通流程。")

    uploaded = st.file_uploader(
        "上传实验数据 CSV / Excel",
        type=["csv", "xlsx", "xls"],
        key="pilot_experiment_upload",
    )
    frame = None
    filename = ""
    if uploaded is not None:
        try:
            frame = load_experiment_table(uploaded.getvalue(), uploaded.name)
            filename = uploaded.name
            st.success(f"读取 {len(frame)} 行 × {len(frame.columns)} 列")
            st.dataframe(frame, use_container_width=True, height=260)
        except Exception as exc:
            st.error(f"读取失败：{exc}")

    if st.button("生成 30 条演示实验数据", key="pilot_demo_experiments"):
        demo = demo_experiments()
        demo.to_csv(EXPERIMENT_PATH, index=False, encoding="utf-8-sig")
        set_flash("已生成 30 条演示实验数据")
        st.rerun()

    if frame is not None:
        columns = [str(column) for column in frame.columns]
        target_default = guess_target(columns)
        target_index = columns.index(target_default) if target_default in columns else 0
        target_column = st.selectbox(
            "目标列（黏附强度）",
            columns,
            index=target_index,
            key="pilot_target_column",
        )
        target_unit_in = st.selectbox(
            "目标列原始单位",
            ["kPa", "MPa", "N/cm", "N/25mm", "gf/25mm", "N/mm"],
            key="pilot_target_unit_in",
        )
        st.caption("标准字段映射：留空表示该字段不使用")
        mapping: dict[str, str | None] = {}
        left, right = st.columns(2)
        for index, field in enumerate(ALL_FIELDS):
            with (left if index % 2 == 0 else right):
                default = find_column(columns, FIELD_ALIASES.get(field, []))
                options = ["(不使用)"] + columns
                value = st.selectbox(
                    f"{FIELD_LABELS.get(field, field)}",
                    options,
                    index=options.index(default) if default in options else 0,
                    key=f"pilot_map_{field}",
                )
                mapping[field] = None if value == "(不使用)" else value

        if st.button("保存实验数据", type="primary", key="pilot_save_experiments"):
            try:
                standardized = standardize_experiments(
                    frame,
                    mapping,
                    target_column,
                    target_unit,
                    target_unit_in,
                )
                standardized = standardized[standardized["adhesion_target"].notna()]
                if not len(standardized):
                    st.error("目标列没有可转换的数值，请检查目标列或单位")
                else:
                    standardized.to_csv(EXPERIMENT_PATH, index=False, encoding="utf-8-sig")
                    set_flash(f"已保存 {len(standardized)} 条实验数据")
                    st.rerun()
            except Exception as exc:
                st.error(f"保存失败：{exc}")

    if EXPERIMENT_PATH.exists():
        saved = pd.read_csv(EXPERIMENT_PATH)
        st.subheader("已保存的实验数据")
        st.dataframe(saved, use_container_width=True, height=300)
        st.caption(f"目标均值：{saved['adhesion_target'].mean():.3f} {target_unit}")
    else:
        st.info("还没有实验数据")


def render_modeling_tab(target_unit: str) -> None:
    st.subheader("3. 训练与预测")
    if not EXPERIMENT_PATH.exists():
        st.info("请先在“实验数据”页保存数据")
        return
    experiments = pd.read_csv(EXPERIMENT_PATH)
    literature = load_literature()

    if "target_unit" in experiments.columns and experiments["target_unit"].notna().any():
        effective_unit = str(experiments["target_unit"].dropna().iloc[0])
    else:
        effective_unit = target_unit
    if effective_unit != target_unit:
        st.info(
            f"已保存实验数据的目标单位是 {effective_unit}，"
            f"本次建模使用 {effective_unit}；如果要改单位，请回实验页重新保存。"
        )

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
        "测试类型（不同测试方法不要混在一起）",
        test_types,
        index=test_types.index(default_type),
        key="pilot_test_type",
    )
    if test_type == "全部":
        st.warning("混用 lap shear / peel / tack 会让模型学到错误的量纲关系，建议只选一种。")

    use_literature = st.checkbox(
        "把文献数据加入训练集",
        value=not literature.empty,
        disabled=literature.empty,
        key="pilot_use_literature",
    )
    frame, numeric_features, categorical_features = build_training_frame(
        experiments, literature, effective_unit, test_type, use_literature
    )
    if not len(frame):
        st.warning("没有可训练的数据（检查目标值、测试类型和文献记录）")
        return
    st.caption(
        f"训练样本：{len(frame)} 条"
        f"（实验 {(frame['source_type'] == 'experiment').sum()}，"
        f"文献 {(frame['source_type'] == 'literature').sum()}）"
    )
    display_columns = [
        "sample_id",
        "source_type",
        "test_type",
        "adhesion_target",
    ] + numeric_features + categorical_features
    display_columns = list(dict.fromkeys(display_columns))
    st.dataframe(
        frame[[column for column in display_columns if column in frame.columns]],
        use_container_width=True,
        height=220,
    )

    model_name = st.selectbox(
        "模型",
        ["高斯过程回归", "随机森林", "梯度提升树", "Ridge 回归"],
        key="pilot_model",
    )
    st.caption("30 条左右数据优先试 GPR / Ridge；树模型容易过拟合，结果只作参考。")
    if st.button("开始训练", type="primary", key="pilot_train"):
        try:
            features = numeric_features + categorical_features
            X = frame[features]
            y = frame["adhesion_target"].astype(float)
            pipeline = make_pipeline(numeric_features, categorical_features, model_name)
            n_splits = min(5, len(frame))
            cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
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
            st.session_state["pilot_bundle"] = {
                "pipeline": pipeline,
                "features": features,
                "numeric_features": numeric_features,
                "categorical_features": categorical_features,
                "model_name": model_name,
                "target_unit": effective_unit,
                "test_type": test_type,
                "metrics": metrics,
                "y_test": y_test.to_numpy(),
                "y_pred": y_pred,
                "frame": frame,
            }
            st.success("训练完成")
        except Exception as exc:
            st.error(f"训练失败：{exc}")

    bundle = st.session_state.get("pilot_bundle")
    if not bundle:
        st.info("点击“开始训练”后查看结果")
        return
    metrics = bundle["metrics"]
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("测试集 R²", f"{metrics['r2']:.3f}" if np.isfinite(metrics["r2"]) else "—")
    col2.metric("MAE", f"{metrics['mae']:.3f}")
    col3.metric("RMSE", f"{metrics['rmse']:.3f}")
    col4.metric("CV R²", f"{metrics['cv_r2']:.3f}")
    plot_scatter(bundle["y_test"], bundle["y_pred"])

    # 特征重要性 / 置换重要性
    model = bundle["pipeline"].named_steps["model"]
    preprocessor = bundle["pipeline"].named_steps["preprocessor"]
    try:
        if hasattr(model, "feature_importances_"):
            feature_names = [
                name.split("__", 1)[-1]
                for name in preprocessor.get_feature_names_out()
            ]
            importance = pd.Series(model.feature_importances_, index=feature_names)
            importance = importance.groupby(level=0).sum().sort_values(ascending=True)
            fig, ax = plt.subplots(figsize=(7, max(2.2, 0.32 * len(importance))))
            importance.plot.barh(ax=ax, color="#36B37E")
            ax.set_title("特征重要性")
            st.pyplot(fig)
            plt.close(fig)
    except Exception:
        pass

    st.subheader("预测新配方")
    st.caption("留空/默认值使用训练集中位数；类别使用下拉选择。")
    input_data = {}
    columns = st.columns(min(len(bundle["features"]), 4))
    frame = bundle["frame"]
    for index, field in enumerate(bundle["features"]):
        with columns[index % 4]:
            if field in bundle["numeric_features"]:
                default = float(frame[field].median()) if frame[field].notna().any() else 0.0
                input_data[field] = st.number_input(
                    FIELD_LABELS.get(field, field),
                    value=default,
                    key=f"pilot_predict_{field}",
                )
            else:
                options = sorted(
                    {str(value) for value in frame[field].dropna().tolist()}
                ) or ["unknown"]
                input_data[field] = st.selectbox(
                    FIELD_LABELS.get(field, field),
                    options,
                    key=f"pilot_predict_{field}",
                )
    if st.button("预测", type="secondary", key="pilot_predict_btn"):
        try:
            row = pd.DataFrame([input_data])
            prediction, std = predict_bundle(bundle, row)
            if std is not None:
                st.success(
                    f"预测黏附强度：**{prediction:.3f} ± {1.96 * std:.3f} "
                    f"{bundle['target_unit']}**（95% 预测区间）"
                )
            else:
                st.success(
                    f"预测黏附强度：**{prediction:.3f} {bundle['target_unit']}**"
                )
            st.session_state["pilot_last_prediction"] = {
                "prediction": prediction,
                "std": std,
                "input": input_data,
                "unit": bundle["target_unit"],
            }
        except Exception as exc:
            st.error(f"预测失败：{exc}")

    last = st.session_state.get("pilot_last_prediction")
    if last:
        st.caption(
            f"最近一次预测：{last['prediction']:.3f} {last['unit']}"
            + (
                f" ± {1.96 * last['std']:.3f}"
                if last.get("std") is not None
                else ""
            )
        )

    # 相似文献
    literature = load_literature()
    if len(literature):
        available = [field for field in bundle["numeric_features"] if field in literature.columns]
        if available:
            base = literature.dropna(subset=available).copy()
            if len(base):
                med = frame[available].median()
                scale = frame[available].std().replace(0, 1).fillna(1)
                distance = (
                    ((base[available].fillna(med) - med) / scale)
                    .pow(2)
                    .sum(axis=1)
                    .pow(0.5)
                )
                base["_distance"] = distance
                similar = base.nsmallest(5, "_distance")
                st.subheader("最相似的文献记录")
                show = [
                    column
                    for column in ["title", "doi", "page", "test_type", "adhesion_target", "quote"]
                    if column in similar.columns
                ]
                st.dataframe(similar[show], use_container_width=True, height=220)


def render_help_tab() -> None:
    st.subheader("4. 说明与回退")
    st.markdown(
        """
### 这是独立试用版

- 目录：`pilot_adhesion/`
- 数据：`pilot_adhesion/data/pilot.db`、`experiments.csv`
- 不会修改主应用 `app.py`，也不会写入 Supabase。
- 回退：在仓库执行 `git switch main`，试用版目录会保留但主应用回到原版本。
- 删除试用数据：侧边栏“清空试用数据”，或手动删除 `pilot_adhesion/data/`。

### 文献抽取字段

目标字段包括：多巴胺含量、聚合物浓度、pH、固化温度/时间、交联剂、基材、测试类型、
黏附强度、测试温度、湿度。AI 只负责抽取，**必须人工检查**后再用于训练。

### 当前限制

- 3 篇文献 + 30 条实验数据属于极小样本，模型结果只能作为趋势参考，不能替代实验。
- 不同测试方法（lap shear / peel / tack）量纲不同，请只用同一种测试类型训练。
- 文献中的 N/cm、gf/25mm 等单位需要正确选择目标单位；无法换算的记录会被忽略。
- 没有做 DOI 查重、单位全自动校验和图表数字化，后续可以补。
        """
    )
    key, base_url, model = ai_settings()
    st.code(
        f"AI 配置：{'已配置' if key else '未配置'}\n"
        f"Base URL：{base_url or '-'}\n"
        f"Model：{model or '-'}\n"
        f"数据库：{DB_PATH}\n"
        f"实验数据：{EXPERIMENT_PATH}",
        language="text",
    )


def main() -> None:
    st.set_page_config(
        page_title="PAA-DA 黏附预测（试用版）",
        page_icon="🧪",
        layout="wide",
    )
    st.title("PAA-DA 黏附性能：文献 + 实验小样本预测")
    st.caption(
        "独立试用版：文献抽取 → 人工检查 → 与 30 条实验数据联合训练 → 预测。"
        "不会影响主应用。"
    )

    with st.sidebar:
        st.header("试用版控制台")
        key, _, model = ai_settings()
        st.write("AI 抽取：", "已配置" if key else "未配置")
        st.caption(f"数据库：`{DB_PATH}`")
        st.caption(f"实验数据：`{EXPERIMENT_PATH}`")
        target_unit = st.selectbox(
            "目标单位",
            TARGET_UNITS,
            key="pilot_target_unit",
            help="lap shear 通常用 kPa；peel 通常用 N/cm。",
        )
        st.divider()
        confirm_reset = st.checkbox(
            "我确认清空试用数据", key="pilot_reset_confirm"
        )
        if st.button(
            "清空试用数据",
            disabled=not confirm_reset,
            key="pilot_reset",
        ):
            # 不要直接删除正在使用的 SQLite 文件（Windows 会锁文件），
            # 清表即可；实验 CSV 单独删除。
            try:
                clear_literature()
            except Exception as exc:
                st.error(f"清空文献数据失败：{exc}")
            try:
                if EXPERIMENT_PATH.exists():
                    EXPERIMENT_PATH.unlink()
            except OSError as exc:
                st.error(f"删除实验文件失败：{exc}")
            st.session_state.pop("pilot_bundle", None)
            st.session_state.pop("pilot_last_prediction", None)
            set_flash("试用数据已清空")
            st.rerun()

    show_flash()
    tab_literature, tab_experiment, tab_model, tab_help = st.tabs(
        ["1 文献数据", "2 实验数据", "3 训练与预测", "4 说明与回退"]
    )
    with tab_literature:
        render_literature_tab(target_unit)
    with tab_experiment:
        render_experiment_tab(target_unit)
    with tab_model:
        render_modeling_tab(target_unit)
    with tab_help:
        render_help_tab()


main()
