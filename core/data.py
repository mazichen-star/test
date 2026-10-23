"""数据读取、表格清洗与建模前预处理。"""

from __future__ import annotations

import hashlib
import io
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


class DataError(RuntimeError):
    """数据读取或预处理错误。"""


CSV_ENCODINGS = ("utf-8-sig", "utf-8", "gb18030", "gbk")
MISSING_CATEGORY = "缺失"

HEADER_KEYWORDS = [
    "长度", "宽度", "厚度", "直径", "强度", "载荷", "应力", "应变", "模量",
    "硬度", "延伸", "伸长", "密度", "温度", "时间", "成分", "编号", "试样",
    "mm", "mpa", "gpa", "kg", "n)", "uts", "yield", "tensile", "strength",
    "modulus", "elongation", "hardness", "density", "temperature", "time",
    "sample", "alloy", "composition",
]
UNIT_TOKENS = [
    "mm", "cm", "μm", "um", "mpa", "gpa", "kg", "n)", "°c", "℃",
    "s/m", "w/(m", "w/mk", "%",
]
FOOTER_KEYWORDS = ("签名", "审核", "日期", "备注", "实验人", "记录人", "批准", "签字")
UNIT_ONLY = {
    "%",
    "°c",
    "℃",
    "mpa",
    "gpa",
    "mm",
    "cm",
    "um",
    "μm",
    "n",
    "kg",
    "s/m",
    "w/(m·k)",
    "w/(m*k)",
    "w/mk",
}


def _is_unit_only(value: Any) -> bool:
    text = str(value).strip().lower().replace(" ", "")
    if not text:
        return False
    if text in UNIT_ONLY:
        return True
    if len(text) <= 12 and any(token in text for token in UNIT_TOKENS):
        return bool(re.fullmatch(r"[%°℃a-zμ/()·*0-9.\-]+", text))
    return False


def _is_number(value: Any) -> bool:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return False
    try:
        float(str(value).replace(",", "").strip())
        return True
    except (TypeError, ValueError):
        return False


def _read_csv(content: bytes) -> pd.DataFrame:
    last_error: Exception | None = None
    for encoding in CSV_ENCODINGS:
        try:
            frame = pd.read_csv(io.BytesIO(content), encoding=encoding)
            # 单列但包含常见分隔符时，尝试自动嗅探
            if frame.shape[1] == 1 and any(
                sep in str(frame.columns[0]) for sep in (";", "\t", "|")
            ):
                raise pd.errors.ParserError("single column")
            return frame
        except UnicodeDecodeError as exc:
            last_error = exc
        except pd.errors.ParserError as exc:
            last_error = exc

    try:
        return pd.read_csv(
            io.BytesIO(content), sep=None, engine="python", encoding="utf-8-sig"
        )
    except Exception as exc:
        raise DataError(f"CSV 读取失败：{exc}") from exc


def _dedupe_columns(columns: list[Any]) -> list[str]:
    result: list[str] = []
    seen: dict[str, int] = {}
    for index, value in enumerate(columns):
        name = str(value).strip() if value is not None else ""
        if not name or name.lower() == "nan":
            name = f"col_{index}"
        if name in seen:
            seen[name] += 1
            name = f"{name}_{seen[name]}"
        else:
            seen[name] = 1
        result.append(name)
    return result


def _header_values(row: pd.Series) -> list[str]:
    return _dedupe_columns(row.tolist())


def smart_clean_excel(raw: pd.DataFrame) -> pd.DataFrame:
    """从实验报告式表格中识别表头、跳过单位行、去掉空行空列。

    这个函数同时兼容“第一行就是表头”的普通 Excel，以及前面带标题、
    参数行、单位行的实验报告。
    """

    frame = raw.copy()
    frame = frame.dropna(axis=0, how="all").dropna(axis=1, how="all")
    if frame.empty:
        return frame.reset_index(drop=True)
    frame = frame.reset_index(drop=True)

    best_index: int | None = None
    best_score = float("-inf")
    for index in range(min(len(frame), 30)):
        row = frame.iloc[index]
        non_empty = [
            str(value).strip()
            for value in row
            if pd.notna(value) and str(value).strip() != ""
        ]
        if len(non_empty) < 2:
            continue

        joined = " ".join(non_empty).lower()
        hits = sum(1 for keyword in HEADER_KEYWORDS if keyword.lower() in joined)
        unit_cells = sum(1 for value in non_empty if _is_unit_only(value))

        numeric_cells = 0
        total_cells = 0
        if index + 1 < len(frame):
            for _, next_row in frame.iloc[index + 1 : index + 4].iterrows():
                for value in next_row:
                    if pd.isna(value) or str(value).strip() == "":
                        continue
                    total_cells += 1
                    if _is_number(value):
                        numeric_cells += 1
        numeric_ratio = numeric_cells / total_cells if total_cells else 0.0

        score = hits * 4 + len(non_empty) * 0.2 + numeric_ratio * 3
        if len(non_empty) >= 3 and numeric_ratio >= 0.5:
            score += 2
        if hits == 0 and numeric_ratio < 0.3:
            score -= 4
        # 单位行（%, MPa, °C ...）通常紧邻表头，必须明显降权
        score -= unit_cells * 6

        if score > best_score:
            best_score = score
            best_index = index

    if best_index is None:
        best_index = 0

    first_row = frame.iloc[best_index]
    first_values = [
        value for value in first_row if pd.notna(value) and str(value).strip() != ""
    ]
    headers = _header_values(first_row)

    # 如果第一行全是数字，它更可能是数据而不是表头
    if first_values and all(_is_number(value) for value in first_values):
        headers = [f"col_{index}" for index in range(frame.shape[1])]
        data = frame.copy()
    else:
        data = frame.iloc[best_index + 1 :].copy()

    data.columns = headers

    # 跳过单位行
    if len(data):
        first = data.iloc[0]
        unit_cells = [
            str(value).strip()
            for value in first
            if pd.notna(value) and str(value).strip() != ""
        ]
        if unit_cells and sum(
            any(token in value.lower() for token in UNIT_TOKENS)
            for value in unit_cells
        ) >= max(1, len(unit_cells) // 2):
            data = data.iloc[1:]

    data = data.dropna(axis=0, how="all").dropna(axis=1, how="all")
    data = data.reset_index(drop=True)

    # 去掉尾部签名/审核行（只删尾部，避免误伤中间稀疏数据）
    while len(data):
        row = data.iloc[-1]
        non_empty = [value for value in row if pd.notna(value) and str(value).strip()]
        text = " ".join(str(value) for value in non_empty)
        if len(non_empty) <= 2 and any(word in text for word in FOOTER_KEYWORDS):
            data = data.iloc[:-1]
        else:
            break

    # 把明显是数字的列转成数值，方便后续选目标列
    for column in data.columns:
        if data[column].dtype == object:
            converted = pd.to_numeric(data[column], errors="coerce")
            non_null = data[column].notna().sum()
            if non_null and converted.notna().sum() >= max(1, int(non_null * 0.9)):
                data[column] = converted

    data = data.reset_index(drop=True)
    data.columns = _dedupe_columns(data.columns.tolist())
    return data


def load_table_bytes(content: bytes, name: str) -> pd.DataFrame:
    """根据扩展名读取 CSV / Excel，并给出统一的错误信息。"""

    if not content:
        raise DataError("文件内容为空")
    suffix = Path(name or "").suffix.lower()
    if suffix in (".csv", ".txt"):
        frame = _read_csv(content)
    elif suffix in (".xlsx", ".xls"):
        try:
            raw = pd.read_excel(io.BytesIO(content), header=None)
        except ImportError as exc:
            raise DataError(
                "读取 .xls 需要安装 xlrd；也可以先把文件另存为 .xlsx。"
            ) from exc
        except Exception as exc:
            raise DataError(f"Excel 读取失败：{exc}") from exc
        frame = smart_clean_excel(raw)
    else:
        raise DataError("只支持 CSV、XLSX、XLS 文件")

    if frame.empty:
        raise DataError("表格为空，或清洗后没有数据行")
    frame.columns = _dedupe_columns(frame.columns.tolist())
    return frame


def dataset_signature(content: bytes, name: str) -> str:
    digest = hashlib.sha256()
    digest.update(str(name or "").encode("utf-8"))
    digest.update(b"\0")
    digest.update(content)
    return digest.hexdigest()


def frame_signature(frame: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    digest.update(str(frame.shape).encode("utf-8"))
    digest.update(str(list(frame.columns)).encode("utf-8"))
    try:
        hashed = pd.util.hash_pandas_object(frame, index=True).values.tobytes()
        digest.update(hashed)
    except Exception:
        digest.update(frame.to_csv(index=False).encode("utf-8"))
    return digest.hexdigest()


def default_target_column(frame: pd.DataFrame) -> str | None:
    if frame.shape[1] == 0:
        return None
    target_keywords = [
        "抗拉", "拉伸", "屈服", "强度", "延伸", "伸长", "模量", "硬度",
        "tensile", "yield", "strength", "modulus", "elongation", "hardness",
        "uts", "ys",
    ]
    best_column: str | None = None
    best_score = -1
    for column in frame.columns:
        name = str(column).lower()
        # 关键词权重必须高于“数值列”这个弱信号，否则 C/Mn 会抢走目标列
        score = sum(1 for keyword in target_keywords if keyword in name) * 5
        if pd.api.types.is_numeric_dtype(frame[column]):
            score += 1
        if score > best_score:
            best_score = score
            best_column = str(column)
    if best_column is not None:
        return best_column
    numeric = frame.select_dtypes(include=[np.number]).columns
    return str(numeric[0]) if len(numeric) else str(frame.columns[0])


def default_feature_columns(frame: pd.DataFrame, target: str) -> list[str]:
    numeric = [
        str(column)
        for column in frame.select_dtypes(include=[np.number]).columns
        if str(column) != target
    ]
    if numeric:
        return numeric
    return [str(column) for column in frame.columns if str(column) != target]


def preprocess(
    frame: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    *,
    fit: bool = True,
    encoders: dict[str, LabelEncoder] | None = None,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, dict[str, LabelEncoder], dict[str, int]]:
    """清洗缺失值、编码类别特征并返回 ``X, y, data, encoders, stats``。"""

    feature_cols = [str(column) for column in feature_cols]
    target_col = str(target_col)
    if not feature_cols:
        raise DataError("请至少选择一个特征列")
    if target_col in feature_cols:
        raise DataError("目标列不能同时作为特征列")
    missing = [
        column
        for column in feature_cols + [target_col]
        if column not in frame.columns
    ]
    if missing:
        raise DataError(f"表格中找不到列：{', '.join(missing)}")

    data = frame[feature_cols + [target_col]].copy()
    target_numeric = pd.to_numeric(data[target_col], errors="coerce")
    target_dropped = int(target_numeric.isna().sum())
    data[target_col] = target_numeric
    data = data.loc[data[target_col].notna()].copy()

    encoders = dict(encoders or {})
    for column in feature_cols:
        if fit:
            if pd.api.types.is_numeric_dtype(data[column]):
                data[column] = pd.to_numeric(data[column], errors="coerce")
            else:
                values = data[column].astype("string").fillna(MISSING_CATEGORY)
                encoder = LabelEncoder()
                data[column] = encoder.fit_transform(values.astype(str))
                encoders[column] = encoder
        else:
            if column in encoders:
                encoder = encoders[column]
                mapping = {
                    str(label): index for index, label in enumerate(encoder.classes_)
                }
                values = data[column].astype("string").fillna(MISSING_CATEGORY)
                data[column] = (
                    values.astype(str).map(mapping).fillna(-1).astype(float)
                )
            else:
                data[column] = pd.to_numeric(data[column], errors="coerce")

    before_drop = len(data)
    data = data.dropna(subset=feature_cols).reset_index(drop=True)
    feature_dropped = before_drop - len(data)

    X = data[feature_cols].to_numpy(dtype=float)
    y = data[target_col].to_numpy(dtype=float)
    stats = {
        "total_rows": int(len(frame)),
        "target_dropped": target_dropped,
        "feature_dropped": int(feature_dropped),
        "valid_rows": int(len(data)),
    }
    return X, y, data, encoders, stats


def make_fill_values(
    X: np.ndarray,
    feature_cols: list[str],
    encoders: dict[str, LabelEncoder] | None = None,
) -> dict[str, float]:
    encoders = encoders or {}
    values: dict[str, float] = {}
    for index, column in enumerate(feature_cols):
        column_values = X[:, index]
        column_values = column_values[~np.isnan(column_values)]
        if column in encoders:
            if len(column_values):
                codes, counts = np.unique(column_values.astype(int), return_counts=True)
                values[column] = float(codes[int(np.argmax(counts))])
            else:
                values[column] = 0.0
        else:
            values[column] = float(np.mean(column_values)) if len(column_values) else 0.0
    return values


def make_feature_ranges(
    X: np.ndarray,
    feature_cols: list[str],
    encoders: dict[str, LabelEncoder] | None = None,
) -> dict[str, tuple[float, float]]:
    encoders = encoders or {}
    ranges: dict[str, tuple[float, float]] = {}
    for index, column in enumerate(feature_cols):
        values = X[:, index]
        values = values[~np.isnan(values)]
        if not len(values):
            ranges[column] = (0.0, 1.0)
            continue
        low = float(np.min(values))
        high = float(np.max(values))
        if column in encoders:
            low, high = float(int(low)), float(int(high))
        if low == high:
            high = low + 1.0
        ranges[column] = (low, high)
    return ranges


def training_signature(
    *,
    data_signature: str,
    target_col: str,
    feature_cols: list[str],
    model_names: list[str],
    test_size: float,
    n_estimators: int,
    gpr_kernel: str,
) -> str:
    digest = hashlib.sha256()
    digest.update(data_signature.encode("utf-8"))
    digest.update(target_col.encode("utf-8"))
    digest.update("|".join(feature_cols).encode("utf-8"))
    digest.update("|".join(model_names).encode("utf-8"))
    digest.update(f"{test_size}|{n_estimators}|{gpr_kernel}".encode("utf-8"))
    return digest.hexdigest()
