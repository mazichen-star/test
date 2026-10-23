"""核心逻辑测试：不启动 Streamlit，也不依赖 Supabase。"""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
import pytest

from core.config import supabase_url
from core.data import (
    datasets_signature,
    default_target_column,
    merge_tables,
    preprocess,
    smart_clean_excel,
    training_signature,
)
from core.modeling import train_models
from core.reporting import excel_bytes, make_word_report
from core.security import hash_password, password_needs_rehash, verify_password
from core.storage import SQLiteStorage


def make_frame(rows: int = 24) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    frame = pd.DataFrame(
        {
            "C": rng.uniform(0.1, 1.2, rows),
            "Mn": rng.uniform(0.2, 2.0, rows),
            "Temp": rng.integers(800, 1200, rows),
            "Structure": rng.choice(["FCC", "BCC"], rows),
        }
    )
    frame["Tensile"] = (
        300
        + 200 * frame["C"]
        + 80 * frame["Mn"]
        + 0.1 * frame["Temp"]
        + rng.normal(0, 8, rows)
    )
    return frame


def test_supabase_url_accepts_rest_suffix(monkeypatch):
    monkeypatch.setenv(
        "SUPABASE_URL", "https://example.supabase.co/rest/v1/"
    )
    assert supabase_url() == "https://example.supabase.co"


def test_password_hash_and_legacy_compatibility():
    hashed = hash_password("abc12345")
    assert hashed.startswith("pbkdf2_sha256$")
    assert verify_password("abc12345", hashed)
    assert not verify_password("wrong", hashed)
    assert not password_needs_rehash(hashed)

    legacy = hashlib.sha256(b"abc12345").hexdigest()
    assert verify_password("abc12345", legacy)
    assert password_needs_rehash(legacy)


def test_sqlite_storage_crud(tmp_path):
    storage = SQLiteStorage(tmp_path / "app.db")

    storage.add_user("tester", "abc12345", "tester@example.com")
    assert storage.user_exists("tester")
    assert storage.verify_user("tester", "abc12345")
    assert not storage.verify_user("tester", "bad")

    with pytest.raises(Exception):
        storage.add_user("tester", "abc12345", "other@example.com")

    storage.save_user_file("tester", "data.csv", b"a,b\n1,2\n")
    storage.save_user_file("tester", "data.csv", b"a,b\n3,4\n")
    files = storage.list_user_files("tester")
    assert len(files) == 1
    assert storage.load_user_file("tester", "data.csv") == b"a,b\n3,4\n"

    storage.add_material(
        "Al2O3", "ceramic", {"tensile_strength": 350}, "实验", "tester", "doi"
    )
    materials = storage.list_materials()
    assert len(materials) == 1
    assert materials[0]["properties"]["tensile_strength"] == 350
    assert storage.find_material("al2o3") is not None

    material_id = materials[0]["id"]
    assert not storage.delete_material(material_id, "other")
    assert storage.delete_material(material_id, "tester")

    storage.change_password("tester", "new12345")
    assert storage.verify_user("tester", "new12345")
    storage.delete_account("tester")
    assert not storage.user_exists("tester")


def test_preprocess_drops_missing_and_encodes_categories():
    frame = make_frame()
    frame.loc[0, "Mn"] = np.nan
    X, y, data, encoders, stats = preprocess(
        frame, ["C", "Mn", "Temp", "Structure"], "Tensile"
    )
    assert X.shape[1] == 4
    assert len(X) == len(y) == len(data)
    assert stats["feature_dropped"] == 1
    assert "Structure" in encoders
    assert set(encoders["Structure"].classes_) == {"BCC", "FCC"}


def test_default_target_column_prefers_property_name():
    frame = make_frame()
    frame["Tensile"] = frame["Tensile"].astype(object)
    assert default_target_column(frame) == "Tensile"


def test_smart_clean_excel_skips_title_and_unit_rows():
    raw = pd.DataFrame(
        [
            ["实验报告", None, None, None],
            ["试样长度 10mm", None, None, None],
            ["C", "Mn", "Temp", "Tensile"],
            ["%", "%", "°C", "MPa"],
            [0.2, 1.1, 900, 450],
            [0.4, 1.3, 950, 510],
            ["审核", "张三", None, None],
        ]
    )
    cleaned = smart_clean_excel(raw)
    assert list(cleaned.columns) == ["C", "Mn", "Temp", "Tensile"]
    assert len(cleaned) == 2
    assert cleaned.iloc[0]["Tensile"] == 450


def test_train_models_and_exports():
    frame = make_frame()
    signature = training_signature(
        data_signature="data",
        target_col="Tensile",
        feature_cols=["C", "Mn", "Temp", "Structure"],
        model_names=["线性回归"],
        test_size=0.2,
        n_estimators=100,
        gpr_kernel="RBF + 白噪声",
    )
    result = train_models(
        frame,
        ["C", "Mn", "Temp", "Structure"],
        "Tensile",
        ["线性回归"],
        test_size=0.2,
        n_estimators=100,
        data_signature="data",
        train_signature=signature,
    )
    assert result["model_name"] == "线性回归"
    assert result["metrics"]["r2"] > 0.8
    assert result["metrics"]["cv_r2"] is not None
    assert len(result["result_df"]) > 0
    assert "data_out" in result

    excel = excel_bytes(
        result["data_out"], metrics=result["metrics"], model_name=result["model_name"]
    )
    assert excel[:2] == b"PK"
    report = make_word_report(
        target_col="Tensile",
        data_source="toy.csv",
        stats=result["stats"],
        feature_cols=result["feature_cols"],
        model_name=result["model_name"],
        metrics=result["metrics"],
        cmp_df=result["cmp_df"],
        y_test=result["y_test"],
        y_pred=result["y_pred"],
        result_df=result["result_df"],
        importance=result["importance"],
    )
    assert report is not None
    assert report[:2] == b"PK"


def test_merge_tables_same_schema_different_order():
    first = pd.DataFrame({"C": [0.1, 0.2], "Mn": [1.0, 1.1], "Tensile": [500, 510]})
    second = pd.DataFrame({"Tensile": [520], "C": [0.3], "Mn": [1.2]})
    merged, info = merge_tables([first, second], ["a.csv", "b.csv"])
    assert info["same_schema"] is True
    assert info["same_order"] is False
    assert list(merged.columns) == ["C", "Mn", "Tensile"]
    assert len(merged) == 3
    assert info["row_counts"] == [
        {"name": "a.csv", "rows": 2},
        {"name": "b.csv", "rows": 1},
    ]


def test_merge_tables_union_reports_differences():
    first = pd.DataFrame({"C": [0.1], "Tensile": [500]})
    second = pd.DataFrame({"C": [0.2], "Hardness": [200]})
    merged, info = merge_tables([first, second], ["a.csv", "b.csv"])
    assert info["same_schema"] is False
    assert "Hardness" in info["missing"]["a.csv"]
    assert "Tensile" in info["missing"]["b.csv"]
    assert "Hardness" in info["extra"]["b.csv"]
    assert merged["Hardness"].isna().sum() == 1
    assert merged["Tensile"].isna().sum() == 1


def test_datasets_signature_is_stable_and_order_sensitive():
    one = datasets_signature([("a.csv", b"a"), ("b.csv", b"b")])
    two = datasets_signature([("a.csv", b"a"), ("b.csv", b"b")])
    reverse = datasets_signature([("b.csv", b"b"), ("a.csv", b"a")])
    assert one == two
    assert one != reverse
