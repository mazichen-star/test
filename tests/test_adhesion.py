"""PAA-DA 黏附模块测试。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.adhesion import (
    TARGET_FIELD,
    build_training_frame,
    demo_experiments,
    normalize_record,
    normalize_test_type,
    parse_records_json,
    standardize_experiments,
    to_kpa,
    to_n_cm,
    train_adhesion_model,
)
from core.storage import SQLiteStorage, SupabaseStorage


def test_unit_conversions():
    assert to_kpa(1.0, "MPa") == pytest.approx(1000.0)
    assert to_kpa(2.0, "kPa") == pytest.approx(2.0)
    assert to_kpa(1.0, "N/cm2") == pytest.approx(10.0)
    assert to_n_cm(1.0, "N/cm") == pytest.approx(1.0)
    assert to_n_cm(1.0, "N/25mm") == pytest.approx(0.4)
    assert normalize_test_type("lap shear") == "lap_shear"
    assert normalize_test_type("peel") == "peel"


def test_normalize_record_and_json():
    record = normalize_record(
        {
            "title": "demo",
            "page": "3",
            "adhesion_value": 0.05,
            "adhesion_unit": "MPa",
            "test_type": "lap shear",
        },
        "kPa",
    )
    assert record["adhesion_target"] == pytest.approx(50.0)
    assert record["page"] == 3
    parsed = parse_records_json(
        '[{"adhesion_value": 20, "adhesion_unit": "kPa", "test_type": "peel"}]',
        "kPa",
    )
    assert parsed[0]["adhesion_target"] == 20
    assert parsed[0]["test_type"] == "peel"


def test_standardize_and_train():
    frame = pd.DataFrame(
        {
            "多巴胺含量": [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            "浓度": [15, 18, 20, 22, 25, 28, 30, 32],
            "pH": [5, 5.5, 6, 6, 6.5, 7, 7, 7.5],
            "固化温度": [40, 45, 50, 55, 60, 65, 70, 75],
            "固化时间": [30, 40, 50, 60, 70, 80, 90, 100],
            "交联剂": ["EDC/NHS"] * 8,
            "基材": ["glass"] * 8,
            "测试类型": ["lap_shear"] * 8,
            "黏附强度": [30, 35, 38, 42, 45, 48, 52, 55],
        }
    )
    mapping = {
        "dopamine_content": "多巴胺含量",
        "polymer_concentration": "浓度",
        "ph": "pH",
        "curing_temp_c": "固化温度",
        "curing_time_min": "固化时间",
        "crosslinker": "交联剂",
        "substrate": "基材",
        "test_type": "测试类型",
    }
    standardized = standardize_experiments(
        frame, mapping, "黏附强度", "kPa", "kPa"
    )
    assert standardized[TARGET_FIELD].notna().sum() == 8

    literature = pd.DataFrame(
        [
            normalize_record(
                {
                    "title": "paper",
                    "test_type": "lap_shear",
                    "dopamine_content": 4.0,
                    "polymer_concentration": 20,
                    "ph": 6,
                    "curing_temp_c": 50,
                    "curing_time_min": 50,
                    "crosslinker": "EDC/NHS",
                    "substrate": "glass",
                    "adhesion_value": 40,
                    "adhesion_unit": "kPa",
                },
                "kPa",
            )
        ]
    )
    combined, numeric, categorical = build_training_frame(
        standardized, literature, "kPa", "lap_shear", True
    )
    assert len(combined) == 9
    bundle = train_adhesion_model(
        combined, numeric, categorical, "Ridge 回归", "kPa", "lap_shear"
    )
    assert "metrics" in bundle
    assert bundle["metrics"]["mae"] >= 0


def test_supabase_adhesion_payload(monkeypatch):
    storage = SupabaseStorage("https://example.supabase.co", "test-key")
    calls = []

    def fake_request(method, table, **kwargs):
        calls.append((method, table, kwargs))
        if method == "POST":
            return kwargs.get("json_body", [])
        return []

    monkeypatch.setattr(storage, "_request", fake_request)
    record = normalize_record(
        {"adhesion_value": 50, "adhesion_unit": "kPa"}, "kPa"
    )
    count = storage.save_adhesion_literature([record], "tester")
    assert count == 1
    assert calls[0][1] == "adhesion_literature"
    assert calls[0][2]["json_body"][0]["username"] == "tester"
    assert calls[0][2]["json_body"][0]["adhesion_target"] == 50

    calls.clear()
    rows = [{"sample_id": "E001", "adhesion_target": 30, "target_unit": "kPa"}]
    assert storage.replace_adhesion_experiments("tester", rows) == 1
    assert calls[0][0] == "DELETE"
    assert calls[1][0] == "POST"
    assert calls[1][2]["json_body"][0]["username"] == "tester"


def test_sqlite_adhesion_storage(tmp_path):
    storage = SQLiteStorage(tmp_path / "app.db")
    storage.add_user("tester", "abc12345", "tester@example.com")

    record = normalize_record(
        {
            "title": "paper",
            "page": 2,
            "test_type": "lap_shear",
            "adhesion_value": 42.5,
            "adhesion_unit": "kPa",
        },
        "kPa",
    )
    assert storage.save_adhesion_literature([record], "tester") == 1
    literature = storage.list_adhesion_literature("tester")
    assert len(literature) == 1
    assert literature[0]["adhesion_target"] == 42.5

    experiments = demo_experiments().head(3)
    assert (
        storage.replace_adhesion_experiments(
            "tester", experiments.to_dict("records")
        )
        == 3
    )
    assert len(storage.list_adhesion_experiments("tester")) == 3
    storage.delete_adhesion_experiments("tester")
    assert storage.list_adhesion_experiments("tester") == []

    storage.delete_account("tester")
    assert not storage.user_exists("tester")
