"""Tests for the sealed split and the data loader. No model, no metric, no feature-target statistic.

The only real dataset touched is the first development dataset. All rule tests use small synthetic
DataFrames and a fake dataset name.
"""

import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import data_access as da  # noqa: E402

FAKE = "SyntheticTestSet"


# ---- seal, exclusion and hash checks ----

def test_first_development_dataset_loads():
    name = da.list_datasets("development")[0]
    X, y, ids = da.load_dataset(name)
    assert set(np.unique(y)) <= {0, 1}
    assert len(X) == len(y) == len(ids)


def test_confirmation_dataset_is_sealed():
    if da.stage2_frozen():
        pytest.skip("stage 2 is frozen; the seal no longer applies")
    name = da.list_datasets("confirmation")[0]
    with pytest.raises(da.SealedDatasetError, match="sealed until stage 2"):
        da.load_dataset(name)


def test_excluded_dataset_raises():
    with pytest.raises(da.DatasetExcludedError):
        da.load_dataset("Zhang2023")


def test_changed_byte_raises_hash_error(tmp_path, monkeypatch):
    name = da.list_datasets("development")[0]
    src = da.RAW_DIR / f"{name}.parquet"
    dst = tmp_path / src.name
    shutil.copyfile(src, dst)
    data = bytearray(dst.read_bytes())
    data[len(data) // 2] ^= 0xFF
    dst.write_bytes(bytes(data))
    monkeypatch.setattr(da, "RAW_DIR", tmp_path)
    with pytest.raises(da.HashMismatchError):
        da.load_dataset(name)


# ---- apply_rules on synthetic data ----

def test_drop_columns_removes_listed_columns():
    df = pd.DataFrame({"ID": ["P0000_1", "P0000_2"], "Target": [0, 1],
                       "IDHmt_mean": [0.1, 0.2], "IDHwt_sd": [0.3, 0.4], "f1": [1.0, 2.0]})
    rules = {"drop_columns": {FAKE: {"columns": ["IDHmt_mean", "IDHwt_sd"], "reason": "test"}}}
    out = da.apply_rules(df, FAKE, rules)
    assert not [c for c in out.columns if c.startswith(("IDHmt_", "IDHwt_"))]
    assert list(out.columns) == ["ID", "Target", "f1"]
    assert "IDHmt_mean" in df.columns  # input not modified


def test_drop_columns_missing_column_raises():
    df = pd.DataFrame({"ID": ["P0000_1"], "Target": [0], "f1": [1.0]})
    rules = {"drop_columns": {FAKE: {"columns": ["not_there"], "reason": "test"}}}
    with pytest.raises(da.IntegrityError):
        da.apply_rules(df, FAKE, rules)


def test_drop_duplicate_rows_keeps_first():
    df = pd.DataFrame({"ID": ["P0000_1", "P0000_2", "P0000_3"], "Target": [1, 1, 0],
                       "f1": [5.0, 5.0, 6.0], "f2": [7.0, 7.0, 8.0]})
    rules = {"drop_duplicate_rows": {FAKE: {"keep": "first", "reason": "test"}}}
    out = da.apply_rules(df, FAKE, rules)
    assert len(out) == 2
    assert list(out["ID"]) == ["P0000_1", "P0000_3"]


def test_strip_id_whitespace():
    df = pd.DataFrame({"ID": [" P0000_1", "P0000_2 ", None], "Target": [0, 1, 0], "f1": [1.0, 2.0, 3.0]})
    out = da.apply_rules(df, FAKE, {"strip_id_whitespace": [FAKE]})
    assert list(out["ID"][:2]) == ["P0000_1", "P0000_2"]
    assert pd.isna(out["ID"].iloc[2])


def test_target_to_int():
    df = pd.DataFrame({"ID": ["P0000_1", "P0000_2"], "Target": [True, False], "f1": [1.0, 2.0]})
    rules = {"target_to_int": {"datasets": [FAKE], "mapping": {"True": 1, "False": 0}}}
    out = da.apply_rules(df, FAKE, rules)
    assert list(out["Target"]) == [1, 0]


def test_no_rules_no_change():
    df = pd.DataFrame({"ID": [" P0000_1"], "Target": [True], "f1": [1.0]})
    out = da.apply_rules(df, FAKE, {})
    pd.testing.assert_frame_equal(out, df)
