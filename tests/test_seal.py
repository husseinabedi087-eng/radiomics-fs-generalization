"""Tests for the sealed split and the data loader. No model, no metric, no feature-target statistic."""

import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import data_access as da  # noqa: E402

NEUTRAL_SKIP = "not checked in this run"


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


def test_arita2018_has_no_idh_columns():
    if "Arita2018" not in da.list_datasets("development"):
        pytest.skip(NEUTRAL_SKIP)
    X, _, _ = da.load_dataset("Arita2018")
    assert not [c for c in X.columns if c.startswith(("IDHmt_", "IDHwt_"))]


def test_song2020_has_259_rows():
    if "Song2020" not in da.list_datasets("development"):
        pytest.skip(NEUTRAL_SKIP)
    X, y, ids = da.load_dataset("Song2020")
    assert len(X) == len(y) == len(ids) == 259
