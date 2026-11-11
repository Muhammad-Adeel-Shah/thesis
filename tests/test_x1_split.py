"""X1 (exploratory): the generation-only probe never sees a test image in training.

The probe trains on train-split reals and ``gen_matched`` only, chooses ``C`` on a
group-wise hold-out of the train split, and is tested on the val split. These tests pin
that no image id -- and no COCO group (a real, its reconstructions and its matched
generation share one) -- is in both training and test, and that a reconstruction is
always paired with its own val real.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location("x1_probe",
                                                  ROOT / "scripts" / "x1_gen_only_probe.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["x1_probe"] = module
    spec.loader.exec_module(module)
    return module


x1 = _load()
MEMBERS = ("sd15", "flux")


def _rows(n_train: int = 40, n_val: int = 10):
    rows, group = [], {}
    for view, n in (("train", n_train), ("val", n_val)):
        for i in range(n):
            real = f"{view}r{i}"
            gen = f"coco_{view}{i}"
            rows.append({"image_id": real, "label": "real", "view": view,
                         "relative_path": f"reals/t_real/{real}.png"})
            rows.append({"image_id": gen, "label": "fake", "view": view,
                         "relative_path": f"gen_matched/{gen}.jpg"})
            group[real] = group[gen] = f"g-{view}{i}"
            for member in MEMBERS:
                recon = f"{real}__{member}"
                rows.append({"image_id": recon, "label": "fake", "view": view,
                             "relative_path": f"pairs/{member}/{real}.png"})
                group[recon] = group[real]
    return rows, group


def test_no_image_id_is_in_both_training_and_test() -> None:
    rows, group = _rows()
    split = x1.build_split(rows, group)
    training = set(split["fit"]) | set(split["tune"])
    test = set(split["test_reals"]) | set(split["test_gen"]) | {
        i for pairs in split["test_recon"].values() for pair in pairs for i in pair}
    assert training and test and not training & test
    assert not set(split["fit"]) & set(split["tune"])


def test_training_holds_reals_and_full_generations_only() -> None:
    rows, group = _rows()
    split = x1.build_split(rows, group)
    training = set(split["fit"]) | set(split["tune"])
    assert not any("__" in i for i in training)                     # no reconstruction
    assert {i.startswith(("trainr", "coco_train")) for i in training} == {True}
    assert split["tune"], "C must be chosen on a non-empty train hold-out"


def test_every_reconstruction_is_paired_with_its_own_val_real() -> None:
    rows, group = _rows()
    split = x1.build_split(rows, group)
    for member, pairs in split["test_recon"].items():
        assert all(recon == f"{real}__{member}" for real, recon in pairs)
        assert [p[0] for p in pairs] == split["test_reals"]


def test_an_id_in_both_splits_raises() -> None:
    rows, group = _rows()
    leaked = dict(next(r for r in rows if r["image_id"] == "trainr0"), view="val")
    with pytest.raises(x1.SplitError, match="trainr0"):
        x1.build_split([*rows, leaked], group)


def test_a_shared_coco_group_raises() -> None:
    """A val generation of a COCO image whose real trains the probe is a content leak."""
    rows, group = _rows()
    group["coco_val0"] = group["trainr0"]
    with pytest.raises(x1.SplitError, match="COCO group"):
        x1.build_split(rows, group)


def test_a_reconstruction_without_its_val_real_raises() -> None:
    rows, group = _rows()
    rows = [r for r in rows if r["image_id"] != "valr3"]
    with pytest.raises(x1.SplitError, match="no val real"):
        x1.build_split(rows, group)
