"""P4 option A (user decision 2026-11-09): pixel-balanced shards, append-only checkpoints,
the resume check's tolerance, the merge, and the fail-fast projection."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from dbp.baselines import shards  # noqa: E402
from dbp.baselines.common import BaselineError  # noqa: E402

COLUMNS = ["list", "image_id", "score", "flag", "lpips2_sd15"]


def _script(name: str):
    spec = importlib.util.spec_from_file_location(f"_s_{name}", ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[f"_s_{name}"] = module
    spec.loader.exec_module(module)
    return module


def _row(i: int, name: str = "panel_b") -> dict:
    return {"list": name, "image_id": f"id{i:03d}", "score": -0.1 * i - 1 / 3, "flag": i % 2,
            "lpips2_sd15": 0.1 * i + 1 / 3}


def test_the_capped_pixel_count_is_the_768_cap_then_a_multiple_of_8() -> None:
    assert shards.capped_pixels(1563, 941) == 768 * 768
    assert shards.capped_pixels(900, 700) == 768 * 696
    assert shards.capped_pixels(384, 384) == 384 * 384
    assert shards.capped_pixels(1000, 3000) == 768 * 768


def test_the_deal_balances_pixels_and_keeps_each_shards_order() -> None:
    items = [{"order": i, "pixels": p} for i, p in enumerate([9, 1, 8, 2, 7, 3, 6, 4, 5])]
    first, second = shards.assign(items, 2)
    loads = [sum(i["pixels"] for i in s) for s in (first, second)]
    assert abs(loads[0] - loads[1]) <= 1 and sum(loads) == 45
    assert all([i["order"] for i in s] == sorted(i["order"] for i in s) for s in (first, second))
    assert shards.assign(items, 2) == [first, second]                 # deterministic
    assert sorted(i["order"] for s in (first, second) for i in s) == list(range(9))


def test_parts_are_appended_never_overwritten_and_reload_exactly(tmp_path: Path) -> None:
    store = shards.ShardStore(tmp_path / "s", COLUMNS)
    store.append([_row(0), _row(1)])
    store.append([_row(2)])
    rows = store.load()
    assert [r["image_id"] for r in rows] == ["id000", "id001", "id002"]
    assert float(rows[1]["score"]) == _row(1)["score"]                # full precision
    assert [e["part"] for e in store.entries()] == ["part_00000.csv", "part_00001.csv"]
    (tmp_path / "s" / "part_00002.csv").write_text("x", encoding="utf-8")  # a planted part
    with pytest.raises(BaselineError, match="not in the ledger"):
        store.load()


def test_a_changed_part_stops_the_load(tmp_path: Path) -> None:
    store = shards.ShardStore(tmp_path / "s", COLUMNS)
    store.append([_row(0), _row(1)])
    part = tmp_path / "s" / "part_00000.csv"
    part.write_text(part.read_text("utf-8").replace("id001", "id999"), encoding="utf-8")
    with pytest.raises(BaselineError, match="no longer matches"):
        store.load()


def test_an_existing_part_name_is_never_written_over(tmp_path: Path) -> None:
    store = shards.ShardStore(tmp_path / "s", COLUMNS)
    store.append([_row(0)])
    store.ledger.write_text("", encoding="utf-8")                     # ledger lost its line
    with pytest.raises(BaselineError, match="never overwritten"):
        store.append([_row(1)])


def test_the_resume_sample_spreads_over_what_is_saved() -> None:
    assert shards.resume_sample(1000, 20)[0] == 0 and shards.resume_sample(1000, 20)[-1] == 999
    assert len(shards.resume_sample(1000, 20)) == 20
    assert shards.resume_sample(5, 20) == [0, 1, 2, 3, 4]
    assert shards.resume_sample(0, 20) == []


def test_fp32_rounding_is_tolerated_and_a_real_difference_is_not() -> None:
    assert shards.within_fp32(-0.0123456789, -0.0123456789 * (1 + 3e-7))
    assert not shards.within_fp32(-0.0123456789, -0.0123456789 * (1 + 1e-5))
    assert shards.within_fp32(0.0, 5e-10) and not shards.within_fp32(0.0, 1e-8)
    assert not shards.within_fp32(float("nan"), float("nan"))


def test_the_merge_restores_list_order_and_needs_every_image_once() -> None:
    lists = {"panel_b": [{"image_id": f"id{i:03d}"} for i in range(4)],
             "luma": [{"image_id": "L0"}]}
    saved = [{"list": "panel_b", "image_id": i} for i in ("id002", "id000", "id003", "id001")]
    saved.append({"list": "luma", "image_id": "L0"})
    merged = shards.merge(saved, lists)
    assert [r["image_id"] for r in merged["panel_b"]] == ["id000", "id001", "id002", "id003"]
    with pytest.raises(BaselineError, match="not scored"):
        shards.merge(saved[1:], lists)
    with pytest.raises(BaselineError, match="twice"):
        shards.merge([*saved, saved[0]], lists)
    with pytest.raises(BaselineError, match="in no list"):
        shards.merge([*saved, {"list": "luma", "image_id": "stray"}], lists)


def _plan(tmp_path: Path) -> dict:
    lists = {}
    for name, n in (("panel_b", 4), ("luma", 2), ("threshold_reals", 2)):
        path = tmp_path / f"{name}.csv"
        path.write_text("image_id,role\n" + "".join(f"{name}{i},test\n" for i in range(n)),
                        encoding="utf-8")
        lists[name] = str(path)
    return {"list_files": lists, "jobs": ["0:dda_native,aeroblade#0", "1:aeroblade#1"],
            "rows": {"dda_native": {"lists": ["panel_b", "luma", "threshold_reals"]}},
            "units": {"dda_native": {"row": "dda_native", "shard": None},
                      "aeroblade#0": {"row": "aeroblade", "shard": 0},
                      "aeroblade#1": {"row": "aeroblade", "shard": 1}},
            "shards": {"aeroblade": {"resume_rescore": 20, "units": {
                "aeroblade#0": {"pixels": 3_600_000, "items": 10, "saved_rows": 0,
                                "saved_pixels": 0},
                "aeroblade#1": {"pixels": 3_600_000, "items": 10, "saved_rows": 4,
                                "saved_pixels": 1_440_000}}}}}


def _smoke(root: Path, unit: str, seconds: float, images: int, pixels: int, load: float) -> None:
    path = root / "reports" / "p4_baselines" / f"{unit}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"timing": {"seconds": seconds, "images": images,
                                           "pixels": pixels}, "load_seconds": load}), "utf-8")


EXPECT = {"deadline_hours": 10.5, "session_limit_hours": 12.0, "margin_hours": 0.25}


def test_the_projection_uses_seconds_per_image_or_per_pixel(tmp_path: Path) -> None:
    fail = _script("p4_failfast")
    plan = _plan(tmp_path)
    smoke = tmp_path / "smoke"
    _smoke(smoke, "dda_native", 1.0, 5, 0, 10.0)              # 0.2 s/image x 8 images + 10 s
    _smoke(smoke, "aeroblade#0", 18.0, 1, 1_000_000, 30.0)    # 18 us/pixel
    _smoke(smoke, "aeroblade#1", 18.0, 1, 1_000_000, 30.0)
    record = fail.project(plan, smoke, EXPECT, elapsed_hours=0.5)
    gpu0, gpu1 = record["jobs"]
    assert gpu0["units"][0]["seconds"] == pytest.approx(10.0 + 0.2 * 8)
    assert gpu0["units"][1]["seconds"] == pytest.approx(30.0 + 18e-6 * 3_600_000)
    # Shard 1 has 4 rows saved: only what is left, plus re-scoring the 4 saved rows.
    assert gpu1["units"][0]["seconds"] == pytest.approx(
        30.0 + 18e-6 * 2_160_000 + 4 * 360_000 * 18e-6)
    assert record["available_hours"] == pytest.approx(10.5) and record["ok"]


def test_a_projection_over_the_time_left_is_refused(tmp_path: Path) -> None:
    fail = _script("p4_failfast")
    plan = _plan(tmp_path)
    smoke = tmp_path / "smoke"
    _smoke(smoke, "dda_native", 1.0, 5, 0, 10.0)
    _smoke(smoke, "aeroblade#0", 18.0, 1, 1_000, 30.0)        # 18 ms/pixel: 18 h
    _smoke(smoke, "aeroblade#1", 18.0, 1, 1_000_000, 30.0)
    record = fail.project(plan, smoke, EXPECT, elapsed_hours=0.5)
    assert not record["ok"] and [j["fits"] for j in record["jobs"]] == [False, True]
    late = fail.project(_plan(tmp_path), smoke, EXPECT, elapsed_hours=11.9)
    assert late["available_hours"] < 0 and not late["ok"]


def test_a_unit_without_a_timed_smoke_cannot_be_projected(tmp_path: Path) -> None:
    fail = _script("p4_failfast")
    with pytest.raises(ValueError, match="no timed image"):
        fail.project(_plan(tmp_path), tmp_path / "empty", EXPECT, elapsed_hours=0.0)
