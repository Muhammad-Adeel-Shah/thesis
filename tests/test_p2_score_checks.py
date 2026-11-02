"""``p2_score.py``: hard checks before the panel is scored once (I-056).

The committed scorer joined S2 and S3 by id but turned a missing row into a NaN that
then silently left both fusions, and it read per-view panel files that do not exist.
These tests run the rewritten script on a tiny fixture shaped like the real inputs (the
real twelve-member bank, real head file formats) and pin:

* a missing id, a duplicate id, a wrong digest, a sha256 that differs between the frozen
  list and the metadata, and a missing input path each stop it, naming the problem;
* ``--check-only`` reads no label (it passes on a list with no label column at all)
  and writes no score;
* a shuffled stream file gives identical scores;
* a second scoring of the same run is refused without ``--crash-rerun``.
"""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dbp.decoders.registry import load_bank_config  # noqa: E402
from dbp.eval.fusion import FittedFusion, RobustMeanFusion  # noqa: E402
from dbp.streams.s3_probe import FEATURE_NAMES, build_set_head  # noqa: E402

RID = "p_a_sd1_seed0"
MEMBERS = [s.key for s in load_bank_config(ROOT / "configs" / "decoder_bank.yaml")
           .distinct_members()]
E7 = [("FLUX.1", "fake"), ("sdxl-vae", "fake"), ("sd-vae-ft-ema", "fake"), ("real", "real")]


def _csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows({c: r[c] for c in columns} for r in rows)


def _digest(rows: list[dict]) -> str:
    lines = sorted(f"{r['image_id']}\t{r['sha256']}" for r in rows)
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()


def build(tmp: Path) -> dict:
    """Inputs for one run; returns the paths and the synthetic counts/digests."""
    import torch

    rng = np.random.default_rng(0)
    panel, meta = [], {"e7_dda_coco": [], "e8_evalgen": []}
    for subset, label in E7:
        for i in range(4):
            image_id = f"e7{subset[:4]}{i}"
            row = {"image_id": image_id, "view": "e7_dda_coco", "label": label,
                   "sha256": hashlib.sha256(image_id.encode()).hexdigest(),
                   "relative_path": f"e7/{image_id}.png"}
            panel.append(row)
            meta["e7_dda_coco"].append({**row, "source_vae": subset})
    for i in range(5):
        image_id = f"e8got{i}"
        row = {"image_id": image_id, "view": "e8_evalgen", "label": "fake",
               "sha256": hashlib.sha256(image_id.encode()).hexdigest(),
               "relative_path": f"e8/{image_id}.png"}
        panel.append(row)
        meta["e8_evalgen"].append({**row, "generator": "GoT"})
    # One extra metadata row: the metadata is the pre-I-047 draw, so a superset.
    meta["e8_evalgen"].append({**meta["e8_evalgen"][0], "image_id": "dropped_by_i047"})
    luma = [{"image_id": r["image_id"], "view": "e7_dda_coco_luma", "label": r["label"],
             "sha256": hashlib.sha256((r["image_id"] + "L").encode()).hexdigest(),
             "relative_path": f"luma/{r['image_id']}.png"}
            for r in panel if r["view"] == "e7_dda_coco"
            and r["image_id"].startswith(("e7FLUX", "e7real"))]
    luma_ids = [f"{r['image_id']}__luma" for r in luma]
    all_ids = [r["image_id"] for r in panel] + luma_ids

    columns = ["image_id", "relative_path", "label", "view", "sha256"]
    _csv(tmp / "lists" / "frozen.csv", panel, columns)
    _csv(tmp / "lists" / "luma.csv", luma, columns)
    _csv(tmp / "lists" / "luma_ids.csv", [{"image_id": i} for i in luma_ids], ["image_id"])
    for view, rows in meta.items():
        _csv(tmp / "sets" / f"p2_panel_{view}.csv", rows, list(rows[0]))

    dims = 8
    s1 = tmp / "s1"
    s1.mkdir()
    np.savez(s1 / "p2_s1_dinov3-vitb16_panel.npz",
             image_ids=np.array([r["image_id"] for r in panel]),
             features=rng.normal(size=(len(panel), dims)).astype(np.float16))
    np.savez(s1 / "p2_s1_dinov3-vitb16_panel_luma.npz", image_ids=np.array(luma_ids),
             features=rng.normal(size=(len(luma), dims)).astype(np.float16))
    probe = tmp / "probe"
    probe.mkdir()
    extra = ["train_only_row"]  # S3 caches are supersets
    for member in MEMBERS:
        np.savez(probe / f"p2_probe_{member}.npz", image_ids=np.array(all_ids + extra),
                 features=rng.normal(size=(len(all_ids) + 1, len(FEATURE_NAMES)))
                 .astype(np.float32))
    s2 = tmp / "s2" / RID / "scores"
    _csv(s2 / "panel.csv", [{"image_id": r["image_id"], "score": f"{rng.normal():.6f}"}
                            for r in panel], ["image_id", "score"])
    _csv(s2 / "luma.csv", [{"image_id": i, "score": f"{rng.normal():.6f}"} for i in luma_ids],
         ["image_id", "score"])

    heads = tmp / "heads"
    heads.mkdir()
    np.savez(heads / f"s1_probe_{RID}.npz", coef=rng.normal(size=(1, dims)),
             intercept=np.array([0.1]))
    width = len(MEMBERS) * len(FEATURE_NAMES)
    np.savez(heads / f"s3_lr_{RID}.npz", coef=rng.normal(size=(1, width)),
             intercept=np.array([0.0]), mean=np.zeros(width, np.float32),
             scale=np.ones(width, np.float32), members=np.array(MEMBERS))
    torch.save(build_set_head().state_dict(), heads / f"s3_set_{RID}.pt")
    streams = ("aeroblade", "s1", "s2", "s3_lr", "s3_set")
    fusions = {
        "f_mean_rz": RobustMeanFusion(("s1", "s2", "s3_set"), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0),
                                      ("mad", "mad", "mad"), f"val_{RID}", 491).to_dict(),
        "f_mean": FittedFusion("mean", streams, (1.0,) * 5, f"val_{RID}", 932).to_dict(),
        "f_stack": FittedFusion("stack", streams, (1.0,) * 5, f"val_{RID}", 932,
                                (0.2,) * 5, 0.0).to_dict(),
    }
    entry = {"fpr_1": {"threshold": 2.0, "rule": "target_fpr_on_reals"},
             "fpr_5": {"threshold": 1.0, "rule": "target_fpr_on_reals"}}
    report = {"run_id": RID, "fitted_on": f"val_{RID}", "fusions": fusions,
              "calibrations": {n: entry for n in (*streams, *fusions)}}
    (heads / f"p2_heads_{RID}.json").write_text(json.dumps(report), encoding="utf-8")
    return {"tmp": tmp, "panel": panel, "luma": luma,
            "synthetic": [str(len(panel)), _digest(panel), str(len(luma)), _digest(luma)]}


def output(result: subprocess.CompletedProcess) -> str:
    """The log goes to stdout; tracebacks to stderr."""
    return result.stdout + result.stderr


def score(fx: dict, *extra: str,
          synthetic: list[str] | None = None) -> subprocess.CompletedProcess:
    tmp = fx["tmp"]
    command = [sys.executable, str(ROOT / "scripts" / "p2_score.py"), "--protocol", "p_a_sd1",
               "--seed", "0", "--panel-list", str(tmp / "lists" / "frozen.csv"),
               "--luma-list", str(tmp / "lists" / "luma.csv"),
               "--luma-ids", str(tmp / "lists" / "luma_ids.csv"),
               "--panel-sets", str(tmp / "sets"), "--s1-cache", str(tmp / "s1"),
               "--probe-dir", str(tmp / "probe"), "--s2-dir", str(tmp / "s2"),
               "--heads", str(tmp / "heads"), "--heads-reports", str(tmp / "heads"),
               "--out-dir", str(tmp / "out"), "--synthetic", *(synthetic or fx["synthetic"]),
               *extra]
    return subprocess.run(command, capture_output=True, text=True, timeout=300, check=False)


@pytest.fixture()
def fx(tmp_path: Path) -> dict:
    return build(tmp_path)


def _scores(fx: dict) -> list[dict]:
    with (fx["tmp"] / "out" / f"scores_{RID}.csv").open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_a_full_scoring_writes_one_finite_row_per_image(fx: dict) -> None:
    result = score(fx)
    assert result.returncode == 0, output(result)[-2000:]
    rows = _scores(fx)
    assert len(rows) == len(fx["panel"]) + len(fx["luma"])
    assert all(r[c] != "" for r in rows for c in ("s1", "s2", "s3_set", "f_mean_rz", "f_stack"))
    luma_flux = [r for r in rows if r["view"] == "e7_dda_coco_luma" and r["label"] == "fake"]
    assert {r["subset"] for r in luma_flux} == {"FLUX.1"}
    assert {r["in_bank"] for r in rows if r["subset"] == "sdxl-vae"} == {"out-of-bank (excluded)"}
    assert all(r["parent_id"] + "__luma" == r["image_id"] for r in luma_flux)


def test_check_only_reads_no_label_and_writes_no_score(fx: dict) -> None:
    tmp = fx["tmp"]
    columns = ["image_id", "relative_path", "view", "sha256"]          # no label column
    _csv(tmp / "lists" / "frozen.csv", fx["panel"], columns)
    _csv(tmp / "lists" / "luma.csv", fx["luma"], columns)
    for view in ("e7_dda_coco", "e8_evalgen"):
        path = tmp / "sets" / f"p2_panel_{view}.csv"
        with path.open(encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        _csv(path, rows, [c for c in rows[0] if c != "label"])
    result = score(fx, "--check-only")
    assert result.returncode == 0, output(result)[-2000:]
    check = json.loads((tmp / "out" / f"check_{RID}.json").read_text())
    assert check["result"] == "PASSED" and check["labels_read"] is False
    assert not (tmp / "out" / f"scores_{RID}.csv").exists()
    assert score(fx).returncode != 0          # the full run needs the labels


def test_a_missing_s2_id_is_an_error_naming_it(fx: dict) -> None:
    path = fx["tmp"] / "s2" / RID / "scores" / "panel.csv"
    lines = path.read_text(encoding="utf-8").splitlines()
    dropped = lines[3].split(",")[0]
    path.write_text("\n".join(lines[:3] + lines[4:]) + "\n", encoding="utf-8")
    result = score(fx)
    assert result.returncode == 1
    assert "missing 1" in output(result) and dropped in output(result)


def test_a_duplicate_s1_id_is_an_error(fx: dict) -> None:
    path = fx["tmp"] / "s1" / "p2_s1_dinov3-vitb16_panel.npz"
    with np.load(path) as blob:
        ids, features = list(blob["image_ids"]), blob["features"]
    ids[1] = ids[0]
    np.savez(path, image_ids=np.array(ids), features=features)
    result = score(fx)
    assert result.returncode == 1 and "duplicate" in output(result)


def test_a_wrong_digest_is_an_error(fx: dict) -> None:
    wrong = list(fx["synthetic"])
    wrong[1] = "0" * 64
    result = score(fx, synthetic=wrong)
    assert result.returncode == 1 and "digest" in output(result)


def test_the_real_panel_digest_is_the_committed_one(fx: dict) -> None:
    """Without --synthetic the frozen panel's committed digest is required."""
    tmp = fx["tmp"]
    command = [sys.executable, str(ROOT / "scripts" / "p2_score.py"), "--protocol", "p_a_sd1",
               "--panel-list", str(tmp / "lists" / "frozen.csv"),
               "--luma-list", str(tmp / "lists" / "luma.csv"),
               "--luma-ids", str(tmp / "lists" / "luma_ids.csv"),
               "--panel-sets", str(tmp / "sets"), "--s1-cache", str(tmp / "s1"),
               "--probe-dir", str(tmp / "probe"), "--s2-dir", str(tmp / "s2"),
               "--heads", str(tmp / "heads"), "--heads-reports", str(tmp / "heads"),
               "--out-dir", str(tmp / "out"), "--check-only"]
    result = subprocess.run(command, capture_output=True, text=True, timeout=300, check=False)
    assert result.returncode == 1 and "11775" in output(result)


def test_a_metadata_hash_disagreement_is_an_error(fx: dict) -> None:
    path = fx["tmp"] / "sets" / "p2_panel_e8_evalgen.csv"
    text = path.read_text(encoding="utf-8")
    first = fx["panel"][-1]["sha256"]
    path.write_text(text.replace(first, "f" * 64, 1), encoding="utf-8")
    result = score(fx)
    assert result.returncode == 1 and "sha256" in output(result)


def test_every_missing_input_path_is_listed(fx: dict) -> None:
    (fx["tmp"] / "probe" / f"p2_probe_{MEMBERS[3]}.npz").unlink()
    (fx["tmp"] / "s2" / RID / "scores" / "luma.csv").unlink()
    result = score(fx, "--check-only")
    assert result.returncode == 1
    assert "2 input(s) missing" in output(result) and MEMBERS[3] in output(result)


def test_a_shuffled_stream_file_gives_identical_scores(fx: dict) -> None:
    assert score(fx).returncode == 0
    first = _scores(fx)
    path = fx["tmp"] / "s2" / RID / "scores" / "panel.csv"
    lines = path.read_text(encoding="utf-8").splitlines()
    body = lines[1:]
    np.random.default_rng(5).shuffle(body)
    path.write_text("\n".join([lines[0], *body]) + "\n", encoding="utf-8")
    assert score(fx, "--crash-rerun", "test: shuffled input").returncode == 0
    assert _scores(fx) == first


def test_a_second_scoring_is_refused_without_a_crash_reason(fx: dict) -> None:
    assert score(fx).returncode == 0
    result = score(fx)
    assert result.returncode == 1 and "scored once" in output(result)


def test_luma_parents_must_be_exactly_the_flux_and_real_rows_of_e7(fx: dict) -> None:
    """Checked before scoring: the bootstrap resamples luma rows with their parents."""
    tmp = fx["tmp"]
    sdxl = next(r for r in fx["panel"] if r["image_id"].startswith("e7sdxl"))
    luma = [*fx["luma"], {**sdxl, "view": "e7_dda_coco_luma"}]
    _csv(tmp / "lists" / "luma.csv", luma,
         ["image_id", "relative_path", "label", "view", "sha256"])
    _csv(tmp / "lists" / "luma_ids.csv", [{"image_id": f"{r['image_id']}__luma"} for r in luma],
         ["image_id"])
    synthetic = [fx["synthetic"][0], fx["synthetic"][1], str(len(luma)), _digest(luma)]
    result = score(fx, "--check-only", synthetic=synthetic)
    assert result.returncode == 1
    assert "luma parents differ" in output(result) and sdxl["image_id"] in output(result)
