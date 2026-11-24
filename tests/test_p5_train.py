"""P5 training pairs and training stages: the pure parts (CPU; no network, no weights)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

pytest.importorskip("av")

import p5_build_pairs as B  # noqa: E402
import p5_train as T  # noqa: E402
from dbp.data import video as V  # noqa: E402


def test_pair_draw_is_preregistered_and_order_free():
    ids = [f"5000_0_{i:05d}" for i in range(4000)]
    draws = [B.pair_draw(i, "wan21") for i in ids]
    assert draws == [B.pair_draw(i, "wan21") for i in ids]               # reproducible
    share = sum(d["degrade"] for d in draws) / len(draws)
    h264 = sum(d["codec"] == "libx264" for d in draws) / len(draws)
    assert abs(share - B.P_REENCODE) < 0.03 and abs(h264 - B.H264_SHARE) < 0.03
    assert {d["crf"] for d in draws} == set(range(18, 36))                 # 18..35 inclusive
    assert (B.P_REENCODE, B.H264_SHARE, B.CRF_RANGE) == (0.80, 0.70, (18, 35))


def test_format_match_picks_the_closest_crf():
    rng = np.random.default_rng(0)
    base = rng.integers(0, 256, (72, 72, 3), dtype=np.uint8)
    frames = np.stack([np.roll(base, 2 * t, axis=1)[:64, :64] for t in range(25)])
    target = 0.25
    data, crf, bpp = B.format_match(frames, target, "30")
    every = {c: B.clip_bpp(V.encode_clip(frames, "libx264", c, "30"), frames)
             for c in range(B.CRF_MIN, B.CRF_MAX + 1)}
    best = min(every, key=lambda c: abs(np.log(every[c] / target)))
    assert abs(np.log(bpp / target)) <= abs(np.log(every[best] / target)) + 1e-9
    assert V.decode_clip(data, expect_frames=25).shape == frames.shape


def make_clip(path: Path, seed: int) -> None:
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 256, (80, 80, 3), dtype=np.uint8)
    V.write_png_clip(np.stack([np.roll(base, t, axis=0)[:64, :64] for t in range(25)]), path)


def test_build_one_degrades_train_pairs_only_and_writes_hashed_files(tmp_path):
    train_id = next(f"s{i}" for i in range(100) if B.pair_draw(f"s{i}", "wan21")["degrade"])
    row = {"sample_id": train_id, "video_id": "v1", "clip_dir": f"train/{train_id}",
           "width": "1920", "height": "1080", "fps": "30.0", "fps_fraction": "30/1",
           "bit_rate": "8000000"}
    make_clip(tmp_path / "reals" / row["clip_dir"], 1)
    make_clip(tmp_path / "recon" / "wan21" / train_id, 2)
    job = {"row": row, "member": "wan21", "reals_root": str(tmp_path / "reals"),
           "recon_root": str(tmp_path / "recon"), "out_root": str(tmp_path / "pairs")}
    rec = B.build_one({**job, "split": "train"})
    assert rec["degraded"] and {"fm", "real_deg", "recon_deg"} <= set(rec)
    for key in ("fm", "real_deg", "recon_deg"):
        assert B.file_ok(tmp_path / "pairs", rec[key])
        frames = V.decode_clip((tmp_path / "pairs" / rec[key]["path"]).read_bytes())
        assert frames.shape == (25, 64, 64, 3)
    assert rec["fm"]["source_bpp"] == pytest.approx(8e6 / (1920 * 1080 * 30), rel=1e-4)
    val = B.build_one({**job, "split": "validation"})
    assert not val["degraded"] and "real_deg" not in val      # V1-a's set is never degraded
    # the diagnostic copy (user, 2026-11-24): re-encoded with the pair's own draw, never trained on
    for key in ("real_diag", "recon_diag"):
        assert B.file_ok(tmp_path / "pairs", val[key])
        assert V.decode_clip((tmp_path / "pairs" / val[key]["path"]).read_bytes()).shape[0] == 25


def test_runs_and_lolo_folds_follow_f1():
    assert T.run_members("s2v_f") == set(T.BANK) and T.run_members("s2v_1") == {"wan21"}
    assert T.LOLO_FOLDS == ("wan21", "hunyuan", "ltx090")              # F1, HunyuanVideo kept
    for fold in T.LOLO_FOLDS:
        members = T.run_members(f"lolo_{fold}")
        assert fold not in members and T.ALWAYS_IN in members and len(members) == 5
    with pytest.raises(SystemExit, match="unknown run"):
        T.run_members("lolo_cogvx")
    assert (T.SCHEDULE_STEPS, T.STOP_AT, T.DEGRADE_P) == (6000, 5000, 0.35)
    assert T.TRAIN_FRAMES == (1, 24) and list(T.FRAMES_SCORED) == list(range(1, 17))


def test_temporal_inputs_and_head_shapes():
    import torch

    f = torch.arange(2 * 17 * 3, dtype=torch.float32).view(2, 17, 3)
    x = T.temporal_inputs(f)
    assert x.shape == (2, 16, 6)
    assert torch.equal(x[:, :, :3], f[:, 1:17]) and torch.equal(x[:, :, 3:], f[:, 1:] - f[:, :-1])
    head = T.temporal_head(3)
    assert head(x).shape == (2,)


def test_load_pairs_takes_the_last_record_per_pair(tmp_path):
    path = tmp_path / "p.jsonl"
    rows = [{"pair_id": "a:wan21", "v": 1}, {"pair_id": "b:wan21", "v": 1},
            {"pair_id": "a:wan21", "v": 2}]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    assert [(r["pair_id"], r["v"]) for r in T.load_pairs(path)] == [("a:wan21", 2), ("b:wan21", 1)]
