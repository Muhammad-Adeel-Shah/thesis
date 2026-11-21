"""P5 video: the bank registry, the video reconstruction kinds, video I/O, the Pexels fetch,
and the video pin check. CPU only; no network; no weights (dry-run stand-ins)."""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

av = pytest.importorskip("av")

from dbp.data import video as V  # noqa: E402
from dbp.decoders.registry import BankConfigError, load_video_bank  # noqa: E402

BANK_PATH = ROOT / "configs" / "video_bank.yaml"


def make_mp4(path: Path, frames: int = 30, size: tuple[int, int] = (360, 640),
             fps: int = 30, static: bool = False) -> None:
    """A small H.264 file with a moving gradient (or a still)."""
    height, width = size
    yy, xx = np.mgrid[0:height, 0:width]
    with av.open(str(path), "w") as container:
        stream = container.add_stream("libx264", rate=fps)
        stream.width, stream.height, stream.pix_fmt = width, height, "yuv420p"
        for t in range(frames):
            shift = 0 if static else 3 * t
            img = np.stack([(xx + shift) % 256, (yy + shift) % 256, (xx + yy) % 256], -1)
            frame = av.VideoFrame.from_ndarray(img.astype(np.uint8), format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


# ------------------------------------------------------------------ registry ---

def test_shipped_video_bank_is_valid_and_complete():
    bank = load_video_bank(BANK_PATH)
    assert (bank.frames, bank.crop, bank.min_short_side) == (25, 320, 320)
    assert [s.key for s in bank.select(("bank",))] == ["wan21", "hunyuan", "cogvx", "ltx090",
                                                        "mochi", "sd15_pf"]
    assert [s.key for s in bank.select(("reserve",))] == ["wan22_5b"]
    assert [s.key for s in bank.select(("excluded",))] == ["allegro"]
    for spec in bank.members:
        assert spec.accepts_frames(25) and 320 % spec.spatial_factor == 0
        assert len(spec.revision) == 40 and len(spec.vae_sha256) == 64


@pytest.mark.parametrize("change, message", [
    ({"cls": "AutoencoderKLEvil"}, "role/kind/cls"),
    ({"revision": "main"}, "full 40-hex"),
    ({"temporal_factor": 7}, r"not 1 \+ k"),
    ({"spatial_factor": 48}, "not a multiple"),
    ({"role": "excluded"}, "excluded_reason"),
    ({"surprise": 1}, "unknown keys"),
])
def test_video_bank_rejects_bad_members(tmp_path, change, message):
    raw = yaml.safe_load(BANK_PATH.read_text(encoding="utf-8"))
    raw["members"][0].update(change)
    raw["members"][0].pop("excluded_reason", None)
    path = tmp_path / "bank.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(BankConfigError, match=message):
        load_video_bank(path)


def test_min_short_side_below_crop_is_refused(tmp_path):
    raw = yaml.safe_load(BANK_PATH.read_text(encoding="utf-8"))
    raw["clip"]["min_short_side"] = 256
    path = tmp_path / "bank.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(BankConfigError, match="force a resize"):
        load_video_bank(path)


# --------------------------------------------------------- reconstruction kinds ---

def test_video_tensor_round_trip_is_exact():
    from dbp.decoders.reconstruct import video_to_tensor, video_to_uint8

    clip = np.random.default_rng(1).integers(0, 256, (5, 16, 24, 3), dtype=np.uint8)
    x = video_to_tensor(clip, "cpu")
    assert tuple(x.shape) == (1, 3, 5, 16, 24)
    assert np.array_equal(video_to_uint8(x), clip)


def test_reconstruct_video_kl_refuses_a_dropped_frame():
    import torch

    from dbp.decoders.reconstruct import reconstruct_video_kl

    class Drops:
        def encode(self, x):
            return SimpleNamespace(latent_dist=SimpleNamespace(mode=lambda: x))

        def decode(self, z):
            return SimpleNamespace(sample=z[:, :, 1:])

    with pytest.raises(RuntimeError, match="every frame"):
        reconstruct_video_kl(Drops(), torch.zeros(1, 3, 25, 8, 8))


def test_framewise_flags_are_switched_off_and_asserted():
    """HunyuanVideo's diffusers default temporal-tiles any clip over 16 frames."""
    from dbp.decoders.loaders import MemberLoadError, assert_video_splitting_disabled

    model = SimpleNamespace(use_tiling=True, use_framewise_encoding=True,
                            use_framewise_decoding=True)
    state = assert_video_splitting_disabled(model, "hunyuan")
    assert state == {"use_tiling": False, "use_framewise_encoding": False,
                     "use_framewise_decoding": False}

    class Stuck:
        use_framewise_decoding = property(lambda self: True, lambda self, v: None)

    with pytest.raises(MemberLoadError, match="still on"):
        assert_video_splitting_disabled(Stuck(), "x")


def test_excluded_member_never_loads():
    from dbp.decoders.loaders import MemberLoadError, load_video_member

    with pytest.raises(MemberLoadError, match="excluded"):
        load_video_member(load_video_bank(BANK_PATH).by_key("allegro"))


def test_every_stand_in_returns_25_frames_deterministically(monkeypatch):
    """Each member's class, in a tiny random-init form, keeps (T, H, W) and repeats exactly;
    this is where a member that drops or pads frames would show up (e.g. Mochi's 6k + 1)."""
    from dbp.decoders.loaders import load_video_member

    monkeypatch.setenv("DBP_DRYRUN", "1")
    clip = np.random.default_rng(0).integers(0, 256, (25, 64, 64, 3), dtype=np.uint8)
    for spec in load_video_bank(BANK_PATH).select(("bank", "reserve")):
        member = load_video_member(spec, "cpu")
        assert member.dry_run_random_init
        out = member.reconstruct(clip)
        assert out.shape == clip.shape and out.dtype == np.uint8, spec.key
        assert np.array_equal(out, member.reconstruct(clip)), spec.key
        assert set(member.submodule_sha256()) >= {"encoder", "decoder"}, spec.key


def test_member_refuses_non_native_length_and_bad_crop(monkeypatch):
    from dbp.decoders.loaders import MemberLoadError, load_video_member

    monkeypatch.setenv("DBP_DRYRUN", "1")
    member = load_video_member(load_video_bank(BANK_PATH).by_key("ltx090"), "cpu")
    with pytest.raises(MemberLoadError, match="not native"):
        member.reconstruct(np.zeros((24, 64, 64, 3), np.uint8))
    with pytest.raises(MemberLoadError, match="never resize"):
        member.reconstruct(np.zeros((25, 48, 64, 3), np.uint8))


# ------------------------------------------------------------------- video I/O ---

def test_probe_window_crop_and_png_round_trip(tmp_path):
    path = tmp_path / "v.mp4"
    make_mp4(path, frames=40)
    info = V.probe(path)
    assert (info["width"], info["height"], info["codec"], info["rotation"]) == (640, 360, "h264", 0)
    assert V.exclusion(info, 320) is None
    start = V.window_start("vid1", info["frames_reported"] or 40, 25)
    assert start == V.window_start("vid1", info["frames_reported"] or 40, 25)
    assert 0 <= start <= 40 - 25
    clip = V.centre_crop_clip(V.decode_window(path, start, 25), 320)
    assert clip.shape == (25, 320, 320, 3) and not V.is_static(clip)
    written = V.write_png_clip(clip, tmp_path / "clip")
    assert written["frames"] == 25
    assert np.array_equal(V.read_png_clip(tmp_path / "clip"), clip)
    with pytest.raises(V.VideoError, match="written once"):
        V.write_png_clip(clip, tmp_path / "clip")


def test_short_clip_small_frame_and_still_are_excluded(tmp_path):
    short, small, still = tmp_path / "s.mp4", tmp_path / "m.mp4", tmp_path / "t.mp4"
    make_mp4(short, frames=10)
    make_mp4(small, frames=30, size=(240, 320))
    make_mp4(still, frames=30, static=True)
    with pytest.raises(V.ShortClipError):
        V.decode_window(short, 0, 25)
    assert V.exclusion(V.probe(small), 320) == "short_side"
    with pytest.raises(V.VideoError, match="never resize"):
        V.centre_crop_clip(np.zeros((25, 240, 320, 3), np.uint8), 320)
    assert V.is_static(V.decode_window(still, 0, 25))
    assert V.exclusion({"rotation": 90, "pix_fmt": "yuv420p", "width": 640, "height": 360},
                       320) == "rotated"
    assert V.exclusion({"rotation": 0, "pix_fmt": "yuv420p10le", "width": 640, "height": 360},
                       320) == "not_8bit"


@pytest.mark.parametrize("codec", ["libx264", "libx265"])
def test_encode_roundtrip_keeps_frames_and_is_deterministic(codec):
    clip = np.random.default_rng(3).integers(0, 256, (25, 64, 64, 3), dtype=np.uint8)
    a, size_a = V.encode_roundtrip(clip, codec, 28, "30000/1001")
    b, size_b = V.encode_roundtrip(clip, codec, 28, "30000/1001")
    assert a.shape == clip.shape and size_a == size_b and np.array_equal(a, b)
    with pytest.raises(V.VideoError, match="only libx264"):
        V.encode_roundtrip(clip, "mpeg4", 28, 30)


# ------------------------------------------------------------------ Pexels fetch ---

def build_source(root: Path, per_shard: int = 3, shards: int = 3) -> None:
    """Shards laid out as the Hub repo is (videos/*.tar + a manifest), one bad video."""
    rows = []
    (root / "videos").mkdir(parents=True)
    for s in range(shards):
        name = f"5000_0_shard_{s:04d}.tar"
        with tarfile.open(root / "videos" / name, "w") as tar:
            for k in range(per_shard):
                sample = f"5000_0_{s}{k:02d}"
                tmp = root / f"{sample}.mp4"
                make_mp4(tmp, frames=30, size=(240, 320) if (s, k) == (0, 0) else (360, 640))
                tar.add(tmp, arcname=f"{sample}.mp4")
                tmp.unlink()
                rows.append({"sample_id": sample, "video_id": f"v{s}{k}", "shard": name})
    with (root / "manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "video_id", "shard"])
        writer.writeheader()
        writer.writerows(rows)


def test_fetch_reals_pools_are_disjoint_and_exclusions_recorded(tmp_path):
    import p5_fetch_reals as F

    names = [f"videos/s{i}.tar" for i in range(5)]
    assert F.shard_plan(names) == F.shard_plan(list(reversed(names)))
    source = tmp_path / "src"
    build_source(source)
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"), PYTHONUTF8="1")
    seen = {}
    for pool, shards, n in (("pilot", "0:1", 2), ("train", "1:3", 4)):
        reports = tmp_path / "reports"
        done = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "p5_fetch_reals.py"), "--source",
             str(source), "--pool", pool, "--shards", shards, "--n", str(n), "--out-root",
             str(tmp_path / "out"), "--cache", str(tmp_path / "cache"), "--reports",
             str(reports), "--workers", "1", "--skip-pins"],
            capture_output=True, text=True, env=env, check=False)
        assert done.returncode == 0, done.stdout + done.stderr
        summary = json.loads((reports / f"p5_reals_{pool}_summary.json").read_text())
        assert summary["kept"] == n and summary["complete"]
        with (reports / f"p5_reals_{pool}_manifest.csv").open(encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        seen[pool] = {r["shard"] for r in rows}
        for r in rows:
            if r["status"] == "kept":
                clip = V.read_png_clip(tmp_path / "out" / r["clip_dir"])
                assert clip.shape == (25, 320, 320, 3)
        assert all(r["exclusion"] for r in rows if r["status"] == "excluded")
    assert not seen["pilot"] & seen["train"]
    all_rows = []
    for pool in ("pilot", "train"):
        with (tmp_path / "reports" / f"p5_reals_{pool}_manifest.csv").open(encoding="utf-8") as h:
            all_rows += list(csv.DictReader(h))
    assert {r["exclusion"] for r in all_rows} >= {"short_side"}


# ---------------------------------------------------------------- video pins ---

def test_verify_video_pins_catches_a_changed_vae_file():
    import p0_verify_pins as P

    bank = load_video_bank(BANK_PATH)

    def api_with(sha_for: dict[str, str]):
        def model_info(repo, revision, token, files_metadata):
            spec = next(s for s in bank.members if s.repo == repo)
            lfs = SimpleNamespace(sha256=sha_for.get(spec.key, spec.vae_sha256))
            return SimpleNamespace(siblings=[SimpleNamespace(
                rfilename=f"{spec.subfolder}/{spec.vae_file}", size=spec.vae_bytes, lfs=lfs)])
        return SimpleNamespace(model_info=model_info)

    good = P.verify_video_pins(bank, api_with({}))
    assert [r.status for r in good] == [P.STATUS_RESOLVES] * 7
    bad = P.verify_video_pins(bank, api_with({"hunyuan": "0" * 64}))
    flagged = [r for r in bad if r.status == P.STATUS_HASH_MISMATCH]
    assert [r.key for r in flagged] == ["hunyuan"] and flagged[0].is_fatal


def test_pilot_expect_hashes_git_bytes_not_the_crlf_working_tree():
    """The first P5 dry-run refused at the pre-flight: the expectation held the sha256 of
    the Windows (CRLF) working copy, the code zip carries git's (LF) bytes."""
    import hashlib

    expect = json.loads((ROOT / "env" / "p5_pilot_expect.json").read_text(encoding="utf-8"))
    data = (ROOT / expect["thresholds"]["path"]).read_bytes().replace(b"\r\n", b"\n")
    assert hashlib.sha256(data).hexdigest() == expect["thresholds"]["sha256"]
    record = json.loads((ROOT / "env" / "p3_final_expect.json").read_text(encoding="utf-8"))
    committed = {m["id"]: m["expect"] for m in record["models"]}
    for model in expect["models"]:
        assert committed[model["id"]] == {"lora_sha256": model["lora_sha256"],
                                          "head_sha256": model["head_sha256"]}


def test_requirements_pin_av_only():
    text = (ROOT / "requirements" / "base.txt").read_text(encoding="utf-8")
    lines = [line.split("#")[0].strip() for line in text.splitlines()]
    assert "av==19.0.2" in lines and not any(line.startswith("pyarrow") for line in lines)


def test_encoder_inventory_reports_both_encoders():
    inventory = V.encoder_inventory()
    assert inventory["libx264"] and inventory["libx265"]
    assert inventory["pyav"] == av.__version__
