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
    assert bank.diffusers == "0.42.0" and {s.diffusers for s in bank.members} == {"0.42.0"}
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


@pytest.mark.parametrize("same_device", [True, False])
def test_move_clip_dir_works_across_devices(tmp_path, monkeypatch, same_device):
    """Kaggle, 2026-11-21: os.replace('/tmp/...', '/kaggle/working/...') raised EXDEV
    (Invalid cross-device link) and stopped the pilot's fetch after 4 shards."""
    import errno

    real_replace = os.replace

    def replace(src, dst):
        if Path(src).parent != Path(dst).parent and not same_device:
            raise OSError(errno.EXDEV, "Invalid cross-device link")
        return real_replace(src, dst)

    monkeypatch.setattr(V, "_same_device", lambda a, b: same_device)
    monkeypatch.setattr(V.os, "replace", replace)
    clip = np.random.default_rng(5).integers(0, 256, (3, 8, 8, 3), dtype=np.uint8)
    src = tmp_path / "tmp_device" / "unpacked" / "c1"
    V.write_png_clip(clip, src)
    dst = tmp_path / "working" / "pilot" / "c1"
    V.move_clip_dir(src, dst)
    assert np.array_equal(V.read_png_clip(dst), clip)
    assert not src.exists() and not dst.with_name("c1.tmp").exists()
    with pytest.raises(V.VideoError, match="written once"):
        V.move_clip_dir(dst, dst)


def test_fetch_never_renames_a_clip_across_directories_directly():
    text = (ROOT / "scripts" / "p5_fetch_reals.py").read_text(encoding="utf-8")
    assert ".replace(final)" not in text and "move_clip_dir(" in text


def test_h264_roundtrip_keeps_frames_and_is_byte_deterministic():
    clip = np.random.default_rng(3).integers(0, 256, (25, 64, 64, 3), dtype=np.uint8)
    runs = [V.encode_roundtrip(clip, "libx264", 28, "30000/1001") for _ in range(5)]
    a, size_a = runs[0]
    assert a.shape == clip.shape
    assert all(size == size_a and np.array_equal(frames, a) for frames, size in runs[1:])
    with pytest.raises(V.VideoError, match="only libx264"):
        V.encode_roundtrip(clip, "mpeg4", 28, 30)


def test_h265_roundtrip_checks_settings_and_frames_not_bytes():
    """User decision 2026-11-24 (option a): libx265 is NOT byte-deterministic (3 outputs in
    10 encodes, 2026-11-23), so each H.265-degraded clip is encoded once and hashed. This
    test checks what IS guaranteed: identical settings, the frame count and the shape."""
    assert V.X265_BYTE_DETERMINISTIC is False
    clip = np.random.default_rng(3).integers(0, 256, (25, 64, 64, 3), dtype=np.uint8)
    settings = V.encoder_options("libx265", 28)
    assert settings == V.encoder_options("libx265", 28)              # one pair, one setting
    assert settings["crf"] == "28" and "frame-threads=1" in settings["x265-params"]
    for _ in range(3):
        frames, size = V.encode_roundtrip(clip, "libx265", 28, "30000/1001")
        assert frames.shape == clip.shape and frames.dtype == np.uint8 and size > 0
    with pytest.raises(V.VideoError, match="only libx264"):
        V.encoder_options("mpeg4", 28)


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


@pytest.mark.parametrize("phase", ["smoke", "reconstruct"])
def test_vae_phases_run_deterministic_true_fp32(monkeypatch, phase):
    """Kaggle pilot 2026-11-21: cogvx's smoke was 1 grey level off between two runs
    ('FAILED: non-deterministic'). The VAE phases never applied the determinism contract
    every image data-building script applies (seed_everything(2026, deterministic=True)).
    On the A10G, cuDNN would also use TF32 for convolutions unless switched off."""
    import torch

    import p5_pilot as P

    saved = (torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic,
             torch.are_deterministic_algorithms_enabled(),
             torch.is_deterministic_algorithms_warn_only_enabled(),
             torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32)
    seen = {}

    def capture(args, bank, clips):
        seen.update(deterministic=torch.backends.cudnn.deterministic,
                    benchmark=torch.backends.cudnn.benchmark,
                    algorithms=torch.are_deterministic_algorithms_enabled(),
                    cudnn_tf32=torch.backends.cudnn.allow_tf32,
                    matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
                    recorded=args.numerics)

    torch.backends.cudnn.deterministic, torch.backends.cudnn.allow_tf32 = False, True
    monkeypatch.setattr(P, f"phase_{phase}", capture)
    monkeypatch.setattr(P, "kept_clips", lambda manifest: [{"sample_id": "x"}])
    monkeypatch.setattr(sys, "argv", ["p5_pilot.py", "--phase", phase, "--expect",
                                      str(ROOT / "env" / "p5_pilot_expect.json")])
    try:
        assert P.main() == 0
    finally:
        (torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic, algorithms,
         warn_only, torch.backends.cudnn.allow_tf32,
         torch.backends.cuda.matmul.allow_tf32) = saved
        torch.use_deterministic_algorithms(algorithms, warn_only=warn_only)
    assert seen["deterministic"] and not seen["benchmark"] and seen["algorithms"]
    assert not seen["cudnn_tf32"] and not seen["matmul_tf32"]
    assert seen["recorded"]["seed"] == 2026 and seen["recorded"]["cudnn_allow_tf32"] is False


def test_p5_splits_are_fixed_sized_seeded_and_order_free():
    ids = [str(10_000_000 + 7 * i) for i in range(2200)]
    split = V.p5_splits(ids)
    counts = {name: sum(v == name for v in split.values()) for name in set(split.values())}
    assert counts == {"threshold": 200, "validation": 200, "train": 1800}
    assert V.p5_splits(list(reversed(ids))) == split            # the id set decides, not order
    with pytest.raises(ValueError, match="duplicate"):
        V.p5_splits(ids + ids[:1])
    with pytest.raises(ValueError, match="need more than"):
        V.p5_splits(ids[:400])


PILOT = ROOT / "reports" / "kaggle_20261121" / "p5_pilot"
PASSING_SMOKE = {"status": "OK", "fp16_build_allowed": True, "fp16_reason": "ok",
                 "fp16_determinism_max_abs_delta": 0}


@pytest.mark.parametrize("key, smoke, expected", [
    ("wan21", PASSING_SMOKE, "fp16"),
    ("ltx090", PASSING_SMOKE, "fp16"),
    ("sd15_pf", PASSING_SMOKE, "fp16"),
    ("mochi", PASSING_SMOKE, "fp32"),     # user, 2026-11-22: its pilot fp16 check failed
    ("cogvx", PASSING_SMOKE, "fp32"),     # user, 2026-11-22: fp32 for determinism
    ("hunyuan", PASSING_SMOKE, "fp32"),   # no pilot check (OOM on the T4), no A10G record
    # user, 2026-11-23: hunyuan's fp16 check is the A10G smoke's, same rule
    ("hunyuan", {**PASSING_SMOKE, "gpu_name": "NVIDIA A10G"}, "fp16"),
    ("hunyuan", {**PASSING_SMOKE, "gpu_name": "Tesla T4"}, "fp32"),
    ("hunyuan", {**PASSING_SMOKE, "gpu_name": "NVIDIA A10G", "fp16_build_allowed": False},
     "fp32"),
    ("hunyuan", {**PASSING_SMOKE, "gpu_name": "NVIDIA A10G",
                 "fp16_determinism_max_abs_delta": 2}, "fp32"),
    ("mochi", {**PASSING_SMOKE, "gpu_name": "NVIDIA A10G"}, "fp32"),
    ("wan21", {**PASSING_SMOKE, "fp16_build_allowed": False}, "fp32"),
    ("wan21", {**PASSING_SMOKE, "fp16_determinism_max_abs_delta": 1}, "fp32"),
    ("wan21", {**PASSING_SMOKE, "fp16_determinism_max_abs_delta": None}, "fp32"),
])
def test_recon_precision_follows_the_pilot_and_this_smoke(key, smoke, expected):
    import p5_pilot as P

    assert P.recon_precision(key, PILOT, smoke)[0] == expected
    if key not in P.FP16_CHECK_ON_A10G:
        assert P.recon_precision(key, None, smoke)[0] == "fp32"


def test_reconstruct_skips_threshold_reals_and_never_mixes_precision(tmp_path, monkeypatch):
    import dbp.decoders.loaders as L
    import p5_pilot as P

    monkeypatch.setattr(V, "P5_SPLIT_SIZES", (("threshold", 2), ("validation", 2)))
    clips = []
    for i in range(7):
        frames = np.full((25, 8, 8, 3), 10 * i, dtype=np.uint8)
        V.write_png_clip(frames, tmp_path / "reals" / f"c{i}")
        clips.append({"sample_id": f"s{i}", "video_id": f"v{i}", "clip_dir": f"c{i}"})
    built = []

    class Fake:
        def __init__(self, precision):
            self.precision = precision

        def reconstruct(self, clip):
            built.append(self.precision)
            return clip

        def free(self):
            return None

    monkeypatch.setattr(L, "load_video_member", lambda spec, device, precision: Fake(precision))
    reports = tmp_path / "reports"
    reports.mkdir()
    (reports / "smoke_wan21.json").write_text(json.dumps(PASSING_SMOKE), encoding="utf-8")
    bank = SimpleNamespace(by_key=lambda key: SimpleNamespace(key=key))
    args = SimpleNamespace(members="wan21", device="cpu", reports=reports,
                           reals_root=tmp_path / "reals", recon_root=tmp_path / "recon",
                           splits="train,validation", fp16_evidence=PILOT)
    P.phase_reconstruct(args, bank, clips)
    split = {r.split(",")[0]: r.split(",")[2] for r in
             (reports / "p5_train_splits.csv").read_text(encoding="utf-8").splitlines()[1:]}
    rows = [json.loads(x) for x in (reports / "recon_wan21.jsonl").read_text().splitlines()]
    assert sorted(r["sample_id"] for r in rows) == sorted(
        s for s, v in split.items() if v in ("train", "validation"))
    assert len(rows) == 5 and not any(split[r["sample_id"]] == "threshold" for r in rows)
    assert {r["precision"] for r in rows} == {"fp16"} and set(built) == {"fp16"}
    # a resume in another precision is refused, not mixed
    args.fp16_evidence = None
    with pytest.raises(SystemExit, match="two precisions"):
        P.phase_reconstruct(args, bank, clips)


def test_requirements_pin_av_only():
    text = (ROOT / "requirements" / "base.txt").read_text(encoding="utf-8")
    lines = [line.split("#")[0].strip() for line in text.splitlines()]
    assert "av==19.0.2" in lines and not any(line.startswith("pyarrow") for line in lines)


def test_one_diffusers_version_everywhere(monkeypatch, tmp_path):
    """User, 2026-11-21: one diffusers for the pilot, AWS and all Kaggle scoring; the loader
    refuses any other version."""
    import diffusers

    from dbp.decoders.loaders import MemberLoadError, load_video_member

    text = (ROOT / "requirements" / "base.txt").read_text(encoding="utf-8")
    lines = [line.split("#")[0].strip() for line in text.splitlines()]
    bank = load_video_bank(BANK_PATH)
    assert f"diffusers=={bank.diffusers}" in lines
    monkeypatch.setenv("DBP_DRYRUN", "1")
    monkeypatch.setattr(diffusers, "__version__", "0.41.9")
    with pytest.raises(MemberLoadError, match="pinned to 0.42.0"):
        load_video_member(bank.by_key("wan21"), "cpu")
    raw = yaml.safe_load(BANK_PATH.read_text(encoding="utf-8"))
    raw["diffusers"] = ">=0.42"
    path = tmp_path / "bank.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(BankConfigError, match="exact X.Y.Z"):
        load_video_bank(path)


def test_encoder_inventory_reports_both_encoders():
    inventory = V.encoder_inventory()
    assert inventory["libx264"] and inventory["libx265"]
    assert inventory["pyav"] == av.__version__
