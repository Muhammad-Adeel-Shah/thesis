"""Video I/O for P5: probe, the pre-declared clip window, crop, lossless frames, re-encode.

The clip protocol (approval A5, 2026-11-21):

- **Native frames only.** ``frames`` consecutive decoded frames, at the stream's own
  frame rate; no frame is dropped, duplicated or interpolated.
- **Crop, never resize.** A centre ``crop x crop`` window, identical in every frame.
- **The window start** is a hash of the clip's id (:func:`window_start`), inside the first
  :data:`START_HORIZON` frames, so the decode cost is bounded and the choice cannot be
  tuned.
- **Pre-declared exclusions** (:func:`exclusion`, :data:`EXCLUSIONS`), applied before any
  model sees a clip.
- **Lossless storage.** Frames are PNG (:func:`write_png_clip`), named ``f000.png`` ...
- **One path for both members of a pair.** Real and reconstruction share decode and
  crop; :func:`encode_roundtrip` is the only re-encode, applied with identical settings
  to both clips of a pair (training degradation) or to reals and fakes alike (the
  format-matched test view, on Kaggle only).

Frames are RGB ``uint8`` from PyAV's ``to_ndarray(format="rgb24")``; nothing here uses
OpenCV.
"""

from __future__ import annotations

import hashlib
import io
import os
import shutil
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np

#: The window starts within the first this-many frames (5 s at 30 fps).
START_HORIZON = 150
#: Decoded pixel formats accepted: 8-bit only (no HDR / 10-bit tone-mapping question).
EIGHT_BIT_FORMATS = frozenset({"yuv420p", "yuvj420p", "yuv422p", "yuvj422p", "yuv444p",
                               "yuvj444p", "nv12", "nv21", "rgb24", "bgr24", "gbrp"})
#: Every reason a clip can be excluded, in the order they are tested.
EXCLUSIONS = ("decode_error", "no_video_stream", "rotated", "not_8bit", "short_side",
              "too_few_frames", "static")


class VideoError(RuntimeError):
    """A clip that cannot be read or written as the protocol requires."""


class ShortClipError(VideoError):
    """The stream ended before the window did (exclusion ``too_few_frames``)."""


def probe(path: Path) -> dict[str, Any]:
    """Container and stream facts of a video file (no frames are kept).

    Returns:
        ``codec, profile, pix_fmt, width, height, fps`` (float), ``fps_fraction`` (str),
        ``frames_reported`` (0 when the container does not say), ``duration_s``,
        ``bit_rate``, ``rotation`` (degrees, 0 when none), ``format`` and ``bytes``.

    Raises:
        VideoError: If the file cannot be opened or has no video stream.
    """
    import av

    try:
        with av.open(str(path)) as container:
            if not container.streams.video:
                raise VideoError(f"{path}: no video stream")
            stream = container.streams.video[0]
            ctx = stream.codec_context
            rate = stream.average_rate or stream.guessed_rate
            rotation = 0
            for frame in container.decode(stream):
                rotation = int(getattr(frame, "rotation", 0) or 0)
                break
            duration = (float(stream.duration * stream.time_base)
                        if stream.duration is not None and stream.time_base else None)
            return {
                "codec": ctx.name, "profile": ctx.profile or "", "pix_fmt": ctx.pix_fmt or "",
                "width": int(ctx.width), "height": int(ctx.height),
                "fps": float(rate) if rate else 0.0, "fps_fraction": str(rate) if rate else "",
                "frames_reported": int(stream.frames or 0), "duration_s": duration,
                "bit_rate": int(ctx.bit_rate or container.bit_rate or 0),
                "rotation": rotation, "format": container.format.name,
                "bytes": Path(path).stat().st_size,
            }
    except VideoError:
        raise
    except Exception as exc:
        raise VideoError(f"{path}: cannot probe ({type(exc).__name__}: {exc})") from exc


def window_start(clip_id: str, frames_reported: int, frames: int, seed: int = 2026) -> int:
    """The pre-declared window start: a hash of ``seed:clip_id``.

    The start lies in ``[0, min(frames_reported, START_HORIZON) - frames]``; when the
    container reports no frame count, the start is 0.
    """
    span = min(frames_reported, START_HORIZON) - frames + 1 if frames_reported else 1
    digest = hashlib.sha256(f"{seed}:{clip_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % max(1, span)


def decode_window(path: Path, start: int, frames: int) -> np.ndarray:
    """Decode ``frames`` consecutive frames from index ``start``: ``(T, H, W, 3)`` uint8.

    Decoding is sequential from the first frame, so the index is the true frame number.

    Raises:
        VideoError: On a decode error, or if the stream ends before the window does.
    """
    import av

    out: list[np.ndarray] = []
    try:
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"
            for index, frame in enumerate(container.decode(stream)):
                if index >= start:
                    out.append(frame.to_ndarray(format="rgb24"))
                if len(out) == frames:
                    break
    except Exception as exc:
        raise VideoError(f"{path}: decode failed ({type(exc).__name__}: {exc})") from exc
    if len(out) != frames:
        raise ShortClipError(f"{path}: {len(out)} frames from {start}, need {frames}")
    return np.stack(out)


def centre_crop_clip(frames: np.ndarray, size: int) -> np.ndarray:
    """The centre ``size x size`` window of every frame. Never resizes.

    Raises:
        VideoError: If either side is below ``size``.
    """
    height, width = frames.shape[1:3]
    if min(height, width) < size:
        raise VideoError(f"{height}x{width} is below the {size} crop; crop, never resize")
    top, left = (height - size) // 2, (width - size) // 2
    return np.ascontiguousarray(frames[:, top: top + size, left: left + size])


#: A window is a still when no frame differs from frame 0 by a mean absolute difference of
#: this many grey levels or more. Not exact equality: a still encoded as H.264 decodes with
#: small P-frame differences (measured on a synthetic still, test_p5_video.py).
STATIC_MAD = 1.0


def is_static(frames: np.ndarray) -> bool:
    """True when the window is a still: every frame within :data:`STATIC_MAD` of frame 0."""
    first = frames[0].astype(np.int16)
    return all(float(np.mean(np.abs(f.astype(np.int16) - first))) < STATIC_MAD
               for f in frames[1:])


def exclusion(info: dict[str, Any] | None, min_short_side: int) -> str | None:
    """The first container-level exclusion a probed clip meets, or None.

    ``too_few_frames`` and ``static`` need the decoded window and are decided by the
    caller with :func:`decode_window` / :func:`is_static`.
    """
    if info is None:
        return "decode_error"
    if info.get("rotation", 0) not in (0, None):
        return "rotated"
    if info.get("pix_fmt") not in EIGHT_BIT_FORMATS:
        return "not_8bit"
    if min(int(info["width"]), int(info["height"])) < min_short_side:
        return "short_side"
    return None


def png_sha256(path: Path) -> str:
    """SHA-256 of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_png_clip(frames: np.ndarray, out_dir: Path) -> dict[str, Any]:
    """Write ``(T, H, W, 3)`` uint8 frames as ``f000.png`` ... atomically (temp dir, rename).

    Returns:
        ``frames``, ``bytes`` and ``digest`` (sha256 over the frames' sha256s in order).

    Raises:
        VideoError: If ``out_dir`` already exists (a clip is written once).
    """
    from PIL import Image

    if frames.dtype != np.uint8 or frames.ndim != 4 or frames.shape[3] != 3:
        raise VideoError(f"write_png_clip expects (T, H, W, 3) uint8, got {frames.shape}")
    if out_dir.exists():
        raise VideoError(f"{out_dir} exists; a clip is written once")
    tmp = out_dir.with_name(out_dir.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    hashes, size = [], 0
    for index, frame in enumerate(frames):
        path = tmp / f"f{index:03d}.png"
        Image.fromarray(frame, "RGB").save(path, format="PNG", compress_level=6)
        hashes.append(png_sha256(path))
        size += path.stat().st_size
    os.replace(tmp, out_dir)
    return {"frames": len(hashes), "bytes": size, "digest": clip_digest(hashes)}


def move_clip_dir(src: Path, dst: Path) -> None:
    """Move a finished clip directory to ``dst``, atomically at the destination.

    Same filesystem: one ``os.replace``. Across filesystems (Kaggle's ``/tmp`` and
    ``/kaggle/working`` are different devices; ``os.replace`` fails there with EXDEV, which
    stopped the first P5 pilot session): copy to a temp sibling of ``dst`` on the target
    device, ``os.replace`` it into place, then delete ``src``.

    Raises:
        VideoError: If ``dst`` already exists (a clip is written once).
    """
    if dst.exists():
        raise VideoError(f"{dst} exists; a clip is written once")
    dst.parent.mkdir(parents=True, exist_ok=True)
    if _same_device(src, dst.parent):
        os.replace(src, dst)
        return
    tmp = dst.with_name(dst.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    shutil.copytree(src, tmp)
    os.replace(tmp, dst)
    shutil.rmtree(src)


def _same_device(a: Path, b: Path) -> bool:
    return os.stat(a).st_dev == os.stat(b).st_dev


def clip_digest(frame_hashes: list[str]) -> str:
    """One digest for a clip: sha256 over its frames' sha256s, in frame order."""
    return hashlib.sha256("\n".join(frame_hashes).encode()).hexdigest()


#: The training pool's split (docs/P5_PLAN.md: 1,800 train + 200 validation + 200
#: threshold-held-out). Fixed 2026-11-22, before any training-pool clip was reconstructed.
P5_SPLIT_SALT = "2026:p5-split:"
P5_SPLIT_SIZES: tuple[tuple[str, int], ...] = (("threshold", 200), ("validation", 200))


def p5_splits(video_ids: list[str]) -> dict[str, str]:
    """Each training-pool video id -> ``threshold`` | ``validation`` | ``train``.

    The ids are ordered by ``sha256(P5_SPLIT_SALT + id)``, a salt distinct from the one that
    chose the kept clips, so the split cannot follow the selection order. The first 200 are
    the threshold-held-out reals (k = 0, never reconstructed: they are only ever used as
    reals), the next 200 are validation, the rest train. Depends on the id set only.

    Raises:
        ValueError: duplicate ids, or too few ids to fill the fixed splits.
    """
    if len(set(video_ids)) != len(video_ids):
        raise ValueError("p5_splits: duplicate video ids (one clip per video)")
    fixed = sum(n for _, n in P5_SPLIT_SIZES)
    if len(video_ids) <= fixed:
        raise ValueError(f"p5_splits: {len(video_ids)} ids, need more than {fixed}")
    order = sorted(video_ids,
                   key=lambda v: hashlib.sha256((P5_SPLIT_SALT + v).encode()).hexdigest())
    out: dict[str, str] = {}
    start = 0
    for name, n in P5_SPLIT_SIZES:
        out.update(dict.fromkeys(order[start:start + n], name))
        start += n
    out.update(dict.fromkeys(order[start:], "train"))
    return out


def read_png_clip(clip_dir: Path) -> np.ndarray:
    """Read a clip written by :func:`write_png_clip`: ``(T, H, W, 3)`` uint8.

    Raises:
        VideoError: If the directory holds no ``f*.png`` frames or a gap in the numbering.
    """
    from PIL import Image

    paths = sorted(clip_dir.glob("f[0-9][0-9][0-9].png"))
    if not paths or [p.name for p in paths] != [f"f{i:03d}.png" for i in range(len(paths))]:
        raise VideoError(f"{clip_dir}: frames missing or not numbered f000..")
    frames = []
    for path in paths:
        with Image.open(path) as img:
            if img.mode != "RGB":
                raise VideoError(f"{path}: mode {img.mode}, expected RGB")
            frames.append(np.asarray(img, dtype=np.uint8))
    return np.stack(frames)


#: libx265's output is NOT byte-deterministic, even single-threaded: 10 encodes of one
#: clip in one process gave 3 different outputs (2026-11-23). User decision (2026-11-24,
#: option a): every H.265-degraded clip is encoded ONCE, written and hashed, and the hashes
#: are the record; both clips of a pair get identical settings (:func:`encoder_options`).
#: H.264 (libx264) stays byte-checked. A thesis limitation.
X265_BYTE_DETERMINISTIC = False


def encoder_options(codec: str, crf: int, preset: str = "medium") -> dict[str, str]:
    """The complete encoder options for one re-encode: the settings two clips of a pair
    share, and the record of what was applied.

    Raises:
        VideoError: For a codec other than libx264 / libx265.
    """
    if codec not in ("libx264", "libx265"):
        raise VideoError(f"codec {codec!r}: only libx264 and libx265 are used")
    options = {"crf": str(int(crf)), "preset": preset}
    if codec == "libx265":
        # single-threaded: it narrows x265's run-to-run variation but does NOT remove it
        # (X265_BYTE_DETERMINISTIC); kept so every encode uses the same settings
        options["x265-params"] = ("log-level=error:pools=none:frame-threads=1:no-wpp=1:"
                                  "lookahead-slices=0")
    return options


def encode_roundtrip(frames: np.ndarray, codec: str, crf: int, fps: float | str,
                     preset: str = "medium") -> tuple[np.ndarray, int]:
    """Encode ``(T, H, W, 3)`` uint8 frames with ``libx264``/``libx265`` and decode them back.

    The settings are the arguments and nothing else (``yuv420p``, mp4 container, x264/x265
    defaults otherwise), so applying the same call to two clips treats them identically.

    Returns:
        The decoded ``(T, H, W, 3)`` uint8 frames and the encoded size in bytes.

    Raises:
        VideoError: For an unknown codec, or if the decode does not return T frames.
    """
    import av

    options = encoder_options(codec, crf, preset)
    rate = Fraction(str(fps)).limit_denominator(1001)
    buffer = io.BytesIO()
    with av.open(buffer, mode="w", format="mp4") as container:
        stream = container.add_stream(codec, rate=rate)
        stream.width, stream.height = int(frames.shape[2]), int(frames.shape[1])
        stream.pix_fmt = "yuv420p"
        stream.options = options
        for frame in frames:
            for packet in stream.encode(av.VideoFrame.from_ndarray(frame, format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    data = buffer.getvalue()
    with av.open(io.BytesIO(data)) as container:
        decoded = [f.to_ndarray(format="rgb24") for f in container.decode(video=0)]
    if len(decoded) != len(frames):
        raise VideoError(f"{codec} crf {crf}: {len(decoded)} frames back from {len(frames)}")
    return np.stack(decoded), len(data)


def encoder_inventory() -> dict[str, Any]:
    """PyAV and FFmpeg versions and whether the libx264/libx265 encoders exist (A3)."""
    import av

    available = set(av.codecs_available)
    return {
        "pyav": av.__version__,
        "ffmpeg_libraries": {k: ".".join(map(str, v)) for k, v in av.library_versions.items()},
        "libx264": "libx264" in available, "libx265": "libx265" in available,
    }
