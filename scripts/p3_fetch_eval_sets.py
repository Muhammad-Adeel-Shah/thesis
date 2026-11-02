"""P3 Phase B: fetch the evaluation sets Panel B is drawn from, exactly as P1 did.

Every source, revision, archive and extraction root is P1's (``env/kaggle_run_p1_eval.md``
Steps 2–6; the roots are those recorded in the committed P1 summaries,
``reports/kaggle_20260928/p1_eval_*_summary.json``). The same roots matter: a rebuilt
manifest records paths, so an identical root is what lets its sha256 be compared with
P1's.

| set | source | root |
|---|---|---|
| E3 | HF dataset `zr-zhang/MLLM-Generated-Image-Detection-Dataset` @ `1498eead…` | `<tmp>/e3` |
| E4 | Google Drive `1A0xoL44Yg68ixd-FuIJn2VC4vdZ6M2gn` via gdown (§6a(c)) | `<tmp>/wildrf/WildRF` |
| E5 | the attached private Kaggle dataset `chameleon-dataset` | as attached |
| E7 | HF `Junwei-Xi/DDA-COCO` @ `8c9330a3…` + COCO `val2017` reals | `<tmp>/e7x/DDA-COCO` |
| E8 | HF dataset `Junwei-Xi/EvalGEN` @ `86c719a9…` (`EvalGEN.zip`) | `<tmp>/e8x` |

Archives are deleted after extraction (disk). Downloads go to ``<tmp>`` (``/tmp`` on
Kaggle), never to ``/kaggle/working`` and never to the local machine (§6a(a)).

**E7's reals are not in the archive.** DDA-COCO ships the six VAE reconstructions only;
P1 placed MSCOCO ``val2017`` (the attached ``awsaf49/coco-2017-dataset``) under
``DDA-COCO/real/`` (``docs/data_cards/e7_dda_coco.md``, "Measured on ingestion").
``--coco-val DIR`` repeats that step: it copies every ``*.jpg`` byte for byte into
``<root>/real/``, refuses if the count is not ``--coco-val-count`` or if ``real/`` already
holds files, and is required whenever E7 is in ``--sets``. The first Panel B session
(2026-10-11) lacked it and the view check stopped the run: 0 reals, 500 P2 rows missing.

``--local NAME=DIR`` uses an existing directory as a set's root instead of fetching it —
the dry-run fixture, and E5's attached dataset. The result is ``roots.json``: set ->
root, which the view builder reads.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

THESIS_DBP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(THESIS_DBP / "src"))

from dbp.utils.logging import setup_logging  # noqa: E402

#: P1's sources, verbatim from env/kaggle_run_p1_eval.md (read 2026-09-20).
HF_SOURCES = {
    "e3": {"repo_id": "zr-zhang/MLLM-Generated-Image-Detection-Dataset",
           "revision": "1498eead24292a9a2d134476b2c559193b68b9de",
           "download_to": "e3", "zip": None, "root": "e3"},
    "e7": {"repo_id": "Junwei-Xi/DDA-COCO",
           "revision": "8c9330a3b374bcac46a8045a0e3c09ebcf7868fb",
           "download_to": "e7", "zip": "DDA-COCO.zip", "extract_to": "e7x",
           "root": "e7x/DDA-COCO"},
    "e8": {"repo_id": "Junwei-Xi/EvalGEN",
           "revision": "86c719a9bfc72b7e5d91bfbb04432774feb4e3b7",
           "download_to": "e8", "zip": "EvalGEN.zip", "extract_to": "e8x", "root": "e8x"},
}
GDOWN_SOURCES = {
    "e4": {"file_id": "1A0xoL44Yg68ixd-FuIJn2VC4vdZ6M2gn", "download_to": "wildrf",
           "archive": "wildrf.zip", "root": "wildrf/WildRF"},
}
ALL_SETS = ("e3", "e4", "e5", "e7", "e8")
#: COCO val2017 holds 5,000 images; P1's E7 manifest has 5,000 reals.
COCO_VAL_COUNT = 5000


class FetchError(RuntimeError):
    """A set could not be fetched or extracted."""


def extract(archive: Path, destination: Path, log) -> None:
    """Extract a zip, refusing members that would escape the destination."""
    destination.mkdir(parents=True, exist_ok=True)
    base = destination.resolve()
    with zipfile.ZipFile(archive) as handle:
        for member in handle.namelist():
            target = (destination / member).resolve()
            if not target.is_relative_to(base):
                raise FetchError(f"{archive.name}: member {member!r} escapes {destination}")
        handle.extractall(destination)
    log.info("extracted %s -> %s", archive.name, destination)


def fetch_hf(name: str, tmp: Path, token: str | None, log) -> Path:
    """Snapshot a pinned HF dataset revision, extract its zip if any, delete the zip."""
    from huggingface_hub import snapshot_download

    spec = HF_SOURCES[name]
    local = tmp / spec["download_to"]
    snapshot_download(repo_id=spec["repo_id"], repo_type="dataset",
                      revision=spec["revision"], local_dir=str(local), token=token)
    log.info("%s: %s @ %s -> %s", name, spec["repo_id"], spec["revision"][:12], local)
    if spec["zip"]:
        archive = local / spec["zip"]
        if not archive.is_file():
            raise FetchError(f"{name}: {spec['zip']} not in the snapshot")
        extract(archive, tmp / spec["extract_to"], log)
        archive.unlink()  # disk: E8's archive alone is 13.5 GiB
    return tmp / spec["root"]


def fetch_gdown(name: str, tmp: Path, log) -> Path:
    """Fetch a Google Drive archive with gdown (installed by the bootstrap's opt-in)."""
    spec = GDOWN_SOURCES[name]
    folder = tmp / spec["download_to"]
    folder.mkdir(parents=True, exist_ok=True)
    archive = folder / spec["archive"]
    if not archive.is_file():
        if shutil.which("gdown") is None:
            raise FetchError("gdown is not installed: run the bootstrap with DBP_WITH_GDOWN=1")
        subprocess.run(["gdown", "--id", spec["file_id"], "-O", str(archive)], check=True)
    log.info("%s: Drive %s -> %s (%d bytes)", name, spec["file_id"], archive,
             archive.stat().st_size)
    extract(archive, folder, log)
    archive.unlink()
    return tmp / spec["root"]


def add_e7_reals(root: Path, coco_val: Path, expected: int, log) -> int:
    """Copy COCO ``val2017`` into ``<root>/real/`` as P1 did. Returns the count copied."""
    if not coco_val.is_dir():
        raise FetchError(f"e7: COCO val2017 directory {coco_val} does not exist")
    images = sorted(p for p in coco_val.glob("*.jpg") if p.is_file())
    if len(images) != expected:
        raise FetchError(f"e7: {coco_val} holds {len(images)} jpg(s), expected {expected}")
    real = root / "real"
    if real.is_dir() and any(real.iterdir()):
        raise FetchError(f"e7: {real} already holds files; refusing to mix in COCO val2017")
    real.mkdir(parents=True, exist_ok=True)
    for image in images:
        shutil.copyfile(image, real / image.name)
    log.info("e7: %d COCO val2017 real(s) %s -> %s", len(images), coco_val, real)
    return len(images)


def main() -> int:
    """Fetch the requested sets and write roots.json. Returns 0 on success."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--tmp", required=True, type=Path, help="/tmp on Kaggle.")
    parser.add_argument("--sets", default=",".join(ALL_SETS))
    parser.add_argument("--local", action="append", default=[], metavar="SET=DIR",
                        help="Use DIR as the set's root instead of fetching (E5 always).")
    parser.add_argument("--roots-out", required=True, type=Path)
    parser.add_argument("--coco-val", type=Path, default=None,
                        help="COCO val2017 images: E7's real class (required with e7).")
    parser.add_argument("--coco-val-count", type=int, default=COCO_VAL_COUNT)
    args = parser.parse_args()
    log = setup_logging("p3_fetch_eval_sets")
    import os

    token = os.environ.get("HF_TOKEN")
    log.info("HF_TOKEN present: %s", token is not None)
    local = {}
    for item in args.local:
        name, _, path = item.partition("=")
        local[name.strip()] = Path(path.strip())
    roots: dict[str, str] = {}
    try:
        for name in [s.strip() for s in args.sets.split(",") if s.strip()]:
            if name not in ALL_SETS:
                raise FetchError(f"unknown set {name!r}")
            if name in local:
                root = local[name]
            elif name in HF_SOURCES:
                root = fetch_hf(name, args.tmp, token, log)
            elif name in GDOWN_SOURCES:
                root = fetch_gdown(name, args.tmp, log)
            else:
                raise FetchError(f"{name}: no fetch route; pass --local {name}=DIR")
            if not root.is_dir():
                raise FetchError(f"{name}: root {root} does not exist after the fetch")
            if name == "e7":
                if args.coco_val is None:
                    raise FetchError("e7: --coco-val is required; DDA-COCO ships no reals")
                add_e7_reals(root, args.coco_val, args.coco_val_count, log)
            roots[name] = str(root)
            log.info("%s root: %s", name, root)
    except (FetchError, subprocess.CalledProcessError, OSError) as exc:
        log.error("FETCH FAILED: %s", exc)
        return 1
    args.roots_out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.roots_out.with_suffix(".tmp")
    tmp.write_text(json.dumps(roots, indent=2) + "\n", encoding="utf-8")
    tmp.replace(args.roots_out)
    log.info("roots -> %s", args.roots_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
