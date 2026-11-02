"""Pre-flight for a GPU stage: prove the run can work before anything long starts.

Every check here is something that has cost a Kaggle session when it was discovered
late instead: a module missing from the image (``spandrel`` left one GPU idle for 3.5
hours), an input attached at the wrong version (``gen_matched`` at 2,475 instead of
2,488), a list that does not resolve (I-044), a cache built for a different list.

Cheap checks run first and are all reported, so one run names every problem rather than
the first. The model smoke test runs last and only if the cheap checks passed: it loads
each named member and pushes N real images through the exact code path of the stage --
``p2_probe_features.py --calibrate N``, which writes nothing to the cache -- so a member
that cannot load fails here in minutes, not hours into a shard.

Exit status is 0 only if every requested check passed. A JSON record of each check is
written to ``reports/p2_preflight_<stage>.json``.

Usage:
    python scripts/p2_preflight.py --stage s3 --imports spandrel,lpips \\
        --smoke movq_k22,qwen_image --smoke-images 2 --smoke-list reports/x.csv ...
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import os
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path, PurePosixPath

THESIS_DBP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(THESIS_DBP / "src"))
sys.path.insert(0, str(THESIS_DBP / "scripts"))

from dbp.data.manifest import file_sha256  # noqa: E402
from dbp.utils.logging import setup_logging  # noqa: E402

#: Lines of a failing smoke run to print. Enough to show the traceback.
TAIL_LINES = 30


@dataclass
class Check:
    """One pre-flight check and its verdict."""

    name: str
    ok: bool
    detail: str
    seconds: float = 0.0
    data: dict = field(default_factory=dict)


def _timed(name: str, fn: Callable[[], tuple[bool, str]], data: dict | None = None) -> Check:
    """Run one check, turning an exception into a failed verdict rather than a crash.

    ``data`` is filled in by the check itself when it has more to record than one line.
    """
    started = time.perf_counter()
    try:
        ok, detail = fn()
    except Exception as exc:  # a check that cannot run has failed, and says why
        ok, detail = False, f"{type(exc).__name__}: {exc}"
    return Check(name, ok, detail, round(time.perf_counter() - started, 2), data or {})


def check_paths(required: dict[str, str]) -> Check:
    """Every named path exists."""
    def run() -> tuple[bool, str]:
        missing = {k: v for k, v in required.items() if not Path(v).exists()}
        if missing:
            return False, f"missing: {missing}"
        return True, f"{len(required)} present"
    return _timed("paths", run)


def check_imports(modules: list[str]) -> Check:
    """Every named module imports, with its version recorded."""
    def run() -> tuple[bool, str]:
        found: list[str] = []
        failed: list[str] = []
        for name in modules:
            try:
                module = importlib.import_module(name)
            except Exception as exc:
                failed.append(f"{name} ({type(exc).__name__}: {exc})")
                continue
            found.append(f"{name} {getattr(module, '__version__', '?')}")
        if failed:
            return False, f"cannot import: {failed}; ok: {found}"
        return True, ", ".join(found)
    return _timed("imports", run)


def check_gpus(expected: int) -> Check:
    """Exactly ``expected`` CUDA devices are visible (0 on a CPU dry-run)."""
    def run() -> tuple[bool, str]:
        import torch

        found = torch.cuda.device_count() if torch.cuda.is_available() else 0
        names = [torch.cuda.get_device_name(i) for i in range(found)]
        if found != expected:
            return False, f"{found} CUDA device(s) {names}, expected {expected}"
        return True, f"{found} CUDA device(s) {names}"
    return _timed("gpu count", run)


def check_hf_token() -> Check:
    """The HF token is present (environment or Kaggle Secrets). Its value is never shown."""
    def run() -> tuple[bool, str]:
        from dbp.utils.hf import get_hf_token

        present = get_hf_token() is not None
        return present, f"HF_TOKEN present: {str(present).lower()}"
    return _timed("hf token", run)


def check_sha256(expected: dict[str, str]) -> Check:
    """Every named input file hashes to its recorded sha256."""
    def run() -> tuple[bool, str]:
        wrong = {}
        for path, want in expected.items():
            if not Path(path).is_file():
                wrong[Path(path).name] = "missing"
            elif (got := file_sha256(path)) != want:
                wrong[Path(path).name] = f"{got[:16]}... != {want[:16]}..."
        if wrong:
            return False, f"inputs differ from the recorded hashes: {wrong}"
        return True, f"{len(expected)} input file(s) match their sha256"
    return _timed("input hashes", run)


def count_rows(path: Path) -> int:
    """Data rows in a CSV."""
    with path.open(encoding="utf-8", newline="") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def check_rows(expected: dict[str, int], base: Path) -> Check:
    """Each CSV has exactly the expected number of rows."""
    def run() -> tuple[bool, str]:
        wrong = {}
        for name, want in expected.items():
            got = count_rows(base / name)
            if got != want:
                wrong[name] = f"{got} (expected {want})"
        if wrong:
            return False, f"row counts differ: {wrong}"
        return True, ", ".join(f"{Path(k).name}={v}" for k, v in expected.items())
    return _timed("row counts", run)


def check_list_digest(
    list_path: Path, expected: str | None, sidecar_dir: Path | None, expect_sidecars: int | None
) -> Check:
    """The image list hashes to the expected digest, and every cache was built on it."""
    import p2_probe_features as probe

    def run() -> tuple[bool, str]:
        digest = probe.image_list_digest(probe.read_image_list(list_path))
        problems: list[str] = []
        if expected and digest != expected:
            problems.append(f"digest {digest} != expected {expected}")
        sidecars: list[Path] = []
        if sidecar_dir is not None:
            sidecars = sorted(sidecar_dir.glob("*.done.json"))
            stale = [
                s.name for s in sidecars
                if json.loads(s.read_text(encoding="utf-8")).get("images_sha256") != digest
            ]
            if stale:
                problems.append(f"sidecars built on a different list: {stale}")
            if expect_sidecars is not None and len(sidecars) != expect_sidecars:
                problems.append(f"{len(sidecars)} sidecar(s), expected {expect_sidecars}")
        if problems:
            return False, "; ".join(problems)
        return True, f"digest {digest[:16]}..., {len(sidecars)} sidecar(s) agree"
    return _timed("list digest", run)


def check_resolution(list_path: Path, root: Path, panel_root: Path | None) -> Check:
    """Every row of the image list resolves to an existing file."""
    import p2_probe_features as probe

    def run() -> tuple[bool, str]:
        rows = probe.read_image_list(list_path)
        roots = probe.resolve_roots(root, panel_root)
        unresolved: list[str] = []
        for row in rows:
            try:
                path = probe.row_path(row, roots)
            except KeyError as exc:
                unresolved.append(f"{row['image_id']} ({exc})")
                continue
            if not path.is_file():
                unresolved.append(row["relative_path"])
        if unresolved:
            return False, f"{len(unresolved)} of {len(rows)} unresolved, e.g. {unresolved[:5]}"
        return True, f"all {len(rows)} rows resolve"
    return _timed("list resolves", run)


def check_gen_matched(reference: Path, directory: Path, expected: int) -> Check:
    """Every regenerated file matches the ``jpg_sha256`` the reference manifest records.

    The reference must be the committed manifest, not the one the matcher just wrote:
    comparing a file with the hash its own writer recorded proves nothing. The members
    already restored were computed on the original files, so a single differing byte
    means the remaining members would be computed on different images.
    """
    def run() -> tuple[bool, str]:
        with reference.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        problems: list[str] = []
        if len(rows) != expected:
            problems.append(f"reference has {len(rows)} rows, expected {expected}")
        on_disk = {p.name for p in directory.glob("*.jpg")}
        if len(on_disk) != expected:
            problems.append(f"{len(on_disk)} .jpg on disk, expected {expected}")
        missing, differing = [], []
        for row in rows:
            name = PurePosixPath(str(row["jpg_path"]).replace("\\", "/")).name
            path = directory / name
            if not path.is_file():
                missing.append(name)
            elif file_sha256(path) != row["jpg_sha256"]:
                differing.append(name)
        if missing:
            problems.append(f"{len(missing)} missing, e.g. {missing[:3]}")
        if differing:
            problems.append(f"{len(differing)} differ from the reference, e.g. {differing[:3]}")
        if problems:
            return False, "; ".join(problems)
        return True, f"all {len(rows)} files match the reference sha256"
    return _timed("gen_matched hashes", run)


def check_caches(probe_dir: Path, list_path: Path, members: list[str]) -> Check:
    """Every member's probe cache is complete, uncorrupted and built on this list.

    Per member: the sidecar's sha256 and size match the file; the cache holds exactly
    the list's rows, in the list's order, built on the list's digest; every value is
    finite; the feature names are the stage's. The per-member sha256 and the mean PSNR
    and LPIPS go into the record, so the log can cite them from a command's output.
    """
    import numpy as np

    import p2_probe_features as probe
    from dbp.streams.s3_probe import FEATURE_NAMES

    data: dict = {}

    def run() -> tuple[bool, str]:
        rows = probe.read_image_list(list_path)
        digest = probe.image_list_digest(rows)
        ids = np.array([str(r["image_id"]) for r in rows])
        data.update({"list_rows": len(rows), "list_digest": digest, "members": {}})
        problems: list[str] = []
        for member in members:
            npz = probe_dir / f"p2_probe_{member}.npz"
            side_path = probe_dir / f"p2_probe_{member}.done.json"
            if not npz.is_file() or not side_path.is_file():
                problems.append(f"{member}: cache or sidecar missing")
                continue
            side = json.loads(side_path.read_text(encoding="utf-8"))
            sha = file_sha256(npz)
            with np.load(npz, allow_pickle=False) as blob:
                features = blob["features"]
                cache_ids = blob["image_ids"].astype(str)
                names = [str(n) for n in blob["feature_names"]]
            wrong = [
                label for label, good in (
                    ("sidecar sha256", side.get("sha256") == sha),
                    ("sidecar bytes", side.get("bytes") == npz.stat().st_size),
                    ("rows", side.get("rows") == len(rows) == features.shape[0]),
                    ("list digest", side.get("images_sha256") == digest),
                    ("image order",
                     cache_ids.shape == ids.shape and bool((cache_ids == ids).all())),
                    ("finite", bool(np.isfinite(features).all())),
                    ("feature names", names == list(FEATURE_NAMES)),
                ) if not good
            ]
            if wrong:
                problems.append(f"{member}: {', '.join(wrong)}")
            record = {"sha256": sha, "bytes": npz.stat().st_size,
                      "rows": int(features.shape[0]), "ok": not wrong}
            if "feature names" not in wrong:
                record["mean_psnr_db"] = round(
                    float(features[:, names.index("psnr_db")].mean()), 2)
                record["mean_lpips_vgg"] = round(
                    float(features[:, names.index("lpips_vgg")].mean()), 4)
            data["members"][member] = record
        if problems:
            return False, "; ".join(problems)
        return True, (f"{len(members)} cache(s) complete on {len(rows)} rows, "
                      f"digest {digest[:16]}...")
    return _timed("probe caches", run, data)


def smoke_member(
    member: str, images: int, list_path: Path, root: Path, panel_root: Path | None,
    device: str, cache_dir: Path | None, worker: Path, timeout: int,
) -> Check:
    """Load one member and run ``images`` real images through the stage's code path."""
    command = [
        sys.executable, "-u", str(worker),
        "--images", str(list_path), "--root", str(root),
        "--device", device, "--members", member, "--calibrate", str(images),
    ]
    if panel_root is not None:
        command += ["--panel-root", str(panel_root)]
    if cache_dir is not None:
        command += ["--cache-dir", str(cache_dir)]

    def run() -> tuple[bool, str]:
        env = dict(os.environ, PYTHONUNBUFFERED="1")
        result = subprocess.run(
            command, capture_output=True, text=True, timeout=timeout, env=env,
            cwd=THESIS_DBP,
        )
        output = (result.stdout + result.stderr).strip().splitlines()
        if result.returncode != 0:
            tail = "\n".join(output[-TAIL_LINES:])
            print(f"\n----- smoke {member}: last {TAIL_LINES} lines -----\n{tail}\n", flush=True)
            return False, f"exit {result.returncode}: {output[-1] if output else '(no output)'}"
        timing = next((line.split("|", 1)[-1].strip() for line in output
                       if "image(s) in" in line), "completed")
        return True, timing
    return _timed(f"smoke {member}", run)


def parse_pairs(items: list[str], flag: str) -> dict[str, str]:
    """Parse repeated ``key=value`` arguments."""
    out: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"--{flag} expects key=value, got {item!r}")
        key, value = item.split("=", 1)
        out[key.strip()] = value.strip()
    return out


def main() -> int:
    """Run the requested checks. Returns 0 only if all of them passed."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--stage", required=True, help="Names the JSON record.")
    parser.add_argument("--require", action="append", default=[], metavar="LABEL=PATH")
    parser.add_argument("--imports", default="", help="Comma-separated module names.")
    parser.add_argument("--rows", action="append", default=[], metavar="CSV=N")
    parser.add_argument("--list", type=Path, default=None, help="Image list to digest.")
    parser.add_argument("--list-digest", default=None)
    parser.add_argument("--sidecars", type=Path, default=None)
    parser.add_argument("--expect-sidecars", type=int, default=None)
    parser.add_argument("--resolve", type=Path, action="append", default=[],
                        help="Image list to resolve; repeatable.")
    parser.add_argument("--expect-gpus", type=int, default=None)
    parser.add_argument("--hf-token", action="store_true", help="Require HF_TOKEN.")
    parser.add_argument("--sha256", action="append", default=[], metavar="PATH=HASH")
    parser.add_argument("--root", type=Path, default=None, help="Corpus root.")
    parser.add_argument("--panel-root", type=Path, default=None)
    parser.add_argument("--gen-matched-reference", type=Path, default=None)
    parser.add_argument("--gen-matched-dir", type=Path, default=None)
    parser.add_argument("--expect-gen-matched", type=int, default=None)
    parser.add_argument("--verify-caches", type=Path, default=None,
                        help="Probe cache directory to verify against --list.")
    parser.add_argument("--members", default="", help="Members --verify-caches expects.")
    parser.add_argument("--smoke", default="", help="Comma-separated members to smoke-test.")
    parser.add_argument("--smoke-images", type=int, default=2)
    parser.add_argument("--smoke-list", type=Path, default=None)
    parser.add_argument("--smoke-timeout", type=int, default=1200, help="Seconds per member.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--worker", type=Path,
                        default=THESIS_DBP / "scripts" / "p2_probe_features.py",
                        help="The stage script the smoke test drives (a test seam).")
    parser.add_argument("--reports", type=Path, default=THESIS_DBP / "reports")
    args = parser.parse_args()
    log = setup_logging("p2_preflight")

    checks: list[Check] = []
    try:
        if args.require:
            checks.append(check_paths(parse_pairs(args.require, "require")))
        if args.rows:
            expected = {k: int(v) for k, v in parse_pairs(args.rows, "rows").items()}
            checks.append(check_rows(expected, Path.cwd()))
    except ValueError as exc:
        log.error("%s", exc)
        return 1
    if args.expect_gpus is not None:
        checks.append(check_gpus(args.expect_gpus))
    if args.hf_token:
        checks.append(check_hf_token())
    if args.sha256:
        bad = [item for item in args.sha256 if "=" not in item]
        if bad:
            log.error("--sha256 expects PATH=HASH, got %s", bad)
            return 1
        checks.append(check_sha256(dict(item.rsplit("=", 1) for item in args.sha256)))
    if args.imports:
        checks.append(check_imports([m.strip() for m in args.imports.split(",") if m.strip()]))
    if args.list is not None:
        checks.append(check_list_digest(args.list, args.list_digest, args.sidecars,
                                        args.expect_sidecars))
    if args.verify_caches is not None:
        wanted = [m.strip() for m in args.members.split(",") if m.strip()]
        if args.list is None or not wanted:
            log.error("--verify-caches needs --list and --members")
            return 1
        checks.append(check_caches(args.verify_caches, args.list, wanted))
    if args.resolve:
        if args.root is None:
            log.error("--resolve needs --root")
            return 1
        for list_path in args.resolve:
            check = check_resolution(list_path, args.root, args.panel_root)
            check.name = f"resolves {Path(list_path).name}"
            checks.append(check)
    if args.gen_matched_reference is not None:
        if args.gen_matched_dir is None or args.expect_gen_matched is None:
            log.error("--gen-matched-reference needs --gen-matched-dir and --expect-gen-matched")
            return 1
        checks.append(check_gen_matched(args.gen_matched_reference, args.gen_matched_dir,
                                        args.expect_gen_matched))

    members = [m.strip() for m in args.smoke.split(",") if m.strip()]
    if members:
        if checks and not all(c.ok for c in checks):
            for member in members:
                checks.append(Check(f"smoke {member}", False,
                                    "skipped: an earlier check failed"))
        elif args.smoke_list is None or args.root is None:
            log.error("--smoke needs --smoke-list and --root")
            return 1
        else:
            for member in members:
                log.info("smoke-testing %s on %d image(s) ...", member, args.smoke_images)
                checks.append(smoke_member(
                    member, args.smoke_images, args.smoke_list, args.root, args.panel_root,
                    args.device, args.cache_dir, args.worker, args.smoke_timeout,
                ))

    if not checks:
        log.error("no checks requested")
        return 1

    width = max(len(c.name) for c in checks)
    for c in checks:
        log.info("%s  %-*s  %6.1fs  %s", "PASS" if c.ok else "FAIL", width, c.name,
                 c.seconds, c.detail)
    passed = all(c.ok for c in checks)
    record = {"stage": args.stage, "passed": passed, "checks": [asdict(c) for c in checks]}
    args.reports.mkdir(parents=True, exist_ok=True)
    out = args.reports / f"p2_preflight_{args.stage}.json"
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    tmp.replace(out)
    if passed:
        log.info("PRE-FLIGHT PASSED (%s): %d check(s)", args.stage, len(checks))
        return 0
    log.error("PRE-FLIGHT FAILED (%s): %d of %d check(s) failed -- do not start the run",
              args.stage, sum(not c.ok for c in checks), len(checks))
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
