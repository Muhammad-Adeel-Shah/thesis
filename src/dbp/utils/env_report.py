"""Collect the full environment fingerprint required for reproducibility.

Everything the thesis needs in order to explain a number later lives in the JSON this
module produces: OS, Python, torch/CUDA/cuDNN, GPU and driver, every relevant package
version, the pinned Hub revisions, the Real-ESRGAN weight hash, and the Pillow/libjpeg
versions (JPEG encoder differences change augmentation artefacts, [R17]).

No secret is ever included. Only ``hf_token_present: true|false``.
"""

from __future__ import annotations

import importlib.metadata as importlib_metadata
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import tempfile
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dbp.utils.hf import token_present

TRACKED_PACKAGES = (
    "torch",
    "torchvision",
    "numpy",
    "diffusers",
    "transformers",
    "peft",
    "accelerate",
    "safetensors",
    "huggingface_hub",
    "spandrel",
    "lpips",
    "scikit-learn",
    "Pillow",
    "pyyaml",
    "pandas",
    "tqdm",
    "matplotlib",
    "scikit-image",
    "pytest",
    "ruff",
    "einops",
)

ALLOWED_HOSTS = (
    "https://pypi.org/simple/",
    "https://huggingface.co/api/models/stabilityai/sdxl-vae",
    "https://github.com",
)


def _version_or_none(pkg: str) -> str | None:
    try:
        return importlib_metadata.version(pkg)
    except importlib_metadata.PackageNotFoundError:
        return None


def collect_packages() -> dict[str, str | None]:
    """Version of every package P0 depends on, or None when absent."""
    return {pkg: _version_or_none(pkg) for pkg in TRACKED_PACKAGES}


def collect_python() -> dict[str, Any]:
    """Interpreter, platform and virtual-environment state."""
    return {
        "version": sys.version.split()[0],
        "executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "in_venv": sys.prefix != sys.base_prefix,
        "venv_prefix": sys.prefix if sys.prefix != sys.base_prefix else None,
        "conda_prefix": os.environ.get("CONDA_PREFIX"),
    }


def collect_torch() -> dict[str, Any]:
    """torch / torchvision build details and CUDA device properties."""
    try:
        import torch
    except ImportError:
        return {"available": False}

    info: dict[str, Any] = {
        "available": True,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "cuda_is_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count(),
        "devices": [],
    }
    try:
        import torchvision

        info["torchvision_version"] = torchvision.__version__
    except ImportError:
        info["torchvision_version"] = None

    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        major, minor = torch.cuda.get_device_capability(i)
        info["devices"].append(
            {
                "index": i,
                "name": props.name,
                "compute_capability": f"{major}.{minor}",
                "sm": major * 10 + minor,
                "total_memory_mb": round(props.total_memory / 1024**2),
            }
        )
    return info


def collect_nvidia_smi() -> dict[str, Any]:
    """GPU name, driver and memory via nvidia-smi, or a reason it is unavailable."""
    if shutil.which("nvidia-smi") is None:
        return {"available": False, "reason": "nvidia-smi not found on PATH"}
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        )
    except (subprocess.SubprocessError, OSError) as exc:
        return {"available": False, "reason": f"{type(exc).__name__}: {exc}"}
    gpus = [line.strip() for line in out.stdout.splitlines() if line.strip()]
    return {"available": True, "gpus": gpus}


def collect_disk(paths: dict[str, str | Path]) -> dict[str, Any]:
    """Free and total space for each named path."""
    result: dict[str, Any] = {}
    for label, path in paths.items():
        try:
            usage = shutil.disk_usage(str(path))
            result[label] = {
                "path": str(path),
                "total_gb": round(usage.total / 1e9, 1),
                "free_gb": round(usage.free / 1e9, 1),
            }
        except OSError as exc:
            result[label] = {"path": str(path), "error": f"{type(exc).__name__}: {exc}"}
    return result


def collect_ram_gb() -> float | None:
    """Total physical RAM in GB, or None if it cannot be determined."""
    if sys.platform == "win32":
        try:
            import ctypes

            class MemoryStatusEx(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            stat = MemoryStatusEx()
            stat.dwLength = ctypes.sizeof(MemoryStatusEx)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
            return round(stat.ullTotalPhys / 1e9, 1)
        except (AttributeError, OSError):
            return None
    try:
        return round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9, 1)
    except (AttributeError, ValueError, OSError):
        return None


def collect_network(timeout: float = 12.0) -> dict[str, str]:
    """Reachability of the allow-listed hosts. SSL verification is never disabled."""
    previous = socket.getdefaulttimeout()
    socket.setdefaulttimeout(timeout)
    results: dict[str, str] = {}
    try:
        for url in ALLOWED_HOSTS:
            host = url.split("/")[2]
            try:
                with urllib.request.urlopen(url) as response:  # noqa: S310
                    results[host] = f"OK {response.status}"
            except Exception as exc:
                results[host] = f"FAIL {type(exc).__name__}"
    finally:
        socket.setdefaulttimeout(previous)
    return results


def collect_pillow() -> dict[str, Any]:
    """Pillow and its codec versions. libjpeg matters: it changes JPEG artefacts [R17]."""
    try:
        import PIL
        import PIL.features
    except ImportError:
        return {"available": False}
    return {
        "available": True,
        "pillow": PIL.__version__,
        "libjpeg": PIL.features.version("jpg"),
        "zlib": PIL.features.version("zlib"),
        "libpng": PIL.features.version("png"),
    }


def read_commit_file(repo_root: Path) -> str | None:
    """Read the full commit hash from ``COMMIT.txt``, if one was shipped.

    ``git archive --add-virtual-file`` embeds this when the code is packaged for a
    runtime that has no ``.git`` directory, such as a Kaggle dataset upload. Without it
    a Kaggle report would record ``git: unavailable`` and the numbers in it could not be
    tied back to the code that produced them.

    Looked for at ``REPO_ROOT/COMMIT.txt`` and ``REPO_ROOT/thesis_dbp/COMMIT.txt``,
    since the archive prefix determines which one exists.
    """
    for candidate in (repo_root / "COMMIT.txt", repo_root / "thesis_dbp" / "COMMIT.txt"):
        if not candidate.is_file():
            continue
        value = candidate.read_text(encoding="utf-8").strip()
        if len(value) == 40:
            try:
                int(value, 16)
            except ValueError:
                continue
            return value
    return None


def collect_git(repo_root: Path) -> dict[str, Any]:
    """Current branch, commit and dirty state, or a reason it is unavailable."""

    def run(*args: str) -> str | None:
        try:
            out = subprocess.run(
                ["git", "-C", str(repo_root), *args],
                capture_output=True,
                text=True,
                timeout=30,
                check=True,
            )
            return out.stdout.strip()
        except (subprocess.SubprocessError, OSError):
            return None

    commit = run("rev-parse", "HEAD")
    if commit is None:
        # No .git here. Fall back to the COMMIT.txt embedded at packaging time, so a
        # Kaggle run can still be tied to the exact code that produced its numbers.
        from_file = read_commit_file(repo_root)
        if from_file is not None:
            return {
                "available": True,
                "commit": from_file,
                "branch": None,
                "dirty": None,
                "source": "COMMIT.txt",
            }
        return {
            "available": False,
            "reason": "no .git directory and no COMMIT.txt",
            "source": None,
        }
    status = run("status", "--porcelain")
    return {
        "available": True,
        "commit": commit,
        "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(status),
        "source": "git",
    }


def collect_runtime() -> dict[str, Any]:
    """Detect whether this is a Kaggle session and where the caches point."""
    is_kaggle = bool(os.environ.get("KAGGLE_KERNEL_RUN_TYPE")) or Path("/kaggle/working").exists()
    return {
        "is_kaggle": is_kaggle,
        "kaggle_kernel_run_type": os.environ.get("KAGGLE_KERNEL_RUN_TYPE"),
        "hf_home": os.environ.get("HF_HOME"),
        "torch_home": os.environ.get("TORCH_HOME"),
    }


def build_report(repo_root: Path, bank_config: Path | None = None) -> dict[str, Any]:
    """Assemble the full environment report. Contains no secrets."""
    dbp_dir = repo_root / "thesis_dbp"
    disk_targets: dict[str, str | Path] = {
        "repo_root": repo_root,
        "hf_home": os.environ.get("HF_HOME") or (dbp_dir / ".cache" / "hf"),
        "tmp": os.environ.get("TMPDIR") or os.environ.get("TEMP") or "/tmp",
    }

    report: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "runtime": collect_runtime(),
        "python": collect_python(),
        "torch": collect_torch(),
        "nvidia_smi": collect_nvidia_smi(),
        "packages": collect_packages(),
        "pillow": collect_pillow(),
        "disk": collect_disk(disk_targets),
        "ram_gb": collect_ram_gb(),
        "network": collect_network(),
        "hf_token_present": token_present(),
        "git": collect_git(repo_root),
    }
    # Promoted to the top level so every consumer reads the commit from one place,
    # whether it came from a live .git or from a packaged COMMIT.txt.
    report["git_commit"] = report["git"].get("commit")
    report["git_commit_source"] = report["git"].get("source")

    if bank_config is not None and bank_config.exists():
        from dbp.decoders.registry import load_bank_config

        bank = load_bank_config(bank_config)
        report["bank"] = {
            "schema_version": bank.schema_version,
            "member_count": len(bank.members),
            "pinned_revisions": {
                spec.key: spec.revision for spec in bank.members if spec.revision is not None
            },
            "pinned_sha256": {
                spec.key: spec.sha256 for spec in bank.members if spec.kind == "sr_gan"
            },
        }
    return report


def write_report(report: dict[str, Any], dest: Path) -> Path:
    """Write the report as UTF-8 JSON, atomically."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(dest.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(report, handle, indent=2)
            handle.write("\n")
        os.replace(tmp, dest)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return dest
