"""scripts/prepush_audit.py: images, secrets and per-image files, across history.

Each case builds a throwaway git repository. Fake tokens are assembled at run time, so
this file itself never holds a string the audit would flag.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64
JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 64


def _audit_module():
    spec = importlib.util.spec_from_file_location("_prepush", ROOT / "scripts" / "prepush_audit.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["_prepush"] = module
    spec.loader.exec_module(module)
    return module


AUDIT = _audit_module()


def _repo(tmp_path: Path, commits: list[dict[str, bytes]]) -> Path:
    """A repository with one commit per mapping (path -> bytes; None deletes)."""
    repo = tmp_path / "repo"
    repo.mkdir()
    run = lambda *a: subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *a],  # noqa: E731
                                    cwd=repo, check=True, capture_output=True)
    run("init", "-q")
    for number, files in enumerate(commits):
        for path, data in files.items():
            target = repo / path
            if data is None:
                target.unlink()
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        run("add", "-A")
        run("commit", "-q", "-m", f"c{number}")
    return repo


def test_charts_under_figures_pass_and_a_clean_repo_is_ok(tmp_path: Path) -> None:
    repo = _repo(tmp_path, [{"reports/p3/figures/fig.png": PNG,
                             "reports/p3/figures/fig.svg": b"<svg/>",
                             "src/a.py": b"x = 1\n"}])
    record = AUDIT.audit(repo, "HEAD")
    assert record["ok"] and len(record["a_images"]["files"]) == 2
    assert record["a_images"]["violations"] == []


@pytest.mark.parametrize(("path", "data"), [
    ("reports/panel/x.jpg", JPEG),                    # a dataset image
    ("reports/p3/x.png", PNG),                        # a PNG outside figures/
    ("reports/p3/figures/big.png", PNG + b"\x00" * 1_000_000),   # a chart over 1 MB
    ("reports/p3/figures/scan.csv", JPEG),            # a JPEG under another name
    ("data/clip.mp4", b"\x00" * 16),                  # video
], ids=["dataset-jpeg", "png-outside-figures", "chart-over-1MB", "renamed-jpeg", "video"])
def test_an_image_that_is_not_a_chart_fails(tmp_path: Path, path: str, data: bytes) -> None:
    repo = _repo(tmp_path, [{path: data}])
    record = AUDIT.audit(repo, "HEAD")
    assert not record["ok"]
    assert [v["path"] for v in record["a_images"]["violations"]] == [path]


def test_an_image_deleted_later_is_still_caught_in_history(tmp_path: Path) -> None:
    repo = _repo(tmp_path, [{"reports/panel/x.jpg": JPEG}, {"reports/panel/x.jpg": None,
                                                            "a.txt": b"a"}])
    record = AUDIT.audit(repo, "HEAD")
    assert not record["ok"] and record["commits"] == 2


@pytest.mark.parametrize("token", [
    "hf_" + "A1b2C3d4" * 5, "ghp_" + "Z9y8X7w6" * 5, "AKIA" + "ABCDEFGH" * 2,
    "-----BEGIN " + "OPENSSH PRIVATE KEY-----",
])
def test_a_token_in_any_file_fails_and_is_redacted(tmp_path: Path, token: str) -> None:
    repo = _repo(tmp_path, [{"env/run.py": f"TOKEN = '{token}'\n".encode()}])
    record = AUDIT.audit(repo, "HEAD")
    hits = record["b_secrets"]["content_hits"]
    assert not record["ok"] and len(hits) == 1
    assert token not in hits[0]["match"] and hits[0]["match"].endswith("(redacted)")


@pytest.mark.parametrize("name", [".env", "kaggle.json", "deploy.pem", "hf_token.txt"])
def test_a_risky_file_name_fails(tmp_path: Path, name: str) -> None:
    repo = _repo(tmp_path, [{name: b"nothing\n"}])
    record = AUDIT.audit(repo, "HEAD")
    assert not record["ok"] and record["b_secrets"]["risky_names"] == [name]


def test_per_image_csvs_are_summarised_not_judged(tmp_path: Path) -> None:
    repo = _repo(tmp_path, [{
        "reports/p3_final_scores/m/panel_b.csv": b"image_id,score\na,0.1\nb,0.2\n",
        "reports/lists/panel_b.csv": b"image_id,relative_path,sha256\na,x,y\n",
        "reports/summary.csv": b"model,auc\nm,0.9\n"}])
    record = AUDIT.audit(repo, "HEAD")
    per_image = record["c_per_image"]
    assert record["ok"] and per_image["files"] == 2 and per_image["rows"] == 3
    kinds = {i["path"]: i["kind"] for i in per_image["list"]}
    assert kinds == {"reports/p3_final_scores/m/panel_b.csv": "scores",
                     "reports/lists/panel_b.csv": "manifest/list"}


def test_the_exit_code_is_nonzero_on_a_violation_and_2_outside_a_repo(tmp_path: Path) -> None:
    bad = _repo(tmp_path, [{"reports/panel/x.jpg": JPEG}])
    script = ROOT / "scripts" / "prepush_audit.py"
    done = subprocess.run([sys.executable, str(script), "--repo", str(bad), "--ref", "HEAD"],
                          capture_output=True, text=True, check=False)
    assert done.returncode == 1 and "PRE-PUSH AUDIT: FAILED" in done.stdout
    nowhere = tmp_path / "plain"
    nowhere.mkdir()
    done = subprocess.run([sys.executable, str(script), "--repo", str(nowhere)],
                          capture_output=True, text=True, check=False)
    assert done.returncode == 2 and "CANNOT RUN" in done.stdout


def test_a_leading_thesis_dbp_folder_is_the_same_rule(tmp_path: Path) -> None:
    repo = _repo(tmp_path, [{"thesis_dbp/reports/p3/figures/fig.png": PNG}])
    assert AUDIT.audit(repo, "HEAD")["ok"]
