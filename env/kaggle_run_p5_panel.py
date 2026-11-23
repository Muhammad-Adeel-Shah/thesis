# %% [markdown]
# # P5 test panel: AIGVDBench draw + fetch, AEGIS, duplicates, metadata gate (freeze prep)
#
# `docs/P5_PLAN.md` sections 1.2 and 3.3; the draw rule of 2026-11-23/24 (the user's
# principles). **Test data never leaves Kaggle**: the kept clips stay in this notebook's
# output, which becomes a PRIVATE dataset; only the small reports archive is pasted back.
#
# 1. **Pre-flight**: runtime, token, imports, disk, both dataset revisions, the committed
#    listing reference, the pins (CLAUDE.md 7a, 12.2).
# 2. **List** every AIGVDBench zip from its central directory (exact range reads) and
#    refuse unless each listing equals the committed `aigvdbench_listing_summary.json`.
# 3. **Draw** (`Split/test.jsonl` x Real.zip, grouped by source video): calibration 300,
#    closed-source reals 300, open-source pool 900; 150 per slice.
# 4. **Fetch** by range reads, CRC-checked; every clip examined (probe, exclusions, the
#    seeded 17-frame window, centre crop 320, keyframe pHashes 1/8/16); open-source slices
#    keep content-matched pairs (the real and that zip's fake of the same name).
# 5. **AEGIS** hard test: the tar and the mapping, whole; every video examined.
# 6. **Finalize** (cannot fail): manifest, exact + near duplicates, the metadata-only gate
#    per slice, the panel digest, the archive. Then the verdict cell.
#
# **Kaggle:** Accelerator **None** (CPU), Internet **on**, Secret `HF_TOKEN`, Save & Run
# All (Commit), account `muhammadadeelshah`. Input: dataset `dbp-code*` (the code at the
# hand-over commit). Paste back: `p5_panel_reports.zip` and the last cell's text. Then
# publish the output (`p5_panel/` clips) as a PRIVATE dataset `dbp-p5-panel`.

# %% Cell 1 — find the input by its contents, and the helpers every later cell uses
import json
import os
import pathlib
import platform
import shutil
import subprocess
import sys
import time

START = time.time()
IN = pathlib.Path(os.environ.get("DBP_INPUT_ROOT", "/kaggle/input"))
WORK = pathlib.Path(os.environ.get("DBP_WORKING_ROOT", "/kaggle/working"))
TMP = pathlib.Path(os.environ.get("DBP_TMP_ROOT", "/tmp"))


def one(label, hits):
    """Exactly one match, or stop with what was found."""
    hits = sorted(set(hits))
    assert len(hits) == 1, f"expected exactly one {label}, found {hits}"
    return hits[0]


CODE = one("code dataset", IN.glob("datasets/*/dbp-code*/thesis_dbp"))
DBP = WORK / "Implementation" / "thesis_dbp"
CLIPS = WORK / "p5_panel"
REPORTS = DBP / "reports" / "p5_panel"
LISTING = TMP / "p5_panel_listing_full.json"
CACHE = TMP / "p5_panel_cache"
for key in ("HF_HOME", "TORCH_HOME", "PIP_CACHE_DIR"):
    os.environ[key] = str(TMP / key.lower())
    os.makedirs(os.environ[key], exist_ok=True)


def sh(*command):
    """Run a command in the staged repo, streaming its output into this cell."""
    print("$", " ".join(str(c) for c in command), flush=True)
    with subprocess.Popen([str(c) for c in command], cwd=DBP, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True, bufsize=1) as proc:
        for line in proc.stdout:
            print(line, end="", flush=True)
    return proc.returncode


def py(script, *args):
    """Run one committed script."""
    return sh(sys.executable, "-u", f"scripts/{script}", *args)


def hardware(path):
    """Print and save the host: CPU model and cores, GPUs, driver."""
    cpu = platform.processor() or "unknown"
    info = pathlib.Path("/proc/cpuinfo")
    if info.is_file():
        names = [line.split(":", 1)[1].strip() for line in info.read_text().splitlines()
                 if line.startswith("model name")]
        cpu = names[0] if names else cpu
    lines = [f"CPU: {cpu} | logical cores: {os.cpu_count()}"]
    smi = shutil.which("nvidia-smi")
    lines.append("nvidia-smi: present" if smi else "nvidia-smi: not found (CPU session)")
    print("\n".join(lines))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


print("code", CODE)

# %% Cell 2 — stage the code; hardware; the environment (bootstrap only)
if DBP.exists():
    shutil.rmtree(DBP)
shutil.copytree(CODE, DBP)
assert (DBP / "scripts" / "p5_fetch_panel.py").is_file(), "the code dataset predates this runbook"
print("code commit:", (DBP / "COMMIT.txt").read_text().split()[0])
hardware(REPORTS / "hardware.txt")
EXPECT_PATH = pathlib.Path(os.environ.get("DBP_EXPECT", DBP / "env" / "p5_panel_expect.json"))
EXPECT = json.loads(EXPECT_PATH.read_text())
print("expected:", json.dumps(EXPECT["expected"]))
assert sh("bash", DBP / "env" / "kaggle_bootstrap.sh") == 0, "bootstrap FAILED"
DRY = ["--skip-pins"] if os.environ.get("DBP_DRYRUN") == "1" else []
SOURCES = ["--source", EXPECT.get("aigv_source", "hub"),
           "--aegis-source", EXPECT.get("aegis_source", "hub")]
COMMON = ["--expect", EXPECT_PATH, "--reports", REPORTS, *SOURCES]

# %% Cell 3 — PRE-FLIGHT: runtime, token, imports; disk, dataset revisions, reference, pins
checks = ["--expect-gpus", EXPECT["gpus"], "--runtime", "--imports", ",".join(EXPECT["imports"])]
checks += ["--hf-token"] if EXPECT["hf_token"] else []
assert py("p2_preflight.py", "--stage", "p5_panel_env", *checks) == 0, \
    "PRE-FLIGHT (environment) FAILED -- nothing has run"
assert py("p5_fetch_panel.py", "--phase", "preflight", *COMMON,
          "--disk", f"{WORK}={EXPECT['disk']['working_gb']}",
          "--disk", f"{TMP}={EXPECT['disk']['tmp_gb']}", *DRY) == 0, \
    "PRE-FLIGHT (panel) FAILED -- nothing has run"

# %% Cell 4 — list every zip from its directory; refuse unless it equals the committed one
assert py("p5_fetch_panel.py", "--phase", "list", *SOURCES, "--full", "--out", LISTING,
          "--reference", DBP / EXPECT["listing_reference"]) == 0, \
    "the AIGVDBench listing differs from the committed reference -- nothing was drawn"

# %% Cell 5 — the draw: test split x Real.zip, by source video; pools and slice orders
assert py("p5_fetch_panel.py", "--phase", "draw", *COMMON, "--listing", LISTING,
          "--cache", CACHE) == 0, "draw FAILED"

# %% Cell 6 — fetch + examine: pool reals, then every slice (content-matched open pairs)
assert py("p5_fetch_panel.py", "--phase", "fetch", *COMMON, "--listing", LISTING,
          "--out-root", CLIPS, "--cache", CACHE, *DRY) == 0, "fetch FAILED"

# %% Cell 7 — AEGIS hard test: tar and mapping, every video examined
assert py("p5_fetch_panel.py", "--phase", "aegis", *COMMON, "--out-root", CLIPS,
          "--cache", CACHE, *DRY) == 0, "AEGIS FAILED"

# %% Cell 8 — report: manifest, duplicates, metadata gate, digest, archive (cannot fail)
py("p5_fetch_panel.py", "--phase", "finalize", *COMMON, "--zip", WORK / "p5_panel_reports")
print(f"wall: {(time.time() - START) / 3600:.2f} h")

# %% Cell 9 — verdict: this cell fails the notebook unless the panel is complete
summary = json.loads((REPORTS / "p5_panel_summary.json").read_text())
assert "error" not in summary, f"panel INCOMPLETE: {summary['error']}"
print("P5 PANEL READY FOR FREEZE REVIEW:", json.dumps({
    "panel_digest": summary["panel_digest"], "pools": summary["counts"]["pools"],
    "slices": {z.split("/")[-1]: s["kept"] for z, s in summary["counts"]["slices"].items()},
    "gate_flags": len(summary["gate_flags"]),
    "near_duplicates_excluding_own_pairs": summary["duplicates"]["near_excluding_own_pairs"]}))
