# %% [markdown]
# # P5 pilot (EXPLORATORY): does image S2-D per frame already catch video-VAE reconstructions?
#
# `docs/P5_PLAN.md` section 8, approved 2026-11-21. **Exploratory**: it may change bank
# membership only for a technical failure (OOM, crash, wrong frame count,
# non-determinism), never because a VAE is "already caught". It never touches test data.
#
# 1. **Pre-flight** (`p2_preflight.py`, `p5_preflight.py`): runtime, GPUs, token, imports,
#    libx264/libx265 round-trips and the FFmpeg build, both frozen S2-D models' hashes
#    (against this expectation and the committed P3 record), the threshold file's hash,
#    disk, and every video-bank pin and VAE file hash (CLAUDE.md 7a).
# 2. **Pexels pilot pool** (`p5_fetch_reals.py`): shards 0:4 of the pre-declared plan,
#    25 native frames, centre crop 320, pre-declared exclusions, 200 clips kept.
# 3. **Smoke**, all 7 bank and reserve members on 3 clips (both T4s, fail-fast).
# 4. **Lineage** (CPU): Qwen-Image's VAE against wan21's, tensor by tensor.
# 5. **Reconstruct** the 200 clips by wan21, hunyuan, ltx090 (both T4s, fail-fast).
# 6. **Score** with S2-D seeds 0 and 1 per frame (frames 1..24), raw and H.264-matched:
#    a 2-clip smoke first, then everything.
# 7. **Summary** and archive (cannot fail), then the verdict cell.
#
# **Kaggle:** GPU **T4 x2**, Internet **on**, Secret `HF_TOKEN`, Save & Run All (Commit),
# account `muhammadadeelshah`. Inputs, found by their contents:
#
# | input | what |
# |---|---|
# | dataset `dbp-code*` | the code at the hand-over commit |
# | notebook output `dbp-p3-s2d` | `outputs/p3/s2/s2d_p35_seed{0,1}/snapshots/step_05000` |
#
# Paste back: the archive `p5_pilot_reports.zip` from the output, and the last cell's text.

# %% Cell 1 — find the inputs by their contents, and the helpers every later cell uses
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


INPUTS = {
    "code": one("code dataset", IN.glob("datasets/*/dbp-code*/thesis_dbp")),
    "s2d": one("S2-D output (outputs/p3/s2/s2d_p35_seed1)",
               [p.parents[3] for p in IN.glob(
                   "*/*/*/Implementation/thesis_dbp/outputs/p3/s2/s2d_p35_seed1")]),
}
DBP = WORK / "Implementation" / "thesis_dbp"
REALS = WORK / "p5_reals"
RECON = TMP / "p5_recon"
SHARDS = TMP / "p5_shards"
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
    if smi:
        done = subprocess.run([smi, "--query-gpu=index,name,driver_version,memory.total",
                               "--format=csv"], capture_output=True, text=True, check=False)
        lines += (done.stdout or done.stderr).strip().splitlines()
    else:
        lines.append("nvidia-smi: not found")
    print("\n".join(lines))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


for name, path in INPUTS.items():
    print(f"{name:5s} {path}")

# %% Cell 2 — stage the code; hardware; the environment (bootstrap only)
if DBP.exists():
    shutil.rmtree(DBP)
shutil.copytree(INPUTS["code"], DBP)
assert (DBP / "scripts" / "p5_pilot.py").is_file(), "the code dataset predates this runbook"
print("code commit:", (DBP / "COMMIT.txt").read_text().split()[0])
hardware(DBP / "reports" / "p5_pilot" / "hardware.txt")
EXPECT_PATH = pathlib.Path(os.environ.get("DBP_EXPECT", DBP / "env" / "p5_pilot_expect.json"))
EXPECT = json.loads(EXPECT_PATH.read_text())
print("expected:", json.dumps(EXPECT["expected"]))
assert sh("bash", DBP / "env" / "kaggle_bootstrap.sh") == 0, "bootstrap FAILED"

# %% Cell 3 — PRE-FLIGHT: runtime, GPUs, token, imports; encoders, models, thresholds, disk, pins
GPUS = int(os.environ.get("DBP_GPUS", EXPECT["gpus"]))
checks = ["--expect-gpus", GPUS, "--runtime", "--imports", ",".join(EXPECT["imports"])]
checks += ["--hf-token"] if EXPECT["hf_token"] else []
assert py("p2_preflight.py", "--stage", "p5_pilot_env", *checks) == 0, \
    "PRE-FLIGHT (environment) FAILED -- nothing has run"
DRY = ["--skip-pins"] if os.environ.get("DBP_DRYRUN") == "1" else []
assert py("p5_preflight.py", "--expect", EXPECT_PATH, "--s2d-root", INPUTS["s2d"],
          "--disk", f"{WORK}={EXPECT['disk']['working_gb']}",
          "--disk", f"{TMP}={EXPECT['disk']['tmp_gb']}", *DRY) == 0, \
    "PRE-FLIGHT (P5) FAILED -- nothing has run"

# %% Cell 4 — the Pexels pilot pool: shards 0:4, 25 native frames, crop 320, 200 kept
source = EXPECT["pexels_source"]
assert py("p5_fetch_reals.py", "--source", source, "--pool", "pilot",
          "--shards", EXPECT["pilot_shards"], "--n", EXPECT["pilot_clips"],
          "--out-root", REALS, "--cache", SHARDS, "--workers", EXPECT["fetch_workers"],
          *DRY) == 0, "Pexels pilot pool INCOMPLETE -- see the summary above"
MANIFEST = DBP / "reports" / "p5_reals" / "p5_reals_pilot_manifest.csv"
COMMON = ["--worker-arg=--expect", f"--worker-arg={EXPECT_PATH}",
          f"--worker-arg=--reals-root={REALS}", f"--worker-arg=--manifest={MANIFEST}",
          f"--worker-arg=--recon-root={RECON}", f"--worker-arg=--s2d-root={INPUTS['s2d']}"]


def shards(phase, jobs, deadline, *extra):
    """One fail-fast parallel phase (CLAUDE.md 12.4)."""
    command = ["--worker", DBP / "scripts" / "p5_pilot.py", "--log-dir",
               DBP / "reports" / "p5_pilot" / "logs", "--log-prefix", f"p5_{phase}",
               "--status", DBP / "reports" / "p5_pilot" / f"status_{phase}.json",
               "--deadline-hours", deadline, "--poll-seconds", EXPECT["poll_seconds"],
               f"--worker-arg=--phase={phase.split('_')[0]}", *COMMON, *extra]
    for job in jobs:
        command += ["--job", job]
    return py("p2_run_shards.py", *command)


# %% Cell 5 — smoke: all 7 bank and reserve members on 3 clips (failures are recorded)
assert shards("smoke", EXPECT["smoke_jobs"], EXPECT["deadlines_hours"]["smoke"],
              f"--worker-arg=--smoke-clips={EXPECT['smoke_clips']}") == 0, "smoke CRASHED"
for path in sorted((DBP / "reports" / "p5_pilot").glob("smoke_*.json")):
    record = json.loads(path.read_text())
    print(f"{record['member']:9s} {record['status'][:70]:70s} "
          f"{record.get('warm_seconds_per_clip')} s/clip  peak {record.get('peak_mb_max')} MB  "
          f"fp16 {record.get('fp16_build_allowed')}")

# %% Cell 6 — lineage: Qwen-Image's VAE against wan21's (CPU)
assert py("p5_pilot.py", "--phase", "lineage", "--expect", EXPECT_PATH) == 0, "lineage FAILED"

# %% Cell 7 — reconstruct the 200 clips by wan21, hunyuan, ltx090 (a failed smoke is skipped)
assert shards("reconstruct", EXPECT["reconstruct_jobs"],
              EXPECT["deadlines_hours"]["reconstruct"]) == 0, "reconstruction FAILED"

# %% Cell 8 — score: a 2-clip smoke of both models, then every clip, raw and H.264-matched
assert shards("score_smoke", EXPECT["score_jobs"], EXPECT["deadlines_hours"]["score_smoke"],
              f"--worker-arg=--limit={EXPECT['score_smoke_clips']}") == 0, "score smoke FAILED"
assert shards("score", EXPECT["score_jobs"], EXPECT["deadlines_hours"]["score"]) == 0, \
    "scoring FAILED"

# %% Cell 9 — report: the summary and the archive (cannot fail the run)
py("p5_pilot.py", "--phase", "summary", "--expect", EXPECT_PATH, "--zip",
   WORK / "p5_pilot_reports")
print(f"wall: {(time.time() - START) / 3600:.2f} h")

# %% Cell 10 — verdict: this cell fails the notebook unless the pilot is complete
summary = json.loads((DBP / "reports" / "p5_pilot" / "p5_pilot_summary.json").read_text())
missing = [m["id"] for m in EXPECT["models"] if m["id"] not in summary.get("scores", {})]
assert "error" not in summary and not missing, f"pilot INCOMPLETE: {summary.get('error')} {missing}"
print("P5 PILOT COMPLETE:", json.dumps({m: {v: {k: e.get("clip_auc") for k, e in d.items()
                                                 if isinstance(e, dict)}
                                             for v, d in s.items() if isinstance(d, dict)}
                                         for m, s in summary["scores"].items()}))
