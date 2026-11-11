# %% [markdown]
# # P3b — ViT-L training: S2-D on DINOv3 ViT-L/16, seeds 0 and 1 (one per T4)
#
# **Pre-registered** in `docs/P3b_VITL_PREREG.md`, frozen at the tag `p3b-vitl-prereg-frozen`
# before any ViT-L training. The S2-D recipe is unchanged -- the bank, the data, the 3,600
# Open Images train reals, degradation p = 0.35, LoRA r 8 on q/k/v/o, lr, weight decay,
# warm-up, crop 320, the batch mix, the 6,000-step schedule stopped at N* = 5,000, gradient
# checkpointing on -- and **only the backbone changes**: `--backbone dinov3-vitl16`
# (pinned). **The trainer is byte-identical** (user decision 2026-10-21).
#
# The inputs, staging and pre-flight are the S2-D session's, unchanged, plus the run record
# of the runtime (torch, CUDA, GPU names, driver; prereg section 1). Both seeds train in
# parallel, one per T4, into `outputs/p3b/s2` -- a directory no ViT-B run ever used, so an
# attached ViT-B output can never be taken for a ViT-L run. A run cut short resumes from its
# 500-step checkpoint when this notebook is re-run with this version's output attached.
#
# **The failure path** (prereg section 5) is checked on the training reports in the report
# cell (`p3b_train_check.py`): a NaN or divergent run, or a validation AUC at step 5,000
# more than 5 points below ViT-B's for the same seed, is reported and **never scored**.
# **Panel B is not touched.**
#
# **Kaggle:** Accelerator **GPU T4 x2**. **Internet:** on. **Secret:** `HF_TOKEN` (the
# DINOv3 weights are gated). **Mode:** Save & Run All (Commit). **Account:**
# `muhammadadeelshah` only. **Run it alone.** torch, torchvision and numpy are never
# reinstalled (the bootstrap enforces it).
#
# | input | what |
# |---|---|
# | dataset `dbp-code*` | the code at a commit holding this runbook |
# | dataset `dbp-p2-s2-inputs` | pairs manifest, split, generation manifest (hashed) |
# | dataset holding `p1_reals/` and `p1_generated/` (slug `report`) | corpus |
# | dataset `thesis-dbp-1-dataset` | corpus pair reconstructions |
# | dataset `dbp-p3-open-images` | the Open Images crops (`oi/reals/t_oi/`) |
# | the `dbp-p3-reconstruct` output (notebook or dataset) | `oi_pairs/` and its manifest |
# | *(a resumed session only)* this notebook's previous version | unfinished runs resume |

# %% Cell 1 — find the inputs, and the helpers every later cell uses
import json
import os
import pathlib
import platform
import shutil
import subprocess
import sys

IN = pathlib.Path(os.environ.get("DBP_INPUT_ROOT", "/kaggle/input"))
WORK = pathlib.Path(os.environ.get("DBP_WORKING_ROOT", "/kaggle/working"))
TMP = pathlib.Path(os.environ.get("DBP_TMP_ROOT", "/tmp"))


def find(pattern):
    """The one attached input matching ``pattern``, whichever account owns it."""
    hits = sorted(p for p in IN.glob(pattern) if p.is_dir())
    assert len(hits) == 1, f"expected exactly one input matching {pattern!r}, found {hits}"
    return hits[0]


# The step-4a output, found by layout: the root holding oi_pairs/ and its manifest.
OI_ROOTS = sorted({p.parents[3] for p in IN.glob(
    "*/*/*/Implementation/thesis_dbp/reports/p3_oi_pairs_manifest.csv")
    if (p.parents[3] / "oi_pairs").is_dir()})
assert len(OI_ROOTS) == 1, f"expected exactly one step-4a output (oi_pairs/), found {OI_ROOTS}"
OI_ROOT = OI_ROOTS[0]
INPUTS = {
    "code": find("datasets/*/dbp-code*/thesis_dbp"),
    "s2_inputs": find("datasets/*/dbp-p2-s2-inputs"),
    "reals": find("datasets/*/*/p1_reals"),
    "generated": find("datasets/*/*/p1_generated"),
    "pairs": find("datasets/*/thesis-dbp-1-dataset/Implementation/thesis_dbp/outputs/p1/pairs"),
    "oi_reals": find("datasets/*/*/oi/reals"),
    "oi_pairs": OI_ROOT / "oi_pairs",
}
# This notebook's previous versions only: their ViT-L runs live under outputs/p3b/s2.
PREVIOUS = sorted(p.parents[2] for p in
                  IN.glob("notebooks/*/*/Implementation/thesis_dbp/outputs/p3b/s2"))
PREVIOUS += [pathlib.Path(p) for p in os.environ.get("DBP_RESUME_DIRS", "").split(os.pathsep)
             if p]
DBP = WORK / "Implementation" / "thesis_dbp"
P1 = WORK / "p1"
S2IN = INPUTS["s2_inputs"]
MERGED = TMP / "dbp_s2d_inputs"
LOCAL = TMP / "dbp_s2d_corpus"
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
    """Print and save the host: CPU model and cores, GPUs, driver, clocks, power, temperature."""
    cpu = platform.processor() or "unknown"
    info = pathlib.Path("/proc/cpuinfo")
    if info.is_file():
        names = [line.split(":", 1)[1].strip() for line in info.read_text().splitlines()
                 if line.startswith("model name")]
        cpu = names[0] if names else cpu
    lines = [f"CPU: {cpu} | logical cores: {os.cpu_count()}"]
    smi = shutil.which("nvidia-smi")
    if smi:
        query = ("index,name,driver_version,clocks.sm,clocks.max.sm,clocks.mem,clocks.max.mem,"
                 "power.draw,power.limit,temperature.gpu")
        done = subprocess.run([smi, f"--query-gpu={query}", "--format=csv"],
                              capture_output=True, text=True, check=False)
        lines += (done.stdout or done.stderr).strip().splitlines()
    else:
        lines.append("nvidia-smi: not found")
    print("\n".join(lines))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


for name, path in INPUTS.items():
    print(f"{name:10s} {path}")
print("resume from:", [str(p) for p in PREVIOUS] or "nothing (first session)")

# %% Cell 2 — stage the code; hardware at session start; the environment (bootstrap only)
if DBP.exists():
    shutil.rmtree(DBP)
shutil.copytree(INPUTS["code"], DBP)
assert (DBP / "scripts" / "p3b_train_check.py").is_file(), "the code dataset predates this runbook"
print("code commit:", (DBP / "COMMIT.txt").read_text().split()[0])
hardware(DBP / "reports" / "p3b_vitl_train_hardware.txt")
EXPECT_PATH = pathlib.Path(os.environ.get("DBP_EXPECT",
                                          DBP / "env" / "p3b_vitl_train_expect.json"))
EXPECT = json.loads(EXPECT_PATH.read_text())
CHOICE = DBP / EXPECT["choice"]
assert CHOICE.is_file(), f"{EXPECT['choice']} is not in this code"
print("expected:", json.dumps(EXPECT["expected"]))
assert sh("bash", DBP / "env" / "kaggle_bootstrap.sh") == 0, "bootstrap FAILED"

# %% Cell 3 — corpus + Open Images; PRE-FLIGHT 1: runtime record, GPUs, token, imports, hashes
links = []
for name in ("reals", "generated", "pairs"):
    links += ["--link", f"{name}={INPUTS[name]}", "--expect", f"{name}={EXPECT['corpus'][name]}"]
links += ["--link", f"oi_reals={INPUTS['oi_reals']}", "--link", f"oi_pairs={INPUTS['oi_pairs']}"]
assert py("p2_stage_corpus.py", "--root", P1, *links) == 0, "corpus staging FAILED"

assert py("p1_match_generations.py",
          "--gen-manifest", S2IN / "p1_gen_manifest.csv",
          "--reals-manifest", "reports/p1_reals_manifest_t_real.csv",
          "--gen-root", P1 / "generated", "--out-root", P1, "--set-id", "gen_matched") == 0, \
    "gen_matched regeneration FAILED"

OI = EXPECT["oi"]
GPUS = int(os.environ.get("DBP_GPUS", EXPECT["gpus"]))
checks = ["--expect-gpus", GPUS, "--runtime", "--imports", ",".join(EXPECT["imports"])]
checks += ["--hf-token"] if EXPECT["hf_token"] else []
for name, spec in EXPECT["inputs"].items():
    checks += ["--sha256", f"{S2IN / name}={spec['sha256']}"]
    checks += ["--rows", f"{S2IN / name}={spec['rows']}"] if "rows" in spec else []
checks += ["--sha256", f"{DBP / EXPECT['nstar']}={EXPECT['nstar_sha256']}"]
checks += ["--sha256", f"{DBP / OI['reals_manifest']}={OI['reals_manifest_sha256']}"]
checks += ["--sha256", f"{DBP / OI['pairs_digest']}={OI['pairs_digest_sha256']}"]
# The reference is the read-only committed manifest, not the one the matcher just wrote.
assert py("p2_preflight.py", "--stage", "p3b_vitl_train_data", *checks, "--root", P1,
          "--gen-matched-reference", INPUTS["code"] / "reports" / "p1_gen_matched_manifest.csv",
          "--gen-matched-dir", P1 / "gen_matched",
          "--expect-gen-matched", EXPECT["gen_matched"]) == 0, \
    "PRE-FLIGHT (data) FAILED -- nothing long has started"

S2D_IN = ["--expect", EXPECT_PATH, "--oi-manifest", DBP / OI["reals_manifest"],
          "--out-dir", MERGED]
assert py("p3_s2d_inputs.py", "build", *S2D_IN,
          "--oi-pairs-manifest",
          OI_ROOT / "Implementation/thesis_dbp/reports/p3_oi_pairs_manifest.csv",
          "--oi-pairs-digest", DBP / OI["pairs_digest"], "--local-root", LOCAL,
          "--pairs-manifest", S2IN / "p1_pairs_manifest.csv",
          "--splits", S2IN / "p2_splits.csv", "--choice", CHOICE, "--root", P1,
          "--workers", OI["hash_workers"],
          "--out", DBP / "reports" / "p3_s2d_inputs.json") == 0, \
    "S2-D INPUTS FAILED -- see reports/p3_s2d_inputs.json; nothing long has started"
# PRE-FLIGHT: no held-out Open Images id in any merged file.
assert py("p3_s2d_inputs.py", "check", *S2D_IN,
          "--built", DBP / "reports" / "p3_s2d_inputs.json", "--local-root", LOCAL,
          "--out", DBP / "reports" / "p3_s2d_inputs_check.json") == 0, \
    "PRE-FLIGHT (held-out exclusion) FAILED -- nothing long has started"
BUILT = json.loads((DBP / "reports" / "p3_s2d_inputs.json").read_text())
assert BUILT["queue"] == EXPECT["queue_resolved"], \
    f"queue {BUILT['queue']}, the pre-registration trains {EXPECT['queue_resolved']}"
print("queue:", BUILT["queue"], "| backbone:", EXPECT["backbone"],
      "| pools:", json.dumps(BUILT["pools"]))

# %% Cell 4 — the queue: both seeds, one per T4, into outputs/p3b/s2 (ViT-L only)
OUT = DBP / EXPECT["out_dir"]
QUEUE = ["--devices", EXPECT["devices"], "--root", LOCAL,
         "--pairs-manifest", MERGED / "p1_pairs_manifest.csv",
         "--splits", MERGED / "p2_splits.csv",
         "--scratch", TMP / "dbp_p3b_train_smoke", "--poll-seconds", EXPECT["poll_seconds"],
         "--train-only", "--out-dir", OUT, "--log-dir", DBP / EXPECT["log_dir"]]
for run in BUILT["queue"]:
    QUEUE += ["--run", run]
QUEUE += [f"--train-arg={a}" for a in EXPECT["train_args"]]
QUEUE += [f"--train-arg=--manifests={MERGED / 'manifests'}"]
print("trainer options:", EXPECT["train_args"])

# %% Cell 5 — PRE-FLIGHT 2: each run on each GPU (2 steps of ViT-L, save, reload)
assert py("p2_run_queue.py", *QUEUE, "--smoke", EXPECT["smoke_steps"],
          "--status", DBP / "reports" / "p3b_vitl_train_smoke_status.json",
          "--deadline-hours", EXPECT["smoke_deadline_hours"]) == 0, \
    "PRE-FLIGHT (models) FAILED -- nothing long has started"

# %% Cell 6 — train both seeds: the 6,000-step schedule, stopped at N* = 5000
resume = []
for previous in PREVIOUS:
    resume += ["--resume-from", previous]
code = py("p2_run_queue.py", *QUEUE, *resume, "--status",
          DBP / "reports" / "p3b_vitl_train_status.json",
          "--deadline-hours", EXPECT["deadline_hours"])
print(f"queue exited {code} (0 = runs complete, 1 = failed, 2 = deadline; resumable)")

# %% Cell 7 — report and the failure path (cannot fail the run)
failure_path = py("p3b_train_check.py", "--expect", EXPECT_PATH,
                  "--reports-dir", DBP / "reports",
                  "--out", DBP / "reports" / "p3b_train_check.json")
py("p2_report.py", "--stage", "s2", "--status", "reports/p3b_vitl_train_status.json",
   "--s2-dir", OUT, "--zip", WORK / "p3b_vitl_train_reports")
throughput = []
for run in BUILT["queue"]:
    path = DBP / "reports" / f"p2_s2_{run.replace(':', '_seed')}.json"
    report = json.loads(path.read_text()) if path.is_file() else {}
    print(run, report.get("status"), "| backbone", report.get("backbone"), "| steps",
          report.get("steps_run"), "| val AUC",
          [(p["step"], p["val_auc"]) for p in report.get("curve", [])][-1:])
    throughput.append(f"{run} {report.get('steps_run')} steps "
                      f"{report.get('measured_step_seconds_train_loop')} s/step "
                      f"{report.get('gpu_hours')} GPU-h")
status = DBP / "reports" / "p3b_vitl_train_status.json"
wall = json.loads(status.read_text()).get("wall_hours") if status.is_file() else None
staged = BUILT.get("staged", {})
print("THROUGHPUT |", f"staging {staged.get('files')} files {staged.get('gb')} GB in "
      f"{staged.get('seconds')} s |", " | ".join(throughput), f"| queue wall {wall} h")
print("FAILURE PATH:", "every seed passes" if failure_path == 0 else
      "STOP AND REPORT -- see reports/p3b_train_check.json (prereg section 5)")

# %% Cell 8 — verdict: this cell fails the notebook if a run did not complete
assert code == 0, (f"queue exited {code}: see reports/p3b_vitl_train_status.json. An "
                   f"unfinished run resumes from its last checkpoint when this notebook is "
                   f"re-run with this version's output attached.")
print("P3b ViT-L RUNS COMPLETE:", BUILT["queue"], "| failure path:",
      "PASS (both seeds cleared for scoring)" if failure_path == 0 else
      "FAIL -- no Panel B scoring of a failed seed (prereg section 5)")
