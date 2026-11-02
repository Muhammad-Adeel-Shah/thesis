# %% [markdown]
# # P3 — non-LOLO S2 runs, two per session (one per T4), plus the LOLO rescoring
#
# **The P3 recipe** (user decision 2026-10-17, the reading of "same recipe" in
# pre-registration section 3): P2's S2 trainer and hyper-parameters on the **6,000-step**
# cosine schedule, **stopped at `N*` = 5000** (`reports/p3_nstar.json`, selected once),
# no new reals, no degradation. In-protocol validation every 500 steps; every member a
# run does not train on is scored **once, at the final step**, per image
# (`reports/p3_heldout_scores/`). Nothing is selected inside a run. **Panel B is not
# touched**: it is scored once, at the end, for every frozen model together.
#
# **The queue** is committed in `env/p3_train_expect.json` (session priority 2026-10-17):
# `BANK-1` + `S2-B` seed 0 first, then the bank-size runs, `BANK-1L` and `S2-B` seed 1.
# Each session trains the first two unfinished runs (`p3_train_next.py`), one per T4; a
# run cut short resumes from its 500-step checkpoint; finished runs are carried forward.
#
# **The LOLO rescoring** (validation data only) runs in the first session: each LOLO
# fold's snapshot at its frozen `N*(-f)` is re-scored per image (`p3_score_heldout.py`),
# so the H1-a CI and H1-c can be computed locally right after. Later sessions carry the
# files forward and skip it.
#
# **Platform-neutral.** Every path comes from the environment, with Kaggle's as defaults:
# `DBP_INPUT_ROOT`, `DBP_WORKING_ROOT`, `DBP_TMP_ROOT`, `DBP_RESUME_DIRS`, `DBP_GPUS`,
# `DBP_DEVICE` (the scorer's device; `cuda` on Kaggle).
#
# **Kaggle:** Accelerator **GPU T4 x2**. **Internet:** on. **Secret:** `HF_TOKEN`.
# **Mode:** Save & Run All (Commit). **Account:** `muhammadadeelshah` only. **Run it alone:**
# no other GPU session at the same time (2026-10-18: a session run beside another one was
# lost with no log and no output).
#
# **Every step is time-bounded:** the queue by its deadline, and the LOLO rescoring by a
# watchdog (`--deadline-hours`) that ends it even if a load or a forward pass hangs.
#
# | input | what |
# |---|---|
# | dataset `dbp-code*` | the code at the commit that holds this runbook |
# | dataset `dbp-p2-s2-inputs` | pairs manifest, split, generation manifest (hashed) |
# | dataset holding `p1_reals/` and `p1_generated/` (slug `report`) | corpus |
# | dataset `thesis-dbp-1-dataset` | corpus pair reconstructions |
# | the final LOLO output (dataset or notebook) | `lolo_f*` snapshots, for the rescoring |
# | *(session 2 on)* this notebook's previous version | done runs skipped; unfinished resume |

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
DEVICE = os.environ.get("DBP_DEVICE", "cuda")


def find(pattern):
    """The one attached input matching ``pattern``, whichever account owns it."""
    hits = sorted(p for p in IN.glob(pattern) if p.is_dir())
    assert len(hits) == 1, f"expected exactly one input matching {pattern!r}, found {hits}"
    return hits[0]


INPUTS = {
    "code": find("datasets/*/dbp-code*/thesis_dbp"),
    "s2_inputs": find("datasets/*/dbp-p2-s2-inputs"),
    "reals": find("datasets/*/*/p1_reals"),
    "generated": find("datasets/*/*/p1_generated"),
    "pairs": find("datasets/*/thesis-dbp-1-dataset/Implementation/thesis_dbp/outputs/p1/pairs"),
}
# The LOLO output: the thesis_dbp root holding outputs/p3/s2/lolo_f8_seed0.
LOLO = sorted({p.parents[3] for p in
               IN.glob("*/*/*/Implementation/thesis_dbp/outputs/p3/s2/lolo_f8_seed0")})
# This notebook's previous versions: p3 outputs that are NOT the LOLO output.
PREVIOUS = sorted(p.parents[2] for p in
                  IN.glob("notebooks/*/*/Implementation/thesis_dbp/outputs/p3/s2")
                  if p.parents[2] not in LOLO)
PREVIOUS += [pathlib.Path(p) for p in os.environ.get("DBP_RESUME_DIRS", "").split(os.pathsep)
             if p]
DBP = WORK / "Implementation" / "thesis_dbp"
P1 = WORK / "p1"
S2IN = INPUTS["s2_inputs"]
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
    """Print and save the host: CPU model and cores, GPUs, clocks, power, temperature."""
    cpu = platform.processor() or "unknown"
    info = pathlib.Path("/proc/cpuinfo")
    if info.is_file():
        names = [line.split(":", 1)[1].strip() for line in info.read_text().splitlines()
                 if line.startswith("model name")]
        cpu = names[0] if names else cpu
    lines = [f"CPU: {cpu} | logical cores: {os.cpu_count()}"]
    smi = shutil.which("nvidia-smi")
    if smi:
        query = ("index,name,clocks.sm,clocks.max.sm,clocks.mem,clocks.max.mem,"
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
print("LOLO output:", [str(p) for p in LOLO] or "none attached")
print("resume from:", [str(p) for p in PREVIOUS] or "nothing (first session)")

# %% Cell 2 — stage the code; set up the environment (the bootstrap is the only way)
if DBP.exists():
    shutil.rmtree(DBP)
shutil.copytree(INPUTS["code"], DBP)
assert (DBP / "scripts" / "p3_train_next.py").is_file(), "the code dataset predates this runbook"
print("code commit:", (DBP / "COMMIT.txt").read_text().split()[0])
hardware(DBP / "reports" / "p3_train_hardware.txt")
EXPECT = json.loads(pathlib.Path(
    os.environ.get("DBP_EXPECT", DBP / "env" / "p3_train_expect.json")).read_text())
NSTAR = DBP / EXPECT["nstar"]
print("expected:", json.dumps(EXPECT["expected"]))
assert sh("bash", DBP / "env" / "kaggle_bootstrap.sh") == 0, "bootstrap FAILED"

# %% Cell 3 — corpus and gen_matched; PRE-FLIGHT 1: GPUs, token, imports, hashes, paths
links = []
for name in ("reals", "generated", "pairs"):
    links += ["--link", f"{name}={INPUTS[name]}", "--expect", f"{name}={EXPECT['corpus'][name]}"]
assert py("p2_stage_corpus.py", "--root", P1, *links) == 0, "corpus staging FAILED"

assert py("p1_match_generations.py",
          "--gen-manifest", S2IN / "p1_gen_manifest.csv",
          "--reals-manifest", "reports/p1_reals_manifest_t_real.csv",
          "--gen-root", P1 / "generated", "--out-root", P1, "--set-id", "gen_matched") == 0, \
    "gen_matched regeneration FAILED"

GPUS = int(os.environ.get("DBP_GPUS", EXPECT["gpus"]))
checks = ["--expect-gpus", GPUS, "--imports", ",".join(EXPECT["imports"])]
checks += ["--hf-token"] if EXPECT["hf_token"] else []
for name, spec in EXPECT["inputs"].items():
    checks += ["--sha256", f"{S2IN / name}={spec['sha256']}"]
    checks += ["--rows", f"{S2IN / name}={spec['rows']}"] if "rows" in spec else []
checks += ["--sha256", f"{NSTAR}={EXPECT['nstar_sha256']}"]
# The reference is the read-only committed manifest, not the one the matcher just wrote.
assert py("p2_preflight.py", "--stage", "p3_train_data", *checks, "--root", P1,
          "--gen-matched-reference", INPUTS["code"] / "reports" / "p1_gen_matched_manifest.csv",
          "--gen-matched-dir", P1 / "gen_matched",
          "--expect-gen-matched", EXPECT["gen_matched"]) == 0, \
    "PRE-FLIGHT (data) FAILED -- nothing long has started"

# %% Cell 4 — which runs this session trains (first two unfinished); is the rescoring due?
choose = ["--expect", pathlib.Path(os.environ.get("DBP_EXPECT", DBP / "env" /
                                                  "p3_train_expect.json")),
          "--count", EXPECT["runs_per_session"]]
for previous in PREVIOUS:
    choose += ["--resume-from", previous]
chosen = py("p3_train_next.py", *choose)
PLAN = json.loads((DBP / "reports" / "p3_train_plan.json").read_text())
assert chosen in (0, 3), "run choice FAILED"
assert PLAN["selected"] or PLAN["lolo_rescore_due"], "EVERY QUEUED RUN IS COMPLETE"
print("this session trains:", PLAN["selected"], "| complete:", PLAN["complete"],
      "| LOLO rescoring due:", PLAN["lolo_rescore_due"])
if PLAN["lolo_rescore_due"]:
    assert len(LOLO) == 1, f"the LOLO rescoring needs exactly one LOLO output attached: {LOLO}"

QUEUE = ["--devices", EXPECT["devices"], "--root", P1,
         "--pairs-manifest", S2IN / "p1_pairs_manifest.csv", "--splits", S2IN / "p2_splits.csv",
         "--scratch", TMP / "dbp_p3_smoke", "--poll-seconds", EXPECT["poll_seconds"],
         "--train-only", "--out-dir", DBP / "outputs" / "p3" / "s2",
         "--log-dir", DBP / "outputs" / "p3" / "logs"]
for run in PLAN["selected"]:
    QUEUE += ["--run", run]
QUEUE += [f"--train-arg={a}" for a in EXPECT["train_args"]]
RESCORE = ["--lolo-root", LOLO[0] if LOLO else "", "--nstar", NSTAR, "--root", P1,
           "--pairs-manifest", S2IN / "p1_pairs_manifest.csv", "--splits",
           S2IN / "p2_splits.csv", "--device", DEVICE]

# %% Cell 5 — PRE-FLIGHT 2: each run on each GPU (2 steps, save, reload); every LOLO snapshot
if PLAN["selected"]:
    assert py("p2_run_queue.py", *QUEUE, "--smoke", EXPECT["smoke_steps"],
              "--status", DBP / "reports" / "p3_train_smoke_status.json",
              "--deadline-hours", EXPECT["smoke_deadline_hours"]) == 0, \
        "PRE-FLIGHT (models) FAILED -- nothing long has started"
if PLAN["lolo_rescore_due"]:
    assert py("p3_score_heldout.py", *RESCORE, "--limit", 2,
              "--deadline-hours", EXPECT["rescore_smoke_deadline_hours"],
              "--out-dir", TMP / "dbp_p3_rescore_smoke") == 0, \
        "PRE-FLIGHT (LOLO snapshots) FAILED -- nothing long has started"

# %% Cell 6 — train the runs, one per GPU: the 6,000-step schedule, stopped at N* = 5000
code = 0
if PLAN["selected"]:
    resume = []
    for previous in PREVIOUS:
        resume += ["--resume-from", previous]
    code = py("p2_run_queue.py", *QUEUE, *resume, "--status",
              DBP / "reports" / "p3_train_status.json",
              "--deadline-hours", EXPECT["deadline_hours"])
print(f"queue exited {code} (0 = runs complete, 1 = failed, 2 = deadline; resumable)")

# %% Cell 7 — the LOLO rescoring at N*(-f), validation data only (first session only)
rescore = 0
if PLAN["lolo_rescore_due"]:
    rescore = py("p3_score_heldout.py", *RESCORE,
                 "--deadline-hours", EXPECT["rescore_deadline_hours"],
                 "--out-dir", DBP / "reports" / "p3_heldout_scores")
print(f"LOLO rescoring exited {rescore}" if PLAN["lolo_rescore_due"]
      else "LOLO rescoring: already done (carried forward)")

# %% Cell 8 — report (cannot fail the run)
py("p2_report.py", "--stage", "s2", "--status", "reports/p3_train_status.json",
   "--s2-dir", DBP / "outputs" / "p3" / "s2", "--zip", WORK / "p3_train_reports")
for run in PLAN["selected"]:
    path = DBP / "reports" / f"p2_s2_{run.replace(':', '_seed')}.json"
    if path.is_file():
        report = json.loads(path.read_text())
        print(run, report.get("status"), "| steps", report.get("steps_run"), "| val AUC",
              [(p["step"], p["val_auc"]) for p in report.get("curve", [])][-1:],
              "| held-out", report.get("heldout_scores"))
summary = DBP / "reports" / "p3_heldout_scores" / "p3_lolo_rescore_summary.json"
if summary.is_file():
    for fold in json.loads(summary.read_text())["folds"]:
        print(fold)
throughput = []
for run in PLAN["selected"]:
    path = DBP / "reports" / f"p2_s2_{run.replace(':', '_seed')}.json"
    report = json.loads(path.read_text()) if path.is_file() else {}
    throughput.append(f"{run} {report.get('steps_run')} steps "
                      f"{report.get('measured_step_seconds_train_loop')} s/step "
                      f"{report.get('gpu_hours')} GPU-h")
status = DBP / "reports" / "p3_train_status.json"
wall = json.loads(status.read_text()).get("wall_hours") if status.is_file() else None
print("THROUGHPUT |", " | ".join(throughput), f"| queue wall {wall} h")

# %% Cell 9 — verdict: this cell fails the notebook if a run or the rescoring did not complete
assert code == 0, (f"queue exited {code}: see reports/p3_train_status.json. An unfinished "
                   f"run resumes from its last checkpoint when this notebook is re-run with "
                   f"this version's output attached.")
assert rescore == 0, "the LOLO rescoring FAILED: see the Cell 7 output"
print("P3 RUNS COMPLETE:", PLAN["selected"], "| all complete so far:",
      sorted(set(PLAN["complete"]) | set(PLAN["selected"])))
