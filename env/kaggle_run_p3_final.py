# %% [markdown]
# # P3 — the final scoring: every frozen model on Panel B, once; then H5 robustness
#
# **Pre-registration sections 2, 5, 6 and 8.** Panel B is scored **once per frozen
# model**; a re-run is allowed only after a crash, recorded in the log with its reason.
# Nothing here is tuned: every section-5 value was committed before this runs, and the
# pre-flight refuses to plan any scoring otherwise.
#
# **Two sessions of this one notebook** (user decision 2026-10-22):
# 1. **clean** -- all 24 frozen models on Panel B (9,405), its luma companion (700) and the
#    891 threshold reals (491 COCO validation + 400 held-out Open Images: the union k = 0
#    threshold), with the deployed protocol (centre crop, head logit);
# 2. **robust** (H5) -- S2-D seed 0, P2 `full_bank` and P2 P-A on Panel B's 8,655 test rows
#    under the 12 conditions of `dbp.eval.perturb` (definitions fixed 2026-10-22).
#
# Session 2 attaches session 1's output: complete stages are carried forward and never
# scored again. A session interrupted mid-model leaves no summary for it; the next session
# scores it again and the plan lists it under `incomplete_previous` (log the reason).
#
# **The pre-flight** (`p3_final_preflight.py`): section 5 committed (degradation p, `N*`,
# `N_max`, the new real source, the k-shot methods); Panel B's four frozen digests; all 24
# models present, finished, and equal to their recorded hashes (the others are recorded);
# every image copied once to local disk with its sha256 checked in the same read. Then a
# 2-row smoke of every model due, then the scoring, one worker per T4, fail-fast.
#
# **Kaggle:** GPU **T4 x2**, Internet on, Secret `HF_TOKEN`, Save & Run All (Commit),
# account `muhammadadeelshah`, **run alone**. Every input is found by its contents:
#
# | input | what |
# |---|---|
# | dataset `dbp-code*` | the code (holds the section-5 records and this runbook) |
# | dataset `dbp-p3-panel-b` | Panel B (`panel_b/<view>/`), private |
# | dataset `report` | the COCO reals (`p1_reals`): the 491 validation reals |
# | dataset `dbp-p3-open-images` | `oi/reals`: the 400 held-out Open Images reals |
# | dataset `dbp-p3-lolo-output` | the 8 LOLO folds |
# | notebook output `dbp-p3-train` (latest) | BANK-k, BANK-1L, S2-B seeds 0 and 1 |
# | notebook output `dbp-p3-tune` | S2-B+deg (`s2b_deg_p35`) |
# | notebook output `dbp-p3-s2d` (after its session 2) | S2-B+reals, S2-D seeds 0 and 1 |
# | dataset `dbp-p2-reference-models` | P2 `full_bank_seed0` and `p_a_sd1_seed0` |
# | *(session 2)* this notebook's previous version | the clean stage, carried forward |

# %% Cell 1 — find the inputs by their contents, and the helpers every later cell uses
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


def one(label, hits):
    """Exactly one match, or stop with what was found."""
    hits = sorted(set(hits))
    assert len(hits) == 1, f"expected exactly one {label}, found {hits}"
    return hits[0]


def runs_root(rid):
    """The thesis_dbp root whose outputs/p3/s2 holds run ``rid`` (dataset or notebook)."""
    pattern = f"*/*/*/Implementation/thesis_dbp/outputs/p3/s2/{rid}"
    return one(f"output holding {rid}", [p.parents[3] for p in IN.glob(pattern)])


INPUTS = {
    "code": one("code dataset", IN.glob("datasets/*/dbp-code*/thesis_dbp")),
    "panel": one("Panel B (a directory holding e7_dda_coco_luma)",
                 [p.parent for g in ("*/*/*/e7_dda_coco_luma", "*/*/*/*/e7_dda_coco_luma")
                  for p in IN.glob(g)]),
    "reals": one("COCO reals (p1_reals)", IN.glob("datasets/*/*/p1_reals")),
    "oi_reals": one("Open Images reals (oi/reals)", IN.glob("datasets/*/*/oi/reals")),
}
SOURCES = {"lolo": runs_root("lolo_f8_seed0"), "train": runs_root("s2b_seed1"),
           "tune": runs_root("s2b_deg_p35_seed0"), "s2d": runs_root("s2d_p35_seed1"),
           "p2ref": one("P2 reference models",
                        [p.parents[2] for p in
                         IN.glob("*/*/*/full_bank_seed0/lora/adapter_model.safetensors")])}
PREVIOUS = sorted({p.parents[1] for p in IN.glob("notebooks/*/*/Implementation/thesis_dbp/"
                                                  "reports/p3_final_plan.json")})
DBP = WORK / "Implementation" / "thesis_dbp"
CORPUS = WORK / "p3_final_corpus"
LOCAL = TMP / "dbp_p3_final_local"
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


for name, path in {**INPUTS, **SOURCES}.items():
    print(f"{name:9s} {path}")
print("previous versions:", [str(p) for p in PREVIOUS] or "none (session 1)")

# %% Cell 2 — stage the code; hardware at session start; the environment (bootstrap only)
if DBP.exists():
    shutil.rmtree(DBP)
shutil.copytree(INPUTS["code"], DBP)
assert (DBP / "scripts" / "p3_final_preflight.py").is_file(), \
    "the code dataset predates this runbook"
print("code commit:", (DBP / "COMMIT.txt").read_text().split()[0])
hardware(DBP / "reports" / "p3_final_hardware.txt")
EXPECT_PATH = pathlib.Path(os.environ.get("DBP_EXPECT", DBP / "env" / "p3_final_expect.json"))
EXPECT = json.loads(EXPECT_PATH.read_text())
print("expected:", json.dumps(EXPECT["expected"]))
assert sh("bash", DBP / "env" / "kaggle_bootstrap.sh") == 0, "bootstrap FAILED"

# %% Cell 3 — PRE-FLIGHT 1: GPUs, token, imports; section 5, Panel B, the 24 models, staging
links = ["--link", f"panel={INPUTS['panel']}", "--link", f"reals={INPUTS['reals']}",
         "--link", f"oi_reals={INPUTS['oi_reals']}"]
assert py("p2_stage_corpus.py", "--root", CORPUS, *links) == 0, "corpus links FAILED"
GPUS = int(os.environ.get("DBP_GPUS", EXPECT["gpus"]))
checks = ["--expect-gpus", GPUS, "--imports", ",".join(EXPECT["imports"])]
checks += ["--hf-token"] if EXPECT["hf_token"] else []
assert py("p2_preflight.py", "--stage", "p3_final_env", *checks) == 0, \
    "PRE-FLIGHT (environment) FAILED -- nothing has been scored"
plan_args = ["--expect", EXPECT_PATH, "--root", CORPUS, "--local-root", LOCAL,
             "--devices", EXPECT["devices"], "--workers", EXPECT["workers"]]
for name, path in SOURCES.items():
    plan_args += ["--source", f"{name}={path}"]
for previous in PREVIOUS:
    plan_args += ["--previous", previous]
assert py("p3_final_preflight.py", *plan_args) == 0, \
    "PRE-FLIGHT (section 5 / Panel B / models / staging) FAILED -- see " \
    "reports/p3_final_plan.json; nothing has been scored"
PLAN = json.loads((DBP / "reports" / "p3_final_plan.json").read_text())
print("STAGE:", PLAN["stage"], "| due:", PLAN["due"], "| jobs:", PLAN["jobs"])
print("incomplete earlier (scored again; log the reason):", PLAN["incomplete_previous"])
assert PLAN["stage"] != "complete", "EVERY MODEL AND STAGE IS ALREADY SCORED"

SHARDS = ["--worker", DBP / "scripts" / "p3_final_score.py", "--log-dir",
          DBP / "outputs" / "p3" / "logs", "--poll-seconds", EXPECT["poll_seconds"]]
for job in PLAN["jobs"]:
    SHARDS += ["--job", job]
WORKER = [f"--worker-arg={a}" for a in ("--plan", DBP / "reports" / "p3_final_plan.json",
                                        "--batch", EXPECT["batch"])]

# %% Cell 4 — PRE-FLIGHT 2: every model due, 2 rows per list, on its own GPU
assert py("p2_run_shards.py", *SHARDS, *WORKER, "--worker-arg=--limit", "--worker-arg=2",
          "--worker-arg=--out-root", f"--worker-arg={TMP / 'dbp_p3_final_smoke'}",
          "--log-prefix", "final_smoke", "--status", DBP / "reports" /
          "p3_final_smoke_status.json", "--deadline-hours", EXPECT["smoke_deadline_hours"]) == 0, \
    "PRE-FLIGHT (models) FAILED -- nothing has been scored"

# %% Cell 5 — score: Panel B once per model (clean), or H5 (robust); one worker per T4
code = py("p2_run_shards.py", *SHARDS, *WORKER, "--log-prefix", f"final_{PLAN['stage']}",
          "--status", DBP / "reports" / "p3_final_status.json",
          "--deadline-hours", EXPECT["deadline_hours"])
print(f"scoring exited {code} (0 = complete, 1 = a worker failed, 2 = deadline)")

# %% Cell 6 — report (cannot fail the run)
py("p2_report.py", "--stage", "final", "--status", "reports/p3_final_status.json",
   "--zip", WORK / "p3_final_reports")
staged = PLAN.get("staged", {})
times = []
for model in PLAN["due"]:
    path = DBP / "reports" / "p3_final" / f"{model}_{PLAN['stage']}.json"
    summary = json.loads(path.read_text()) if path.is_file() else {}
    seconds = sum(v.get("seconds", 0) for v in summary.get("lists", {}).values())
    times.append(f"{model} {round(seconds)} s")
print("THROUGHPUT |", f"staging {staged.get('files')} files {staged.get('gb')} GB in "
      f"{staged.get('seconds')} s ({staged.get('files_per_s')} files/s) |", " | ".join(times))

# %% Cell 7 — verdict: this cell fails the notebook if a model due was not scored
missing = [m for m in PLAN["due"]
           if not (DBP / "reports" / "p3_final" / f"{m}_{PLAN['stage']}.json").is_file()]
assert code == 0 and not missing, (f"scoring exited {code}; not scored: {missing}. Re-run with "
                                   f"this version attached: complete models are carried "
                                   f"forward, the rest are scored (log the reason).")
print("FINAL SCORING STAGE COMPLETE:", PLAN["stage"], len(PLAN["due"]), "model(s)")
