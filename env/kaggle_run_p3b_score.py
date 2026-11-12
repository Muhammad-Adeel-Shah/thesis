# %% [markdown]
# # P3b — the frozen ViT-L models on Panel B, once: clean, then jpeg50 + resize050
#
# **Pre-registered** in `docs/P3b_VITL_PREREG.md` section 2 (tag `p3b-vitl-prereg-frozen`).
# **Panel B is scored once per frozen model**, on P3's final-scoring path
# (`p3_final_score.py --backbone dinov3-vitl16`, the deployed protocol: centre crop 320,
# head logit):
# 1. **clean** -- Panel B (9,405), its luma companion (700) and the 891 threshold reals
#    (491 COCO validation + 400 held-out Open Images: the union k = 0 threshold);
# 2. **robust** -- Panel B's 8,655 test rows under the two H5 decision conditions only,
#    `jpeg50` and `resize050`.
#
# **Before anything is scored** (`p3b_score_preflight.py`): section 5 and Panel B's frozen
# digests; for each model, its training report in the attached output is the committed
# one, the failure path (section 5) passes, and its LoRA, head and snapshot hashes equal the
# **committed model record** -- a mismatch refuses the model. Every image is copied once to
# local disk with its sha256 checked. Then, per stage: a smoke of every model on its own T4,
# the **fail-fast projection** (refused if it cannot finish in the time left), the scoring.
#
# **Once only, per list:** every finished list is recorded the moment it ends; a re-run with
# this version attached keeps every finished list and scores only the rest.
#
# **Kaggle:** GPU **T4 x2**, Internet on, Secret `HF_TOKEN`, Save & Run All (Commit),
# account `muhammadadeelshah`, **run alone**. Every input is found by its contents:
#
# | input | what |
# |---|---|
# | dataset `dbp-code*` | the code (holds the model record and this runbook) |
# | dataset `dbp-p3-panel-b` | Panel B (`panel_b/<view>/`), private |
# | dataset `report` | the COCO reals (`p1_reals`): the 491 validation reals |
# | dataset `dbp-p3-open-images` | `oi/reals`: the 400 held-out Open Images reals |
# | the ViT-L training notebook's output | `outputs/p3b/s2/s2d_p35_seed{0,1}` and its reports |
# | *(a resumed session only)* this notebook's previous version | finished lists, kept |

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
    "panel": one("Panel B (a directory holding e7_dda_coco_luma)",
                 [p.parent for g in ("*/*/*/e7_dda_coco_luma", "*/*/*/*/e7_dda_coco_luma")
                  for p in IN.glob(g)]),
    "reals": one("COCO reals (p1_reals)", IN.glob("datasets/*/*/p1_reals")),
    "oi_reals": one("Open Images reals (oi/reals)", IN.glob("datasets/*/*/oi/reals")),
    "train": one("ViT-L training output (outputs/p3b/s2)",
                 [p.parents[3] for p in IN.glob(
                     "*/*/*/Implementation/thesis_dbp/outputs/p3b/s2/s2d_p35_seed1")]),
}
PREVIOUS = sorted({p.parents[1] for p in IN.glob("notebooks/*/*/Implementation/thesis_dbp/"
                                                  "reports/p3b_score_plan.json")})
DBP = WORK / "Implementation" / "thesis_dbp"
CORPUS = WORK / "p3b_score_corpus"
LOCAL = TMP / "dbp_p3b_score_local"
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
    print(f"{name:9s} {path}")
print("previous versions:", [str(p) for p in PREVIOUS] or "none (first session)")

# %% Cell 2 — stage the code; hardware at session start; the environment (bootstrap only)
if DBP.exists():
    shutil.rmtree(DBP)
shutil.copytree(INPUTS["code"], DBP)
assert (DBP / "scripts" / "p3b_score_preflight.py").is_file(), \
    "the code dataset predates this runbook"
print("code commit:", (DBP / "COMMIT.txt").read_text().split()[0])
hardware(DBP / "reports" / "p3b_score_hardware.txt")
EXPECT_PATH = pathlib.Path(os.environ.get("DBP_EXPECT", DBP / "env" / "p3b_score_expect.json"))
EXPECT = json.loads(EXPECT_PATH.read_text())
print("expected:", json.dumps(EXPECT["expected"]))
assert sh("bash", DBP / "env" / "kaggle_bootstrap.sh") == 0, "bootstrap FAILED"

# %% Cell 3 — PRE-FLIGHT 1: runtime record, GPUs, token, imports; the stage helper
links = ["--link", f"panel={INPUTS['panel']}", "--link", f"reals={INPUTS['reals']}",
         "--link", f"oi_reals={INPUTS['oi_reals']}"]
assert py("p2_stage_corpus.py", "--root", CORPUS, *links) == 0, "corpus links FAILED"
GPUS = int(os.environ.get("DBP_GPUS", EXPECT["gpus"]))
checks = ["--expect-gpus", GPUS, "--runtime", "--imports", ",".join(EXPECT["imports"])]
checks += ["--hf-token"] if EXPECT["hf_token"] else []
assert py("p2_preflight.py", "--stage", "p3b_score_env", *checks) == 0, \
    "PRE-FLIGHT (environment) FAILED -- nothing has been scored"
PLAN_FILE = DBP / "reports" / "p3b_score_plan.json"
PLAN_ARGS = ["--expect", EXPECT_PATH, "--train-root", INPUTS["train"], "--root", CORPUS,
             "--local-root", LOCAL, "--devices", EXPECT["devices"],
             "--workers", EXPECT["workers"]]
for previous in PREVIOUS:
    PLAN_ARGS += ["--previous", previous]


def stage_round(number):
    """One stage: plan, smoke, fail-fast projection, scoring. Nothing long starts unless the
    first three pass. Returns what happened; never raises (the verdict cell judges)."""
    out = {"round": number, "plan": py("p3b_score_preflight.py", *PLAN_ARGS)}
    plan = json.loads(PLAN_FILE.read_text()) if PLAN_FILE.is_file() else {}
    if PLAN_FILE.is_file():   # each round's plan is kept; the next round's overwrites the file
        shutil.copyfile(PLAN_FILE, PLAN_FILE.with_name(f"p3b_score_plan_round{number}.json"))
    out.update({"stage": plan.get("stage"), "due": plan.get("due", []),
                "resumed": plan.get("resumed"), "problems": plan.get("problems")})
    if out["plan"] != 0 or plan.get("stage") in (None, "complete"):
        return out
    stage = plan["stage"]
    shards = ["--worker", DBP / "scripts" / "p3_final_score.py", "--log-dir",
              DBP / "outputs" / "p3b" / "logs", "--poll-seconds", EXPECT["poll_seconds"]]
    for job in plan["jobs"]:
        shards += ["--job", job]
    worker = [f"--worker-arg={a}" for a in ("--plan", PLAN_FILE, "--batch", EXPECT["batch"],
                                            "--backbone", EXPECT["backbone"])]
    smoke_root = TMP / f"dbp_p3b_smoke_{stage}"
    if smoke_root.exists():
        shutil.rmtree(smoke_root)
    out["smoke"] = py("p2_run_shards.py", *shards, *worker, "--worker-arg=--limit",
                      f"--worker-arg={EXPECT['smoke_limit']}", "--worker-arg=--out-root",
                      f"--worker-arg={smoke_root}", "--log-prefix", f"p3b_smoke_{stage}",
                      "--status", DBP / "reports" / f"p3b_smoke_status_{stage}.json",
                      "--deadline-hours", EXPECT["smoke_deadline_hours"])
    if out["smoke"] != 0:
        return out
    out["failfast"] = py("p3b_failfast.py", "--plan", PLAN_FILE, "--expect", EXPECT_PATH,
                         "--smoke-root", smoke_root,
                         "--elapsed-hours", f"{(time.time() - START) / 3600:.3f}")
    if out["failfast"] != 0:
        return out
    projection = json.loads((DBP / "reports" / f"p3b_projection_{stage}.json").read_text())
    out["score"] = py("p2_run_shards.py", *shards, *worker, "--worker-arg=--resume-lists",
                      "--log-prefix", f"p3b_{stage}",
                      "--status", DBP / "reports" / f"p3b_score_status_{stage}.json",
                      "--deadline-hours", projection["deadline_hours"])
    return out


# %% Cell 4 — stage 1 (clean, unless an earlier version finished it): plan, smoke, fail-fast, score
first = stage_round(1)
print("STAGE ROUND 1:", json.dumps(first))

# %% Cell 5 — stage 2 (robust: jpeg50 + resize050), only after stage 1 completed cleanly
second = stage_round(2) if first.get("score", 0) == 0 and first["plan"] == 0 else \
    {"round": 2, "skipped": "stage 1 did not complete"}
print("STAGE ROUND 2:", json.dumps(second))

# %% Cell 6 — report (cannot fail the run)
latest = "robust" if (DBP / "reports" / "p3b_score_status_robust.json").is_file() else "clean"
py("p2_report.py", "--stage", "final", "--status", f"reports/p3b_score_status_{latest}.json",
   "--zip", WORK / "p3b_score_reports")
times = []
for model in [m["id"] for m in EXPECT["models"]]:
    for stage in ("clean", "robust"):
        path = DBP / "reports" / "p3_final" / f"{model}_{stage}.json"
        summary = json.loads(path.read_text()) if path.is_file() else {}
        lists = summary.get("lists", {})
        times.append(f"{model} {stage} {sorted(lists)} "
                     f"{round(sum(v.get('seconds', 0) for v in lists.values()))} s "
                     f"kept {summary.get('kept_lists')}")
print("THROUGHPUT |", " | ".join(times))

# %% Cell 7 — verdict: this cell fails the notebook unless both stages are complete
done = {f"{m['id']}_{s}": (DBP / "reports" / "p3_final" / f"{m['id']}_{s}.json").is_file()
        for m in EXPECT["models"] for s in ("clean", "robust")}
assert all(done.values()), (f"not complete: {[k for k, v in done.items() if not v]}; rounds "
                            f"{first} / {second}. Re-run with this version attached: finished "
                            f"lists are kept, the rest are scored.")
print("P3b SCORING COMPLETE:", sorted(done))
