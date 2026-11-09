# %% [markdown]
# # P4 — the baselines on Panel B: every row of the frozen spec, once (descriptive)
#
# **Spec `docs/P4_BASELINES_SPEC.md`** (frozen, tag `p4-baselines-frozen`; the pre-flight
# checks its sha256). Seven rows, each scored once, on Panel B (9,405), its luma
# companion (700) and the 891 threshold reals -- the P3 final scoring's committed lists:
#
# | row | what |
# |---|---|
# | `dda_native`, `dda_crop308` | DDA at `CenterCrop(336)` / `(308)` |
# | `dda_jpeg50`, `dda_resize050` | DDA native, Panel B test rows, the two H5 conditions |
# | `cospy_native` | CO-SPY, its own transform, the fused logit |
# | `fire_native` | FIRE, `--resize`, one image at a time, the seed rule |
# | `aeroblade` | AEROBLADE (ours), 768 cap, `sd15` + `sd21` + KD2.1 MoVQ, LPIPS layer 2 |
#
# Rows already scored are in the code (`reports/p4_baselines/`) and are never scored again.
# **AEROBLADE is split** (user decision 2026-11-09, execution only): its images are dealt
# to both T4s by capped pixels (`aeroblade#0`, `aeroblade#1`), saved in append-only
# checkpoints every 200 images, and merged into the normal `aeroblade` row at the end. A
# session that stops early keeps every checkpoint: attach its version and run again; each
# shard re-scores 20 saved rows first and stops if any differs beyond fp32 rounding.
#
# **Fail fast** (every row): the smoke times each unit after one warm-up image, and the run
# is refused if any GPU's projection exceeds the time left in the session.
#
# **Kaggle:** GPU **T4 x2**, Internet on, Secret `HF_TOKEN`, Save & Run All (Commit),
# account `muhammadadeelshah`, **run alone**. Inputs, found by their contents:
#
# | input | what |
# |---|---|
# | dataset `dbp-code*` | the code (with the weights session's sha256 values committed) |
# | dataset `dbp-p3-panel-b` | Panel B (`panel_b/<view>/`), private |
# | dataset `report` | the COCO reals (`p1_reals`) |
# | dataset `dbp-p3-open-images` | `oi/reals`: the 400 held-out Open Images reals |
# | notebook output `dbp-p4-weights` | `p4_weights/` and its report (step 1) |
# | *(a re-run)* this notebook's previous version | complete rows and checkpoints, carried |

# %% Cell 1 — find the inputs by their contents, and the helpers every later cell uses
import json
import os
import pathlib
import platform
import shutil
import subprocess
import sys
import time

SESSION_START = time.time()
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
    "weights": one("the weights session's output (p4_weights/)",
                   [p.parent for p in IN.glob("*/*/*/p4_weights/dda/DDA_ckpt.pth")]
                   ).parent,
}
PREVIOUS = sorted({p.parents[1] for p in IN.glob("notebooks/*/*/Implementation/thesis_dbp/"
                                                  "reports/p4_plan.json")})
DBP = WORK / "Implementation" / "thesis_dbp"
CORPUS = WORK / "p4_corpus"
LOCAL = TMP / "dbp_p4_local"
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
    print(f"{name:9s} {path}")
print("previous versions:", [str(p) for p in PREVIOUS] or "none (session 1)")

# %% Cell 2 — stage the code; hardware at session start; the environment (bootstrap only)
if DBP.exists():
    shutil.rmtree(DBP)
shutil.copytree(INPUTS["code"], DBP)
assert (DBP / "scripts" / "p4_preflight.py").is_file(), "the code dataset predates this runbook"
print("code commit:", (DBP / "COMMIT.txt").read_text().split()[0])
hardware(DBP / "reports" / "p4_hardware.txt")
EXPECT_PATH = pathlib.Path(os.environ.get("DBP_EXPECT", DBP / "env" / "p4_baselines_expect.json"))
EXPECT = json.loads(EXPECT_PATH.read_text())
print("expected:", json.dumps(EXPECT["expected"]))
assert sh("bash", DBP / "env" / "kaggle_bootstrap.sh") == 0, "bootstrap FAILED"

# %% Cell 3 — PRE-FLIGHT 1: GPUs, token, imports; the frozen spec, lists, weights, staging
links = ["--link", f"panel={INPUTS['panel']}", "--link", f"reals={INPUTS['reals']}",
         "--link", f"oi_reals={INPUTS['oi_reals']}"]
assert py("p2_stage_corpus.py", "--root", CORPUS, *links) == 0, "corpus links FAILED"
GPUS = int(os.environ.get("DBP_GPUS", EXPECT["gpus"]))
checks = ["--expect-gpus", GPUS, "--imports", ",".join(EXPECT["imports"])]
checks += ["--hf-token"] if EXPECT["hf_token"] else []
assert py("p2_preflight.py", "--stage", "p4_env", *checks) == 0, \
    "PRE-FLIGHT (environment) FAILED -- nothing has been scored"
plan_args = ["--expect", EXPECT_PATH, "--weights", INPUTS["weights"], "--root", CORPUS,
             "--local-root", LOCAL, "--workers", EXPECT["workers"]]
for previous in PREVIOUS:
    plan_args += ["--previous", previous]
assert py("p4_preflight.py", *plan_args) == 0, \
    "PRE-FLIGHT (spec / lists / weights / staging) FAILED -- see reports/p4_plan.json; " \
    "nothing has been scored"
PLAN = json.loads((DBP / "reports" / "p4_plan.json").read_text())
print("STAGE:", PLAN["stage"], "| units due:", PLAN["due"], "| jobs:", PLAN["jobs"],
      "| merge due:", PLAN["merge_due"])
for row, spec in PLAN.get("shards", {}).items():
    print(f"{row}: {spec['pixels'] / 1e9:.3f} Gpx;",
          {u: f"{s['items']} images {s['pixels'] / 1e9:.3f} Gpx, {s['saved_rows']} saved"
           for u, s in spec["units"].items()},
          f"| {spec['carried_files']} checkpoint files carried")
print("incomplete earlier (scored again; log the reason):", PLAN["incomplete_previous"])
assert PLAN["stage"] != "complete", "EVERY ROW IS ALREADY SCORED"

SHARDS = ["--worker", DBP / "scripts" / "p4_score.py", "--log-dir",
          DBP / "outputs" / "p4" / "logs", "--poll-seconds", EXPECT["poll_seconds"]]
for job in PLAN["jobs"]:
    SHARDS += ["--job", job]
WORKER = [f"--worker-arg={a}" for a in ("--plan", DBP / "reports" / "p4_plan.json")]
SMOKE = TMP / "dbp_p4_smoke"

# %% Cell 4 — PRE-FLIGHT 2: every unit due, timed on 2 images per list; FAIL FAST on time
if PLAN["due"]:
    assert py("p2_run_shards.py", *SHARDS, *WORKER, "--worker-arg=--limit", "--worker-arg=2",
              "--worker-arg=--out-root", f"--worker-arg={SMOKE}", "--log-prefix", "p4_smoke",
              "--status", DBP / "reports" / "p4_smoke_status.json",
              "--deadline-hours", EXPECT["smoke_deadline_hours"]) == 0, \
        "PRE-FLIGHT (rows) FAILED -- nothing has been scored"
    elapsed = (time.time() - SESSION_START) / 3600
    assert py("p4_failfast.py", "--plan", DBP / "reports" / "p4_plan.json", "--expect",
              EXPECT_PATH, "--smoke-root", SMOKE, "--elapsed-hours", f"{elapsed:.4f}") == 0, \
        "FAIL-FAST: the projection exceeds the time left in this session -- REFUSED, " \
        "nothing long has started (see reports/p4_projection.json)"
    PROJECTION = json.loads((DBP / "reports" / "p4_projection.json").read_text())
    print("projected:", {j["device"]: j["hours"] for j in PROJECTION["jobs"]},
          "h; available", PROJECTION["available_hours"], "h")

# %% Cell 5 — score every unit due, once; one worker per T4
code = 0
if PLAN["due"]:
    code = py("p2_run_shards.py", *SHARDS, *WORKER, "--log-prefix", "p4_score",
              "--status", DBP / "reports" / "p4_status.json",
              "--deadline-hours", PROJECTION["deadline_hours"])
print(f"scoring exited {code} (0 = complete, 1 = a worker failed, 2 = deadline)")

# %% Cell 6 — merge the shards into their rows (the work step; verifies every image)
merge_code = py("p4_merge.py", "--plan", DBP / "reports" / "p4_plan.json") \
    if PLAN.get("shards") else 0
print(f"merge exited {merge_code} (0 = merged or nothing to merge, 2 = a shard is not done, "
      f"1 = verification FAILED)")

# %% Cell 7 — report (cannot fail the run)
py("p2_report.py", "--stage", "p4", "--status", "reports/p4_status.json",
   "--zip", WORK / "p4_reports")
staged = PLAN.get("staged", {})
times = []
for row in PLAN["rows_due"]:
    path = DBP / "reports" / "p4_baselines" / f"{row}.json"
    summary = json.loads(path.read_text()) if path.is_file() else {}
    rates = {k: v.get("seconds_per_image") for k, v in summary.get("lists", {}).items()}
    times.append(f"{row} {rates or 'not complete'}")
print("THROUGHPUT |", f"staging {staged.get('files')} files {staged.get('gb')} GB in "
      f"{staged.get('seconds')} s |", " | ".join(times))

# %% Cell 8 — verdict: this cell fails the notebook if a row due was not completed
missing = [r for r in PLAN["rows_due"]
           if not (DBP / "reports" / "p4_baselines" / f"{r}.json").is_file()]
assert code == 0 and merge_code == 0 and not missing, (
    f"scoring exited {code}, merge {merge_code}; not complete: {missing}. Re-run with this "
    f"version attached: complete rows and every checkpoint are carried forward.")
print("P4 BASELINE SCORING COMPLETE:", PLAN["rows_due"])
