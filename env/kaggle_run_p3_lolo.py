# %% [markdown]
# # P3 Phase C — two LOLO folds per session, one per T4 (S2)
#
# Leave-one-lineage-out, pre-registration section 3 (FROZEN 2026-10-08): fold f trains S2
# with lineage f's members held out (fold 1 also without `gen_matched`), the S2-B recipe
# (P2's trainer and hyper-parameters, fixed steps, no new reals, no degradation), for
# exactly `N_max` = 6,000 steps, evaluating the held-out lineage every 500 steps and
# saving a snapshot there. **Nothing is selected here**: `N*` is chosen only after all
# eight folds exist (nested rule), and the folds are scored on Panel B only then.
#
# **The session picks its own folds**: the two lowest-numbered folds without an `OK`
# report in this session or any attached previous one (`p3_lolo_next.py`), one per T4
# through the queue (one worker per GPU), so a fold interrupted by a session limit is
# picked again first and resumes from its 500-step checkpoint. Every finished fold is
# carried forward into this session's output. No cell is edited between sessions.
#
# **Platform-neutral.** Every path comes from the environment, with Kaggle's as defaults:
# `DBP_INPUT_ROOT` (inputs, laid out as `datasets/<owner>/<slug>/`), `DBP_WORKING_ROOT`,
# `DBP_TMP_ROOT`, `DBP_RESUME_DIRS` (extra previous-session `thesis_dbp` roots, separated
# by the OS path separator), `DBP_GPUS` (visible GPUs; the expect file says 2 for a Kaggle
# T4 x2) and `DBP_FOLD` (override the choice).
#
# **Kaggle:** Accelerator **GPU T4 x2** (one fold per GPU). **Internet:** on.
# **Secret:** `HF_TOKEN` (DINOv3 is gated). **Mode:** Save & Run All (Commit).
# **Account:** `muhammadadeelshah` only.
#
# | input | what |
# |---|---|
# | dataset `dbp-code*` | the code at the commit that holds this runbook |
# | dataset `dbp-p2-s2-inputs` | pairs manifest, split, generation manifest (hashed) |
# | dataset holding `p1_reals/` and `p1_generated/` (slug `report`) | corpus |
# | dataset `thesis-dbp-1-dataset` | corpus pair reconstructions |
# | *(session 2 on)* this notebook's previous version | done folds skipped; unfinished resumes |

# %% Cell 1 — find the inputs, and the helpers every later cell uses
import json
import os
import pathlib
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


INPUTS = {
    "code": find("datasets/*/dbp-code*/thesis_dbp"),
    "s2_inputs": find("datasets/*/dbp-p2-s2-inputs"),
    "reals": find("datasets/*/*/p1_reals"),
    "generated": find("datasets/*/*/p1_generated"),
    "pairs": find("datasets/*/thesis-dbp-1-dataset/Implementation/thesis_dbp/outputs/p1/pairs"),
}
PREVIOUS = sorted(p.parents[2] for p in
                  IN.glob("notebooks/*/*/Implementation/thesis_dbp/outputs/p3/s2"))
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


for name, path in INPUTS.items():
    print(f"{name:10s} {path}")
print("resume from:", [str(p) for p in PREVIOUS] or "nothing (first session)")

# %% Cell 2 — stage the code; set up the environment (the bootstrap is the only way)
if DBP.exists():
    shutil.rmtree(DBP)
shutil.copytree(INPUTS["code"], DBP)
assert (DBP / "scripts" / "p3_lolo_next.py").is_file(), "the code dataset predates this runbook"
print("code commit:", (DBP / "COMMIT.txt").read_text().split()[0])
EXPECT = json.loads(pathlib.Path(
    os.environ.get("DBP_EXPECT", DBP / "env" / "p3_lolo_expect.json")).read_text())
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
# The reference is the read-only committed manifest, not the one the matcher just wrote.
assert py("p2_preflight.py", "--stage", "p3_lolo_data", *checks, "--root", P1,
          "--gen-matched-reference", INPUTS["code"] / "reports" / "p1_gen_matched_manifest.csv",
          "--gen-matched-dir", P1 / "gen_matched",
          "--expect-gen-matched", EXPECT["gen_matched"]) == 0, \
    "PRE-FLIGHT (data) FAILED -- nothing long has started"

# %% Cell 4 — which folds this session trains (two lowest unfinished; DBP_FOLD overrides)
choose = ["--count", EXPECT["folds_per_session"]]
choose += ["--fold", os.environ["DBP_FOLD"]] if os.environ.get("DBP_FOLD") else []
for previous in PREVIOUS:
    choose += ["--resume-from", previous]
chosen = py("p3_lolo_next.py", *choose)
PLAN = json.loads((DBP / "reports" / "p3_lolo_plan.json").read_text())
assert chosen in (0, 3), "fold choice FAILED"
assert not PLAN["all_complete"], "ALL EIGHT FOLDS ARE COMPLETE -- run the N* selection instead"
print("this session trains:", PLAN["selected"], "| complete:", PLAN["complete"])

QUEUE = ["--devices", EXPECT["devices"], "--root", P1,
         "--pairs-manifest", S2IN / "p1_pairs_manifest.csv", "--splits", S2IN / "p2_splits.csv",
         "--scratch", TMP / "dbp_lolo_smoke", "--poll-seconds", EXPECT["poll_seconds"],
         "--train-only", "--out-dir", DBP / "outputs" / "p3" / "s2",
         "--log-dir", DBP / "outputs" / "p3" / "logs"]
for run in PLAN["selected"]:
    QUEUE += ["--run", run]
QUEUE += [f"--train-arg={a}" for a in EXPECT["train_args"]]

# %% Cell 5 — PRE-FLIGHT 2: each fold on each GPU -- 2 steps, held-out eval, save, reload
assert py("p2_run_queue.py", *QUEUE, "--smoke", EXPECT["smoke_steps"],
          "--status", DBP / "reports" / "p3_lolo_smoke_status.json",
          "--deadline-hours", EXPECT["smoke_deadline_hours"]) == 0, \
    "PRE-FLIGHT (models) FAILED -- nothing long has started"

# %% Cell 6 — train the folds, one per GPU: 6,000 fixed steps, held-out AUC + snapshot / 500
resume = []
for previous in PREVIOUS:
    resume += ["--resume-from", previous]
code = py("p2_run_queue.py", *QUEUE, *resume, "--status",
          DBP / "reports" / "p3_lolo_status.json", "--deadline-hours", EXPECT["deadline_hours"])
print(f"queue exited {code} (0 = folds complete, 1 = failed, 2 = deadline; resumable)")

# %% Cell 7 — report (cannot fail the run)
py("p2_report.py", "--stage", "s2", "--status", "reports/p3_lolo_status.json",
   "--s2-dir", DBP / "outputs" / "p3" / "s2", "--zip", WORK / "p3_lolo_reports")
for run in PLAN["selected"]:
    path = DBP / "reports" / f"p2_s2_{run.replace(':', '_seed')}.json"
    if path.is_file():
        report = json.loads(path.read_text())
        print(run, report.get("status"), "| steps", report.get("steps_run"), "| held-out AUC",
              [(p["step"], p["heldout_auc"]) for p in report.get("curve", [])])

# %% Cell 8 — verdict: this cell fails the notebook if a fold did not complete
assert code == 0, (f"LOLO queue exited {code}: see reports/p3_lolo_status.json. An "
                   f"unfinished fold resumes from its last checkpoint when this notebook is "
                   f"re-run with this version's output attached.")
print("LOLO FOLDS COMPLETE:", PLAN["selected"], "| all complete so far:",
      sorted(set(PLAN["complete"]) | {r.split(":")[0] for r in PLAN["selected"]}))
