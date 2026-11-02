# %% [markdown]
# # P2 Session D — the four S2 runs, trained and scored
#
# One queue, one worker per T4. Each worker takes the next run as soon as its GPU is
# free, trains it (`p2_train_s2.py`, checkpoint every 500 steps) and scores the panel,
# the luma companion and the validation rows with that run's selected model
# (`p2_s2_infer.py`). The runs, in queue order: `p_a_sd1` seed 0, `p_a_sd1` seed 1,
# `4ch_only` seed 0, `full_bank` seed 0.
#
# **Accelerator:** GPU T4 x2. **Internet:** on. **Secret:** `HF_TOKEN` (DINOv3 is gated).
# **Mode:** Save & Run All (Commit). **Account:** your own only (CLAUDE.md section 13).
#
# **Inputs to attach** (Cell 1 finds each by name, under whichever account owns it):
#
# | input | what |
# |---|---|
# | dataset `dbp-code*` | the code: `dbp-code`, or a per-commit `dbp-code-<commit>` |
# | dataset `dbp-p2-s2-inputs` | pairs manifest, split, image lists (hashed in the expect file) |
# | notebook `phase-2` | the frozen panel |
# | dataset holding `p1_reals/` and `p1_generated/` (slug `report`, title `p1-data`) | corpus |
# | dataset `thesis-dbp-1-dataset` | corpus pair reconstructions |
# | *(resume only)* this notebook's last version | its output: unfinished runs resume |
#
# Every cell is a thin call to a committed, tested script (CLAUDE.md section 12). Two
# pre-flights run before anything long: data (paths, hashes, GPUs, token, imports), then
# models (a 2-step train, save, reload and a 2-row score for every run on every GPU).
# This file is executed verbatim by `scripts/runbook_dryrun.py` before hand-over.

# %% Cell 1 — find the inputs, and the one helper every later cell uses
import json
import os
import pathlib
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
    "code": find("datasets/*/dbp-code*/thesis_dbp"),  # e.g. dbp-code-ed7a8cb
    "s2_inputs": find("datasets/*/dbp-p2-s2-inputs"),
    "panel": find("notebooks/*/phase-2/panel"),
    "reals": find("datasets/*/*/p1_reals"),  # the p1-data upload; its slug is `report`
    "generated": find("datasets/*/*/p1_generated"),
    "pairs": find("datasets/*/thesis-dbp-1-dataset/Implementation/thesis_dbp/outputs/p1/pairs"),
}
PREVIOUS = sorted(p.parents[2] for p in
                  IN.glob("notebooks/*/*/Implementation/thesis_dbp/outputs/p2/s2"))

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
assert (DBP / "scripts" / "p2_run_queue.py").is_file(), "the code dataset predates this runbook"
print("code commit:", (DBP / "COMMIT.txt").read_text().split()[0])

EXPECT = json.loads(pathlib.Path(
    os.environ.get("DBP_EXPECT", DBP / "env" / "p2_s2_expect.json")).read_text())
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

checks = ["--expect-gpus", EXPECT["gpus"], "--imports", ",".join(EXPECT["imports"])]
checks += ["--hf-token"] if EXPECT["hf_token"] else []
for name, spec in EXPECT["inputs"].items():
    checks += ["--sha256", f"{S2IN / name}={spec['sha256']}"]
    checks += ["--rows", f"{S2IN / name}={spec['rows']}"] if "rows" in spec else []
for name in EXPECT["resolve"]:
    checks += ["--resolve", S2IN / name]
# The reference is the read-only committed manifest, not the one the matcher just wrote.
assert py("p2_preflight.py", "--stage", "s2_data", *checks,
          "--root", P1, "--panel-root", INPUTS["panel"],
          "--gen-matched-reference", INPUTS["code"] / "reports" / "p1_gen_matched_manifest.csv",
          "--gen-matched-dir", P1 / "gen_matched",
          "--expect-gen-matched", EXPECT["gen_matched"]) == 0, \
    "PRE-FLIGHT (data) FAILED -- nothing long has started"

QUEUE = ["--devices", EXPECT["devices"], "--root", P1, "--panel-root", INPUTS["panel"],
         "--pairs-manifest", S2IN / "p1_pairs_manifest.csv", "--splits", S2IN / "p2_splits.csv",
         "--scratch", TMP / "dbp_s2_smoke", "--poll-seconds", EXPECT["poll_seconds"]]
for run in EXPECT["runs"]:
    QUEUE += ["--run", run]
for name, (csv_name, view) in EXPECT["lists"].items():
    QUEUE += ["--list", f"{name}={S2IN / csv_name}"]
    QUEUE += ["--only", f"{name}={view}"] if view else []
QUEUE += [f"--train-arg={a}" for a in EXPECT["train_args"]]
QUEUE += [f"--infer-arg={a}" for a in EXPECT["infer_args"]]

# %% Cell 4 — PRE-FLIGHT 2: every run on every GPU -- 2 steps, save, reload, score 2 rows
assert py("p2_run_queue.py", *QUEUE, "--smoke", EXPECT["smoke_steps"],
          "--deadline-hours", EXPECT["smoke_deadline_hours"]) == 0, \
    "PRE-FLIGHT (models) FAILED -- nothing long has started"

# %% Cell 5 — S2: train and score every run from one queue (a failed run does not stop the rest)
resume = []
for previous in PREVIOUS:
    resume += ["--resume-from", previous]
code = py("p2_run_queue.py", *QUEUE, *resume, "--deadline-hours", EXPECT["deadline_hours"])
print(f"queue exited {code} (0 = all complete, 1 = a run failed, 2 = deadline)")

# %% Cell 6 — report (cannot fail the run)
py("p2_report.py", "--stage", "s2", "--status", "reports/p2_s2_status.json",
   "--zip", WORK / "p2_s2_reports")

# %% Cell 7 — verdict: this cell fails the notebook if any run did not complete
assert code == 0, (f"S2 queue exited {code}: see the status table above and "
                   f"reports/p2_s2_status.json. Completed runs are saved and will be skipped "
                   f"when this notebook is re-run with this version's output attached.")
