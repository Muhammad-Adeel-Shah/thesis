# %% [markdown]
# # P3 §4b — degradation tuning: two S2-B+deg candidates, one per T4, and the committed rule
#
# **What runs** (rule committed 2026-10-17, before either candidate trained; unchanged):
# 1. `S2-B` seed 0 (from the `dbp-p3-train` output) is scored on the **development**
#    panel (the P2 panel) and the 491 COCO validation reals, unless a previous version of
#    this notebook already did -- **only the rows the rule reads** (user, 2026-10-21):
#    6,717 dev rows (E3 reals + the two E3 unseen generators, E4, E5) and the 491 reals,
#    selected by `p3_tune_lists.py` with the rule's own definitions;
# 2. the two candidates `s2b_deg_p70` and `s2b_deg_p35` (S2-B + the real-side degradation
#    chain at p = 0.7 / 0.35; 4 operations at 1/4, parameters uniform, JPEG chroma 50/50;
#    one chain per pair, identical for both members) train in parallel, one per T4, on the
#    P3 recipe (6,000-step schedule stopped at 5,000), and are scored the same way;
# 3. `p3_tune_degradation.py` applies the rule: dev macro achieved FPR over E4 and E5 at
#    the COCO-only k = 0 threshold, guarded by the dev macro AUC of the four unseen slices
#    (no more than 1 point below S2-B seed 0). The chosen candidate IS the pre-registered
#    `S2-B+deg` ablation.
#
# **Development data only. Panel B is never read.** The smoke measures the degradation's
# data-loader cost (plan §4b). Every long step is time-bounded.
#
# **Kaggle:** GPU **T4 x2**, Internet on, Secret `HF_TOKEN`, Save & Run All (Commit),
# account `muhammadadeelshah`. **Run it alone** -- no other GPU session at the same time.
#
# | input | what |
# |---|---|
# | dataset `dbp-code*` | the code at the commit that holds this runbook |
# | dataset `dbp-p2-s2-inputs` | pairs, split, generations, **the dev panel and val lists** |
# | dataset holding `p1_reals/` and `p1_generated/` (slug `report`) | corpus |
# | dataset `thesis-dbp-1-dataset` | corpus pair reconstructions |
# | the P2 dev panel (`dbp-p2-panel`, or the `phase-2` notebook's `panel/`) | the panel images |
# | notebook output `dbp-p3-train` (latest version) | the trained `S2-B` seed 0 |
# | *(only to resume)* this notebook's previous version | done candidates skipped |

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


def by_layout(*patterns):
    """The one directory whose layout matches (by content, not by name)."""
    hits = sorted({p.parent for pattern in patterns for p in IN.glob(pattern) if p.is_dir()})
    assert len(hits) == 1, f"expected exactly one input with {patterns}, found {hits}"
    return hits[0]


INPUTS = {
    "code": find("datasets/*/dbp-code*/thesis_dbp"),
    "s2_inputs": find("datasets/*/dbp-p2-s2-inputs"),
    "reals": find("datasets/*/*/p1_reals"),
    "generated": find("datasets/*/*/p1_generated"),
    "pairs": find("datasets/*/thesis-dbp-1-dataset/Implementation/thesis_dbp/outputs/p1/pairs"),
    # The dev panel root holds one directory per view.
    "panel": by_layout("*/*/*/e5_chameleon_pipeline", "*/*/*/*/e5_chameleon_pipeline"),
}
# This notebook's previous versions: p3 outputs holding a candidate.
PREVIOUS = sorted({p.parents[3] for p in
                   IN.glob("notebooks/*/*/Implementation/thesis_dbp/outputs/p3/s2/s2b_deg_*")})
# The dbp-p3-train output: the thesis_dbp root holding the trained S2-B seed 0 -- not a
# previous version of this notebook, which carries a copy of it.
TRAIN = sorted({p.parents[3] for p in
                IN.glob("*/*/*/Implementation/thesis_dbp/outputs/p3/s2/s2b_seed0")}
               - set(PREVIOUS))
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
print("S2-B source:", [str(p) for p in TRAIN] or "NONE")
print("resume from:", [str(p) for p in PREVIOUS] or "nothing (first session)")

# %% Cell 2 — stage the code; set up the environment (the bootstrap is the only way)
if DBP.exists():
    shutil.rmtree(DBP)
shutil.copytree(INPUTS["code"], DBP)
assert (DBP / "scripts" / "p3_tune_degradation.py").is_file(), \
    "the code dataset predates this runbook"
print("code commit:", (DBP / "COMMIT.txt").read_text().split()[0])
hardware(DBP / "reports" / "p3_tune_hardware.txt")
EXPECT = json.loads(pathlib.Path(
    os.environ.get("DBP_EXPECT", DBP / "env" / "p3_tune_expect.json")).read_text())
print("expected:", json.dumps(EXPECT["expected"]))
assert sh("bash", DBP / "env" / "kaggle_bootstrap.sh") == 0, "bootstrap FAILED"

# %% Cell 3 — corpus, gen_matched; PRE-FLIGHT 1: GPUs, token, imports, hashes, every dev row
links = []
for name in ("reals", "generated", "pairs"):
    links += ["--link", f"{name}={INPUTS[name]}", "--expect", f"{name}={EXPECT['corpus'][name]}"]
assert py("p2_stage_corpus.py", "--root", P1, *links) == 0, "corpus staging FAILED"
assert py("p1_match_generations.py",
          "--gen-manifest", S2IN / "p1_gen_manifest.csv",
          "--reals-manifest", "reports/p1_reals_manifest_t_real.csv",
          "--gen-root", P1 / "generated", "--out-root", P1, "--set-id", "gen_matched") == 0, \
    "gen_matched regeneration FAILED"
# Only the rows the committed rule reads are scored (user, 2026-10-21): derived with the
# rule's own definitions and checked against the committed sha256 and row counts.
TUNE_LISTS = DBP / "reports" / "p3_tune_lists"
assert py("p3_tune_lists.py", "--panel-list", S2IN / "p2_panel_images.csv",
          "--val-list", S2IN / "p2_s1_images.csv", "--out-dir", TUNE_LISTS,
          "--expect", pathlib.Path(os.environ.get("DBP_EXPECT", DBP / "env" /
                                                  "p3_tune_expect.json"))) == 0, \
    "tuning lists FAILED -- nothing long has started"
GPUS = int(os.environ.get("DBP_GPUS", EXPECT["gpus"]))
checks = ["--expect-gpus", GPUS, "--imports", ",".join(EXPECT["imports"])]
checks += ["--hf-token"] if EXPECT["hf_token"] else []
for name, spec in EXPECT["inputs"].items():
    checks += ["--sha256", f"{S2IN / name}={spec['sha256']}"]
    checks += ["--rows", f"{S2IN / name}={spec['rows']}"] if "rows" in spec else []
for name in EXPECT["resolve"]:
    checks += ["--resolve", TUNE_LISTS / name]
assert py("p2_preflight.py", "--stage", "p3_tune_data", *checks,
          "--root", P1, "--panel-root", INPUTS["panel"],
          "--gen-matched-reference", INPUTS["code"] / "reports" / "p1_gen_matched_manifest.csv",
          "--gen-matched-dir", P1 / "gen_matched",
          "--expect-gen-matched", EXPECT["gen_matched"]) == 0, \
    "PRE-FLIGHT (data) FAILED -- nothing long has started"

# %% Cell 4 — bring in the trained S2-B seed 0 (and any finished work of a previous version)
OUT = DBP / "outputs" / "p3" / "s2"
assert len(TRAIN) == 1, f"attach exactly one dbp-p3-train output holding S2-B seed 0: {TRAIN}"
for source in [*PREVIOUS, *TRAIN]:              # a previous version's copy may hold scores
    if not (OUT / "s2b_seed0").exists() and (source / "outputs/p3/s2/s2b_seed0").is_dir():
        shutil.copytree(source / "outputs/p3/s2/s2b_seed0", OUT / "s2b_seed0")
        for name in ("p2_s2_s2b_seed0.json", "p2_s2_scores_s2b_seed0.json"):
            if (source / "reports" / name).is_file():
                shutil.copy2(source / "reports" / name, DBP / "reports" / name)
        print("S2-B seed 0 from", source)
BASE = json.loads((DBP / "reports/p2_s2_s2b_seed0.json").read_text())
assert BASE["status"] == "OK" and BASE["stop_at"] == EXPECT["base_stop_at"] \
    and BASE["steps_run"] == EXPECT["base_stop_at"], \
    f"the attached S2-B seed 0 is not the finished P3-recipe run: {BASE['status']}"
# Scored = a full scoring summary (no --limit) naming every list; a smoke never counts.
SUMMARY = DBP / "reports" / "p2_s2_scores_s2b_seed0.json"
BASE_SCORED = SUMMARY.is_file() and json.loads(SUMMARY.read_text()).get("limit") is None \
    and set(json.loads(SUMMARY.read_text()).get("lists", {})) >= set(EXPECT["lists"])
LISTS = []
for name, (csv_name, view) in EXPECT["lists"].items():
    LISTS += ["--list", f"{name}={TUNE_LISTS / csv_name}"]
    LISTS += ["--only", f"{name}={view}"] if view else []
SCORE_BASE = ["--protocol", "s2b", "--seed", 0, "--root", P1, "--panel-root", INPUTS["panel"],
              *LISTS, "--out-dir", OUT, "--reports-dir", DBP / "reports", "--device", DEVICE,
              *EXPECT["infer_args"]]
QUEUE = ["--devices", EXPECT["devices"], "--root", P1, "--panel-root", INPUTS["panel"],
         "--pairs-manifest", S2IN / "p1_pairs_manifest.csv", "--splits", S2IN / "p2_splits.csv",
         "--scratch", TMP / "dbp_p3_tune_smoke", "--poll-seconds", EXPECT["poll_seconds"],
         "--out-dir", OUT, "--log-dir", DBP / "outputs" / "p3" / "logs", *LISTS]
for run in EXPECT["runs"]:
    QUEUE += ["--run", run]
QUEUE += [f"--train-arg={a}" for a in EXPECT["train_args"]]
QUEUE += [f"--infer-arg={a}" for a in EXPECT["infer_args"]]
print("S2-B seed 0 dev scores present:", BASE_SCORED)

# %% Cell 5 — PRE-FLIGHT 2: each candidate on each GPU (2 steps + scoring; degradation cost)
assert py("p2_run_queue.py", *QUEUE, "--smoke", EXPECT["smoke_steps"],
          "--status", DBP / "reports" / "p3_tune_smoke_status.json",
          "--deadline-hours", EXPECT["smoke_deadline_hours"]) == 0, \
    "PRE-FLIGHT (models) FAILED -- nothing long has started"
if not BASE_SCORED:
    assert py("p2_s2_infer.py", *SCORE_BASE, "--limit", 2,
              "--deadline-hours", EXPECT["base_smoke_deadline_hours"]) == 0, \
        "PRE-FLIGHT (S2-B scoring) FAILED -- nothing long has started"
for path in sorted((TMP / "dbp_p3_tune_smoke").rglob("p2_s2_s2b_deg_*.json")):
    report = json.loads(path.read_text())
    print(path.parent.name, report.get("status"), "| degradation cost:", report.get("degrade"))

# %% Cell 6 — score S2-B seed 0 on the dev panel and the val reals (skipped if done)
base = 0
if not BASE_SCORED:
    base = py("p2_s2_infer.py", *SCORE_BASE,
              "--deadline-hours", EXPECT["base_score_deadline_hours"])
print(f"S2-B seed 0 scoring exited {base}" if not BASE_SCORED else "S2-B seed 0: already scored")

# %% Cell 7 — train and score both candidates, one per GPU (resumable)
resume = []
for previous in PREVIOUS:
    resume += ["--resume-from", previous]
code = py("p2_run_queue.py", *QUEUE, *resume, "--status",
          DBP / "reports" / "p3_tune_status.json", "--deadline-hours", EXPECT["deadline_hours"])
print(f"queue exited {code} (0 = complete, 1 = failed, 2 = deadline; resumable)")

# %% Cell 8 — apply the committed rule; report (cannot fail the run)
if base == 0 and code == 0:
    py("p3_tune_degradation.py", "--scores-root", OUT,
       "--panel-list", TUNE_LISTS / "p3_tune_panel_rows.csv",
       "--val-list", S2IN / "p2_s1_images.csv")
py("p2_report.py", "--stage", "s2", "--status", "reports/p3_tune_status.json",
   "--s2-dir", OUT, "--zip", WORK / "p3_tune_reports")
choice = DBP / "reports" / "p3_degradation_choice.json"
print(choice.read_text() if choice.is_file() else "no choice written")
throughput = []
for run in EXPECT["runs"]:
    path = DBP / "reports" / f"p2_s2_{run.replace(':', '_seed')}.json"
    report = json.loads(path.read_text()) if path.is_file() else {}
    throughput.append(f"{run} {report.get('steps_run')} steps "
                      f"{report.get('measured_step_seconds_train_loop')} s/step "
                      f"{report.get('gpu_hours')} GPU-h")
status = DBP / "reports" / "p3_tune_status.json"
wall = json.loads(status.read_text()).get("wall_hours") if status.is_file() else None
print("THROUGHPUT |", " | ".join(throughput), f"| queue wall {wall} h")

# %% Cell 9 — verdict: this cell fails the notebook if anything did not complete
assert base == 0, f"S2-B seed 0 scoring exited {base}: re-run with this version attached"
assert code == 0, f"queue exited {code}: see reports/p3_tune_status.json (resumable)"
assert choice.is_file(), "the rule was not applied: see the Cell 8 output"
decision = json.loads(choice.read_text())
print("CHOSEN:", decision["chosen"], "p =", decision["chosen_p"], "--", decision["reason"])
