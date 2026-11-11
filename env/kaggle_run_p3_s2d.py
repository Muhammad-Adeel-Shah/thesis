# %% [markdown]
# # P3 — S2-D (seeds 0 and 1) and S2-B+reals: COCO + Open Images train reals, pooled by real
#
# **The models** (pre-registration section 3):
# - **`S2-D`** = S2-B + the 3,600 Open Images V7 train reals with their reconstructions
#   by all 12 members + the degradation chain at the p the **committed** tuning rule
#   chose (`reports/p3_degradation_choice.json`, read here, never typed);
# - **`S2-B+reals`** = S2-B + the same reals, no degradation.
#
# **Mixing** (user decision 2026-10-21, option (a)): one real pool -- the COCO train reals
# S2-B uses and the Open Images train reals -- sampled **uniformly by real, exactly as the
# trainer does today**; each real's partners are its own reconstructions; gen_matched
# stays at 25% of fakes. **The trainer is unchanged** (byte-identical): the reals reach it
# through merged input files that `p3_s2d_inputs.py` builds after hashing **every** Open
# Images train real and reconstruction against the committed manifests. **The 400
# held-out Open Images reals and their reconstructions never enter training**: they are
# never read, and the pre-flight refuses if any of their ids is in a merged file.
#
# **Staging** (user, 2026-10-21): the first read of a file from `/kaggle/input` cost up to
# about 1 s in session 4. Before training, every file the runs read (their COCO and Open
# Images pairs and reals, gen_matched) is read once by a parallel pass and copied to local
# disk (`/tmp`), from which the trainer reads; the `THROUGHPUT` line reports the staging
# time, files/s and each run's training s/step. A copy rather than a page-cache warm-up:
# about 17.5-20 GB could not be shown to stay cached in 33.7 GB of RAM beside two trainers.
#
# **The recipe** is the P3 one: the 6,000-step cosine schedule stopped at `N*` = 5000,
# in-protocol validation every 500 steps. These runs train on all 12 members, so like
# S2-B they hold no lineage out and write no per-image held-out file. **Panel B is not
# touched.**
#
# **The queue:** `S2-D` s0 + `S2-B+reals` first, then `S2-D` s1 -- two per session, one
# per T4 (`p3_train_next.py`). A run cut short resumes from its 500-step checkpoint;
# finished runs are carried forward.
#
# **Kaggle:** Accelerator **GPU T4 x2**. **Internet:** on. **Secret:** `HF_TOKEN`.
# **Mode:** Save & Run All (Commit). **Account:** `muhammadadeelshah` only. **Run it alone:**
# no other GPU session at the same time.
#
# | input | what |
# |---|---|
# | dataset `dbp-code*` | the code at a commit holding this runbook **and** the tuning result |
# | dataset `dbp-p2-s2-inputs` | pairs manifest, split, generation manifest (hashed) |
# | dataset holding `p1_reals/` and `p1_generated/` (slug `report`) | corpus |
# | dataset `thesis-dbp-1-dataset` | corpus pair reconstructions |
# | dataset `dbp-p3-open-images` | the Open Images crops (`oi/reals/t_oi/`) |
# | the `dbp-p3-reconstruct` output (notebook or dataset) | `oi_pairs/` and its manifest |
# | *(session 2)* this notebook's previous version | done runs skipped; unfinished resume |

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
# This notebook's previous versions: p3 S2 outputs that are not the step-4a output.
PREVIOUS = sorted(p.parents[2] for p in
                  IN.glob("notebooks/*/*/Implementation/thesis_dbp/outputs/p3/s2")
                  if p.parents[4] != OI_ROOT)
PREVIOUS += [pathlib.Path(p) for p in os.environ.get("DBP_RESUME_DIRS", "").split(os.pathsep)
             if p]
DBP = WORK / "Implementation" / "thesis_dbp"
P1 = WORK / "p1"
S2IN = INPUTS["s2_inputs"]
MERGED = TMP / "dbp_s2d_inputs"
# Every file the runs read, copied once from /kaggle/input to local disk (user,
# 2026-10-21): the trainer reads from here, not from the slow input mount.
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
print("resume from:", [str(p) for p in PREVIOUS] or "nothing (first session)")

# %% Cell 2 — stage the code; hardware at session start; the environment (bootstrap only)
if DBP.exists():
    shutil.rmtree(DBP)
shutil.copytree(INPUTS["code"], DBP)
assert (DBP / "scripts" / "p3_s2d_inputs.py").is_file(), "the code dataset predates this runbook"
print("code commit:", (DBP / "COMMIT.txt").read_text().split()[0])
hardware(DBP / "reports" / "p3_s2d_hardware.txt")
EXPECT = json.loads(pathlib.Path(
    os.environ.get("DBP_EXPECT", DBP / "env" / "p3_s2d_expect.json")).read_text())
CHOICE = DBP / EXPECT["choice"]
assert CHOICE.is_file(), (f"{EXPECT['choice']} is not in this code: build the code zip "
                          f"from a commit after the tuning result was committed")
print("tuning result:", CHOICE.read_text())
print("expected:", json.dumps(EXPECT["expected"]))
assert sh("bash", DBP / "env" / "kaggle_bootstrap.sh") == 0, "bootstrap FAILED"

# %% Cell 3 — corpus + Open Images; PRE-FLIGHT 1: GPUs, token, imports, every input hash
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
checks = ["--expect-gpus", GPUS, "--imports", ",".join(EXPECT["imports"])]
checks += ["--hf-token"] if EXPECT["hf_token"] else []
for name, spec in EXPECT["inputs"].items():
    checks += ["--sha256", f"{S2IN / name}={spec['sha256']}"]
    checks += ["--rows", f"{S2IN / name}={spec['rows']}"] if "rows" in spec else []
checks += ["--sha256", f"{DBP / EXPECT['nstar']}={EXPECT['nstar_sha256']}"]
checks += ["--sha256", f"{DBP / OI['reals_manifest']}={OI['reals_manifest_sha256']}"]
checks += ["--sha256", f"{DBP / OI['pairs_digest']}={OI['pairs_digest_sha256']}"]
# The reference is the read-only committed manifest, not the one the matcher just wrote.
assert py("p2_preflight.py", "--stage", "p3_s2d_data", *checks, "--root", P1,
          "--gen-matched-reference", INPUTS["code"] / "reports" / "p1_gen_matched_manifest.csv",
          "--gen-matched-dir", P1 / "gen_matched",
          "--expect-gen-matched", EXPECT["gen_matched"]) == 0, \
    "PRE-FLIGHT (data) FAILED -- nothing long has started"

# Every file the runs read is copied once to LOCAL (parallel); the Open Images train reals
# and reconstructions are hash-checked in that same read (the held-out ones never read);
# the merged files written; S2-D resolved from the committed tuning result.
S2D_IN = ["--expect", pathlib.Path(os.environ.get("DBP_EXPECT", DBP / "env" /
                                                  "p3_s2d_expect.json")),
          "--oi-manifest", DBP / OI["reals_manifest"], "--out-dir", MERGED]
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
print("queue:", BUILT["queue"], "| pools:", json.dumps(BUILT["pools"]))

# %% Cell 4 — which runs this session trains (first two unfinished)
choose = ["--expect", DBP / "reports" / "p3_s2d_queue.json",
          "--count", EXPECT["runs_per_session"]]
for previous in PREVIOUS:
    choose += ["--resume-from", previous]
chosen = py("p3_train_next.py", *choose)
PLAN = json.loads((DBP / "reports" / "p3_train_plan.json").read_text())
assert chosen in (0, 3), "run choice FAILED"
assert PLAN["selected"], "EVERY QUEUED RUN IS COMPLETE"
print("this session trains:", PLAN["selected"], "| complete:", PLAN["complete"])

QUEUE = ["--devices", EXPECT["devices"], "--root", LOCAL,
         "--pairs-manifest", MERGED / "p1_pairs_manifest.csv",
         "--splits", MERGED / "p2_splits.csv",
         "--scratch", TMP / "dbp_p3_s2d_smoke", "--poll-seconds", EXPECT["poll_seconds"],
         "--train-only", "--out-dir", DBP / "outputs" / "p3" / "s2",
         "--log-dir", DBP / "outputs" / "p3" / "logs"]
for run in PLAN["selected"]:
    QUEUE += ["--run", run]
QUEUE += [f"--train-arg={a}" for a in EXPECT["train_args"]]
QUEUE += [f"--train-arg=--manifests={MERGED / 'manifests'}"]

# %% Cell 5 — PRE-FLIGHT 2: each run on each GPU (2 steps, save, reload)
assert py("p2_run_queue.py", *QUEUE, "--smoke", EXPECT["smoke_steps"],
          "--status", DBP / "reports" / "p3_s2d_smoke_status.json",
          "--deadline-hours", EXPECT["smoke_deadline_hours"]) == 0, \
    "PRE-FLIGHT (models) FAILED -- nothing long has started"

# %% Cell 6 — train the runs, one per GPU: the 6,000-step schedule, stopped at N* = 5000
resume = []
for previous in PREVIOUS:
    resume += ["--resume-from", previous]
code = py("p2_run_queue.py", *QUEUE, *resume, "--status",
          DBP / "reports" / "p3_s2d_status.json", "--deadline-hours", EXPECT["deadline_hours"])
print(f"queue exited {code} (0 = runs complete, 1 = failed, 2 = deadline; resumable)")

# %% Cell 7 — report (cannot fail the run)
py("p2_report.py", "--stage", "s2", "--status", "reports/p3_s2d_status.json",
   "--s2-dir", DBP / "outputs" / "p3" / "s2", "--zip", WORK / "p3_s2d_reports")
throughput = []
for run in PLAN["selected"]:
    path = DBP / "reports" / f"p2_s2_{run.replace(':', '_seed')}.json"
    report = json.loads(path.read_text()) if path.is_file() else {}
    pool = BUILT["pools"].get(run.split(":")[0], {})
    print(run, report.get("status"), "| steps", report.get("steps_run"), "| val AUC",
          [(p["step"], p["val_auc"]) for p in report.get("curve", [])][-1:],
          "| degradation", report.get("degrade"), "| pool", pool.get("coco_reals"), "COCO +",
          pool.get("oi_reals"), "Open Images reals, Open Images share",
          pool.get("oi_share_of_reals"))
    throughput.append(f"{run} {report.get('steps_run')} steps "
                      f"{report.get('measured_step_seconds_train_loop')} s/step "
                      f"{report.get('gpu_hours')} GPU-h")
status = DBP / "reports" / "p3_s2d_status.json"
wall = json.loads(status.read_text()).get("wall_hours") if status.is_file() else None
staged = BUILT.get("staged", {})
print("THROUGHPUT |", f"staging {staged.get('files')} files {staged.get('gb')} GB in "
      f"{staged.get('seconds')} s ({staged.get('files_per_s')} files/s, "
      f"{staged.get('mb_per_s')} MB/s) |", " | ".join(throughput), f"| queue wall {wall} h")

# %% Cell 8 — verdict: this cell fails the notebook if a run did not complete
assert code == 0, (f"queue exited {code}: see reports/p3_s2d_status.json. An unfinished "
                   f"run resumes from its last checkpoint when this notebook is re-run with "
                   f"this version's output attached.")
print("S2-D RUNS COMPLETE:", PLAN["selected"], "| all complete so far:",
      sorted(set(PLAN["complete"]) | set(PLAN["selected"])))
