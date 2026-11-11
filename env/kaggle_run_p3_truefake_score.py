# %% [markdown]
# # P3 — TrueFake: the 4 frozen models on the social-media sample, once (secondary, descriptive)
#
# **The TrueFake rule of `docs/P3_PLAN.md` (fixed 2026-10-20).** No decision rule and no
# hypothesis; scored once, on the frozen models, with the deployed protocol (centre crop
# 320, head logit), every version of every id independently.
#
# **The pre-flight** (`p3_truefake_prepare.py`):
# 1. the sample and the fetch manifest are the committed files (16,000 members);
# 2. the 4 models -- `S2-B` s0, `S2-B+deg` (`s2b_deg_p35`), `S2-D` s0 (`s2d_p35`), P2
#    `full_bank` -- are present and equal to their records in `env/p3_final_expect.json`;
# 3. every member is copied once to local disk with its sha256 checked in the same read;
# 4. **disjointness:** every version of the 2,000 sampled reals against our training reals
#    (COCO T-real, Open Images train) by sha256 and pHash (<= 6 bits), including the COCO
#    originals, hashed here from `coco-2017-dataset` after their sha256 is checked; a
#    matching real is dropped in all four versions, no backfill, counted per pool.
#
# Then a 2-row smoke of every model due, then the scoring, one worker per T4, fail-fast.
#
# **Privacy:** TrueFake images stay private. The per-image scores come back in the reports
# zip and stay in `kaggle_raw`; only aggregates are ever committed.
#
# **Kaggle:** GPU **T4 x2**, Internet on, Secret `HF_TOKEN`, Save & Run All (Commit),
# account `muhammadadeelshah`, **run alone**. Every input is found by its contents:
#
# | input | what |
# |---|---|
# | dataset `dbp-code*` | the code (holds this runbook and its expect file) |
# | notebook output of the TrueFake fetch (successful version) | `truefake_sample/` (16,000) |
# | dataset `coco-2017-dataset` (awsaf49) | `coco2017/train2017`: the originals behind T-real |
# | notebook output `dbp-p3-train` (latest) | S2-B seed 0 |
# | notebook output `dbp-p3-tune` | S2-B+deg (`s2b_deg_p35`) |
# | notebook output `dbp-p3-s2d` (after its session 2) | S2-D seed 0 |
# | dataset `dbp-p2-reference-models` | P2 `full_bank_seed0` |
# | *(only after a crash)* this notebook's previous version | complete models, carried forward |
#
# The sample is found by content (a folder holding `PreSocial/Real/FFHQ`), so the fetch
# notebook's output and a private dataset holding the same folder both work -- attach one,
# not both. Its files are not trusted by where they come from: the pre-flight checks every
# one against the committed fetch manifest's sha256.

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
    "sample": one("TrueFake sample (a directory holding PreSocial/Real/FFHQ: the fetch "
                  "notebook's truefake_sample/)",
                  [p.parents[2] for g in ("*/*/*/PreSocial/Real/FFHQ",
                                          "*/*/*/*/PreSocial/Real/FFHQ",
                                          "*/*/*/*/*/PreSocial/Real/FFHQ")
                   for p in IN.glob(g)]),
    "coco": one("COCO 2017 (coco2017/train2017)",
                [p.parent for p in IN.glob("*/*/*/coco2017/train2017")]),
}
# Each source is found by a run only it holds -- the final runbook's markers, proven on
# Kaggle: the tune output also carries s2b_seed0, so s2b_seed0 cannot identify train.
SOURCES = {"train": runs_root("s2b_seed1"), "tune": runs_root("s2b_deg_p35_seed0"),
           "s2d": runs_root("s2d_p35_seed1"),
           "p2ref": one("P2 reference models",
                        [p.parents[2] for p in
                         IN.glob("*/*/*/full_bank_seed0/lora/adapter_model.safetensors")])}
PREVIOUS = sorted({p.parents[1] for p in IN.glob("notebooks/*/*/Implementation/thesis_dbp/"
                                                  "reports/p3_truefake_plan.json")})
DBP = WORK / "Implementation" / "thesis_dbp"
LOCAL = TMP / "dbp_p3_truefake_local"
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
print("previous versions:", [str(p) for p in PREVIOUS] or "none")

# %% Cell 2 — stage the code; hardware at session start; the environment (bootstrap only)
if DBP.exists():
    shutil.rmtree(DBP)
shutil.copytree(INPUTS["code"], DBP)
assert (DBP / "scripts" / "p3_truefake_prepare.py").is_file(), \
    "the code dataset predates this runbook"
print("code commit:", (DBP / "COMMIT.txt").read_text().split()[0])
hardware(DBP / "reports" / "p3_truefake_hardware.txt")
EXPECT_PATH = pathlib.Path(os.environ.get("DBP_EXPECT",
                                          DBP / "env" / "p3_truefake_score_expect.json"))
EXPECT = json.loads(EXPECT_PATH.read_text())
print("expected:", json.dumps(EXPECT["expected"]))
assert sh("bash", DBP / "env" / "kaggle_bootstrap.sh") == 0, "bootstrap FAILED"

# %% Cell 3 — PRE-FLIGHT 1: GPUs, token, imports; sample, models, staging, disjointness
GPUS = int(os.environ.get("DBP_GPUS", EXPECT["gpus"]))
checks = ["--expect-gpus", GPUS, "--imports", ",".join(EXPECT["imports"])]
checks += ["--hf-token"] if EXPECT["hf_token"] else []
assert py("p2_preflight.py", "--stage", "p3_truefake_env", *checks) == 0, \
    "PRE-FLIGHT (environment) FAILED -- nothing has been scored"
plan_args = ["--expect", EXPECT_PATH, "--sample-root", INPUTS["sample"], "--coco-root",
             INPUTS["coco"], "--local-root", LOCAL, "--devices", EXPECT["devices"],
             "--workers", EXPECT["workers"]]
for name, path in SOURCES.items():
    plan_args += ["--source", f"{name}={path}"]
for previous in PREVIOUS:
    plan_args += ["--previous", previous]
assert py("p3_truefake_prepare.py", *plan_args) == 0, \
    "PRE-FLIGHT (sample / models / staging / disjointness) FAILED -- see " \
    "reports/p3_truefake_plan.json; nothing has been scored"
PLAN = json.loads((DBP / "reports" / "p3_truefake_plan.json").read_text())
print("STAGE:", PLAN["stage"], "| due:", PLAN["due"], "| jobs:", PLAN["jobs"])
print("dropped real ids per pool:", {k: v["dropped_ids"]
                                     for k, v in PLAN["disjointness"]["pools"].items()})
print("list:", {k: v for k, v in PLAN["list"].items() if k != "path"})
assert PLAN["stage"] != "complete", "EVERY MODEL IS ALREADY SCORED"

SHARDS = ["--worker", DBP / "scripts" / "p3_truefake_score.py", "--log-dir",
          DBP / "outputs" / "p3" / "logs", "--poll-seconds", EXPECT["poll_seconds"]]
for job in PLAN["jobs"]:
    SHARDS += ["--job", job]
WORKER = [f"--worker-arg={a}" for a in ("--plan", DBP / "reports" / "p3_truefake_plan.json",
                                        "--batch", EXPECT["batch"])]

# %% Cell 4 — PRE-FLIGHT 2: every model due, 2 rows, on its own GPU
assert py("p2_run_shards.py", *SHARDS, *WORKER, "--worker-arg=--limit", "--worker-arg=2",
          "--worker-arg=--out-root", f"--worker-arg={TMP / 'dbp_p3_truefake_smoke'}",
          "--log-prefix", "truefake_smoke", "--status", DBP / "reports" /
          "p3_truefake_smoke_status.json",
          "--deadline-hours", EXPECT["smoke_deadline_hours"]) == 0, \
    "PRE-FLIGHT (models) FAILED -- nothing has been scored"

# %% Cell 5 — score: the 4 models once; one worker per T4
code = py("p2_run_shards.py", *SHARDS, *WORKER, "--log-prefix", "truefake",
          "--status", DBP / "reports" / "p3_truefake_status.json",
          "--deadline-hours", EXPECT["deadline_hours"])
print(f"scoring exited {code} (0 = complete, 1 = a worker failed, 2 = deadline)")

# %% Cell 6 — report (cannot fail the run)
py("p2_report.py", "--stage", "truefake", "--status", "reports/p3_truefake_status.json",
   "--zip", WORK / "p3_truefake_reports")
staged = PLAN.get("staged", {})
times = []
for model in PLAN["due"]:
    path = DBP / "reports" / "p3_truefake" / f"{model}.json"
    summary = json.loads(path.read_text()) if path.is_file() else {}
    seconds = sum(v.get("seconds", 0) for v in summary.get("lists", {}).values())
    times.append(f"{model} {round(seconds)} s")
print("THROUGHPUT |", f"staging {staged.get('files')} files {staged.get('gb')} GB in "
      f"{staged.get('seconds')} s ({staged.get('files_per_s')} files/s) |", " | ".join(times))

# %% Cell 7 — verdict: this cell fails the notebook if a model due was not scored
missing = [m for m in PLAN["due"]
           if not (DBP / "reports" / "p3_truefake" / f"{m}.json").is_file()]
assert code == 0 and not missing, (f"scoring exited {code}; not scored: {missing}. Re-run with "
                                   f"this version attached: complete models are carried "
                                   f"forward, the rest are scored (log the reason).")
print("TRUEFAKE SCORING COMPLETE:", len(PLAN["due"]), "model(s)")
