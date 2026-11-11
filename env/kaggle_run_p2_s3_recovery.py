# %% [markdown]
# # P2 Session C — S3 recovery
#
# The first S3 commit finished 9 of the 12 bank members, then one shard died on a missing
# module and the other was stopped by its deadline. This runbook restores the 9 and runs
# the 3 that remain: `movq_k22`, `qwen_image`, `realesrgan_x4`.
#
# **Accelerator:** GPU T4 x2. **Internet:** on. **Mode:** Save & Run All (Commit).
#
# **Inputs to attach** (Cell 1 checks every one of them before anything else runs):
#
# | input | what |
# |---|---|
# | `adeelshah29/dbp-code` | the code, at the commit the hand-over names |
# | `adeelshah29/dbp-p2-s3-partial` | the 9 finished members, the S1 caches, the image lists |
# | notebook `adeelshah29/phase-2` | the frozen panel |
# | `adeelshah29/p1-data` | corpus reals and generations |
# | `muhammadadeelshah/thesis-dbp-1-dataset` | corpus pair reconstructions |
#
# Every cell is a thin call to a committed, tested script (CLAUDE.md section 12). The
# environment is set up by `env/kaggle_bootstrap.sh` and nothing else. Two pre-flights
# run before the long job: models first (fails in minutes), then data. This file is
# executed verbatim by `scripts/runbook_dryrun.py` before it is handed over.
#
# None of the three members is gated, so this stage needs no HF token.

# %% Cell 1 — inputs, and the one helper every later cell uses
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

INPUTS = {
    "code": IN / "datasets/adeelshah29/dbp-code/thesis_dbp",
    "bundle": IN / "datasets/adeelshah29/dbp-p2-s3-partial",
    "panel": IN / "notebooks/adeelshah29/phase-2/panel",
    "reals": IN / "datasets/adeelshah29/p1-data/p1_reals",
    "generated": IN / "datasets/adeelshah29/p1-data/p1_generated",
    "pairs": IN / "datasets/muhammadadeelshah/thesis-dbp-1-dataset"
             "/Implementation/thesis_dbp/outputs/p1/pairs",
}
missing = {name: str(path) for name, path in INPUTS.items() if not path.is_dir()}
assert not missing, f"input not attached, or attached under another name: {missing}"

DBP = WORK / "Implementation" / "thesis_dbp"
P1 = WORK / "p1"
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


print("inputs present:", ", ".join(INPUTS))

# %% Cell 2 — stage the code; set up the environment (the bootstrap is the only way)
if DBP.exists():
    shutil.rmtree(DBP)
shutil.copytree(INPUTS["code"], DBP)
assert (DBP / "scripts" / "p2_run_shards.py").is_file(), "the code dataset predates this runbook"
print("code commit:", (DBP / "COMMIT.txt").read_text().split()[0])

EXPECT = json.loads(pathlib.Path(
    os.environ.get("DBP_EXPECT", DBP / "env" / "p2_s3_recovery_expect.json")).read_text())
print("expected:", json.dumps(EXPECT["expected"]))

assert sh("bash", DBP / "env" / "kaggle_bootstrap.sh") == 0, "bootstrap FAILED"

# %% Cell 3 — restore the finished members; PRE-FLIGHT 1: every model loads and runs
assert py("p2_restore_partial.py", "--bundle-root", INPUTS["bundle"],
          "--expect-members", EXPECT["restored_members"]) == 0, "restore FAILED"

assert py("p2_preflight.py", "--stage", "s3_models",
          "--imports", ",".join(EXPECT["imports"]),
          "--smoke", ",".join(EXPECT["smoke_members"]),
          "--smoke-images", EXPECT["smoke_images"],
          "--smoke-list", "reports/p2_probe_images.csv",
          "--root", P1, "--panel-root", INPUTS["panel"],
          "--device", DEVICE, "--cache-dir", TMP / "dbp_weights") == 0, \
    "PRE-FLIGHT (models) FAILED -- nothing long has started"

# %% Cell 4 — corpus and gen_matched; PRE-FLIGHT 2: counts, hashes, every path resolves
links = []
for name in ("reals", "generated", "pairs"):
    links += ["--link", f"{name}={INPUTS[name]}", "--expect", f"{name}={EXPECT['corpus'][name]}"]
assert py("p2_stage_corpus.py", "--root", P1, *links) == 0, "corpus staging FAILED"

assert py("p1_match_generations.py",
          "--gen-manifest", "reports/p1_gen_manifest.csv",
          "--reals-manifest", "reports/p1_reals_manifest_t_real.csv",
          "--gen-root", P1 / "generated", "--out-root", P1, "--set-id", "gen_matched") == 0, \
    "gen_matched regeneration FAILED"

# The reference is the read-only committed manifest, not the one the matcher just wrote.
assert py("p2_preflight.py", "--stage", "s3_data",
          "--rows", f"reports/p2_probe_images.csv={EXPECT['probe_rows']}",
          "--list", "reports/p2_probe_images.csv", "--list-digest", EXPECT["list_digest"],
          "--sidecars", "outputs/p2/probe", "--expect-sidecars", EXPECT["restored_members"],
          "--resolve", "reports/p2_probe_images.csv",
          "--root", P1, "--panel-root", INPUTS["panel"],
          "--gen-matched-reference", INPUTS["code"] / "reports" / "p1_gen_matched_manifest.csv",
          "--gen-matched-dir", P1 / "gen_matched",
          "--expect-gen-matched", EXPECT["gen_matched"]) == 0, \
    "PRE-FLIGHT (data) FAILED -- nothing long has started"

# %% Cell 5 — S3: the remaining members, one job per GPU, fail-fast
jobs = []
for device, members in EXPECT["jobs"]:
    jobs += ["--job", f"{device}:{members}"]
code = py("p2_run_shards.py", *jobs,
          "--images", "reports/p2_probe_images.csv",
          "--root", P1, "--panel-root", INPUTS["panel"], "--cache-dir", TMP / "dbp_weights",
          "--deadline-hours", EXPECT["deadline_hours"],
          "--poll-seconds", EXPECT["poll_seconds"])
assert code == 0, (f"S3 runner exited {code} (1 = a job failed and the rest were stopped, "
                   f"2 = deadline). The status and logs are already in reports/.")

# %% Cell 6 — report (cannot fail the run)
py("p2_report.py", "--zip", WORK / "p2_s3_recovery_reports")
