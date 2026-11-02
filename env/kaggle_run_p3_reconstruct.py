# %% [markdown]
# # P3 step 4a — the 4,000 Open Images reals through all 12 bank decoders (T4 x2)
#
# The new training reals (the Open Images dataset, accepted 2026-10-17) become aligned
# real/reconstruction pairs exactly as the P1 corpus did: `scripts/p1_build_pairs.py`, the
# single reconstruction protocol (`dbp.decoders.reconstruct`: never resize, never tile,
# `latent_dist.mode()`, clamp, uint8, PNG), every **distinct** member (aliases skipped by
# the registry), each at the precision **P1's committed bench decided**
# (`reports/kaggle_20260926/p1_throughput.csv`: fp32 for sdxl, realesrgan_x4, movq_k22).
#
# **Two jobs, one per T4, member-sharded and fail-fast** (`p2_run_shards.py`): each job
# loads its six members once; the split balances P1's measured per-member time.
#
# **The output-size gate (before the long job).** The pre-flight reconstructs a sample of
# reals through every member, **measures** each member's mean PNG size, projects the full
# output and refuses to start if it would not fit the output disk with a margin.
#
# **Resumable.** If the session ends early, re-run this notebook with this version's
# output attached: finished pairs and their manifests are copied back and skipped.
#
# **Platform-neutral:** `DBP_INPUT_ROOT`, `DBP_WORKING_ROOT`, `DBP_TMP_ROOT`, `DBP_GPUS`.
# **Kaggle:** Accelerator **GPU T4 x2**. **Internet:** on (pinned weights from the Hub,
# the Real-ESRGAN release URL). **Secret:** `HF_TOKEN`. **Mode:** Save & Run All
# (Commit). **Account:** `muhammadadeelshah` only. **Panel B is not touched.**
#
# | input | what |
# |---|---|
# | dataset `dbp-code*` | the code at the commit that holds this runbook |
# | the Open Images dataset (any name) | `oi/reals/t_oi/*.png` (4,000 crops) |
# | *(only to resume)* this notebook's previous version | finished pairs |

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
    # Found by layout, not by name: the one attached dataset holding oi/reals. Its
    # content is checked file by file against the committed manifest in Cell 3.
    "reals": find("datasets/*/*/oi/reals"),
}
PREVIOUS = sorted(p.parent for p in IN.glob("notebooks/*/*/oi_pairs") if p.is_dir())
DBP = WORK / "Implementation" / "thesis_dbp"
PAIRS = WORK / "oi_pairs"
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
    print(f"{name:6s} {path}")
print("resume from:", [str(p) for p in PREVIOUS] or "nothing (first session)")

# %% Cell 2 — stage the code; set up the environment (the bootstrap is the only way)
if DBP.exists():
    shutil.rmtree(DBP)
shutil.copytree(INPUTS["code"], DBP)
assert (DBP / "scripts" / "p3_oi_pairs_check.py").is_file(), \
    "the code dataset predates this runbook"
print("code commit:", (DBP / "COMMIT.txt").read_text().split()[0])
EXPECT = json.loads(pathlib.Path(
    os.environ.get("DBP_EXPECT", DBP / "env" / "p3_reconstruct_expect.json")).read_text())
MANIFEST = DBP / EXPECT["reals_manifest"]
THROUGHPUT = DBP / EXPECT["throughput_csv"]
print("expected:", json.dumps(EXPECT["expected"]))
assert sh("bash", DBP / "env" / "kaggle_bootstrap.sh") == 0, "bootstrap FAILED"

# %% Cell 3 — PRE-FLIGHT 1: GPUs, token, imports, committed files, every input PNG's sha256
GPUS = int(os.environ.get("DBP_GPUS", EXPECT["gpus"]))
checks = ["--expect-gpus", GPUS, "--imports", ",".join(EXPECT["imports"])]
checks += ["--hf-token"] if EXPECT["hf_token"] else []
checks += ["--sha256", f"{MANIFEST}={EXPECT['reals_manifest_sha256']}",
           "--sha256", f"{THROUGHPUT}={EXPECT['throughput_csv_sha256']}"]
assert py("p2_preflight.py", "--stage", "p3_reconstruct", *checks) == 0, \
    "PRE-FLIGHT (environment) FAILED -- nothing long has started"
assert py("p3_oi_pairs_check.py", "inputs", "--manifest", MANIFEST,
          "--reals-root", INPUTS["reals"], "--rows", EXPECT["n_reals"],
          "--out", DBP / "reports" / "p3_oi_pairs_inputs.json") == 0, \
    "PRE-FLIGHT (inputs) FAILED: the attached reals do not match the committed manifest"

# Resume: bring back finished pairs and their per-member manifests.
for previous in PREVIOUS:
    shutil.copytree(previous / "oi_pairs", PAIRS, dirs_exist_ok=True)
    for manifest in (previous / "Implementation/thesis_dbp/reports").glob(
            f"{EXPECT['name']}_manifest_*.csv"):
        shutil.copy2(manifest, DBP / "reports" / manifest.name)
    print("resumed from", previous)

BUILD = ["--reals-manifest", MANIFEST, "--reals-root", INPUTS["reals"],
         "--throughput-csv", THROUGHPUT, "--cache-dir", TMP / "dbp_weights"]
JOBS = []
for job in EXPECT["jobs"]:
    JOBS += ["--job", job]
MEMBERS = ",".join(m for job in EXPECT["jobs"] for m in job.split(":", 1)[1].split(","))

# %% Cell 4 — PRE-FLIGHT 2: every member on its GPU (a sample), and the OUTPUT-SIZE GATE
assert py("p2_run_shards.py", *JOBS, "--worker", "scripts/p1_build_pairs.py",
          *[f"--worker-arg={a}" for a in [*BUILD, "--name", EXPECT["name"] + "_sample",
                                          "--out-root", TMP / "oi_pairs_sample",
                                          "--limit", EXPECT["sample_reals"]]],
          "--log-prefix", "pairs_sample", "--log-dir", DBP / "outputs" / "logs",
          "--status", DBP / "reports" / "p3_oi_pairs_sample_status.json",
          "--deadline-hours", EXPECT["sample_deadline_hours"],
          "--poll-seconds", EXPECT["poll_seconds"]) == 0, \
    "PRE-FLIGHT (members) FAILED -- nothing long has started"
summaries = sorted((DBP / "reports").glob(f"{EXPECT['name']}_sample_summary_*.json"))
assert py("p3_oi_pairs_check.py", "gate", "--manifest", MANIFEST, "--members", MEMBERS,
          *[a for s in summaries for a in ("--summary", s)], "--free-path", WORK,
          "--margin-gb", EXPECT["margin_gb"],
          "--out", DBP / "reports" / "p3_oi_size_gate.json") == 0, \
    "OUTPUT-SIZE GATE FAILED: the projected pairs would not fit -- see p3_oi_size_gate.json"

# %% Cell 5 — reconstruct: two jobs, one per T4, six members each; fail-fast
code = py("p2_run_shards.py", *JOBS, "--worker", "scripts/p1_build_pairs.py",
          *[f"--worker-arg={a}" for a in [*BUILD, "--name", EXPECT["name"],
                                          "--out-root", PAIRS]],
          "--log-prefix", "pairs", "--log-dir", DBP / "outputs" / "logs",
          "--status", DBP / "reports" / "p3_oi_pairs_status.json",
          "--deadline-hours", EXPECT["deadline_hours"], "--poll-seconds", EXPECT["poll_seconds"])
print(f"reconstruction exited {code} (0 = complete, 1 = failed, 2 = deadline; resumable)")

# %% Cell 6 — merge, verify and report (cannot fail the run)
py("p1_build_pairs.py", "--merge", "--name", EXPECT["name"])
py("p3_oi_pairs_check.py", "final", "--manifest", MANIFEST, "--members", MEMBERS,
   "--pairs-manifest", DBP / "reports" / f"{EXPECT['name']}_manifest.csv",
   "--pairs-parent", WORK, "--throughput-csv", THROUGHPUT,
   "--out", DBP / "reports" / "p3_oi_pairs_final.json")
for name in ("p3_oi_size_gate.json", "p3_oi_pairs_status.json", "p3_oi_pairs_final.json"):
    path = DBP / "reports" / name
    print(name, path.read_text() if path.is_file() else "(not written)")

# %% Cell 7 — verdict: this cell fails the notebook if the pairs are not complete and verified
final = DBP / "reports" / "p3_oi_pairs_final.json"
assert code == 0, (f"reconstruction exited {code}: see reports/p3_oi_pairs_status.json. "
                   f"Re-run with this version's output attached to resume.")
assert final.is_file() and json.loads(final.read_text())["ok"], \
    "the merged pairs did not verify: see reports/p3_oi_pairs_final.json"
print("STEP 4a COMPLETE:", json.loads(final.read_text())["rows"], "pairs,",
      json.loads(final.read_text())["gb"], "GB")
