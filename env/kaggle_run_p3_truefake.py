# %% [markdown]
# # P3 — fetch the frozen TrueFake sample (Kaggle CPU)
#
# The rule (`P3_PLAN.md`, "TrueFake -- a secondary, descriptive social-media test", fixed
# 2026-10-20, before Panel B is scored) and the frozen sample
# `reports/truefake/truefake_sample.csv` (4,000 ids x 4 versions = 16,000 files, 7.985 GiB)
# decide what is fetched. `p3_fetch_truefake.py` reads them member by member from
# `TrueFake.zip` by byte range, after verifying the archive's size and its index (central
# directory sha256), and checks every member's CRC-32.
#
# **Any refusal stops the session** (403, 429, 5xx, a quota or sign-in page, a non-range
# reply): nothing is retried and nothing works around it -- report it to Claude instead.
#
# **The images stay private:** publish the output's `truefake_sample/` only as a
# **PRIVATE** dataset (e.g. `dbp-truefake-sample`); never put an image in the thesis, a
# report or a commit. Only `reports/truefake/` (names, sizes, hashes) is copied back.
#
# **Accelerator: None (CPU)** -- no GPU quota; may run beside a GPU session.
# **Internet:** on. **Mode:** Save & Run All (Commit). **Account:** `muhammadadeelshah`.
#
# | input | what |
# |---|---|
# | dataset `dbp-code*` | the code, at the commit that holds this runbook |

# %% Cell 1 — find the input, and the helpers every later cell uses
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


CODE = find("datasets/*/dbp-code*/thesis_dbp")
DBP = WORK / "Implementation" / "thesis_dbp"
OUT = WORK / "truefake_sample"
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


print("code", CODE)

# %% Cell 2 — stage the code; set up the environment (CPU session)
if DBP.exists():
    shutil.rmtree(DBP)
shutil.copytree(CODE, DBP)
assert (DBP / "scripts" / "p3_fetch_truefake.py").is_file(), \
    "the code dataset predates this runbook"
print("code commit:", (DBP / "COMMIT.txt").read_text().split()[0])
EXPECT = json.loads(pathlib.Path(
    os.environ.get("DBP_EXPECT", DBP / "env" / "p3_truefake_expect.json")).read_text())
SAMPLE = pathlib.Path(EXPECT["sample"]) if pathlib.Path(EXPECT["sample"]).is_absolute() \
    else DBP / EXPECT["sample"]
print("expected:", json.dumps(EXPECT["expected"]))
os.environ["DBP_CPU_SESSION"] = "1"
assert sh("bash", DBP / "env" / "kaggle_bootstrap.sh") == 0, "bootstrap FAILED"

# %% Cell 3 — PRE-FLIGHT: the frozen sample is the committed one; enough disk for it
free_gb = shutil.disk_usage(WORK).free / 1e9
print(f"free on {WORK}: {free_gb:.1f} GB; the sample needs about {EXPECT['sample_gb']} GB")
assert free_gb > EXPECT["sample_gb"] + 1.0, "not enough disk for the sample"
assert py("p2_preflight.py", "--stage", "p3_truefake",
          "--sha256", f"{SAMPLE}={EXPECT['sample_sha256']}",
          "--rows", f"{SAMPLE}={EXPECT['sample_rows']}") == 0, \
    "PRE-FLIGHT FAILED -- nothing has been fetched"

# %% Cell 4 — fetch: verify the archive and its index, then every member (stops on refusal)
local = ["--local-zip", EXPECT["local_zip"]] if EXPECT.get("local_zip") else []
code = py("p3_fetch_truefake.py", "--sample", SAMPLE, "--out", OUT,
          "--workers", EXPECT["workers"], "--expect-size", EXPECT["zip_size"],
          "--expect-cd-sha256", EXPECT["cd_sha256"], *local)
print(f"fetch exited {code} (0 = complete, 3 = Google Drive refused -- STOPPED, 1 = failed)")

# %% Cell 5 — report (cannot fail the run)
status = DBP / "reports" / "truefake" / "truefake_fetch_status.json"
print(status.read_text() if status.is_file() else "(no status written)")

# %% Cell 6 — verdict
assert code != 3, ("GOOGLE DRIVE REFUSED: the session stopped as the rule requires. Do not "
                   "re-run or work around it -- send reports/truefake/truefake_fetch_status.json")
assert code == 0, "the fetch did not complete: see the Cell 4 output and the status file"
print("TRUEFAKE SAMPLE FETCHED: publish truefake_sample/ as a PRIVATE dataset")
