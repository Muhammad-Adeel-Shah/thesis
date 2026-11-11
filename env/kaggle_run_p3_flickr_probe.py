# %% [markdown]
# # P3 Phase B step 3 — Flickr probe from Kaggle (CPU, about 5 minutes)
#
# Open Images run 1 got HTTP 429 on all 24 requests. This probe (user decision
# 2026-10-13) sends **6 GETs**, the first 3 `farm*` and the first 3 `c*` rows in seed
# order, **one at a time, 30 s apart**, keeps **no image**, and records each response
# (time, status, headers without cookies, first 200 body bytes) and the session's egress
# IP range (first two octets only, via `api.ipify.org`, approved 2026-10-13).
#
# The verdict picks the pre-decided branch: **A** (mostly 200: a rate limit; rerun on
# Kaggle with ≤ 2 connections and a fixed gap) or **B** (refused on both host groups:
# Kaggle's IPs are blocked; prepare the local route).
#
# **Accelerator: None (CPU).** **Internet:** on. **Mode:** Save & Run All (Commit).
# **Account:** `muhammadadeelshah` only. **Input:** the `dbp-code*` dataset only.

# %% Cell 1 — find the code input, and the helpers every later cell uses
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


INPUTS = {"code": find("datasets/*/dbp-code*/thesis_dbp")}
DBP = WORK / "Implementation" / "thesis_dbp"
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


print("code", INPUTS["code"])

# %% Cell 2 — stage the code; set up the environment (CPU session)
if DBP.exists():
    shutil.rmtree(DBP)
shutil.copytree(INPUTS["code"], DBP)
assert (DBP / "scripts" / "p3_probe_flickr.py").is_file(), "the code predates the probe"
print("code commit:", (DBP / "COMMIT.txt").read_text().split()[0])
EXPECT = json.loads(pathlib.Path(
    os.environ.get("DBP_EXPECT", DBP / "env" / "p3_flickr_probe_expect.json")).read_text())
print("expected:", json.dumps(EXPECT["expected"]))
os.environ["DBP_CPU_SESSION"] = "1"
assert sh("bash", DBP / "env" / "kaggle_bootstrap.sh") == 0, "bootstrap FAILED"

# %% Cell 3 — PRE-FLIGHT: imports and the metadata's hash, before any request
METADATA = DBP / EXPECT["metadata"]["path"]
assert py("p2_preflight.py", "--stage", "p3_flickr_probe",
          "--imports", ",".join(EXPECT["imports"]),
          "--sha256", f"{METADATA}={EXPECT['metadata']['sha256']}",
          "--rows", f"{METADATA}={EXPECT['metadata']['rows']}") == 0, \
    "PRE-FLIGHT FAILED -- no request has been sent"

# %% Cell 4 — the probe: 6 GETs, one at a time, 30 s apart; no image kept
probe = ["--metadata", METADATA, "--farm", EXPECT["farm"], "--c", EXPECT["c"],
         "--gap", EXPECT["gap_s"]]
probe += ["--ip-echo-url", EXPECT["ip_echo_url"]] if EXPECT.get("ip_echo_url") else []
probe += ["--local-source", EXPECT["local_source"]] if EXPECT.get("local_source") else []
assert py("p3_probe_flickr.py", *probe) == 0, "PROBE FAILED TO RUN"

# %% Cell 5 — report (cannot fail the run)
path = DBP / "reports" / "p3_flickr_probe.json"
print(f"===== {path.name}\n" + (path.read_text() if path.is_file() else "(not written)"))

# %% Cell 6 — verdict
PROBE = json.loads((DBP / "reports" / "p3_flickr_probe.json").read_text())
assert len(PROBE["requests"]) == EXPECT["farm"] + EXPECT["c"], "not every request was sent"
print("PROBE:", PROBE["verdict"], "| egress", PROBE["egress_ip_range"], "|",
      json.dumps(PROBE["by_host_group"]))
