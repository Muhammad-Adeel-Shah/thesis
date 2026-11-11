# %% [markdown]
# # P4 — step 1: fetch and check every baseline weight (CPU; before any score)
#
# **Spec `docs/P4_BASELINES_SPEC.md`, sections 1 and 4** (frozen, tag
# `p4-baselines-frozen`). Nothing is scored here. The session:
#
# 1. downloads `DDA_ckpt.pth`, CO-SPY's `sd-v1_4.zip` and SigLIP's
#    `open_clip_model.safetensors` from the Hugging Face Hub at their pinned revisions,
#    each checked against its recorded size and sha256;
# 2. downloads FIRE's "Imagenet with adm" checkpoint from Google Drive with `gdown`
#    (installed by the bootstrap: pinned, `--no-deps`, this session only; torch,
#    torchvision and numpy printed before and after) and **records its sha256**. **If
#    Google Drive refuses, the run stops and says so; no mirror;**
# 3. extracts the three CO-SPY `.pth` files by name and hashes them;
# 4. opens every checkpoint with `torch.load(weights_only=True)` and checks it against
#    the adapters in `dbp.baselines`, so a mismatch costs no GPU time.
#
# **Kaggle:** accelerator **None (CPU)**, Internet on, account `muhammadadeelshah`, Save &
# Run All (Commit). **Inputs:** the code dataset `dbp-code*` only. The output (the weights
# under `p4_weights/` and the reports) stays PRIVATE and is attached to the scoring
# notebook.

# %% Cell 1 — find the inputs by their contents, and the helpers every later cell uses
import json
import os
import pathlib
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


CODE = one("code dataset", IN.glob("datasets/*/dbp-code*/thesis_dbp"))
DBP = WORK / "Implementation" / "thesis_dbp"
WEIGHTS = WORK / "p4_weights"
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


print("code     ", CODE)

# %% Cell 2 — stage the code; the environment (bootstrap only, with gdown)
if DBP.exists():
    shutil.rmtree(DBP)
shutil.copytree(CODE, DBP)
assert (DBP / "scripts" / "p4_weights.py").is_file(), "the code dataset predates this runbook"
print("code commit:", (DBP / "COMMIT.txt").read_text().split()[0])
EXPECT_PATH = pathlib.Path(os.environ.get("DBP_EXPECT", DBP / "env" / "p4_weights_expect.json"))
EXPECT = json.loads(EXPECT_PATH.read_text())
print("expected:", json.dumps(EXPECT["expected"]))
os.environ["DBP_CPU_SESSION"] = "1"
os.environ["DBP_WITH_GDOWN"] = "1"
assert sh("bash", DBP / "env" / "kaggle_bootstrap.sh") == 0, "bootstrap FAILED"
assert py("p2_preflight.py", "--stage", "p4_weights_env",
          "--imports", ",".join(EXPECT["imports"])) == 0, "PRE-FLIGHT (imports) FAILED"

# %% Cell 3 — fetch, hash, extract and inspect (the work step; it writes its own record)
fetch = ["--expect", EXPECT_PATH, "--out", WEIGHTS]
fetch += ["--local-source", EXPECT["local_source"]] if EXPECT.get("local_source") else []
code = py("p4_weights.py", *fetch)
print(f"weights exited {code} (0 = every file present, hash-checked and loadable)")

# %% Cell 4 — report (cannot fail the run)
py("p2_report.py", "--stage", "p4", "--status", "reports/p4_weights.json",
   "--zip", WORK / "p4_weights_reports")
for path in sorted(WEIGHTS.rglob("*")):
    if path.is_file():
        print(f"  {path.relative_to(WEIGHTS)}  {path.stat().st_size:,} B")

# %% Cell 5 — verdict: this cell fails the notebook if any check failed
REPORT = json.loads((DBP / "reports" / "p4_weights.json").read_text())
print("FIRE sha256 (commit it before any scoring):", REPORT["files"].get("fire", {}).get("sha256"))
assert code == 0 and REPORT["ok"], f"weights FAILED: {REPORT['problems']}"
print("P4 WEIGHTS COMPLETE: every file present, hash-checked and loadable")
