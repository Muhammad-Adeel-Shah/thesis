# %% [markdown]
# # P4 — the isolated DDA equivalence check (CPU; its own notebook)
#
# **User decision 2026-11-01 (spec items 11 and 7.12):** our DDA adapter is checked
# against **DDA's own code** on 2 COCO val2017 images. DDA's code builds DINOv2 through
# `torch.hub`, which runs code downloaded from GitHub, so this check runs **only in
# isolation**:
#
# * a **separate, CPU-only** notebook (accelerator **None**), Internet on;
# * **no Secrets** (Add-ons > Secrets: none attached);
# * **no private dataset**: the only inputs are the code-only bundle `dbp-p4-dda-equiv`
#   (this runbook, the check script, `dbp.baselines` and the bootstrap, from git's bytes;
#   no data and no reports) and the public dataset `awsaf49/coco-2017-dataset`;
# * outputs: the two logits from each implementation. **Nothing from this session is
#   reused.**
#
# **Pass:** the logits agree within 1e-3. The tolerance is never loosened; a failure is
# fixed at its cause.

# %% Cell 1 — find the two inputs; refuse to run with anything else attached
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


CODE = one("code-only bundle", IN.glob("datasets/*/dbp-p4-dda-equiv*/thesis_dbp"))
COCO_VAL = one("public COCO 2017 val2017", IN.glob("datasets/*/coco-2017-dataset/*/val2017"))
attached = sorted(p.relative_to(IN).as_posix() for p in IN.glob("*/*/*") if p.is_dir())
print("attached:", attached)
assert len(attached) == 2 and all(
    any(name in a for name in ("dbp-p4-dda-equiv", "coco-2017-dataset")) for a in attached), \
    f"ISOLATION: only the code bundle and public COCO may be attached, found {attached}"
assert not (IN / "notebooks").exists() or not any((IN / "notebooks").iterdir()), \
    "ISOLATION: no notebook output may be attached"
DBP = WORK / "Implementation" / "thesis_dbp"
for key in ("HF_HOME", "TORCH_HOME", "PIP_CACHE_DIR"):
    os.environ[key] = str(TMP / key.lower())
    os.makedirs(os.environ[key], exist_ok=True)


def sh(*command):
    """Run a command in the staged bundle, streaming its output into this cell."""
    print("$", " ".join(str(c) for c in command), flush=True)
    with subprocess.Popen([str(c) for c in command], cwd=DBP, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True, bufsize=1) as proc:
        for line in proc.stdout:
            print(line, end="", flush=True)
    return proc.returncode


def py(script, *args):
    """Run one script of the bundle."""
    return sh(sys.executable, "-u", f"scripts/{script}", *args)


print("bundle   ", CODE)
print("coco val ", COCO_VAL)

# %% Cell 2 — stage the bundle; the environment (bootstrap only)
if DBP.exists():
    shutil.rmtree(DBP)
shutil.copytree(CODE, DBP)
print("code commit:", (DBP / "COMMIT.txt").read_text().split()[0])
EXPECT_PATH = pathlib.Path(os.environ.get("DBP_EXPECT",
                                          DBP / "env" / "p4_dda_equivalence_expect.json"))
EXPECT = json.loads(EXPECT_PATH.read_text())
print("expected:", json.dumps(EXPECT["expected"]))
os.environ["DBP_CPU_SESSION"] = "1"
assert sh("bash", DBP / "env" / "kaggle_bootstrap.sh") == 0, "bootstrap FAILED"

# %% Cell 3 — PRE-FLIGHT: the imports both implementations need; the two images exist
missing = [name for name in EXPECT["images"] if not (COCO_VAL / name).is_file()]
assert not missing, f"PRE-FLIGHT FAILED: not in the public COCO val2017: {missing}"
assert py("p2_preflight.py", "--stage", "p4_dda_equiv_env",
          "--imports", ",".join(EXPECT["imports"])) == 0, "PRE-FLIGHT (imports) FAILED"

# %% Cell 4 — score both implementations on the 2 images (writes its own result)
check = ["--expect", EXPECT_PATH, "--coco-val", COCO_VAL, "--work", TMP / "dda_equiv",
         "--out", WORK / "p4_dda_equivalence.json"]
check += ["--local-archive", EXPECT["local_archive"]] if EXPECT.get("local_archive") else []
check += ["--local-ckpt", EXPECT["local_ckpt"]] if EXPECT.get("local_ckpt") else []
code = py("p4_dda_equivalence.py", *check)

# %% Cell 5 — report: the two logits from each implementation (cannot fail the run)
RESULT_PATH = WORK / "p4_dda_equivalence.json"
RESULT = json.loads(RESULT_PATH.read_text()) if RESULT_PATH.is_file() else {"images": []}
for image in RESULT["images"]:
    print(image["file"], "upstream", image.get("upstream_logit"), "ours",
          image.get("ours_logit"), "|diff|", image.get("abs_diff"))
print(json.dumps(RESULT, indent=2))

# %% Cell 6 — verdict
assert code == 0 and RESULT.get("pass"), f"DDA EQUIVALENCE FAILED: max |diff| " \
    f"{RESULT.get('max_abs_diff')} > {EXPECT['tolerance']} (fix the cause; never the tolerance)"
print("DDA EQUIVALENCE PASSED: max |diff|", RESULT["max_abs_diff"])
