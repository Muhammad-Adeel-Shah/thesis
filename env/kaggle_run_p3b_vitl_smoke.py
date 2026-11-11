# %% [markdown]
# # P3b — ViT-L smoke: DINOv3 ViT-L/16 on the unchanged S2-D recipe (no training run)
#
# **Plan only** (user decision 2026-11-11; `docs/P3b_VITL_PREREG.md`, not yet frozen). The
# S2-D recipe is unchanged -- bank, data, Open Images train reals, degradation p = 0.35,
# LoRA r 8 on q/k/v/o, lr, weight decay, warm-up, crop 320, the batch mix -- and **only
# the backbone changes**: DINOv3 ViT-B/16 -> ViT-L/16 (`dinov3-vitl16`, pinned), so the
# head's input is 2 x 1024. This session answers four questions before any training:
#
# 1. does ViT-L load and train on a T4 (the recipe keeps gradient checkpointing on);
# 2. peak GPU memory;
# 3. seconds per step after warm-up, and the projected hours for 5,000 steps;
# 4. whether gradient checkpointing is **needed**: a second smoke runs with it off, as a
#    measurement only.
#
# **The trainer is byte-identical** (user decision 2026-10-21). Each smoke is the trainer's
# own `--smoke 50` (50 steps of `S2-D` seed 0, one validation, save, reload, compare) run
# in-process by `p3b_smoke_probe.py`, which times the optimiser steps, reads the peak GPU
# memory and -- for the measurement only -- makes transformers' gradient-checkpointing
# switch a no-op. The two smokes run one after the other on the first T4. **Nothing is kept
# as a model, and Panel B is not touched.** The inputs, staging and pre-flight are the
# S2-D session's, unchanged.
#
# **Kaggle:** Accelerator **GPU T4 x2**. **Internet:** on. **Secret:** `HF_TOKEN` (the
# DINOv3 weights are gated). **Mode:** Save & Run All (Commit). **Account:**
# `muhammadadeelshah` only. **Run it alone.** torch, torchvision and numpy are never
# reinstalled (the bootstrap enforces it).
#
# | input | what |
# |---|---|
# | dataset `dbp-code*` | the code at a commit holding this runbook |
# | dataset `dbp-p2-s2-inputs` | pairs manifest, split, generation manifest (hashed) |
# | dataset holding `p1_reals/` and `p1_generated/` (slug `report`) | corpus |
# | dataset `thesis-dbp-1-dataset` | corpus pair reconstructions |
# | dataset `dbp-p3-open-images` | the Open Images crops (`oi/reals/t_oi/`) |
# | the `dbp-p3-reconstruct` output (notebook or dataset) | `oi_pairs/` and its manifest |

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
DBP = WORK / "Implementation" / "thesis_dbp"
P1 = WORK / "p1"
S2IN = INPUTS["s2_inputs"]
MERGED = TMP / "dbp_s2d_inputs"
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

# %% Cell 2 — stage the code; hardware at session start; the environment (bootstrap only)
if DBP.exists():
    shutil.rmtree(DBP)
shutil.copytree(INPUTS["code"], DBP)
assert (DBP / "scripts" / "p3b_smoke_report.py").is_file(), \
    "the code dataset predates this runbook"
print("code commit:", (DBP / "COMMIT.txt").read_text().split()[0])
hardware(DBP / "reports" / "p3b_vitl_smoke_hardware.txt")
EXPECT_PATH = pathlib.Path(os.environ.get("DBP_EXPECT",
                                          DBP / "env" / "p3b_vitl_smoke_expect.json"))
EXPECT = json.loads(EXPECT_PATH.read_text())
CHOICE = DBP / EXPECT["choice"]
assert CHOICE.is_file(), f"{EXPECT['choice']} is not in this code"
print("expected:", json.dumps(EXPECT["expected"]))
assert sh("bash", DBP / "env" / "kaggle_bootstrap.sh") == 0, "bootstrap FAILED"

# %% Cell 3 — corpus + Open Images; PRE-FLIGHT: GPUs, token, imports, every input hash
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
assert py("p2_preflight.py", "--stage", "p3b_vitl_smoke_data", *checks, "--root", P1,
          "--gen-matched-reference", INPUTS["code"] / "reports" / "p1_gen_matched_manifest.csv",
          "--gen-matched-dir", P1 / "gen_matched",
          "--expect-gen-matched", EXPECT["gen_matched"]) == 0, \
    "PRE-FLIGHT (data) FAILED -- nothing long has started"

S2D_IN = ["--expect", EXPECT_PATH, "--oi-manifest", DBP / OI["reals_manifest"],
          "--out-dir", MERGED]
assert py("p3_s2d_inputs.py", "build", *S2D_IN,
          "--oi-pairs-manifest",
          OI_ROOT / "Implementation/thesis_dbp/reports/p3_oi_pairs_manifest.csv",
          "--oi-pairs-digest", DBP / OI["pairs_digest"], "--local-root", LOCAL,
          "--pairs-manifest", S2IN / "p1_pairs_manifest.csv",
          "--splits", S2IN / "p2_splits.csv", "--choice", CHOICE, "--root", P1,
          "--workers", OI["hash_workers"],
          "--out", DBP / "reports" / "p3_s2d_inputs.json") == 0, \
    "S2-D INPUTS FAILED -- see reports/p3_s2d_inputs.json; nothing long has started"
assert py("p3_s2d_inputs.py", "check", *S2D_IN,
          "--built", DBP / "reports" / "p3_s2d_inputs.json", "--local-root", LOCAL,
          "--out", DBP / "reports" / "p3_s2d_inputs_check.json") == 0, \
    "PRE-FLIGHT (held-out exclusion) FAILED -- nothing long has started"
BUILT = json.loads((DBP / "reports" / "p3_s2d_inputs.json").read_text())
PROTOCOL, SEED = BUILT["queue"][0].split(":")
print("smoke run:", PROTOCOL, "seed", SEED, "| backbone:", EXPECT["backbone"],
      "| pools:", json.dumps(BUILT["pools"]))


def smoke(spec):
    """One smoke of the UNCHANGED trainer, run in-process by the probe that measures it."""
    scratch = TMP / f"dbp_p3b_smoke_gc_{spec['gradient_checkpointing']}"
    if scratch.exists():
        shutil.rmtree(scratch)
    trainer = ["--protocol", PROTOCOL, "--seed", SEED, "--root", LOCAL,
               "--pairs-manifest", MERGED / "p1_pairs_manifest.csv",
               "--splits", MERGED / "p2_splits.csv", "--manifests", MERGED / "manifests",
               "--backbone", EXPECT["backbone"], "--device", spec["device"],
               "--smoke", EXPECT["smoke_steps"], "--out-dir", scratch / "out",
               "--reports-dir", scratch / "reports", *EXPECT["train_args"]]
    return py("p3b_smoke_probe.py", "--gradient-checkpointing", spec["gradient_checkpointing"],
              "--out", scratch / "probe.json", "--", *trainer)


RECIPE, MEASURE = EXPECT["smoke_runs"]

# %% Cell 4 — smoke 1, THE RECIPE: ViT-L, gradient checkpointing on, 50 steps
recipe_code = smoke(RECIPE)
print(f"recipe smoke exited {recipe_code} (0 = 50 steps trained, saved, reloaded)")

# %% Cell 5 — smoke 2, MEASUREMENT ONLY: gradient checkpointing off (may run out of memory)
measure_code = smoke(MEASURE)
print(f"measurement smoke exited {measure_code} (non-zero is an answer here: it did not fit)")

# %% Cell 6 — report: memory, s/step, projection (cannot fail the run)
py("p3b_smoke_report.py", "--expect", EXPECT_PATH, "--scratch", TMP,
   "--out", DBP / "reports" / "p3b_vitl_smoke.json")
for mode in ("on", "off"):
    for path in (TMP / f"dbp_p3b_smoke_gc_{mode}").glob("**/*.json"):
        target = DBP / "reports" / "p3b_smoke" / f"gc_{mode}" / path.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
py("p2_report.py", "--stage", "s2", "--status", "reports/p3b_vitl_smoke.json",
   "--zip", WORK / "p3b_vitl_smoke_reports")

# %% Cell 7 — verdict: the recipe smoke must have passed
assert recipe_code == 0, ("the RECIPE smoke (gradient checkpointing on) FAILED: ViT-L does "
                          "not train on a T4 as the recipe stands -- see the report above")
print("P3b SMOKE COMPLETE: see reports/p3b_vitl_smoke.json")
