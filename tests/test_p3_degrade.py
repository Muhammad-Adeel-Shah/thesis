"""P3 section 4b: the degradation chain, its synchronisation, the tuning rule (fixed 2026-10-17)."""

from __future__ import annotations

import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from dbp.decoders.registry import load_bank_config  # noqa: E402
from dbp.eval.run_config import RunConfigError, load_run_configs  # noqa: E402
from dbp.train import degrade as D  # noqa: E402

BANK = load_bank_config(ROOT / "configs" / "decoder_bank.yaml")


def _tune():
    alias = "_p3_tune_test"
    path = ROOT / "scripts" / "p3_tune_degradation.py"
    spec = importlib.util.spec_from_file_location(alias, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    spec.loader.exec_module(module)
    return module


def _image(seed: int, size: int = 384) -> np.ndarray:
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size] / (size - 1)
    base = np.stack([xx, yy, (xx + yy) / 2], axis=-1) * 200 + rng.normal(0, 10, (size, size, 3))
    return np.clip(base, 0, 255).astype(np.uint8)


# --- the fixed shares -----------------------------------------------------------------

def test_values_are_the_ones_fixed_on_2026_10_17() -> None:
    assert D.OPERATIONS == ("jpeg", "webp", "resize", "social")
    assert (D.JPEG_QUALITY, D.WEBP_QUALITY, D.SOCIAL_QUALITY) == ((50, 95), (50, 95), (70, 85))
    assert D.RESIZE_SCALE == (0.5, 1.0) and set(D.SUBSAMPLINGS) == {"4:2:0", "4:4:4"}
    assert D.CANDIDATE_P == (0.7, 0.35)
    # 2026-10-19: the social-media JPEG is 4:2:0 only; a re-share half the time.
    assert D.SOCIAL_SUBSAMPLING == "4:2:0" and D.RESHARE_P == 0.5


@pytest.mark.parametrize("p", [0.7, 0.35])
def test_shares_match_the_rule(p: float) -> None:
    rng = np.random.default_rng(2026)
    chains = [D.sample_chain(rng, p) for _ in range(40_000)]
    applied = [c for c in chains if c.op is not None]
    assert abs(len(applied) / len(chains) - p) < 0.01
    ops = Counter(c.op for c in applied)
    assert all(abs(n / len(applied) - 0.25) < 0.012 for n in ops.values()) and len(ops) == 4
    chroma = Counter(c.subsampling for c in applied if c.op == "jpeg")
    assert abs(chroma["4:2:0"] / sum(chroma.values()) - 0.5) < 0.02
    social = [c for c in applied if c.op == "social"]
    assert {c.subsampling for c in social} == {"4:2:0"}                 # platforms subsample
    reshared = [c for c in social if c.quality2 is not None]
    assert abs(len(reshared) / len(social) - 0.5) < 0.02
    assert all(c.scale2 is not None for c in reshared)
    assert all(c.quality2 is None and c.scale2 is None
               for c in applied if c not in reshared)
    # the second round is drawn independently of the first: |r| within 4 standard errors
    bound = 4.0 / np.sqrt(len(reshared))
    assert abs(np.corrcoef([c.scale for c in reshared], [c.scale2 for c in reshared])[0, 1]) < bound
    assert abs(np.corrcoef([c.quality for c in reshared],
                           [c.quality2 for c in reshared])[0, 1]) < bound
    assert (min(c.quality2 for c in reshared), max(c.quality2 for c in reshared)) == (70, 85)
    q = {op: [c.quality for c in applied if c.op == op] for op in ("jpeg", "webp", "social")}
    assert (min(q["jpeg"]), max(q["jpeg"])) == (50, 95)
    assert (min(q["webp"]), max(q["webp"])) == (50, 95)
    assert (min(q["social"]), max(q["social"])) == (70, 85)
    scales = [c.scale for c in applied if c.op in ("resize", "social")]
    assert 0.5 <= min(scales) < 0.51 and 0.99 < max(scales) <= 1.0


def test_every_draw_consumes_the_same_stream() -> None:
    a, b = np.random.default_rng(7), np.random.default_rng(7)
    D.sample_chain(a, 0.0)          # never applied
    D.sample_chain(b, 1.0)          # always applied
    assert a.random() == b.random()


# --- synchronisation (user request, 2026-10-19) ---------------------------------------

def test_real_and_reconstruction_get_byte_identical_operations_and_parameters(
        monkeypatch) -> None:
    seen: list[str] = []
    original = D.apply_chain

    def spy(img, chain):
        seen.append(chain.to_json())
        return original(img, chain)

    monkeypatch.setattr(D, "apply_chain", spy)
    rng = np.random.default_rng(3)
    paths = Counter()
    for _ in range(300):
        seen.clear()
        real, recon = _image(1, 96), _image(2, 96)
        (_, _), chain = D.degrade_pair([real, recon], rng, 1.0)
        assert len(seen) == 2 and seen[0].encode() == seen[1].encode() == chain.to_json().encode()
        paths[(chain.op or "none") + ("_reshare" if chain.quality2 is not None else "")] += 1
    # every path was exercised, the re-share one included (both rounds in one shared chain)
    assert set(paths) == {"jpeg", "webp", "resize", "social", "social_reshare"}


def test_one_chain_gives_identical_bytes_for_identical_inputs_and_keeps_the_shape() -> None:
    img = _image(5)
    for chain in (D.DegradeChain("jpeg", 60, "4:2:0"), D.DegradeChain("webp", 70),
                  D.DegradeChain("resize", scale=0.6), D.DegradeChain("social", 75, "4:2:0", 0.8),
                  D.DegradeChain("social", 75, "4:2:0", 0.8, quality2=72, scale2=0.6)):
        once, twice = D.apply_chain(img, chain), D.apply_chain(img.copy(), chain)
        assert once.tobytes() == twice.tobytes()
        assert once.shape == img.shape and once.dtype == np.uint8
        assert not np.array_equal(once, img)            # it did something
    assert D.apply_chain(img, D.DegradeChain(None)).tobytes() == img.tobytes()


def test_the_trainer_degrades_every_pair_through_one_shared_chain() -> None:
    text = (ROOT / "scripts" / "p2_train_s2.py").read_text(encoding="utf-8")
    assert "degrade_pair(members, degrade_rng, run.degrade_p)" in text
    assert "augment_pair(degrade([real, fake])" in text
    assert "degrade([load_rgb_uint8(args.root / reals_index[real_id])," in text
    assert "np.random.default_rng([args.seed, DEGRADE_STREAM])" in text   # its own stream
    assert '"rng_degrade": degrade_rng.bit_generator.state' in text        # resumable


def test_bad_inputs_are_refused_and_codecs_are_present() -> None:
    with pytest.raises(D.DegradeError, match="in \\[0, 1\\]"):
        D.sample_chain(np.random.default_rng(0), 1.5)
    with pytest.raises(D.DegradeError, match="uint8 HWC RGB"):
        D.apply_chain(np.zeros((8, 8), np.uint8), D.DegradeChain("jpeg", 60, "4:2:0"))
    with pytest.raises(D.DegradeError, match="malformed"):
        D.apply_chain(_image(0, 32), D.DegradeChain("jpeg"))
    with pytest.raises(D.DegradeError, match="at least two"):
        D.degrade_pair([_image(0, 32)], np.random.default_rng(0), 1.0)
    D.check_codecs()


# --- the run configs ------------------------------------------------------------------

def test_candidates_are_s2b_plus_degradation_and_nothing_else_degrades() -> None:
    p3 = load_run_configs(ROOT / "configs" / "p3_runs.yaml", BANK)
    s2b = p3.by_id("s2b")
    # S2-D (2026-10-21) is S2-B + reals + the same chain: one entry per candidate p, so the
    # runbook resolves it from the committed tuning result. S2-B+reals does not degrade.
    for rid, p in (("s2b_deg_p70", 0.7), ("s2b_deg_p35", 0.35),
                   ("s2d_p70", 0.7), ("s2d_p35", 0.35)):
        run = p3.by_id(rid)
        assert run.degrade_p == p
        assert run.active_members(BANK) == s2b.active_members(BANK)
        assert (run.eval_crop, run.use_gen_matched) == (s2b.eval_crop, s2b.use_gen_matched)
    others = [r for r in p3.runs if not r.run_id.startswith(("s2b_deg_", "s2d_"))]
    assert p3.by_id("s2b_reals") in others
    assert all(r.degrade_p == 0.0 for r in others)
    assert all(r.degrade_p == 0.0 for r in load_run_configs(ROOT / "configs" / "p2_runs.yaml",
                                                            BANK).runs)


@pytest.mark.parametrize("value", ["0.5", True, -0.1, 1.1])
def test_bad_degrade_p_is_refused(tmp_path: Path, value) -> None:
    path = tmp_path / "runs.yaml"
    path.write_text(f"runs:\n  - run_id: x\n    degrade_p: {json.dumps(value)}\n",
                    encoding="utf-8")
    with pytest.raises(RunConfigError, match="degrade_p"):
        load_run_configs(path, BANK)


# --- the committed rule -------------------------------------------------------------

def _m(auc: float, fpr: float) -> dict:
    return {"macro_auc": auc, "macro_fpr": fpr}


@pytest.mark.parametrize(("p70", "p35", "chosen", "why"), [
    (_m(0.70, 0.10), _m(0.795, 0.20), "s2b_deg_p35_seed0", "only candidate"),     # one passes
    (_m(0.80, 0.10), _m(0.80, 0.20), "s2b_deg_p70_seed0", "lower macro FPR"),     # both, far
    (_m(0.80, 0.10), _m(0.80, 0.105), "s2b_deg_p35_seed0", "milder p"),           # both, close
    (_m(0.70, 0.30), _m(0.70, 0.10), "s2b_deg_p35_seed0", "lower macro FPR"),     # neither, far
    (_m(0.70, 0.10), _m(0.70, 0.11), "s2b_deg_p35_seed0", "milder p"),            # neither, close
])
def test_the_rule_in_every_branch(p70, p35, chosen, why) -> None:
    tune = _tune()
    decision = tune.choose(0.80, {"s2b_deg_p70_seed0": p70, "s2b_deg_p35_seed0": p35})
    assert decision["chosen"] == chosen and why in decision["reason"]
    assert decision["chosen_p"] == {"s2b_deg_p70_seed0": 0.7, "s2b_deg_p35_seed0": 0.35}[chosen]


def test_the_guard_is_inclusive_at_one_point() -> None:
    tune = _tune()
    decision = tune.choose(0.80, {"s2b_deg_p70_seed0": _m(0.79, 0.1),
                                  "s2b_deg_p35_seed0": _m(0.7899, 0.1)})
    assert decision["passes_guard"] == {"s2b_deg_p70_seed0": True, "s2b_deg_p35_seed0": False}


def test_dev_metrics_use_p2s_slices_the_val_reals_and_e4_e5_only(tmp_path: Path) -> None:
    tune = _tune()
    assert tune.unseen_slices() == ("e3_matched/GPT-Image2-fake", "e3_matched/Nano-Banana2-fake",
                                    "e4_wildrf_matched", "e5_chameleon_pipeline")
    rows, panel = [], {}
    for view, gen in (("e3_matched", "GPT-Image2-fake"), ("e3_matched", "Nano-Banana2-fake"),
                      ("e4_wildrf_matched", ""), ("e5_chameleon_pipeline", ""),
                      ("e7_dda_coco", "")):
        for k in range(3):
            iid = f"{view}_{gen}_{k}"
            rows.append({"image_id": iid, "view": view, "label": "fake", "subset": "",
                         "generator": gen})
            panel[iid] = (2.0 + k) if view == "e5_chameleon_pipeline" else (10.0 + k)
    for view in ("e3_matched", "e4_wildrf_matched", "e5_chameleon_pipeline", "e7_dda_coco"):
        for k in range(4):
            iid = f"{view}_real_{k}"
            rows.append({"image_id": iid, "view": view, "label": "real", "subset": "",
                         "generator": ""})
            panel[iid] = float(k) if view != "e5_chameleon_pipeline" else 10.0
    result = tune.evaluate(rows, panel, val_reals=[float(x) for x in range(100)])
    assert set(result["fpr"]) == {"e4_wildrf_matched", "e5_chameleon_pipeline"}
    assert result["threshold"] == 95.0 and result["threshold_achieved_fpr_on_val"] == 0.05
    assert result["fpr"]["e5_chameleon_pipeline"]["achieved_fpr"] == 0.0
    assert result["auc"]["e4_wildrf_matched"]["auc"] == 1.0
    assert result["auc"]["e5_chameleon_pipeline"]["auc"] == 0.0


def test_the_tuning_runbook_reads_no_panel_b_and_bounds_every_step() -> None:
    text = (ROOT / "env" / "kaggle_run_p3_tune.py").read_text(encoding="utf-8")
    code_lines = [line for line in text.splitlines() if not line.lstrip().startswith("#")]
    assert not any("panel_b" in line.lower() or "panel-b" in line.lower() for line in code_lines)
    assert text.count('py("p2_s2_infer.py"') == 2
    assert '"--deadline-hours", EXPECT["base_smoke_deadline_hours"]' in text
    assert '"--deadline-hours", EXPECT["base_score_deadline_hours"]' in text
    assert "import dbp" not in text and "from dbp" not in text
    expect = json.loads((ROOT / "env" / "p3_tune_expect.json").read_text(encoding="utf-8"))
    assert expect["runs"] == ["s2b_deg_p70:0", "s2b_deg_p35:0"]
    assert expect["base_stop_at"] == 5000 and "--stop-at=5000" in expect["train_args"]


def test_a_reshare_is_the_social_operation_applied_twice() -> None:
    img = _image(9)
    first = D.DegradeChain("social", 80, "4:2:0", 0.7)
    second = D.DegradeChain("social", 72, "4:2:0", 0.55)
    both = D.DegradeChain("social", 80, "4:2:0", 0.7, quality2=72, scale2=0.55)
    expected = D.apply_chain(D.apply_chain(img, first), second)
    assert D.apply_chain(img, both).tobytes() == expected.tobytes()
    assert D.apply_chain(img, both).tobytes() != D.apply_chain(img, first).tobytes()
    with pytest.raises(D.DegradeError, match="malformed"):
        D.apply_chain(img, D.DegradeChain("social", 80, "4:2:0", 0.7, quality2=72))
