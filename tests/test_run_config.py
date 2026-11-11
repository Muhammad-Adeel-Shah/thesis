"""Tests for the P2 run configuration and the SEF fairness rule (D-034).

The rule this file protects: a number compared against SEF must come from a run that
excludes the lineages SEF never trained on. Everything here is about making that
mechanical rather than remembered — a typo in an excluded lineage must fail loudly,
and the exclusions must be present in the block a report prints.
"""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

from dbp.decoders.registry import load_bank_config
from dbp.eval.run_config import (
    SEF_COMPARABLE_RUN,
    SEF_EXCLUDED_LINEAGES,
    RunConfigError,
    load_run_configs,
)

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
BANK_PATH = CONFIGS / "decoder_bank.yaml"
RUNS_PATH = CONFIGS / "p2_runs.yaml"


@pytest.fixture(scope="module")
def bank():
    return load_bank_config(BANK_PATH)


@pytest.fixture(scope="module")
def runs(bank):
    return load_run_configs(RUNS_PATH, bank)


@pytest.fixture
def raw() -> dict:
    with RUNS_PATH.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _write(tmp_path: Path, data: dict) -> Path:
    dest = tmp_path / "runs.yaml"
    dest.write_text(yaml.safe_dump(data), encoding="utf-8")
    return dest


def test_shipped_config_validates(runs):
    assert {run.run_id for run in runs.runs} == {
        "p_a_sd1",
        "p_a_sd1_chameleon",
        "4ch_only",
        "4ch_only_strict",
        "full_bank",
        "full_bank_chameleon",
        "4ch_only_chameleon",
    }


def test_exactly_one_run_is_sef_comparable(runs):
    """Two would make 'the SEF number' ambiguous; zero would make it unavailable."""
    sef = runs.sef_comparable()
    assert sef.run_id == SEF_COMPARABLE_RUN


def test_the_sef_run_excludes_every_lineage_sef_never_had(runs):
    sef = runs.sef_comparable()
    assert set(SEF_EXCLUDED_LINEAGES) <= set(sef.excluded_lineages)
    assert "kl_16ch" in sef.excluded_lineages
    assert "kl_16ch_qwen" in sef.excluded_lineages
    assert "kl_32ch_flux2" in sef.excluded_lineages


def test_the_sef_run_cannot_train_on_the_excluded_members(runs, bank):
    sef = runs.sef_comparable()
    active = sef.active_members(bank)
    for key in ("sd35", "flux", "qwen_image", "flux2_klein"):
        assert key not in active, f"{key} is reachable in the SEF-comparable run"
    for key in ("sd15", "sd21", "sdxl"):
        assert key in active


def test_the_full_bank_run_excludes_nothing_and_is_labelled(runs, bank):
    full = runs.by_id("full_bank")
    assert full.excluded_lineages == ()
    assert full.is_sef_comparable is False
    assert "deployment configuration" in full.label
    assert len(full.active_members(bank)) == len(bank.distinct_members())


def test_aliases_are_never_active_in_any_run(runs, bank):
    """distinct_members() is the source, so the sd21 alias cannot reappear here."""
    for run in runs.runs:
        assert "sd_ft_mse" not in run.active_members(bank)


def test_subset_status_distinguishes_the_three_cases(runs, bank):
    sef = runs.sef_comparable()
    assert sef.subset_status(bank, "sd15") == "in-bank"
    assert sef.subset_status(bank, "sd35") == "out-of-bank (excluded)"
    assert sef.subset_status(bank, None) == "out-of-bank (not in bank)"
    assert sef.subset_status(bank, "midjourney") == "out-of-bank (not in bank)"

    full = runs.by_id("full_bank")
    assert full.subset_status(bank, "sd35") == "in-bank"


def test_the_report_block_carries_the_exclusions(runs, bank):
    """A run whose exclusions are not printed beside its numbers is not reportable."""
    block = runs.sef_comparable().describe(bank)
    assert block["excluded_lineages"] == list(SEF_EXCLUDED_LINEAGES)
    assert block["sef_comparable"] is True
    assert "kl_16ch" not in block["active_lineages"]
    assert block["n_active_members"] == len(block["active_members"])

    header = runs.sef_comparable().report_header(bank)
    assert "excluded lineages" in header
    assert "kl_16ch" in header


def test_an_unknown_excluded_lineage_is_rejected(tmp_path, raw, bank):
    """A typo would silently exclude nothing and quietly invalidate the comparison."""
    data = copy.deepcopy(raw)
    data["runs"][0]["excluded_lineages"] = ["kl16ch"]  # deliberate typo
    with pytest.raises(RunConfigError, match="not lineages"):
        load_run_configs(_write(tmp_path, data), bank)


def test_duplicate_exclusions_and_ids_are_rejected(tmp_path, raw, bank):
    data = copy.deepcopy(raw)
    data["runs"][0]["excluded_lineages"] = ["kl_16ch", "kl_16ch"]
    with pytest.raises(RunConfigError, match="duplicates"):
        load_run_configs(_write(tmp_path, data), bank)

    data = copy.deepcopy(raw)
    data["runs"].append(copy.deepcopy(data["runs"][0]))
    with pytest.raises(RunConfigError, match="duplicate run_id"):
        load_run_configs(_write(tmp_path, data), bank)


def test_unknown_keys_and_missing_ids_are_rejected(tmp_path, raw, bank):
    data = copy.deepcopy(raw)
    data["runs"][0]["excluded_lineage"] = ["kl_16ch"]  # singular typo
    with pytest.raises(RunConfigError, match="unknown key"):
        load_run_configs(_write(tmp_path, data), bank)

    data = copy.deepcopy(raw)
    data["runs"][0].pop("run_id")
    with pytest.raises(RunConfigError, match="run_id is required"):
        load_run_configs(_write(tmp_path, data), bank)


def test_chameleon_runs_use_a_320_crop_and_a_single_crop(runs):
    """Measured: at 384 only 44.0% of Chameleon reals survive against 94.5% of fakes."""
    for run_id in ("full_bank_chameleon", "4ch_only_chameleon"):
        run = runs.by_id(run_id)
        assert run.eval_crop == 320
        assert run.eval_crops_per_image == 1


def test_no_chameleon_run_claims_to_be_sef_comparable(runs):
    """SEF published no Chameleon figure, so nothing there may carry the label."""
    for run in runs.runs:
        if "chameleon" in run.run_id:
            assert run.is_sef_comparable is False


# --------------------------------------------------------------------- P2 ---


def test_every_run_evaluates_at_320(runs):
    """D-045: one input size for all three streams, or S3's bank runs twice."""
    for run in runs.runs:
        assert run.eval_crop == 320, f"{run.run_id} is at {run.eval_crop}"
        assert run.eval_crops_per_image == 1


def test_the_primary_protocol_trains_on_the_sd1_lineage_and_nothing_else(runs, bank):
    """D-045: without this, SDXL is in-bank while the pre-registration calls it unseen."""
    run = runs.by_id("p_a_sd1")
    assert set(run.active_members(bank)) == {"sd15", "sd21", "sd_ft_ema"}
    assert run.active_lineages(bank) == ["kl_4ch_sd1"]


def test_the_chameleon_variant_of_p_a_has_the_same_bank(runs, bank):
    plain = runs.by_id("p_a_sd1")
    chameleon = runs.by_id("p_a_sd1_chameleon")
    assert chameleon.active_members(bank) == plain.active_members(bank)
    assert chameleon.is_sef_comparable is False


def test_strict_4ch_still_retains_sdxl_which_is_why_p_a_exists(runs, bank):
    """The nearest pre-P2 run could not serve as P-A: it trains on the SDXL lineage."""
    assert "kl_4ch_sdxl" in runs.by_id("4ch_only_strict").active_lineages(bank)


def test_4ch_only_is_still_the_single_sef_comparable_run(runs):
    """D-034 must survive the P2 additions."""
    comparable = [run.run_id for run in runs.runs if run.is_sef_comparable]
    assert comparable == ["4ch_only"]
