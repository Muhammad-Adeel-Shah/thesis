"""Tests for the decoder-bank registry.

The registry is the only thing standing between a typo and a silently wrong decoder
fingerprint, so its validation is tested as carefully as the protocol itself.
"""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

from dbp.decoders.registry import (
    CROP_MULTIPLE_BY_KIND,
    BankConfigError,
    load_bank_config,
)

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "decoder_bank.yaml"

EXPECTED_KEYS = {
    "sd15",
    "sd21",
    "sdxl",
    "sd35",
    "flux",
    "dcae",
    "realesrgan_x4",
    "sd_ft_ema",
    "sd_ft_mse",
    "sdxl_fp16fix",
    "flux2_klein",
    "qwen_image",
    "movq_k22",
}


@pytest.fixture(scope="module")
def config():
    return load_bank_config(CONFIG_PATH)


@pytest.fixture
def raw() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _write(tmp_path: Path, data: dict) -> Path:
    dest = tmp_path / "bank.yaml"
    dest.write_text(yaml.safe_dump(data), encoding="utf-8")
    return dest


def test_shipped_config_validates(config):
    assert config.schema_version == 2
    assert {spec.key for spec in config.members} == EXPECTED_KEYS


def test_every_hub_member_has_a_pinned_40_char_revision(config):
    """A floating ref would let an upstream re-upload change the decoder silently."""
    for spec in config.members:
        if spec.is_hub_member:
            assert spec.revision is not None, f"{spec.key} has no pinned revision"
            assert len(spec.revision) == 40, f"{spec.key}: revision is not a full commit SHA"
            int(spec.revision, 16)


def test_crop_multiple_matches_kind(config):
    for spec in config.members:
        assert spec.crop_multiple == CROP_MULTIPLE_BY_KIND[spec.kind]


def test_sixteen_channel_lineages_are_present(config):
    """The 16-channel members are the core novelty claim against SEF."""
    sixteen = {s.key for s in config.members if s.latent_channels == 16}
    assert sixteen == {"sd35", "flux", "qwen_image"}


def test_lineage_grouping_is_sane(config):
    lineages = {spec.key: spec.lineage for spec in config.members}
    assert lineages["sd15"] == lineages["sd21"] == "kl_4ch_sd1"
    assert lineages["sdxl"] == lineages["sdxl_fp16fix"] == "kl_4ch_sdxl"
    assert lineages["sd35"] == lineages["flux"] == "kl_16ch"
    assert lineages["dcae"] == "dc_ae_f32"
    assert lineages["realesrgan_x4"] == "gan_sr"


def test_by_key_and_select(config):
    assert config.by_key("sd15").repo.endswith("stable-diffusion-v1-5")
    with pytest.raises(KeyError):
        config.by_key("does_not_exist")
    assert len(config.select(tiers=("core",))) == 10
    assert len(config.select(tiers=("optional",))) == 3
    assert {s.key for s in config.select(kinds=("sr_gan",))} == {"realesrgan_x4"}


def test_unknown_member_key_is_rejected(tmp_path, raw):
    data = copy.deepcopy(raw)
    data["members"][0]["latnet_channels"] = 4  # deliberate typo
    with pytest.raises(BankConfigError, match="unknown key"):
        load_bank_config(_write(tmp_path, data))


def test_unknown_top_level_key_is_rejected(tmp_path, raw):
    data = copy.deepcopy(raw)
    data["membres"] = []
    with pytest.raises(BankConfigError, match="unknown top-level key"):
        load_bank_config(_write(tmp_path, data))


def test_missing_revision_is_rejected(tmp_path, raw):
    data = copy.deepcopy(raw)
    data["members"][0].pop("revision")
    with pytest.raises(BankConfigError, match="pinned commit SHA"):
        load_bank_config(_write(tmp_path, data))


def test_short_revision_is_rejected(tmp_path, raw):
    data = copy.deepcopy(raw)
    data["members"][0]["revision"] = "451f4fe"
    with pytest.raises(BankConfigError, match="40-character commit SHA"):
        load_bank_config(_write(tmp_path, data))


def test_non_hex_revision_is_rejected(tmp_path, raw):
    data = copy.deepcopy(raw)
    data["members"][0]["revision"] = "z" * 40
    with pytest.raises(BankConfigError, match="not hexadecimal"):
        load_bank_config(_write(tmp_path, data))


def test_wrong_crop_multiple_is_rejected(tmp_path, raw):
    data = copy.deepcopy(raw)
    data["members"][0]["crop_multiple"] = 16  # vae_kl requires 8
    with pytest.raises(BankConfigError, match="requires 8"):
        load_bank_config(_write(tmp_path, data))


def test_duplicate_member_key_is_rejected(tmp_path, raw):
    data = copy.deepcopy(raw)
    data["members"].append(copy.deepcopy(data["members"][0]))
    with pytest.raises(BankConfigError, match="duplicate member key"):
        load_bank_config(_write(tmp_path, data))


def test_invalid_kind_is_rejected(tmp_path, raw):
    data = copy.deepcopy(raw)
    data["members"][0]["kind"] = "vae_vq"
    with pytest.raises(BankConfigError, match="not in"):
        load_bank_config(_write(tmp_path, data))


def test_sr_gan_requires_an_https_url(tmp_path, raw):
    data = copy.deepcopy(raw)
    for member in data["members"]:
        if member["kind"] == "sr_gan":
            member["url"] = "http://example.com/weights.pth"
    with pytest.raises(BankConfigError, match="requires an https"):
        load_bank_config(_write(tmp_path, data))


def test_missing_config_file_raises():
    with pytest.raises(BankConfigError, match="not found"):
        load_bank_config(Path("no_such_bank.yaml"))


# ---------------------------------------------------------------------------
# P1 schema: aliases and the three added decoder families
# ---------------------------------------------------------------------------


def test_alias_is_declared_in_the_config_not_hard_coded(config):
    """D-016 becomes enforced data rather than prose."""
    assert config.aliases() == {"sd_ft_mse": "sd21"}
    assert config.by_key("sd_ft_mse").is_alias is True
    assert config.by_key("sd21").is_alias is False


def test_distinct_members_excludes_the_alias(config):
    keys = {spec.key for spec in config.distinct_members()}
    assert "sd_ft_mse" not in keys
    assert "sd21" in keys
    assert len(keys) == len(config.members) - 1


def test_distinct_members_still_honours_tier_and_kind_filters(config):
    assert {s.key for s in config.distinct_members(kinds=("sr_gan",))} == {"realesrgan_x4"}
    optional = {s.key for s in config.distinct_members(tiers=("optional",))}
    assert optional == {"sd_ft_ema", "sdxl_fp16fix"}


def test_alias_pointing_at_a_missing_member_is_rejected(tmp_path, raw):
    data = copy.deepcopy(raw)
    data["members"][0]["alias_of"] = "no_such_member"
    with pytest.raises(BankConfigError, match="which is not a member"):
        load_bank_config(_write(tmp_path, data))


def test_alias_chain_is_rejected(tmp_path, raw):
    """Resolving a chain would make distinct_members() order-dependent."""
    data = copy.deepcopy(raw)
    by_key = {m["key"]: m for m in data["members"]}
    by_key["sd15"]["alias_of"] = "sd_ft_mse"  # sd_ft_mse is itself an alias of sd21
    with pytest.raises(BankConfigError, match="Alias chains are not allowed"):
        load_bank_config(_write(tmp_path, data))


def test_self_alias_is_rejected(tmp_path, raw):
    data = copy.deepcopy(raw)
    data["members"][0]["alias_of"] = data["members"][0]["key"]
    with pytest.raises(BankConfigError, match="points at itself"):
        load_bank_config(_write(tmp_path, data))


def test_the_three_added_families_have_their_own_lineages(config):
    assert config.by_key("flux2_klein").lineage == "kl_32ch_flux2"
    assert config.by_key("qwen_image").lineage == "kl_16ch_qwen"
    assert config.by_key("movq_k22").lineage == "vq_movq_4ch"


def test_every_lineage_is_represented_exactly_once_per_group(config):
    """Eight lineages after P1-B3; leave-one-lineage-out holds out a whole group."""
    lineages = {spec.lineage for spec in config.distinct_members()}
    assert lineages == {
        "kl_4ch_sd1",
        "kl_4ch_sdxl",
        "kl_16ch",
        "kl_16ch_qwen",
        "kl_32ch_flux2",
        "dc_ae_f32",
        "vq_movq_4ch",
        "gan_sr",
    }


def test_added_families_are_hub_members_with_a_crop_multiple_of_eight(config):
    for key in ("flux2_klein", "qwen_image", "movq_k22"):
        spec = config.by_key(key)
        assert spec.is_hub_member is True
        assert spec.crop_multiple == 8
        assert spec.subfolder, f"{key}: the weights live in a subfolder of the repo"


def test_added_families_are_not_gated_and_carry_a_permissive_license(config):
    """Chosen deliberately: a gated or `other`-licensed VAE cannot be relied on."""
    for key in ("flux2_klein", "qwen_image", "movq_k22"):
        spec = config.by_key(key)
        assert spec.gated is False, f"{key} is marked gated"
        assert spec.license == "apache-2.0", f"{key}: license is {spec.license!r}"


@pytest.mark.network
def test_every_pinned_revision_actually_resolves_on_the_hub(config):
    """Guard against a fabricated or mistyped SHA.

    A 40-character hex string passes the offline format check while pointing at
    nothing. This asserts the commit exists, not that it equals HEAD: the whole point
    of pinning is that the pin must NOT follow upstream.
    """
    from huggingface_hub import HfApi
    from huggingface_hub.utils import GatedRepoError, RevisionNotFoundError

    from dbp.utils.hf import get_hf_token

    api = HfApi()
    token = get_hf_token()
    unresolved = []
    for spec in config.members:
        if not spec.is_hub_member:
            continue
        try:
            api.model_info(spec.repo, revision=spec.revision, token=token)
        except RevisionNotFoundError:
            unresolved.append(f"{spec.key}: revision {spec.revision} does not exist")
        except GatedRepoError:
            pytest.skip(f"{spec.key} is gated and access is not granted in this runtime")
        except Exception as exc:  # network down, rate limit, etc.
            pytest.skip(f"Hub unreachable for {spec.key}: {type(exc).__name__}")
    assert not unresolved, "\n".join(unresolved)
