"""Decoder-bank registry: schema, validation and loading of ``decoder_bank.yaml``.

Validation is deliberately strict and fails loudly. A silently mistyped key would
produce a bank member that loads from a floating revision or crops on the wrong
multiple, and either error would corrupt every downstream experiment while still
looking like a plausible result.

Schema version 2 (P1) adds three things:

- ``alias_of``: a member whose decoder weights are byte-identical to another member's.
  P0 measured ``sd_ft_mse`` and ``sd21`` to the same decoder digest, so generating both
  would duplicate a probe channel and inflate the apparent size of the bank. Aliases
  are excluded by :meth:`BankConfig.distinct_members`, never by a hard-coded key list.
- Three further kinds -- ``vae_kl_flux2``, ``vae_kl_5d`` and ``vq`` -- for decoder
  families whose diffusers class is not ``AutoencoderKL``.
- Lineage labels for those families.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml

Kind = Literal["vae_kl", "vae_kl_flux2", "vae_kl_5d", "vae_dc", "vq", "sr_gan"]
Tier = Literal["core", "optional"]

VALID_KINDS: frozenset[str] = frozenset(
    {"vae_kl", "vae_kl_flux2", "vae_kl_5d", "vae_dc", "vq", "sr_gan"}
)
VALID_TIERS: frozenset[str] = frozenset({"core", "optional"})
VALID_DTYPE_POLICIES: frozenset[str] = frozenset({"fp32_default", "fp32_only"})

#: Every kind except ``sr_gan`` loads its weights from the Hugging Face Hub.
HUB_KINDS: frozenset[str] = VALID_KINDS - {"sr_gan"}

#: Centre-crop multiple required by each kind. Enforced, not merely documented.
CROP_MULTIPLE_BY_KIND: dict[str, int] = {
    "vae_kl": 8,
    "vae_kl_flux2": 8,
    "vae_kl_5d": 8,
    "vae_dc": 32,
    "vq": 8,
    "sr_gan": 4,
}

#: Lineage label implied by the kind, for kinds that form a lineage on their own.
LINEAGE_BY_KIND: dict[str, str] = {
    "sr_gan": "gan_sr",
    "vae_dc": "dc_ae_f32",
    "vae_kl_flux2": "kl_32ch_flux2",
    "vae_kl_5d": "kl_16ch_qwen",
    "vq": "vq_movq_4ch",
}

#: Every key a member may declare. Anything else is a typo and raises.
ALLOWED_MEMBER_KEYS: frozenset[str] = frozenset(
    {
        "key",
        "tier",
        "kind",
        "repo",
        "subfolder",
        "revision",
        "url",
        "sha256",
        "latent_channels",
        "downsample",
        "gated",
        "license",
        "dtype_policy",
        "crop_multiple",
        "generators",
        "refs",
        "notes",
        "alias_of",
    }
)

REQUIRED_MEMBER_KEYS: frozenset[str] = frozenset(
    {
        "key",
        "tier",
        "kind",
        "gated",
        "license",
        "dtype_policy",
        "crop_multiple",
        "generators",
        "refs",
    }
)

ALLOWED_TOP_LEVEL_KEYS: frozenset[str] = frozenset(
    {"schema_version", "resolved_on", "seed", "members"}
)


class BankConfigError(ValueError):
    """Raised when ``decoder_bank.yaml`` is malformed or internally inconsistent."""


@dataclass(frozen=True)
class DecoderSpec:
    """One member of the decoder bank.

    Hub members (every kind except ``sr_gan``) carry ``repo``/``subfolder``/``revision``.
    The GAN upsampler (``sr_gan``) carries ``url``/``sha256`` instead.
    """

    key: str
    tier: Tier
    kind: Kind
    gated: bool
    license: str
    dtype_policy: str
    crop_multiple: int
    generators: list[str]
    refs: list[str]
    repo: str | None = None
    subfolder: str = ""
    revision: str | None = None
    url: str | None = None
    sha256: str | None = None
    latent_channels: int | None = None
    downsample: int | None = None
    notes: str | None = None
    alias_of: str | None = None

    @property
    def is_hub_member(self) -> bool:
        """True when the weights come from the Hugging Face Hub."""
        return self.kind in HUB_KINDS

    @property
    def is_alias(self) -> bool:
        """True when this member's decoder is byte-identical to another member's."""
        return self.alias_of is not None

    @property
    def lineage(self) -> str:
        """Coarse lineage label used by the leave-one-lineage-out protocol."""
        by_kind = LINEAGE_BY_KIND.get(self.kind)
        if by_kind is not None:
            return by_kind
        if self.latent_channels == 16:
            return "kl_16ch"
        if self.key.startswith("sdxl"):
            return "kl_4ch_sdxl"
        return "kl_4ch_sd1"


@dataclass(frozen=True)
class BankConfig:
    """The validated contents of ``decoder_bank.yaml``."""

    schema_version: int
    resolved_on: str
    seed: int
    members: list[DecoderSpec] = field(default_factory=list)

    def by_key(self, key: str) -> DecoderSpec:
        """Return the member with this key, or raise ``KeyError``."""
        for spec in self.members:
            if spec.key == key:
                return spec
        raise KeyError(f"no bank member named {key!r}; have {[s.key for s in self.members]}")

    def select(
        self, tiers: tuple[str, ...] = ("core", "optional"), kinds: tuple[str, ...] | None = None
    ) -> list[DecoderSpec]:
        """Return members filtered by tier and, optionally, kind."""
        out = [s for s in self.members if s.tier in tiers]
        if kinds is not None:
            out = [s for s in out if s.kind in kinds]
        return out

    def distinct_members(
        self, tiers: tuple[str, ...] = ("core", "optional"), kinds: tuple[str, ...] | None = None
    ) -> list[DecoderSpec]:
        """Like :meth:`select`, but with aliased duplicates removed.

        A member declaring ``alias_of`` has the same decoder weights as its target, so
        reconstructing through both would store two identical probe channels. Every
        data-building stage iterates over this, never over :meth:`select`, and never
        over a list of keys written out by hand.
        """
        return [s for s in self.select(tiers=tiers, kinds=kinds) if not s.is_alias]

    def aliases(self) -> dict[str, str]:
        """Map each aliased member key to the key whose decoder it duplicates."""
        return {s.key: s.alias_of for s in self.members if s.alias_of is not None}


def _validate_member(raw: dict[str, Any], index: int) -> DecoderSpec:
    """Validate one raw member mapping and build its :class:`DecoderSpec`."""
    where = f"members[{index}]"
    if not isinstance(raw, dict):
        raise BankConfigError(f"{where}: expected a mapping, got {type(raw).__name__}")

    key = raw.get("key", f"<missing key at {where}>")
    where = f"members[{index}] ({key})"

    unknown = set(raw) - ALLOWED_MEMBER_KEYS
    if unknown:
        raise BankConfigError(
            f"{where}: unknown key(s) {sorted(unknown)}. "
            f"Allowed keys are {sorted(ALLOWED_MEMBER_KEYS)}."
        )

    missing = REQUIRED_MEMBER_KEYS - set(raw)
    if missing:
        raise BankConfigError(f"{where}: missing required key(s) {sorted(missing)}")

    kind = raw["kind"]
    if kind not in VALID_KINDS:
        raise BankConfigError(f"{where}: kind {kind!r} not in {sorted(VALID_KINDS)}")

    tier = raw["tier"]
    if tier not in VALID_TIERS:
        raise BankConfigError(f"{where}: tier {tier!r} not in {sorted(VALID_TIERS)}")

    dtype_policy = raw["dtype_policy"]
    if dtype_policy not in VALID_DTYPE_POLICIES:
        raise BankConfigError(
            f"{where}: dtype_policy {dtype_policy!r} not in {sorted(VALID_DTYPE_POLICIES)}"
        )

    if not isinstance(raw["gated"], bool):
        raise BankConfigError(f"{where}: gated must be a bool, got {type(raw['gated']).__name__}")

    expected_multiple = CROP_MULTIPLE_BY_KIND[kind]
    if raw["crop_multiple"] != expected_multiple:
        raise BankConfigError(
            f"{where}: crop_multiple is {raw['crop_multiple']} but kind {kind!r} "
            f"requires {expected_multiple}"
        )

    for list_key in ("generators", "refs"):
        if not isinstance(raw[list_key], list) or not raw[list_key]:
            raise BankConfigError(f"{where}: {list_key} must be a non-empty list")

    alias_of = raw.get("alias_of")
    if alias_of is not None:
        if not isinstance(alias_of, str) or not alias_of:
            raise BankConfigError(f"{where}: alias_of must be a non-empty member key")
        if alias_of == raw["key"]:
            raise BankConfigError(f"{where}: alias_of points at itself")

    if kind in HUB_KINDS:
        if not raw.get("repo"):
            raise BankConfigError(f"{where}: kind {kind!r} requires a non-empty 'repo'")
        revision = raw.get("revision")
        if not revision:
            raise BankConfigError(
                f"{where}: 'revision' is required and must be a pinned commit SHA. "
                f"Loading from a floating ref would let an upstream re-upload change "
                f"the decoder fingerprint."
            )
        if not (isinstance(revision, str) and len(revision) == 40):
            raise BankConfigError(
                f"{where}: revision must be a 40-character commit SHA, got {revision!r}"
            )
        try:
            int(revision, 16)
        except ValueError as exc:
            raise BankConfigError(f"{where}: revision {revision!r} is not hexadecimal") from exc
        if raw.get("latent_channels") is None:
            raise BankConfigError(f"{where}: kind {kind!r} requires 'latent_channels'")
        if raw.get("downsample") is None:
            raise BankConfigError(f"{where}: kind {kind!r} requires 'downsample'")
        if raw.get("url") is not None:
            raise BankConfigError(f"{where}: 'url' is only valid for kind 'sr_gan'")
    else:  # sr_gan
        url = raw.get("url")
        if not url or not str(url).startswith("https://"):
            raise BankConfigError(f"{where}: kind 'sr_gan' requires an https 'url', got {url!r}")
        if raw.get("repo") is not None:
            raise BankConfigError(f"{where}: 'repo' is only valid for Hub members")
        sha256 = raw.get("sha256")
        if sha256 is not None and not (isinstance(sha256, str) and len(sha256) == 64):
            raise BankConfigError(
                f"{where}: sha256 must be a 64-character hex digest or null, got {sha256!r}"
            )

    return DecoderSpec(
        key=raw["key"],
        tier=tier,
        kind=kind,
        gated=raw["gated"],
        license=raw["license"],
        dtype_policy=dtype_policy,
        crop_multiple=raw["crop_multiple"],
        generators=list(raw["generators"]),
        refs=list(raw["refs"]),
        repo=raw.get("repo"),
        subfolder=raw.get("subfolder") or "",
        revision=raw.get("revision"),
        url=raw.get("url"),
        sha256=raw.get("sha256"),
        latent_channels=raw.get("latent_channels"),
        downsample=raw.get("downsample"),
        notes=raw.get("notes"),
        alias_of=alias_of,
    )


def _validate_aliases(members: list[DecoderSpec], where: str) -> None:
    """Every ``alias_of`` must name a real, non-aliased member.

    Chains are rejected rather than resolved: ``distinct_members`` would otherwise
    depend on traversal order, and a two-step alias is far more likely to be a mistake
    than an intention.
    """
    by_key = {spec.key: spec for spec in members}
    for spec in members:
        if spec.alias_of is None:
            continue
        target = by_key.get(spec.alias_of)
        if target is None:
            raise BankConfigError(
                f"{where}: member {spec.key!r} aliases {spec.alias_of!r}, which is not a member"
            )
        if target.alias_of is not None:
            raise BankConfigError(
                f"{where}: member {spec.key!r} aliases {spec.alias_of!r}, which is itself an "
                f"alias of {target.alias_of!r}. Alias chains are not allowed."
            )


def load_bank_config(path: str | Path) -> BankConfig:
    """Load and fully validate ``decoder_bank.yaml``.

    Args:
        path: Path to the registry YAML.

    Returns:
        A validated :class:`BankConfig`.

    Raises:
        BankConfigError: On any unknown key, missing field, unpinned revision,
            duplicate member key, kind/crop-multiple mismatch or a dangling alias.
    """
    config_path = Path(path)
    if not config_path.is_file():
        raise BankConfigError(f"bank config not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    if not isinstance(raw, dict):
        raise BankConfigError(f"{config_path}: top level must be a mapping")

    unknown = set(raw) - ALLOWED_TOP_LEVEL_KEYS
    if unknown:
        raise BankConfigError(
            f"{config_path}: unknown top-level key(s) {sorted(unknown)}. "
            f"Allowed: {sorted(ALLOWED_TOP_LEVEL_KEYS)}."
        )

    for required in ("schema_version", "members"):
        if required not in raw:
            raise BankConfigError(f"{config_path}: missing top-level key {required!r}")

    members_raw = raw["members"]
    if not isinstance(members_raw, list) or not members_raw:
        raise BankConfigError(f"{config_path}: 'members' must be a non-empty list")

    members = [_validate_member(m, i) for i, m in enumerate(members_raw)]

    seen: set[str] = set()
    for spec in members:
        if spec.key in seen:
            raise BankConfigError(f"{config_path}: duplicate member key {spec.key!r}")
        seen.add(spec.key)

    _validate_aliases(members, str(config_path))

    return BankConfig(
        schema_version=int(raw["schema_version"]),
        resolved_on=str(raw.get("resolved_on", "")),
        seed=int(raw.get("seed", 2026)),
        members=members,
    )
