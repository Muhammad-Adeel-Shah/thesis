"""Evaluation run configuration — which lineages a run was allowed to train on.

**Why this exists.** SEF trained on the SD2.1 VAE alone. Our bank contains SD3.5, FLUX,
Qwen and FLUX.2 as well. A number produced with the full bank and placed next to SEF's
published DDA-COCO figures (FLUX.1 82.8, SD3.5 88.2) is not a comparison — it is a
better-resourced system beating a smaller one, and reporting it as a win would be
dishonest by omission.

So a run carries the lineages it was **not** allowed to use. Two configurations are
defined for P2:

- ``4ch_only`` — excludes ``kl_16ch``, ``kl_16ch_qwen`` and ``kl_32ch_flux2``. **Every
  number compared against SEF must come from this configuration.**
- ``full_bank`` — excludes nothing, and is reported separately and always labelled
  *"deployment configuration — in-bank decoders"*.

``excluded_lineages`` is validated against the registry, so a typo cannot silently
exclude nothing, and :meth:`RunConfig.describe` produces the block every evaluation
report prints. A run whose exclusions are not printed alongside its numbers is not
reportable.

The per-subset in-bank / out-of-bank question is separate and equally load-bearing:
:meth:`RunConfig.subset_status` answers it for one evaluation subset, so an E7 table can
state, per row, whether that VAE was in the training bank.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from dbp.decoders.registry import BankConfig

#: The configuration every SEF comparison must use. Named here rather than in a script
#: so the rule has one home (D-034).
SEF_COMPARABLE_RUN = "4ch_only"

#: Lineages SEF never saw. Excluded by the SEF-comparable configuration.
SEF_EXCLUDED_LINEAGES: tuple[str, ...] = ("kl_16ch", "kl_16ch_qwen", "kl_32ch_flux2")

ALLOWED_KEYS = frozenset(
    {
        "run_id",
        "excluded_lineages",
        "comparable_to",
        "label",
        "eval_crop",
        "eval_crops_per_image",
        "notes",
        "gen_matched",
        "excluded_members",
        "degrade_p",
    }
)


class RunConfigError(ValueError):
    """Raised when a run configuration is malformed or names an unknown lineage."""


@dataclass(frozen=True)
class RunConfig:
    """One training/evaluation configuration.

    Attributes:
        run_id: Short identifier, used in report filenames and tables.
        excluded_lineages: Lineages this run may **not** train on. Validated against
            the registry.
        comparable_to: The published baseline this configuration is built to be
            comparable with, or None. ``"SEF"`` marks the configuration whose numbers
            may be placed next to SEF's.
        label: The phrase that must appear beside this run's numbers in a table.
        eval_crop: Square crop used at evaluation time.
        eval_crops_per_image: Crops taken per evaluation image.
        notes: Free text carried into the report.
        use_gen_matched: Whether the run trains on ``gen_matched`` (the SD1.5
            generations). False only where that would leak a held-out lineage: P3's
            LOLO fold 1 holds out SD1, and SD1.5 generations are SD1 (P3 pre-registration
            section 3). Default True, so every P2 configuration is unchanged.
        excluded_members: Individual distinct members this run may **not** train on, on
            top of ``excluded_lineages`` (P3 ``BANK-1`` is sd15 alone: the SD1 lineage
            minus sd21 and sd_ft_ema). Validated against the registry; an alias is never
            named, because excluding its distinct twin excludes both (they share one
            decoder). Default empty, so every P2 and LOLO configuration is unchanged.
        degrade_p: Probability that a training pair gets a real-side degradation chain
            (P3 section 4b, :mod:`dbp.train.degrade`). 0 means none: every existing
            configuration is unchanged.
    """

    run_id: str
    excluded_lineages: tuple[str, ...] = ()
    comparable_to: str | None = None
    label: str = ""
    eval_crop: int = 384
    eval_crops_per_image: int = 1
    notes: str = ""
    use_gen_matched: bool = True
    excluded_members: tuple[str, ...] = ()
    degrade_p: float = 0.0

    @property
    def is_sef_comparable(self) -> bool:
        """Whether this run's numbers may be placed beside SEF's published figures."""
        return self.comparable_to == "SEF"

    def active_lineages(self, bank: BankConfig) -> list[str]:
        """Lineages this run may train on (at least one active member), sorted."""
        active = set(self.active_members(bank))
        return sorted({s.lineage for s in bank.distinct_members() if s.key in active})

    def active_members(self, bank: BankConfig) -> list[str]:
        """Distinct member keys this run may train on, in registry order."""
        excluded = set(self.excluded_lineages)
        members = set(self.excluded_members)
        return [s.key for s in bank.distinct_members()
                if s.lineage not in excluded and s.key not in members]

    def subset_status(self, bank: BankConfig, member_key: str | None) -> str:
        """Whether an evaluation subset's generator was in this run's training bank.

        Args:
            bank: The registry.
            member_key: The bank member corresponding to the subset's generator, or
                None when the subset's generator is not in the bank at all.

        Returns:
            ``"in-bank"``, ``"out-of-bank (excluded)"`` or ``"out-of-bank (not in bank)"``.
            Every E7 row prints one of these, so a reader never has to infer it.
        """
        if member_key is None:
            return "out-of-bank (not in bank)"
        try:
            spec = bank.by_key(member_key)
        except KeyError:
            return "out-of-bank (not in bank)"
        if spec.lineage in set(self.excluded_lineages) \
                or (spec.alias_of or spec.key) in set(self.excluded_members):
            return "out-of-bank (excluded)"
        return "in-bank"

    def describe(self, bank: BankConfig) -> dict[str, Any]:
        """The block every evaluation report prints for this run."""
        return {
            "run_id": self.run_id,
            "label": self.label,
            "comparable_to": self.comparable_to,
            "sef_comparable": self.is_sef_comparable,
            "excluded_lineages": list(self.excluded_lineages),
            "excluded_members": list(self.excluded_members),
            "degrade_p": self.degrade_p,
            "active_lineages": self.active_lineages(bank),
            "active_members": self.active_members(bank),
            "n_active_members": len(self.active_members(bank)),
            "eval_crop": self.eval_crop,
            "eval_crops_per_image": self.eval_crops_per_image,
            "notes": self.notes,
        }

    def report_header(self, bank: BankConfig) -> str:
        """A one-paragraph header for a human-readable report."""
        excluded = ", ".join(self.excluded_lineages) or "none"
        return (
            f"run {self.run_id} ({self.label or 'unlabelled'})\n"
            f"  excluded lineages : {excluded}\n"
            f"  excluded members  : {', '.join(self.excluded_members) or 'none'}\n"
            f"  trained on        : {', '.join(self.active_members(bank))}\n"
            f"  comparable to     : {self.comparable_to or 'nothing published'}\n"
            f"  evaluation crop   : {self.eval_crop} x{self.eval_crops_per_image}"
        )


@dataclass(frozen=True)
class RunConfigSet:
    """Every configuration defined for the project."""

    runs: list[RunConfig] = field(default_factory=list)

    def by_id(self, run_id: str) -> RunConfig:
        for run in self.runs:
            if run.run_id == run_id:
                return run
        raise KeyError(f"no run named {run_id!r}; have {[r.run_id for r in self.runs]}")

    def sef_comparable(self) -> RunConfig:
        """The one configuration whose numbers may sit beside SEF's."""
        candidates = [run for run in self.runs if run.is_sef_comparable]
        if len(candidates) != 1:
            raise RunConfigError(
                f"exactly one run must be marked comparable_to: SEF, found {len(candidates)}"
            )
        return candidates[0]


def _validate(raw: dict[str, Any], index: int, bank: BankConfig) -> RunConfig:
    where = f"runs[{index}] ({raw.get('run_id', '<no run_id>')})"
    unknown = set(raw) - ALLOWED_KEYS
    if unknown:
        raise RunConfigError(f"{where}: unknown key(s) {sorted(unknown)}")
    if not raw.get("run_id"):
        raise RunConfigError(f"{where}: run_id is required")

    excluded = tuple(raw.get("excluded_lineages") or ())
    known = {spec.lineage for spec in bank.members}
    unknown_lineages = sorted(set(excluded) - known)
    if unknown_lineages:
        raise RunConfigError(
            f"{where}: excluded_lineages names {unknown_lineages}, which are not lineages "
            f"in the registry. Known lineages: {sorted(known)}. A typo here would "
            f"silently exclude nothing and quietly invalidate the comparison."
        )
    if len(set(excluded)) != len(excluded):
        raise RunConfigError(f"{where}: excluded_lineages contains duplicates")

    members = tuple(raw.get("excluded_members") or ())
    if len(set(members)) != len(members):
        raise RunConfigError(f"{where}: excluded_members contains duplicates")
    specs = {spec.key: spec for spec in bank.members}
    for key in members:
        if key not in specs:
            raise RunConfigError(
                f"{where}: excluded_members names {key!r}, which is not a member of the "
                f"registry. Known members: {sorted(specs)}."
            )
        if specs[key].alias_of is not None:
            raise RunConfigError(
                f"{where}: excluded_members names the alias {key!r}; name "
                f"{specs[key].alias_of!r}, which excludes both (one decoder, never split)"
            )
        if specs[key].lineage in excluded:
            raise RunConfigError(
                f"{where}: {key!r} is already excluded by its lineage "
                f"{specs[key].lineage!r}; list it once"
            )
    if members and not [s for s in bank.distinct_members()
                        if s.lineage not in set(excluded) and s.key not in set(members)]:
        raise RunConfigError(f"{where}: the exclusions leave no member to train on")

    degrade = raw.get("degrade_p", 0.0)
    if isinstance(degrade, bool) or not isinstance(degrade, (int, float)) \
            or not 0.0 <= float(degrade) <= 1.0:
        raise RunConfigError(f"{where}: degrade_p must be a number in [0, 1], got {degrade!r}")

    gen = raw.get("gen_matched", True)
    if not isinstance(gen, bool):
        raise RunConfigError(f"{where}: gen_matched must be true or false, got {gen!r}")

    crops = int(raw.get("eval_crops_per_image", 1))
    if crops < 1:
        raise RunConfigError(f"{where}: eval_crops_per_image must be >= 1")

    return RunConfig(
        run_id=str(raw["run_id"]),
        excluded_lineages=excluded,
        comparable_to=raw.get("comparable_to"),
        label=str(raw.get("label", "")),
        eval_crop=int(raw.get("eval_crop", 384)),
        eval_crops_per_image=crops,
        notes=str(raw.get("notes", "")),
        use_gen_matched=gen,
        excluded_members=members,
        degrade_p=float(degrade),
    )


def load_run_configs(path: str | Path, bank: BankConfig) -> RunConfigSet:
    """Load and validate the run configurations against the registry.

    Args:
        path: Path to ``configs/p2_runs.yaml``.
        bank: The validated bank registry, used to check every excluded lineage exists.

    Returns:
        The validated :class:`RunConfigSet`.

    Raises:
        RunConfigError: On an unknown key, a missing ``run_id``, a duplicate id, or an
            excluded lineage that is not in the registry.
    """
    config_path = Path(path)
    if not config_path.is_file():
        raise RunConfigError(f"run config not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict) or "runs" not in raw:
        raise RunConfigError(f"{config_path}: expected a mapping with a 'runs' list")

    runs = [_validate(item, index, bank) for index, item in enumerate(raw["runs"])]
    seen: set[str] = set()
    for run in runs:
        if run.run_id in seen:
            raise RunConfigError(f"{config_path}: duplicate run_id {run.run_id!r}")
        seen.add(run.run_id)
    return RunConfigSet(runs=runs)
