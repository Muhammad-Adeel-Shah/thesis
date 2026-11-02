"""S2: the learned artifact stream -- the same backbone, with LoRA.

S1 and S2 share a backbone and share training data. The only differences are that S2
adapts the backbone with LoRA [R2] and that it is trained on **content-aligned pairs**
under synchronized augmentation. Everything the two streams disagree about is therefore
attributable to the representation, which is the condition RQ1 needs (§6.1, §6.3).

Two things here are discovered rather than assumed:

* the attention projection module names LoRA attaches to -- ``q_proj, k_proj, v_proj``
  on DINOv3 and ``query, key, value`` on DINOv2, per the P0 smoke test, but
  :func:`discover_attention_projections` reads them off the loaded model so a backbone
  swap cannot silently attach LoRA to nothing and train a bare linear head instead;
* the step budget, which is set from a measured step time against a GPU-hour allowance
  rather than fixed in advance (:func:`steps_within_allowance`), because no committed
  measurement of a training step on this backbone exists.

The batch is formed as **pairs**: a real and one of its reconstructions, always
together, always under one augmentation draw. Splitting a pair across two batches would
let batch statistics differ between the classes, which is a shortcut of exactly the
kind the pairing exists to remove.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

#: LoRA configuration, the setting DDA and SEF both use.
LORA_R = 8
LORA_ALPHA = 8
LORA_DROPOUT = 0.05

#: Module-name suffixes that are attention projections in the DINO family. Used only
#: to search the loaded model; nothing is attached that was not found.
PROJECTION_SUFFIXES: tuple[str, ...] = (
    "q_proj", "k_proj", "v_proj", "o_proj",
    "query", "key", "value",
    "qkv", "proj",
)

#: Below this many steps inside the allowance, the run is undertrained and the
#: comparison between streams stops meaning anything. Approved guard: stop and ask.
MIN_STEPS_PER_RUN = 2000


class ArtifactStreamError(ValueError):
    """Raised when the LoRA stream cannot be built or budgeted as specified."""


def discover_attention_projections(model: Any) -> list[str]:
    """Attention projection module names present in a loaded backbone.

    Args:
        model: The loaded backbone.

    Returns:
        Sorted distinct leaf names, e.g. ``["k_proj", "q_proj", "v_proj"]``.

    Raises:
        ArtifactStreamError: If none are found. LoRA would then attach to nothing and
            the run would silently reduce to training a linear head on frozen
            features -- that is S1, and it would be reported as S2.
    """
    import torch
    from torch import nn

    found: set[str] = set()
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        leaf = name.rsplit(".", 1)[-1]
        if leaf in PROJECTION_SUFFIXES and "mlp" not in name.lower():
            found.add(leaf)
    del torch
    if not found:
        raise ArtifactStreamError(
            "no attention projection modules found; LoRA would attach to nothing and "
            "S2 would collapse into S1. Inspect model.named_modules()."
        )
    return sorted(found)


def attach_lora(
    model: Any,
    target_modules: Sequence[str] | None = None,
    r: int = LORA_R,
    alpha: int = LORA_ALPHA,
    dropout: float = LORA_DROPOUT,
) -> dict[str, Any]:
    """Wrap a backbone in LoRA adapters and report what was actually attached.

    Args:
        model: The loaded backbone.
        target_modules: Module names to adapt. Discovered when omitted.
        r: LoRA rank.
        alpha: LoRA alpha.
        dropout: LoRA dropout.

    Returns:
        ``{"model", "target_modules", "params_trainable", "params_total",
        "trainable_pct"}``.

    Raises:
        ArtifactStreamError: If nothing became trainable.
    """
    from peft import LoraConfig, get_peft_model

    targets = list(target_modules) if target_modules else discover_attention_projections(model)
    config = LoraConfig(
        r=int(r),
        lora_alpha=int(alpha),
        lora_dropout=float(dropout),
        target_modules=targets,
        bias="none",
    )
    wrapped = get_peft_model(model, config)

    trainable = sum(p.numel() for p in wrapped.parameters() if p.requires_grad)
    total = sum(p.numel() for p in wrapped.parameters())
    if trainable == 0:
        raise ArtifactStreamError(
            f"LoRA attached to {targets} but nothing is trainable; the target names "
            "did not match any module in this backbone"
        )
    return {
        "model": wrapped,
        "target_modules": targets,
        "params_trainable": int(trainable),
        "params_total": int(total),
        "trainable_pct": round(100.0 * trainable / total, 4),
    }


def build_head(hidden_size: int) -> Any:
    """Linear classifier over ``[CLS || mean patch]``.

    Args:
        hidden_size: The backbone's hidden width; the head input is twice it.

    Returns:
        A ``torch.nn.Module`` mapping ``(batch, 2 * hidden)`` to ``(batch,)`` logits.
    """
    from torch import nn

    class ArtifactHead(nn.Module):
        """One linear layer, so the comparison with S1's probe stays honest."""

        def __init__(self) -> None:
            super().__init__()
            self.fc = nn.Linear(2 * int(hidden_size), 1)

        def forward(self, features: Any) -> Any:
            """Map pooled features to one logit per sample."""
            return self.fc(features).squeeze(-1)

    return ArtifactHead()


@dataclass(frozen=True)
class StepBudget:
    """How many steps an allowance buys at a measured step time.

    Attributes:
        allowance_hours: GPU-hours granted to this run.
        measured_step_seconds: Wall-clock of one full training step, measured.
        steps: Steps the allowance buys.
        ceiling: The plan's upper bound on steps.
        undertrained: Whether ``steps`` falls below :data:`MIN_STEPS_PER_RUN`.
    """

    allowance_hours: float
    measured_step_seconds: float
    steps: int
    ceiling: int
    undertrained: bool

    def to_dict(self) -> dict[str, Any]:
        """Flat mapping for the report."""
        return {
            "allowance_hours": self.allowance_hours,
            "measured_step_seconds": round(self.measured_step_seconds, 5),
            "steps": self.steps,
            "ceiling": self.ceiling,
            "undertrained": self.undertrained,
            "min_steps_per_run": MIN_STEPS_PER_RUN,
        }


def steps_within_allowance(
    allowance_hours: float, measured_step_seconds: float, ceiling: int = 8000
) -> StepBudget:
    """Convert a GPU-hour allowance and a measured step time into a step budget.

    No committed measurement of a training step on this backbone exists -- P0's LoRA
    entry is explicitly a dry run with no optimizer -- so the step count is derived
    from a measurement taken at the start of the run rather than fixed in the plan.

    Args:
        allowance_hours: GPU-hours for this run.
        measured_step_seconds: Seconds per full step, measured at the real batch size
            with gradient checkpointing on.
        ceiling: The plan's upper bound; a fast step buys no more than this.

    Returns:
        The budget. ``undertrained`` is the approved guard: below
        :data:`MIN_STEPS_PER_RUN`, stop and ask rather than run.

    Raises:
        ArtifactStreamError: If either input is not positive. A zero step time would
            divide into an unbounded step count.
    """
    if allowance_hours <= 0:
        raise ArtifactStreamError(f"allowance_hours must be positive, got {allowance_hours}")
    if measured_step_seconds <= 0:
        raise ArtifactStreamError(
            f"measured_step_seconds must be positive, got {measured_step_seconds}; "
            "the calibration cell did not produce a real measurement"
        )
    affordable = int(allowance_hours * 3600.0 / measured_step_seconds)
    steps = min(affordable, int(ceiling))
    return StepBudget(
        allowance_hours=float(allowance_hours),
        measured_step_seconds=float(measured_step_seconds),
        steps=steps,
        ceiling=int(ceiling),
        undertrained=steps < MIN_STEPS_PER_RUN,
    )


def pair_batch_plan(batch_size: int, gen_matched_share: float = 0.25) -> dict[str, int]:
    """Split a batch into aligned pairs plus the unpaired ``gen_matched`` share.

    A batch of 16 is 8 pairs. ``gen_matched`` rows have no reconstruction partner --
    they are generations matched to a real's JPEG settings -- so they enter as their
    own share of the fake half, keeping the class balance at 50/50.

    Args:
        batch_size: Total samples per step. Must be even.
        gen_matched_share: Share of the **fakes** drawn from ``gen_matched``.

    Returns:
        ``{"pairs", "reals", "reconstruction_fakes", "gen_matched_fakes"}``.

    Raises:
        ArtifactStreamError: If the batch is odd or the share is outside ``[0, 1)``.
    """
    if batch_size % 2 != 0 or batch_size < 2:
        raise ArtifactStreamError(
            f"batch_size must be an even number >= 2 so every real has its pair "
            f"member in the same step, got {batch_size}"
        )
    if not 0.0 <= gen_matched_share < 1.0:
        raise ArtifactStreamError(
            f"gen_matched_share must be in [0, 1), got {gen_matched_share}"
        )
    half = batch_size // 2
    gen_matched = int(round(half * gen_matched_share))
    return {
        "pairs": half - gen_matched,
        "reals": half,
        "reconstruction_fakes": half - gen_matched,
        "gen_matched_fakes": gen_matched,
    }
