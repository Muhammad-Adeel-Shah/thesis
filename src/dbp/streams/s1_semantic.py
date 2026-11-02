"""S1: the frozen semantic stream -- DINOv3 features plus a linear probe.

The backbone never moves. Frozen vision-foundation-model features with a linear probe
are the strongest simple in-the-wild baseline reported in 2026 [R3], and freezing is
also what keeps S1 *different* from S2: fine-tune it and it becomes a second artifact
detector, at which point the complementarity question of RQ1 has no contrast left to
measure (§6.2).

The feature is ``[CLS || mean of the patch tokens]``, and the **register tokens are
excluded**. DINOv3 prepends four of them and DINOv2 none, so the number is read from
the loaded model's config rather than written down here -- a hard-coded 4 would
silently average four register tokens into the patch mean on any other backbone, and
the result would still look like a plausible feature vector.

Training-set features are cached from a fixed centre crop, not from augmented views.
That is a deliberate limitation, not an oversight: S1 exists to be the cheap frozen
baseline, and caching one view per image is what makes it cheap. It is recorded in the
report so that S1's robustness numbers are read with it in mind.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

#: Grid of L2 strengths the probe's C is chosen from, on validation only.
PROBE_C_GRID: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0)

#: Seeds for the train-subsample order, so the probe's stability is measurable.
PROBE_SEEDS: tuple[int, ...] = (0, 1, 2)


class SemanticStreamError(ValueError):
    """Raised when features or a probe cannot be built as specified."""


def register_token_count(config: Any) -> int:
    """Number of register tokens the backbone prepends, read from its own config.

    Args:
        config: A loaded model's config, or the metadata dict of
            :class:`dbp.backbones.loader.LoadedBackbone`.

    Returns:
        The count -- 4 for DINOv3, 0 for DINOv2.

    Raises:
        SemanticStreamError: If the field is absent. Defaulting to 0 would fold the
            register tokens into the patch mean without any sign that it happened.
    """
    if isinstance(config, dict):
        if "num_register_tokens" not in config:
            raise SemanticStreamError(
                "config has no 'num_register_tokens'; the patch mean cannot be taken "
                "without knowing how many leading tokens are not patches"
            )
        value = config["num_register_tokens"]
    else:
        value = getattr(config, "num_register_tokens", None)
        if value is None:
            raise SemanticStreamError(
                f"{type(config).__name__} has no num_register_tokens attribute"
            )
    return int(value or 0)


def expected_patch_tokens(input_size: int, patch_size: int) -> int:
    """Number of patch tokens a square input produces.

    Raises:
        SemanticStreamError: If the input is not a whole number of patches.
    """
    if input_size % patch_size != 0:
        raise SemanticStreamError(
            f"input {input_size} is not a multiple of patch size {patch_size}; the "
            "backbone would pad or crop and the token count would stop being checkable"
        )
    return (input_size // patch_size) ** 2


def pool_tokens(hidden_states: Any, num_register_tokens: int, num_patch_tokens: int) -> Any:
    """``[CLS || mean patch]`` from a ``(batch, tokens, hidden)`` output.

    Args:
        hidden_states: The backbone's last hidden state.
        num_register_tokens: Leading tokens after CLS that are not patches.
        num_patch_tokens: How many patch tokens to expect.

    Returns:
        A ``(batch, 2 * hidden)`` tensor.

    Raises:
        SemanticStreamError: If the token count does not match the layout. The check is
            the point: a silent mismatch is how register tokens end up in the mean.
    """
    if hidden_states.ndim != 3:
        raise SemanticStreamError(
            f"expected (batch, tokens, hidden), got {tuple(hidden_states.shape)}"
        )
    total = int(hidden_states.shape[1])
    expected = 1 + num_register_tokens + num_patch_tokens
    if total != expected:
        raise SemanticStreamError(
            f"the backbone returned {total} tokens but the layout says "
            f"1 CLS + {num_register_tokens} register + {num_patch_tokens} patch = "
            f"{expected}. Refusing to guess which ones are patches."
        )
    cls = hidden_states[:, 0, :]
    patches = hidden_states[:, 1 + num_register_tokens :, :].mean(dim=1)
    import torch

    return torch.cat([cls, patches], dim=1)


def normalise_batch(images_uint8: np.ndarray, mean: Sequence[float], std: Sequence[float]) -> Any:
    """Convert an ``(n, H, W, 3)`` uint8 batch to a normalised ``(n, 3, H, W)`` tensor.

    Args:
        images_uint8: The crops, already at the model's input size. Never resized here.
        mean: Per-channel mean, from the pinned model's own image processor.
        std: Per-channel standard deviation, likewise.

    Returns:
        A float32 tensor.

    Raises:
        SemanticStreamError: On the wrong dtype or shape.
    """
    import torch

    if images_uint8.dtype != np.uint8:
        raise SemanticStreamError(f"expected uint8, got {images_uint8.dtype}")
    if images_uint8.ndim != 4 or images_uint8.shape[3] != 3:
        raise SemanticStreamError(f"expected (n, H, W, 3), got {images_uint8.shape}")
    array = images_uint8.astype(np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(0, 3, 1, 2).contiguous()
    mean_t = torch.tensor(list(mean), dtype=torch.float32).view(1, 3, 1, 1)
    std_t = torch.tensor(list(std), dtype=torch.float32).view(1, 3, 1, 1)
    return (tensor - mean_t) / std_t


def processor_normalisation(repo: str, revision: str, token: str | None = None) -> dict[str, Any]:
    """Read the pinned model's own image mean and std from the Hub.

    Taking them from the model's processor rather than writing ImageNet constants into
    this file means the normalisation can never drift away from the weights it belongs
    to.

    Args:
        repo: Hub repo id.
        revision: The pinned commit SHA.
        token: HF token, for a gated repo.

    Returns:
        ``{"image_mean", "image_std", "source"}``.

    Raises:
        SemanticStreamError: If the processor cannot be read. Substituting a default
            would change every feature vector without changing any code.
    """
    import os

    from transformers import AutoImageProcessor

    if os.environ.get("DBP_DRYRUN") == "1":
        # The local dry-run cannot read the gated processor. It uses the ImageNet
        # constants and labels them, so no dry-run number can pass for a real one.
        return {"image_mean": [0.485, 0.456, 0.406], "image_std": [0.229, 0.224, 0.225],
                "source": "DRY-RUN placeholder (DBP_DRYRUN=1), not the pinned processor"}
    try:
        processor = AutoImageProcessor.from_pretrained(repo, revision=revision, token=token)
    except Exception as exc:
        raise SemanticStreamError(
            f"could not read the image processor for {repo}@{revision[:12]}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    mean = getattr(processor, "image_mean", None)
    std = getattr(processor, "image_std", None)
    if not mean or not std:
        raise SemanticStreamError(
            f"{repo}@{revision[:12]} has no image_mean/image_std on its processor"
        )
    return {
        "image_mean": [float(v) for v in mean],
        "image_std": [float(v) for v in std],
        "source": f"{repo}@{revision}",
    }


def fit_probe(
    train_features: np.ndarray,
    train_labels: Sequence[int] | np.ndarray,
    val_features: np.ndarray,
    val_labels: Sequence[int] | np.ndarray,
    c_grid: Sequence[float] = PROBE_C_GRID,
    seed: int = 0,
    max_iter: int = 2000,
) -> dict[str, Any]:
    """Train the linear probe, choosing ``C`` on validation only.

    Args:
        train_features: ``(n_train, d)`` cached features.
        train_labels: 0/1 labels.
        val_features: ``(n_val, d)`` cached features.
        val_labels: 0/1 labels.
        c_grid: Candidate L2 strengths.
        seed: Seed for the solver and the subsample order.
        max_iter: Solver iteration cap.

    Returns:
        ``{"model", "C", "val_auc", "grid", "seed", "n_train", "n_val"}``.

    Raises:
        SemanticStreamError: If either split has a single class, or the feature widths
            disagree.
    """
    from sklearn.linear_model import LogisticRegression

    from dbp.eval.metrics import roc_auc

    x_train = np.asarray(train_features, dtype=np.float32)
    x_val = np.asarray(val_features, dtype=np.float32)
    y_train = np.asarray(train_labels).astype(int).ravel()
    y_val = np.asarray(val_labels).astype(int).ravel()

    if x_train.ndim != 2 or x_val.ndim != 2:
        raise SemanticStreamError("features must be 2-D (n, d)")
    if x_train.shape[1] != x_val.shape[1]:
        raise SemanticStreamError(
            f"train features are {x_train.shape[1]}-d and val features "
            f"{x_val.shape[1]}-d; they came from different backbones or crops"
        )
    for name, y in (("train", y_train), ("val", y_val)):
        if len(np.unique(y)) < 2:
            raise SemanticStreamError(f"the {name} split has a single class")

    grid: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    for c in c_grid:
        model = LogisticRegression(C=float(c), max_iter=max_iter, random_state=seed)
        model.fit(x_train, y_train)
        auc = roc_auc(y_val, model.decision_function(x_val))
        grid.append({"C": float(c), "val_auc": auc})
        if best is None or auc > best["val_auc"]:
            best = {"model": model, "C": float(c), "val_auc": auc}

    assert best is not None  # c_grid is non-empty by construction
    return {
        **best,
        "grid": grid,
        "seed": seed,
        "n_train": int(x_train.shape[0]),
        "n_val": int(x_val.shape[0]),
    }
