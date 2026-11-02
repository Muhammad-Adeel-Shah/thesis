"""DINO-family backbone loading and the P0 LoRA attach dry-run.

DINOv2/DINOv3 are the strongest backbones currently reported for generalizable
AI-image detection: B-Free and DDA use DINOv2, SEF uses DINOv3 [R10][R11][R12]. ViT-B
is used rather than ViT-L so that the fused model still fits a 16 GB T4 during P2
training, and so that the SEF-lite and DDA-lite parity baselines run on exactly the
same backbone.

This module performs **no training**. ``lora_dry_run`` attaches adapters, counts
parameters and runs a single forward pass. Training hyperparameters are decided in P2;
``r=8``/``alpha=8`` here only mirror the DDA and SEF setting so the parameter counts in
the P0 report are meaningful.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any

import torch

#: Backbones pinned to commit SHAs resolved during P0 (2026-09-19).
BACKBONES: dict[str, dict[str, Any]] = {
    "dinov2-base": {
        "repo": "facebook/dinov2-base",
        "revision": "f9e44c814b77203eaa57a6bdbbd535f21ede1415",
        "patch_size": 14,
        "gated": False,
        "license": "apache-2.0",
        "default": True,
    },
    "dinov3-vitb16": {
        "repo": "facebook/dinov3-vitb16-pretrain-lvd1689m",
        "revision": "5931719e67bbdb9737e363e781fb0c67687896bc",
        "patch_size": 16,
        "gated": True,
        "license": "other",
        "default": False,
    },
}

#: 336 is divisible by both 14 (DINOv2) and 16 (DINOv3), so the two backbones can be
#: compared at an identical input size without resizing differently for each.
DEFAULT_INPUT_SIZE = 336

#: Substrings that identify attention projection modules. Used only to *search*; the
#: names actually attached are always discovered from the loaded model, never assumed.
ATTENTION_NAME_HINTS = ("query", "key", "value", "qkv", "q_proj", "k_proj", "v_proj")


#: The local runbook dry-run (``DBP_DRYRUN=1``) has neither a GPU nor the gated weights.
#: It builds a randomly initialised model of the pinned DINOv3 architecture instead --
#: same class, same patch size, same four register tokens, same attention module names
#: for LoRA -- with two layers rather than twelve so a CPU can train it. It refuses CUDA,
#: so it cannot stand in for the real backbone on Kaggle, and it says what it is in the
#: returned config. Weight download and the token are first exercised by Kaggle's
#: pre-flight.
DRYRUN_DINOV3_CONFIG: dict[str, Any] = {
    "hidden_size": 768, "intermediate_size": 3072, "num_attention_heads": 12,
    "num_hidden_layers": 2, "num_register_tokens": 4, "patch_size": 16,
    "image_size": 224, "layer_norm_eps": 1e-05, "layerscale_value": 1.0,
    "rope_theta": 100.0, "pos_embed_rescale": 2.0, "query_bias": True,
    "key_bias": False, "value_bias": True, "proj_bias": True, "mlp_bias": True,
    "use_gated_mlp": False,
}


def dryrun_enabled() -> bool:
    """Whether this process is the local runbook dry-run (``DBP_DRYRUN=1``)."""
    return os.environ.get("DBP_DRYRUN") == "1"


class BackboneLoadError(RuntimeError):
    """Raised when a backbone cannot be loaded or is gated without access."""


@dataclass
class LoadedBackbone:
    """A loaded backbone plus the metadata the P0 report needs."""

    name: str
    repo: str
    revision: str
    model: Any
    hidden_size: int
    patch_size: int
    num_parameters: int
    config: dict[str, Any] = field(default_factory=dict)


def load_backbone(
    name: str = "dinov2-base",
    device: torch.device | str = "cpu",
    token: str | None = None,
) -> LoadedBackbone:
    """Load a DINO-family backbone in eval mode, pinned to its commit SHA.

    ``trust_remote_code`` is never passed: both models are natively supported by
    transformers, so no remote code needs to execute.

    Args:
        name: A key of :data:`BACKBONES`.
        device: Torch device.
        token: HF token, required only for the gated DINOv3.

    Returns:
        The loaded backbone and its metadata.

    Raises:
        BackboneLoadError: On an unknown name, or when the Hub refuses access.
    """
    if name not in BACKBONES:
        raise BackboneLoadError(f"unknown backbone {name!r}; have {sorted(BACKBONES)}")
    entry = BACKBONES[name]

    from transformers import AutoConfig, AutoModel

    if dryrun_enabled():
        if str(device).startswith("cuda") or name != "dinov3-vitb16":
            raise BackboneLoadError(
                "DBP_DRYRUN=1 builds a random-init DINOv3 for CPU dry-runs only; it never "
                f"stands in on {device!r} or for {name!r}"
            )
        model = AutoModel.from_config(
            AutoConfig.for_model("dinov3_vit", **DRYRUN_DINOV3_CONFIG)
        )
    else:
        try:
            model = AutoModel.from_pretrained(
                entry["repo"],
                revision=entry["revision"],
                use_safetensors=True,
                token=token,
            )
        except Exception as exc:
            raise BackboneLoadError(f"{name}: {type(exc).__name__}: {exc}") from exc

    model.to(device)
    model.eval()

    cfg = model.config
    hidden_size = int(getattr(cfg, "hidden_size", 0))
    patch_size = int(getattr(cfg, "patch_size", entry["patch_size"]))

    return LoadedBackbone(
        name=name,
        repo=entry["repo"],
        revision=entry["revision"],
        model=model,
        hidden_size=hidden_size,
        patch_size=patch_size,
        num_parameters=sum(p.numel() for p in model.parameters()),
        config={
            "model_type": getattr(cfg, "model_type", None),
            "num_hidden_layers": getattr(cfg, "num_hidden_layers", None),
            "num_attention_heads": getattr(cfg, "num_attention_heads", None),
            "num_register_tokens": getattr(cfg, "num_register_tokens", 0),
            "image_size": getattr(cfg, "image_size", None),
            "dry_run_random_init": dryrun_enabled(),
        },
    )


def find_attention_projection_names(model: Any) -> list[str]:
    """Discover the attention projection module names present in ``model``.

    DINOv2 and DINOv3 do not name these the same way (separate ``query``/``key``/
    ``value`` versus a fused ``qkv``), so the names are always read off the loaded
    model rather than hard-coded.

    Returns:
        The sorted unique leaf names of every ``nn.Linear`` that looks like an
        attention projection.
    """
    found: set[str] = set()
    for module_name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        leaf = module_name.rsplit(".", 1)[-1]
        if any(hint in leaf.lower() for hint in ATTENTION_NAME_HINTS):
            found.add(leaf)
    return sorted(found)


@torch.no_grad()
def forward_once(
    model: Any,
    size: int = DEFAULT_INPUT_SIZE,
    device: torch.device | str = "cpu",
    use_fp16_autocast: bool = False,
    warmups: int = 0,
) -> dict[str, Any]:
    """Run a single forward pass and measure shape, latency and peak memory.

    fp16 autocast (never bf16) is used on CUDA, matching the T4 policy: the T4 has
    fp16 tensor cores but no native bf16 [R21].

    Args:
        model: A loaded backbone taking ``pixel_values``.
        size: Square input side length.
        device: Device to run on.
        use_fp16_autocast: Enable fp16 autocast (CUDA only).
        warmups: Forward passes run and discarded before the measured one (issue
            I-009). The first pass on CUDA absorbs context creation and cuDNN
            autotuning, so without this the first backbone measured looks slow and the
            second looks fast. Peak memory is reset after the warm-ups, so the reported
            figure is the steady-state pass, not the allocator's initial growth.
    """
    device = torch.device(device)
    x = torch.zeros(1, 3, size, size, device=device)

    is_cuda = device.type == "cuda"
    for _ in range(warmups):
        if is_cuda and use_fp16_autocast:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                model(pixel_values=x)
        else:
            model(pixel_values=x)

    if is_cuda:
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    start = time.perf_counter()
    if is_cuda and use_fp16_autocast:
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            out = model(pixel_values=x)
    else:
        out = model(pixel_values=x)
    if is_cuda:
        torch.cuda.synchronize(device)
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    hidden = out.last_hidden_state
    return {
        "input_size": size,
        "output_shape": list(hidden.shape),
        "num_tokens": int(hidden.shape[1]),
        "hidden_size": int(hidden.shape[2]),
        "forward_ms": round(elapsed_ms, 2),
        "peak_gpu_mb": (
            round(torch.cuda.max_memory_allocated(device) / 1024**2, 1) if is_cuda else None
        ),
        "fp16_autocast": bool(is_cuda and use_fp16_autocast),
    }


def lora_dry_run(
    model: Any,
    r: int = 8,
    lora_alpha: int = 8,
    size: int = DEFAULT_INPUT_SIZE,
    device: torch.device | str = "cpu",
) -> dict[str, Any]:
    """Attach LoRA adapters, count parameters and run one forward pass.

    No optimizer is created and no training step is taken. This exists purely to prove
    that PEFT can find and wrap this backbone's attention projections before P2 relies
    on it.

    Args:
        model: A loaded backbone.
        r: LoRA rank. 8 matches DDA and SEF [R11][R12][R14].
        lora_alpha: LoRA alpha.
        size: Forward-pass input size.
        device: Torch device.

    Returns:
        The discovered target modules, trainable/total parameter counts and the
        post-attach forward result.
    """
    from peft import LoraConfig, get_peft_model

    target_modules = find_attention_projection_names(model)
    if not target_modules:
        raise BackboneLoadError(
            "no attention projection modules found; refusing to guess target_modules"
        )

    total_before = sum(p.numel() for p in model.parameters())

    config = LoraConfig(r=r, lora_alpha=lora_alpha, target_modules=target_modules, bias="none")
    peft_model = get_peft_model(model, config)

    trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    total_after = sum(p.numel() for p in peft_model.parameters())

    peft_model.eval()
    forward = forward_once(peft_model, size=size, device=device)

    return {
        "r": r,
        "lora_alpha": lora_alpha,
        "target_modules_found": target_modules,
        "params_total_before": total_before,
        "params_total_after": total_after,
        "params_trainable": trainable,
        "trainable_pct": round(100.0 * trainable / total_after, 4),
        "forward_after_attach": forward,
        "note": "dry-run only: no optimizer, no training step",
    }
