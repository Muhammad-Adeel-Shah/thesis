"""P4 baseline adapters (spec docs/P4_BASELINES_SPEC.md): CPU tests on tiny models.

No weights are downloaded. DDA and CO-SPY are checked against the formulas and key names
their upstream code defines (read 2026-10-31 / 2026-11-01); FIRE against its forward and
the seed rule; AEROBLADE against "minus the minimum second-layer distance".
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from dbp.baselines import aeroblade, common, cospy, dda, fire  # noqa: E402

TINY = {"embed_dim": 32, "depth": 2, "num_heads": 2}


def _script(name: str):
    spec = importlib.util.spec_from_file_location(f"_p4_{name}", ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[f"_p4_{name}"] = module
    spec.loader.exec_module(module)
    return module


def _tiny_kl_vae():
    from diffusers import AutoencoderKL

    torch.manual_seed(0)
    return AutoencoderKL(in_channels=3, out_channels=3,
                         down_block_types=("DownEncoderBlock2D",) * 4,
                         up_block_types=("UpDecoderBlock2D",) * 4,
                         block_out_channels=(4, 4, 8, 8), layers_per_block=1,
                         latent_channels=4, norm_num_groups=2, sample_size=64).eval()


@pytest.fixture
def dry(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DBP_DRYRUN", "1")


# ------------------------------------------------------------------ common ---


def test_fire_seed_is_the_first_8_sha256_bytes_masked_to_63_bits() -> None:
    import hashlib

    digest = hashlib.sha256(b"abc").digest()
    assert common.fire_seed("abc") == int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)
    assert common.fire_seed("abc") == common.fire_seed("abc") != common.fire_seed("abd")
    assert 0 <= common.fire_seed("x") < 2 ** 63


def test_the_768_cap_crops_the_centre_and_never_pads() -> None:
    img = np.arange(1000 * 600 * 3, dtype=np.uint32).reshape(1000, 600, 3).astype(np.uint8)
    out = common.center_crop_max(img, 768)
    assert out.shape == (768, 600, 3) and np.array_equal(out, img[116:884])
    assert common.center_crop_max(img[:300, :300], 768).shape == (300, 300, 3)


def test_a_checkpoint_that_is_not_plain_weights_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "bad.pth"
    torch.save({"args": argparse.Namespace(lr=1)}, path)
    with pytest.raises(common.BaselineError, match="weights_only=True"):
        common.load_pth(path, common.file_sha256(path), "bad")


def test_a_checkpoint_with_another_hash_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "w.pth"
    torch.save({"a": torch.zeros(1)}, path)
    with pytest.raises(common.BaselineError, match="sha256"):
        common.load_pth(path, "0" * 64, "w")


# --------------------------------------------------------------------- DDA ---


def test_lora_is_ddas_formula_with_scale_alpha_over_rank() -> None:
    torch.manual_seed(0)
    base = nn.Linear(6, 5)
    layer = dda.LoRALinear(base, rank=8, alpha=1.0)
    nn.init.normal_(layer.lora.lora_B)
    x = torch.randn(3, 6)
    want = base(x) + (x @ layer.lora.lora_A.T) @ layer.lora.lora_B.T * (1.0 / 8)
    assert torch.allclose(layer(x), want, atol=1e-6)


def test_every_blocks_four_linears_are_wrapped_with_ddas_key_names(dry: None) -> None:
    model = dda.DDADetector(TINY)
    keys = set(model.state_dict())
    for block in range(TINY["depth"]):
        for target in dda.LORA_TARGETS:
            stem = f"base_model.model.blocks.{block}.{target}"
            assert {f"{stem}.original_layer.weight", f"{stem}.original_layer.bias",
                    f"{stem}.lora.lora_A", f"{stem}.lora.lora_B"} <= keys
    assert {"base_model.model.mask_token", "base_model.model.pos_embed",
            "base_model.fc.weight", "base_model.fc.bias"} <= keys
    assert model.base_model.model.pos_embed.shape[1] == 1 + 37 * 37


def _dinov2_reference(pos_embed: torch.Tensor, w: int, h: int, patch: int = 14) -> torch.Tensor:
    """DINOv2's interpolate_pos_encoding (vision_transformer.py @ 7764ea0f), with x's
    ``(B, nc, w, h)`` naming: ``w`` is dimension 2 (rows)."""
    n = pos_embed.shape[1] - 1
    dim = pos_embed.shape[-1]
    w0, h0 = w // patch, h // patch
    m = int(math.sqrt(n))
    sx, sy = float(w0 + 0.1) / m, float(h0 + 0.1) / m
    patch_pos = F.interpolate(pos_embed.float()[:, 1:].reshape(1, m, m, dim).permute(0, 3, 1, 2),
                              mode="bicubic", antialias=False, scale_factor=(sx, sy))
    patch_pos = patch_pos.permute(0, 2, 3, 1).view(1, -1, dim)
    return torch.cat((pos_embed.float()[:, 0].unsqueeze(0), patch_pos), dim=1)


@pytest.mark.parametrize("size", [(336, 336), (308, 308), (336, 308), (294, 336)])
def test_the_position_embedding_is_interpolated_as_dinov2_does(size: tuple[int, int]) -> None:
    torch.manual_seed(0)
    pos = torch.randn(1, 1 + 37 * 37, 8)
    ours = dda.interpolate_pos_encoding(pos, *size)
    assert torch.equal(ours, _dinov2_reference(pos, *size))
    assert ours.shape[1] == 1 + (size[0] // 14) * (size[1] // 14)


def test_dinov2s_interpolation_is_not_timms_default_resampling() -> None:
    from timm.layers import resample_abs_pos_embed

    torch.manual_seed(0)
    pos = torch.randn(1, 1 + 37 * 37, 8)
    timm_default = resample_abs_pos_embed(pos, new_size=(24, 24), num_prefix_tokens=1)
    assert not torch.allclose(dda.interpolate_pos_encoding(pos, 336, 336), timm_default,
                              atol=1e-4)


def test_the_checkpoint_loads_strictly_and_reproduces_the_logits(tmp_path: Path,
                                                                 dry: None) -> None:
    torch.manual_seed(0)
    model = dda.DDADetector(TINY).eval()
    path = tmp_path / "DDA_ckpt.pth"
    torch.save({"model": model.state_dict(), "epoch": 1}, path)
    loaded = dda.load_dda(path, common.file_sha256(path), "cpu", TINY)
    x = torch.randn(2, 3, 336, 336)
    with torch.no_grad():
        assert torch.equal(loaded(x), model(x))


@pytest.mark.parametrize("change", ["extra", "missing"])
def test_any_missing_or_unexpected_key_stops_the_load(tmp_path: Path, dry: None,
                                                      change: str) -> None:
    state = dda.DDADetector(TINY).state_dict()
    if change == "extra":
        state["base_model.model.register_tokens"] = torch.zeros(1)
    else:
        state.pop("base_model.model.mask_token")
    path = tmp_path / "DDA_ckpt.pth"
    torch.save({"model": state}, path)
    with pytest.raises(common.BaselineError, match="missing|unexpected"):
        dda.load_dda(path, common.file_sha256(path), "cpu", TINY)


def test_a_tiny_architecture_is_refused_outside_the_dry_run(
        monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DBP_DRYRUN", raising=False)
    with pytest.raises(common.BaselineError, match="dry-run"):
        dda.DDADetector(TINY)


def test_ddas_crop_zero_pads_a_small_image_and_normalises() -> None:
    img = np.full((300, 500, 3), 128, dtype=np.uint8)
    x, padded = dda.dda_input(img, 336)
    assert padded and x.shape == (3, 336, 336)
    zero = (torch.zeros(3) - torch.tensor(dda.CLIP_MEAN)) / torch.tensor(dda.CLIP_STD)
    assert torch.allclose(x[:, 0, 200], zero) and torch.allclose(x[:, 17, 200], zero)
    mid = (torch.full((3,), 128 / 255) - torch.tensor(dda.CLIP_MEAN)) / torch.tensor(dda.CLIP_STD)
    assert torch.allclose(x[:, 168, 168], mid, atol=1e-6)
    assert dda.dda_input(np.zeros((400, 400, 3), np.uint8), 308)[1] is False


# ------------------------------------------------------------------ CO-SPY ---


class _StubClip(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(3, 4)

    def encode_image(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x.mean(dim=(2, 3)))


def _cospy_files(tmp_path: Path, artifact_extra: dict | None = None) -> tuple[dict, dict]:
    torch.manual_seed(0)
    states = {"semantic": {"fc.weight": torch.randn(1, 4), "fc.bias": torch.randn(1)},
              "fusion": {"fc.weight": torch.randn(1, 2), "fc.bias": torch.randn(1)},
              "artifact": {**cospy.ArtifactDetector(None).state_dict(),
                           **(artifact_extra or {})}}
    paths, shas = {}, {}
    for name, state in states.items():
        paths[name] = tmp_path / f"{name}_weights.pth"
        torch.save(state, paths[name])
        shas[name] = common.file_sha256(paths[name])
    return paths, shas


def test_cospy_loads_the_three_files_and_fuses_two_logits(tmp_path: Path) -> None:
    paths, shas = _cospy_files(tmp_path)
    model = cospy.load_cospy(paths, shas, _StubClip(), 4, _tiny_kl_vae(), "cpu")
    assert torch.equal(model.fc.weight, torch.load(paths["fusion"])["fc.weight"])
    x = torch.rand(2, 3, 384, 384)
    with torch.no_grad():
        first, second = model(x), model(x)
    assert first.shape == (2, 1) and torch.equal(first, second)  # latent_dist.mean


def test_a_vae_key_in_the_artifact_file_stops_the_load(tmp_path: Path) -> None:
    paths, shas = _cospy_files(tmp_path, {"artifact_encoder.vae.encoder.conv_in.weight":
                                          torch.zeros(1)})
    with pytest.raises(common.BaselineError, match="VAE"):
        cospy.load_cospy(paths, shas, _StubClip(), 4, _tiny_kl_vae(), "cpu")


def test_the_artifact_branch_sees_the_224_resize_without_antialias() -> None:
    model = cospy.CoSpyFusion(_StubClip(), 4, _tiny_kl_vae())
    seen: dict[str, torch.Tensor] = {}

    def record(name: str):
        def forward(x: torch.Tensor) -> torch.Tensor:
            seen[name] = x
            return torch.zeros(len(x), 1)
        return forward

    model.art.forward, model.sem.forward = record("art"), record("sem")
    x = torch.rand(1, 3, 384, 384)
    model(x)
    from torchvision.transforms import InterpolationMode
    from torchvision.transforms import functional as TF

    want = TF.resize(x, [224], interpolation=InterpolationMode.BILINEAR, antialias=False)
    assert torch.equal(seen["art"], want) and torch.allclose(seen["sem"], (x - 0.5) / 0.5)


def test_cospy_resizes_the_short_side_to_384_and_flags_an_upscale() -> None:
    x, upscaled = cospy.cospy_input(np.zeros((300, 500, 3), np.uint8))
    assert upscaled and x.shape == (3, 384, 384) and float(x.max()) <= 1.0
    assert cospy.cospy_input(np.zeros((400, 600, 3), np.uint8))[1] is False


# -------------------------------------------------------------------- FIRE ---


def test_fire_resizes_the_short_side_to_256_then_crops_the_long_side_to_8() -> None:
    x, cropped = fire.fire_input(np.zeros((300, 500, 3), np.uint8))
    assert x.shape == (3, 256, 424) and cropped
    x, cropped = fire.fire_input(np.zeros((512, 512, 3), np.uint8))
    assert x.shape == (3, 256, 256) and not cropped


def _fire_model() -> fire.FIREDetector:
    torch.manual_seed(0)
    return fire.FIREDetector(_tiny_kl_vae().half()).eval()


def test_the_seed_rule_makes_every_fire_score_reproducible() -> None:
    model = _fire_model()
    img = (np.random.default_rng(0).random((64, 96, 3)) * 255).astype(np.uint8)
    first = fire.score_fire(model, img, "id_a", "cpu")[0]
    assert fire.score_fire(model, img, "id_a", "cpu")[0] == first
    assert fire.score_fire(model, img, "id_b", "cpu")[0] != first


def test_fires_filtered_image_is_divided_by_its_range_without_subtracting_the_min() -> None:
    z = torch.tensor([[2.0, 4.0], [6.0, 10.0]])
    assert torch.equal(fire.FFTFilter._normalised(z), z / 8.0)


@pytest.mark.parametrize("vae", ["none", "full", "partial"])
def test_fire_takes_the_vae_from_the_checkpoint_only_when_it_holds_all_of_it(
        tmp_path: Path, vae: str) -> None:
    model = _fire_model()
    state = model.state_dict()
    if vae == "none":
        state = {k: v for k, v in state.items() if not k.startswith("vae.")}
    elif vae == "partial":
        state.pop(next(k for k in state if k.startswith("vae.")))
    path = tmp_path / "fire.pth"
    torch.save(state, path)
    if vae == "partial":
        with pytest.raises(common.BaselineError, match="part of the VAE"):
            fire.load_fire(path, common.file_sha256(path), _tiny_kl_vae().half(), "cpu")
        return
    _, source = fire.load_fire(path, common.file_sha256(path), _tiny_kl_vae().half(), "cpu")
    assert source == ("checkpoint" if vae == "full" else "pinned sd15")


def test_a_missing_resnet_key_stops_the_fire_load(tmp_path: Path) -> None:
    state = {k: v for k, v in _fire_model().state_dict().items()
             if not k.startswith(("vae.", "resnet.fc."))}
    path = tmp_path / "fire.pth"
    torch.save(state, path)
    with pytest.raises(common.BaselineError, match="missing"):
        fire.load_fire(path, common.file_sha256(path), _tiny_kl_vae().half(), "cpu")


# --------------------------------------------------------------- AEROBLADE ---


class _Spec:
    def __init__(self, key: str) -> None:
        self.key, self.crop_multiple = key, 8


class _Member:
    def __init__(self, key: str, shift: int) -> None:
        self.spec, self.shift = _Spec(key), shift

    def reconstruct(self, img: np.ndarray) -> np.ndarray:
        from dbp.decoders.reconstruct import prepare_input

        return np.clip(prepare_input(img, 8).astype(int) + self.shift, 0, 255).astype(np.uint8)


class _Net:
    """Per-layer distances: layer 2 is the mean absolute difference; layer 1 is huge."""

    def __call__(self, a: torch.Tensor, b: torch.Tensor, retPerLayer: bool,  # noqa: N803
                 normalize: bool) -> tuple:
        assert retPerLayer and not normalize and float(a.min()) >= -1.0
        d = (a - b).abs().mean().reshape(1, 1, 1, 1)
        return d, [d + 100, d, d + 50]


def test_aeroblade_is_minus_the_smallest_second_layer_distance() -> None:
    img = np.full((1000, 500, 3), 100, dtype=np.uint8)
    members = [_Member("sd15", 10), _Member("sd21", 4), _Member("movq_k21", 20)]
    score, distances, cut = aeroblade.score_aeroblade(members, _Net(), img, "cpu")
    assert cut and set(distances) == {"sd15", "sd21", "movq_k21"}
    assert score == -min(distances.values()) == -distances["sd21"]
    assert distances["sd21"] == pytest.approx(4 / 127.5, rel=1e-5)
    assert aeroblade.score_aeroblade(members, _Net(), img[:500], "cpu")[2] is False


def test_the_kandinsky_21_movq_is_validated_like_a_bank_member() -> None:
    from dbp.decoders.registry import load_bank_config

    extra = load_bank_config(ROOT / "configs" / "p4_aeroblade.yaml")
    bank = load_bank_config(ROOT / "configs" / "decoder_bank.yaml")
    spec = extra.by_key("movq_k21")
    assert (spec.kind, spec.crop_multiple, spec.latent_channels) == ("vq", 8, 4)
    assert spec.revision == "04b366659663bf60deacb96fd3e38f1cb39a3be4"
    assert not {s.key for s in extra.members} & {s.key for s in bank.members}


def test_a_kandinsky_21_shaped_movq_round_trips_through_the_vq_path() -> None:
    """The 2.1 MoVQ config (read 2026-11-01): VQModel, spatial norm, vq_embed_dim 4."""
    from diffusers import VQModel

    from dbp.decoders.loaders import _TENSOR_RECONSTRUCTORS
    from dbp.decoders.reconstruct import reconstruct_vq, to_tensor

    torch.manual_seed(0)
    model = VQModel(down_block_types=("DownEncoderBlock2D", "AttnDownEncoderBlock2D"),
                    up_block_types=("AttnUpDecoderBlock2D", "UpDecoderBlock2D"),
                    block_out_channels=(32, 32), layers_per_block=1, latent_channels=4,
                    num_vq_embeddings=16, norm_num_groups=32, vq_embed_dim=4,
                    norm_type="spatial", sample_size=32).eval()
    assert _TENSOR_RECONSTRUCTORS["vq"] is reconstruct_vq
    x = to_tensor(np.full((32, 48, 3), 90, np.uint8), "cpu")
    first, second = reconstruct_vq(model, x), reconstruct_vq(model, x)
    assert first.shape == x.shape and torch.equal(first, second)


# ----------------------------------------------------------------- tooling ---


def test_the_isolated_bundle_holds_only_the_check_and_its_code() -> None:
    zipper = _script("build_code_zip")
    paths = [f"thesis_dbp/{p}" for p in (
        "pyproject.toml", "requirements/base.txt", "env/kaggle_bootstrap.sh",
        "env/kaggle_run_p4_dda_equivalence.py", "env/p4_dda_equivalence_expect.json",
        "scripts/p4_dda_equivalence.py", "scripts/p2_preflight.py", "src/dbp/__init__.py",
        "src/dbp/baselines/dda.py", "src/dbp/data/__init__.py", "src/dbp/data/manifest.py",
        "src/dbp/utils/__init__.py", "src/dbp/utils/logging.py",
        "reports/kaggle_20261011/p3_panel_b_images.csv", "scripts/p4_score.py")]
    chosen = zipper.select(paths, "p4-dda-equiv")
    assert not any("reports/" in p or p.endswith("p4_score.py") for p in chosen)
    assert "thesis_dbp/src/dbp/baselines/dda.py" in chosen
    with pytest.raises(zipper.BuildError, match="not in the tree"):
        zipper.select(paths[:-3], "p4-dda-equiv")


def test_the_preflight_refuses_a_weight_without_a_recorded_hash(tmp_path: Path) -> None:
    pre = _script("p4_preflight")
    report = tmp_path / "Implementation/thesis_dbp/reports/p4_weights.json"
    report.parent.mkdir(parents=True)
    report.write_text('{"ok": true}', encoding="utf-8")
    (tmp_path / "p4_weights").mkdir()
    record, problems = pre.check_weights(tmp_path / "p4_weights", {"weights": {
        "fire": {"file": "fire/x.pth", "sha256": None}}})
    assert record["weights_report"]["ok"] is True
    assert problems and "no recorded sha256" in problems[0]


def test_condition_rows_score_panel_b_test_rows_only() -> None:
    pre = _script("p4_preflight")
    assert pre.row_lists({"condition": "clean"}) == ["panel_b", "luma", "threshold_reals"]
    assert pre.row_lists({"condition": "jpeg50"}) == ["panel_b_test"]


def test_a_row_is_scored_once_and_a_smoke_never_counts(tmp_path: Path) -> None:
    score = _script("p4_score")
    summary = tmp_path / "row.json"
    summary.write_text("{}", encoding="utf-8")
    assert score.refuses_rescore(summary, 0) and not score.refuses_rescore(summary, 2)
    out = tmp_path / "s.csv"
    score.write_scores(out, [{"image_id": "a"}], [{"score": -0.5, "flag": True,
                                                    "lpips2_sd15": 0.5}], ["lpips2_sd15"])
    assert out.read_text(encoding="utf-8") == "image_id,score,flag,lpips2_sd15\n" \
                                              "a,-0.500000,1,0.500000\n"


def test_the_expect_files_carry_the_spec_pins() -> None:
    import json

    spec = (ROOT / "docs" / "P4_BASELINES_SPEC.md").read_text(encoding="utf-8")
    weights = json.loads((ROOT / "env" / "p4_weights_expect.json").read_text("utf-8"))
    for entry in weights["hf"].values():
        assert entry["revision"] in spec and entry["sha256"] in spec
    scoring = json.loads((ROOT / "env" / "p4_baselines_expect.json").read_text("utf-8"))
    assert set(scoring["rows"]) == {"dda_native", "dda_crop308", "dda_jpeg50",
                                    "dda_resize050", "cospy_native", "fire_native",
                                    "aeroblade"}
    jobs = [j.partition(":")[2].split(",") for j in scoring["jobs"]]
    assert sorted(m for j in jobs for m in j) == sorted(scoring["rows"])
    assert scoring["weights"]["dda"]["sha256"] == weights["hf"]["dda"]["sha256"]
