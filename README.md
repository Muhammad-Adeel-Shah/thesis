# Decoder-Bank Probing (DBP)

Cross-generator detection of AI-generated images and videos, by probing images with a
**lineage-diverse bank of generator decoders** and reading the per-decoder residual response.

MS Data Science thesis, FAST-NUCES (24i-8017) —
*Decoder-Induced Spatio-Temporal Forensic Responses for Cross-Generator Detection of
AI-Generated Images and Videos*.

## Why

The FIRE baseline (CVPR 2025) reaches AUC ~100 in-domain on DiffusionForensics
LSUN-Bedroom but collapses to AUC 40–60 on modern generators, and at threshold 0.5 it
labels all 389 real images fake. FIRE reconstructs with the SD-1.5 VAE, yet its classifier
was trained on ADM fakes, which are pixel-space and never pass through a VAE — so it never
learned a decoder fingerprint at all.

DBP builds training pairs by reconstructing **real** images through a bank of decoders
spanning genuinely different lineages, then uses that same bank as a probe at test time.

## The bank

13 members, **12 distinct**, 8 lineages.

| lineage | members | latent ch | downsample |
|---|---|---|---|
| SD1 (KL-f8) | `sd15`, `sd21`, `sd_ft_ema`*, `sd_ft_mse`*† | 4 | 8 |
| SDXL | `sdxl`, `sdxl_fp16fix`* | 4 | 8 |
| 16-channel | `sd35`, `flux` | 16 | 8 |
| 16-channel (Qwen, causal-3D) | `qwen_image` | 16 | 8 |
| 32-channel (FLUX.2) | `flux2_klein` | 32 | 8 |
| VQ de-tokenizer | `movq_k22` (Kandinsky 2.2 MoVQ, 16 384 codes) | 4 | 8 |
| deep-compression | `dcae` (SANA) | 32 | 32 |
| GAN upsampling | `realesrgan_x4` | – | 4 |

`*` = optional tier. `†` = `alias_of: sd21` — P0 measured byte-identical decoder weights,
so it is excluded by `BankConfig.distinct_members()` and never reconstructed separately.
Full registry with pinned commit SHAs: `configs/decoder_bank.yaml`.

## Ground rules

- **Never resize.** Centre-crop to a multiple of the member's factor. Resizing destroys the
  high-frequency generator traces the method depends on, and creates a train/test bias.
- **Never tile.** `enable_tiling()` introduces seam artefacts that are themselves forensic
  signals. `enable_slicing()` is fine.
- **Deterministic encode.** `latent_dist.mode()`, never `.sample()`.
- **fp32 unless proven.** fp16 needs PSNR >= 45 dB against the member's own fp32
  reconstruction for the smoke test, and >= 55 dB **at every crop measured** before it is
  used to build data.
- **Compression history is never confounded with the label.** Every evaluation set carries
  a `label x (format, JPEG, quality, chroma subsampling)` table, and a set where the two
  track each other gets a JPEG-matched view (see `CLAUDE.md` 6b). The same rule is applied
  to the *training* side: the generations get a matched variant drawn from T-real's own
  distribution, and P2 reports raw-vs-matched as an ablation.
- **Several crops of one photograph are not several images.** Rows carry
  `source_image_id`, and every split groups by it.
- `src/dbp/decoders/reconstruct.py` is the **single source of truth** for the protocol.
  Do not reimplement it anywhere else.

## Setup

- Local (CPU, development): [`env/setup_local.md`](env/setup_local.md)
- Kaggle T4 (all reported numbers): [`env/kaggle_bootstrap.sh`](env/kaggle_bootstrap.sh)

## P0 checks

```bash
python scripts/p0_env_check.py           # -> reports/env_report.json
python scripts/p0_bank_smoke_test.py     # -> reports/p0_bank_smoke.csv + outputs/p0/*.png
python scripts/p0_backbone_smoke_test.py # -> reports/p0_backbone_smoke.json
python scripts/p0_verify_integrity.py    # -> reports/p0_integrity.json
pytest -q tests && ruff check .
```

## P1 data & evaluation

```bash
python scripts/p1_throughput_bench.py    # -> reports/p1_throughput.csv  (fp16 + budget)
python scripts/p1_jpeg_equivalence.py    # -> reports/p1_jpeg_equivalence.json  (I-010)
python scripts/p1_build_reals.py         # -> outputs/p1/reals + manifest
python scripts/p1_build_pairs.py         # -> outputs/p1/pairs  (resumable, sharded)
python scripts/p1_build_generations.py   # -> outputs/p1/generated
python scripts/p1_match_generations.py   # -> outputs/p1/gen_matched  (I-023)
python scripts/p1_build_eval.py          # -> eval manifests, matched views, leakage
```

Evaluation sets: **E7 DDA-COCO** (primary head-to-head), **E8 EvalGEN** (5 generators,
3 of them not diffusion), **E3 MLLM**, **E4 WildRF** (test split only). E6 MNW is
blocked on its repository id; E5 Chameleon on access; E2 Synthbuster is deferred to P2.

Runbooks: [`env/kaggle_run_p1_preflight.md`](env/kaggle_run_p1_preflight.md) →
[`kaggle_run_p1_build.md`](env/kaggle_run_p1_build.md) →
[`kaggle_run_p1_eval.md`](env/kaggle_run_p1_eval.md).
Data cards: [`docs/data_cards/`](docs/data_cards/).

## Layout

```
configs/    decoder bank registry (pinned revisions)
src/dbp/    utils | decoders | backbones
scripts/    P0 check entry points
tests/      unit + smoke tests (markers: network, gpu, gated)
reports/    small JSON/CSV, tracked in git
outputs/    images and logs, git-ignored
docs/       IMPLEMENTATION_LOG.md -- updated at the end of every prompt
```

## Status

P0 complete. See [`docs/IMPLEMENTATION_LOG.md`](docs/IMPLEMENTATION_LOG.md).
