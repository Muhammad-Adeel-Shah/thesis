# Implementation Log — Decoder-Bank Probing (DBP)

Thesis: *Decoder-Induced Spatio-Temporal Forensic Responses for Cross-Generator Detection of AI-Generated Images and Videos* (24i-8017)

Rule: this file is append-only below "Prompt entries"; the snapshot tables above it are kept current.

## 1. Current snapshot

| Item | Value |
|---|---|
| Last prompt | **P3-C (2026-10-29)**: every P3 confirmatory hypothesis is decided (`reports/p3_results/tables/t1_verdicts.md`): **H1-a PASS**, **H1-b PASS** (3/3 folds), **H1-c PASS**, **H3 FAIL**, **H3-k PASS** (Q and S), **H2 FAIL**, **H4 PASS**, **H5 FAIL**. Thesis tables and figures committed (`9626eca`). TrueFake (secondary, descriptive) **scored and analysed** (2026-10-31): 4 models, 3,999 ids kept (1 FORLAB id excluded: a truncated source JPEG), `reports/p3_results/p3_truefake_analysis.json`. P4 baseline spec **frozen** 2026-11-01 (tag `p4-baselines-frozen`, `docs/P4_BASELINES_SPEC.md`); baseline code built and dry-run 2026-11-02; Kaggle steps 1 (weights) and 2 (isolated DDA check, PASS) done 2026-11-05; step 3 (GPU scoring) handed over. |
| Runtime / GPU / driver | `local-dev+kaggle-gpu`. **Local:** Windows 11 Pro 10.0.26200, no NVIDIA GPU, 17.0 GB RAM. **Kaggle:** 2x Tesla T4, sm_75, driver 580.159.04, 14 912 MB each, 33.7 GB RAM (device 0 used) |
| Python / torch / CUDA / cuDNN | **Local:** Python 3.13.1, torch 2.11.0+cpu, tv 0.26.0+cpu, CUDA None. **Kaggle:** Python 3.12.13, torch 2.10.0+cu128, tv 0.25.0+cu128, CUDA 12.8, cuDNN 91002 |
| diffusers / transformers / peft / spandrel / lpips / Pillow(libjpeg) | **Local:** 0.40.0 / 5.17.0 / 0.21.0 / 0.4.2 / 0.1.4 / 12.3.0 (**libjpeg 8.0**). **Kaggle:** 0.37.1 / 5.0.0 / 0.19.1 / 0.4.2 / 0.1.4 / 11.3.0 (**libjpeg 6.2**) — the libjpeg difference matters for P1, see I-010. **Kaggle, S3 recovery session 2026-10-04** (pre-flight import check): diffusers 0.40.0, transformers 5.16.1, peft 0.20.0, spandrel 0.4.2; Python 3.13 per the site-packages path in the probe log |
| Other pinned | accelerate 1.15.0, safetensors 0.8.0, huggingface_hub 1.32.0, scikit-learn 1.9.1, **scipy 1.18.1 (declared in `requirements/base.txt` at P2 for McNemar, Spearman and the bootstrap)**, numpy 2.5.2, pandas 3.0.6, ruff 0.16.8, pytest 9.1.1 |
| Git branch / last commit | **Repository re-rooted at `thesis_dbp/` by the user (2026-11-05):** branch `master`, `origin` = github.com/Muhammad-Adeel-Shah/thesis; the earlier history (183 commits, tags `p3-prereg-frozen`, `p4-baselines-frozen`) was removed with the old `.git`; see the 2026-11-05 entry. Code for the TrueFake run: `218da57` (zip `dbp-code-218da57.zip` in kaggle_raw); the TrueFake analysis ran at `8ea9e56`; see the latest prompt entry for later commits. The commit a given run used is stamped into its upload as `COMMIT.txt` |
| Protected files integrity | **0 files checked, 0 changed — CLEAN** in both runtimes, algorithm self-test byte-identical. Kaggle verdict at `reports/kaggle_20260920/p0_integrity.json`. Since P1 the report is rewritten only when its content actually changes, so a CLEAN run no longer dirties the working tree |

## 2. Protected assets (never modified)

- **None exist yet.** `REPO_ROOT` (`F:\thesis-2\Implementation`) was a fresh, empty directory at P0. A depth-limited search of `F:` and of `C:\Users\HP` found no `FIRE`, `Base_implementation` or `DiffusionForensics` anywhere on this machine.
- The FIRE baseline and the zero-shot evaluation results **live on Kaggle and stay there** (user decision, Phase A Q1a).
- Baseline manifest: `reports/p0_protected_manifest_before.json` — 0 entries, `algo_version: 1`.
- The manifest algorithm was validated against a synthetic tree (`Base_implementation/FIRE/train.py`, `prior_results/zero_shot.csv`, `prior_results/weights.pth`): it correctly classified hashed vs fast-mode files and excluded `.git/`, `thesis_dbp/`, `.gitignore` and `CLAUDE.md`. An empty manifest on its own would not have proved the scanner works.
- `p0_verify_integrity.py --self-test` regenerates the baseline with its own copy of the algorithm and asserts byte-identity, so the two sides of the comparison cannot drift together. It reports **True**.
- **Action required in a later prompt:** when `REPO_ROOT/prior_results/` is populated, re-baseline and record the new file count here.

## 3. Decoder bank status

**Measured on a Kaggle T4 (sm_75, CUDA 12.8, driver 580.159.04) on 2026-09-20 from commit `758e4c1f`.**
Source: `reports/kaggle_20260920/p0_bank_smoke.csv`. **20/20 rows OK.** No member OOMed at 1024;
the largest peak was 4 609.7 MB of 14 912 MB.

| key | lineage | ch | ↓ | revision | fp16 ok | PSNR@512 | LPIPS@512 | ms/img@512 | peak MB@512 | status |
|---|---|---|---|---|---|---|---|---|---|---|
| sd15 | kl_4ch_sd1 | 4 | 8 | `451f4fe16113` | **yes** (61.00 dB) | 26.797 | 0.08954 | 2029.0 * | 1158.4 | OK |
| sd21 | kl_4ch_sd1 | 4 | 8 | `bb2154823665` | **yes** (60.55 dB) | 27.737 | 0.07801 | 769.5 | 1159.0 | OK |
| sd_ft_ema | kl_4ch_sd1 | 4 | 8 | `f04b2c4b9831` | **yes** (60.76 dB) | 27.341 | 0.08146 | 744.2 | 1159.0 | OK |
| sd_ft_mse | kl_4ch_sd1 | 4 | 8 | `31f26fdeee13` | **yes** (60.55 dB) | 27.737 | 0.07801 | 786.2 | 1159.0 | OK — **same decoder as sd21** |
| sdxl | kl_4ch_sdxl | 4 | 8 | `6f5909a7e596` | **NO (5.21 dB)** | 28.294 | 0.07468 | 770.9 | 1159.0 | OK, fp32 only |
| sdxl_fp16fix | kl_4ch_sdxl | 4 | 8 | `207b116dae70` | yes (48.58 dB) | 28.212 | 0.07594 | 753.0 | 1159.0 | OK |
| sd35 | kl_16ch | 16 | 8 | `ceddf0a7fdf2` | **yes** (61.31 dB) | 31.939 | 0.03886 | 782.1 | 1160.5 | OK |
| flux | kl_16ch | 16 | 8 | `741f7c3ce8b3` | **yes** (55.42 dB) | 33.766 | 0.02865 | 748.8 | 1160.5 | OK |
| dcae | dc_ae_f32 | 32 | 32 | `ca69e17e9760` | **yes** (58.64 dB) | 25.246 | 0.11002 | 3813.8 | 2101.6 | OK |
| realesrgan_x4 | gan_sr | – | 4 | sha256 `4fa0d38905f7…` | yes (59.86 dB) | 23.821 | 0.23985 | 169.1 | 258.5 | OK |

`*` sd15 ran first and absorbed CUDA context initialisation, so its 2 029 ms is inflated;
comparable members land at 744–786 ms. Recorded as issue I-009.

At crop 1024 every member also passed: PSNR 35.1–40.6 dB, peak VRAM 643–4 610 MB,
fp16 verdicts unchanged except that they improve slightly (sdxl still fails at 5.29 dB).

**Determinism: all 20 rows re-ran fp32 twice with `max_abs_delta_rerun = 0`** — bit-identical,
stronger than the ≤ 1 tolerance the protocol allows.

### Key findings

1. **The fp16 policy (D-003) is vindicated.** `sdxl` reconstructs at **5.21 dB** against its own
   fp32 output — catastrophic overflow, exactly as predicted in §6.4.7 — while every other member
   lands at 55–63 dB. Had fp16 been assumed safe across the board, every SDXL residual in the
   thesis would have been noise. `sdxl_fp16fix` passes at 48.58 dB, but that is only 3.6 dB above
   the 45 dB threshold, so it stays on fp32 for data building.
2. **`sd21` and `sd_ft_mse` are ONE lineage, not two.** Their decoder state dicts hash identically
   (`71cd356c763df913…`), which is why their PSNR, LPIPS and mean|r| agree to four decimals on both
   images. SD 2.1 ships the ft-MSE VAE. **They must never be split across a leave-one-lineage-out
   train/test boundary**, or the split leaks. The bank holds **9 distinct decoders, not 10**.
3. **Both 16-channel lineages load and reconstruct correctly** — the two cases SEF is weakest on
   are now known to work end to end. That is the finding.

   **Smoke-test observation, not a finding** (n = 1 image per crop size, so it carries no
   statistical weight): the 16-channel members separate from the 4-channel ones in **LPIPS at both
   crops** (`flux` 0.0286 / `sd35` 0.0389 vs 0.0747–0.0895 at 512; 0.0114 / 0.0190 vs
   0.0724–0.0862 at 1024 — strictly better in both), but **PSNR is not consistent**: strictly
   better at 512 (astronaut), and *reversed* at 1024 (retina), where `sd21` 40.63 > `sd35` 38.36 >
   `flux` 37.76. Any claim about 16-channel reconstruction quality needs a proper corpus in P1;
   two images cannot support one.
4. **`realesrgan_x4` has by far the highest LPIPS** (0.240 at 512, 0.268 at 1024 — roughly 3x any
   VAE), consistent with the GAN-upsampling arm contributing a different artefact family rather
   than duplicating the VAEs. Same caveat as above: one image per crop.
5. **T4 headroom is ample.** Worst case 4 610 MB of 14 912 MB at 1024, so P1 can raise batch size or
   crop without re-planning memory.

### P1-B3 extension — three families added (2026-09-20)

Audited by command in the P1 session (`HfApi.model_info` + the pinned `config.json`);
**no weights downloaded locally, no reconstruction measured yet.** The T4 numbers are
`PENDING-GPU` until `env/kaggle_run_p1_preflight.md` Cell 7 runs.

| key | lineage | class | ch | ↓ | revision | licence | gated | vae bytes | status |
|---|---|---|---|---|---|---|---|---|---|
| flux2_klein | kl_32ch_flux2 | `AutoencoderKLFlux2` | 32 | 8 | `e7b7dc27f91d` | apache-2.0 | no | 168 120 878 | PENDING-GPU |
| qwen_image | kl_16ch_qwen | `AutoencoderKLQwenImage` | 16 | 8 | `75e0b4be04f6` | apache-2.0 | no | 253 806 966 | PENDING-GPU |
| movq_k22 | vq_movq_4ch | `VQModel` | 4 | 8 | `9ae140d347fe` | apache-2.0 | no | 271 380 364 | PENDING-GPU |

**Bank is now 13 members / 12 distinct / 8 lineages** (`sd_ft_mse` is an `alias_of`
`sd21` and is excluded by `BankConfig.distinct_members()`). All 15 pins verified online
in this session: 0 failed, 0 blocked.

Why each earns a slot:

- **flux2_klein** — a fourth latent width. The bank now spans 4 / 16 / 32 channels at f8
  plus f32 deep-compression, instead of clustering at 4 and 16.
- **qwen_image** — a second, independently trained 16-channel lineage, and the only
  member whose encoder is causal-3D. The protocol feeds it a single frame with `T = 1`
  and squeezes the temporal axis back out, which is directly relevant to the S4 stream.
- **movq_k22** — the vector-quantised de-tokenizer family the P0 bank had **no**
  representative of. `decode` re-quantises to a 16 384-entry codebook by default, so its
  residual carries a quantisation lattice rather than a Gaussian bottleneck.

API facts verified against **both** diffusers 0.40.0 (local) and the v0.37.1 tag
(Kaggle), so nothing here depends on a version only one runtime has:

- all three classes are top-level exports in v0.37.1;
- `AutoencoderKLFlux2._encode` unpacks 4 dims and returns `AutoencoderKLOutput.latent_dist`;
  its `patch_size: [2, 2]` is applied by the FLUX.2 *pipeline*, outside encode/decode, so
  the probe path is plain f8 and the crop multiple is 8;
- `AutoencoderKLQwenImage._encode` unpacks **5** dims — `(B, C, T, H, W)`;
- `VQModel.encode` returns `.latents` (no distribution) and `VQModel` implements **no**
  tiling at all: `enable_tiling()` raises and `use_tiling` does not exist until
  `disable_tiling()` creates it. The loader guard handles that case explicitly.

### Backbone status

Source: `reports/kaggle_20260920/p0_backbone_smoke.json`. Both **OK**.

| item | dinov2-base (default) | dinov3-vitb16 |
|---|---|---|
| revision | `f9e44c814b77…` | `5931719e67bb…` |
| access | open | **GRANTED** |
| params | 86.6 M | 85.7 M |
| hidden / patch | 768 / 14 | 768 / 16 |
| tokens @336 | 577 = 576 + 1 CLS + 0 reg | 446 = 441 + 1 CLS + **4 reg** |
| attention projections | `['key', 'query', 'value']` | **`['k_proj', 'q_proj', 'v_proj']`** |
| LoRA r=8 trainable | 442 368 / 87 022 848 = **0.5083 %** | 442 368 / 86 102 784 = **0.5138 %** |
| forward (fp16 autocast) | 385.6 ms * | 56.9 ms |
| peak VRAM | 540.0 MB | 530.1 MB |

`*` DINOv2 ran first and absorbed CUDA initialisation; a 6.8x gap between two ViT-B models differing
only 1.3x in token count is not credible. Both timings are provisional pending I-009.

**D-010 earned its place.** DINOv3 names its projections `k_proj/q_proj/v_proj` where DINOv2 uses
`key/query/value`. A hard-coded `target_modules` list would have attached LoRA to **nothing** on
DINOv3 and silently trained a frozen backbone. Identical trainable counts across both backbones are
expected: `8 x (768 + 768) x 3 projections x 12 layers = 442 368`.

## 4. Decisions register

| ID | Decision | Rationale | Reference | Prompt |
|---|---|---|---|---|
| D-001 | No resizing; crop to multiple | Resizing erases generator traces / causes bias | B-Free (CVPR'25); Grommelt et al. (ECCVW'24) | P0 |
| D-002 | `latent_dist.mode()`, no tiling, uint8-quantised PNG recon | Determinism; no artificial seams; match generator output format | diffusers docs; Aligned (ICLR'25) | P0 |
| D-003 | fp32 VAEs unless fp16 parity ≥ 45 dB | Avoid fp16 overflow corrupting residuals | — | P0 |
| D-004 | DINO ViT-B + LoRA r=8 on T4 | Same family as B-Free/DDA/SEF; fits 16 GB | [R10–R14] | P0 |
| D-005 | `sd21` loaded from the `sd2-community` mirror | The official `stabilityai/stable-diffusion-2-1` was deleted; verified 401 RepositoryNotFound at P0. Needed for DDA/SEF baseline parity | [R11][R12] | P0 |
| D-006 | Real-ESRGAN via `spandrel`, not `basicsr`/`realesrgan` | Restricted deserialization + arch auto-detection; the legacy packages have broken torchvision imports | [R6] | P0 |
| D-007 | Every Hub member pinned to a full 40-char SHA, and a network test asserts the pin **resolves** | A format-valid but fabricated SHA passes offline checks while pointing at nothing. This actually happened during P0 and was caught | — | P0 |
| D-008 | Local venv is CPU-only and development-only; every reported number comes from Kaggle T4 | fp16 numerics and VRAM limits cannot be reproduced on CPU | [R19][R21] | P0 |
| D-009 | CPU timings live in `cpu_ms_per_img`; `ms_per_img` stays empty on CPU | Makes it structurally impossible to mistake a CPU number for a T4 number | — | P0 |
| D-010 | `target_modules` for LoRA are **discovered** from the loaded model, never hard-coded | DINOv2 uses separate `query`/`key`/`value`; DINOv3 may fuse them. A hard-coded list would silently attach to nothing | [R14] | P0 |
| D-011 | `ruff` is run from inside `thesis_dbp/`, and its cache is pinned to `.cache/ruff` | Run from `REPO_ROOT` it creates `REPO_ROOT/.ruff_cache/`, outside the write scope | — | P0 |
| D-012 | **Never write an identifier or measured value (commit SHA, file hash, URL, version, size, metric) that was not produced by a command run in the current session. In the report, give the command or file that each such value came from.** | Recalling, completing or inferring a value is fabrication even when the result is right. Violated in P0: 12-character SHA prefixes were completed from memory, producing five pins to commits that do not exist. A value that cannot be produced is written as a placeholder marked `NOT-VERIFIED`, never as a plausible-looking value | User instruction, P0 review; see D-007 and the P0 correction record | P0 |
| D-013 | Ask before adding anything outside the approved plan — file, module, dependency, config key or extra output — even inside `thesis_dbp/` and even when clearly beneficial | P0 added five such items (deviations 3–7); they were accepted retroactively and that is not a precedent. Scope creep inside the write scope is still scope creep | User instruction, P0 review | P0 |
| D-014 | `kaggle_run_p0.md` re-verifies all 12 pins online (9 Hub members + `realesrgan_x4` URL/SHA-256 + 2 backbones) and aborts before the 4.9 GB download if any fails | Catching a dead pin three hours into a T4 session wastes the session; and the P0 fabricated-SHA incident proves the failure is real, not hypothetical | User instruction, P0 review | P0 |
| D-015 | Pin verification promoted to `scripts/p0_verify_pins.py` as the single source of truth, unit-tested with a mocked `HfApi`; `kaggle_run_p0.md` Cell 5 calls the script rather than an inline copy; **from P1 on, every data-building script calls it before writing data**. The Kaggle zip is built with `git archive --add-virtual-file=thesis_dbp/COMMIT.txt:<sha>`, and `env_report.py` reads `COMMIT.txt` when `.git` is absent, recording it as `git_commit` with `git_commit_source` | Two copies of a safety check drift, and the copy that drifts is the one that stops catching things. The mocked test fixes the decisive branch: `RevisionNotFoundError` is fatal (registry wrong), `GatedRepoError` is reported but not fatal (licence/token). Kaggle has no `.git`, so without embedded provenance its numbers could not be tied to the code that produced them | User instruction, P0 review; `--add-virtual-file` requires git ≥ 2.38, verified present on 2.54.0.windows.1 | P0 |
| D-016 | **Authoritative lineage map for P1**, derived from `reports/kaggle_20260920/p0_decoder_identity.json`: `kl_4ch_sd1 = {sd15, sd_ft_ema, sd21 == sd_ft_mse}`; `kl_4ch_sdxl = {sdxl, sdxl_fp16fix}`; `kl_16ch = {sd35, flux}`; `dc_ae_f32 = {dcae}`; `gan_sr = {realesrgan_x4}`. **`sd_ft_mse` is an alias of `sd21` and must never be generated as a separate member.** Leave-one-lineage-out holds out a whole group, never part of one | Measured byte-identical decoder hashes, not assumed from model cards. Generating the alias twice would double its weight in training and leak across any split that separated them | `p0_decoder_identity.json`, `byte_identical_decoder_groups: [["sd21","sd_ft_mse"]]` | P0-K |
| D-017 | Raw Kaggle result archives live in `F:\thesis-2\kaggle_raw\`, never inside `REPO_ROOT`; only extracted reports are tracked under `thesis_dbp/reports/kaggle_<date>/` | Anything in `REPO_ROOT` outside `thesis_dbp/` is by definition a protected-set change and fails the integrity check, which is exactly what happened with `p0_results.zip` (I-013) | User instruction, P0-K review; recorded in `CLAUDE.md` 7b | P0-K |
| D-018 | **FLUX.2 is represented by `black-forest-labs/FLUX.2-klein-4B` (`vae`), not `FLUX.2-dev`.** Identity between the two VAEs is recorded as **UNVERIFIED** and no claim is made either way | Measured in this session: FLUX.2-dev is tagged `other` and gated `auto`, and `hf_hub_download` of its `vae/config.json` returned **403 GatedRepoError**. A decoder whose licence is `other` and whose config cannot even be read is not a defensible bank member for a project with a deployment goal. FLUX.2-klein-4B is tagged `apache-2.0` and is not gated. Its config carries `_name_or_path: black-forest-labs/FLUX.2-dev` and the two weight files differ exactly 2x in size (168 120 878 vs 336 213 556 bytes) — suggestive of a dtype difference, but suggestion is not verification, and the dev weights cannot be read to settle it | User instruction Q3; `HfApi.model_info`, `hf_hub_download` | P1 |
| D-019 | Bank extended to **13 members / 12 distinct / 8 lineages** with three new kinds — `vae_kl_flux2`, `vae_kl_5d`, `vq` | The P0 bank clustered at 4 and 16 latent channels and had no de-tokenizer at all. Leave-one-lineage-out over 5 lineages is a weaker test than over 8, and a VQ member probes a genuinely different bottleneck. A new `kind` per family rather than a config-level class name keeps the kind → crop-multiple and kind → loader maps enforced | `configs/decoder_bank.yaml`; `scripts/p0_verify_pins.py` (15/15 resolve) | P1 |
| D-020 | **`alias_of` is a schema field and `BankConfig.distinct_members()` is the only member iterator the build scripts use** | D-016 was prose, and prose does not stop a future script from iterating `select()` and writing `sd_ft_mse` twice. Now the alias is data the registry validates — dangling aliases and alias chains raise — and the exclusion cannot be forgotten. This is what made B5's "aliases are skipped by the registry, never by a hard-coded name" requirement implementable | User instruction Q5; `registry.py`, `test_registry.py` | P1 |
| D-021 | **The P1 prompt's description of E3 is wrong and the measured facts are used instead.** Prompt: "2 904 imgs, 320x320 BMP, already resized". Measured from the Hub listing: **4 356 files = 2 178 unique images x 2 views**, **no BMP at all**, `Raw` native and variable, `Preprocessing` downscaled JPEG and non-square. Balanced 726 real / 726 GPT-Image2 / 726 Nano-Banana2 across Texture 648 / Structure 939 / Hybrid 591 | Building an evaluation set against a false description would produce numbers that look valid. The correction is recorded rather than silently applied so the discrepancy is visible to a reader of the thesis | User instruction; `HfApi.dataset_info` + full file listing, 2026-09-20; `docs/data_cards/e3_mllm.md` | P1 |
| D-022 | **The JPEG-matched-view rule, applied to every E-set.** Each data card carries a `label x (format, JPEG, quality, chroma subsampling)` table. Where compression history is confounded with the label, a derived view re-encodes **only the never-compressed fakes**, quality and subsampling drawn per image from that set's **own** real distribution, seeded by `blake2b(salt + ':' + image_id)`, at native resolution, built on Kaggle, originals untouched. E3 is reported in three views: Raw, **Raw-JPEG-matched (PRIMARY)**, Preprocessing (secondary, resized) | E3's `Raw` view has **0 real PNGs against 1 361 never-compressed fakes** — a 93.7 % spread. A detector can score far above chance there by reading compression history alone [R7]. Loading pixels rather than files stops the *format* leaking but not the *history*. Reals are never re-encoded: replacing a genuine history with a synthetic one is the same error in the other direction | User instruction; `CLAUDE.md` 6b; `dbp/data/jpeg.py`, `scripts/p1_build_eval.py`, `tests/test_eval_sets.py` | P1 |
| D-023 | **Commercial-use constraints are recorded as a first-class limit on the deployment goal**: Synthbuster CC-BY-NC-SA (non-commercial), MNW evaluation-only, WildRF and Chameleon research-only (`NOASSERTION`), DIV2K academic-only. Only COCO and E3 (apache-2.0) are comparatively permissive | The thesis has a deployment goal, and most of the corpus forbids commercial use. Left unrecorded, a perfectly good result would support a claim the data does not license. T-ctrl matters most, because it is *training* data: anything trained on it inherits the restriction | User instruction; `CLAUDE.md` 6c; the four data cards | P1 |
| D-024 | **Two fp16 thresholds, deliberately different.** `FP16_PSNR_THRESHOLD_DB = 45.0` stays the P0 smoke-test gate (D-003). `FP16_BUILD_PSNR_THRESHOLD_DB = 55.0` governs *building data*, and requires parity at **every** crop measured, not on average | A smoke test measures one image per crop; a corpus is thousands, and a pair built in fp16 is permanent. `sdxl_fp16fix` illustrates the gap: 48.58 dB at 512 and 55.56 dB at 1024 — acceptable as a smoke test, not acceptable to build 2 500 permanent pairs with | `reconstruct.py:fp16_build_allowed`; P1 section 6.2 | P1 |
| D-025 | **Zero new dependencies.** The perceptual hash is implemented in `dbp/data/phash.py` over `scipy.fft.dct` rather than adding `imagehash` | `ImageHash==4.3.2` is preinstalled on Kaggle and **absent** from the local lock file. Adding it would mean the two environments could not run the same code, for about twenty lines. The tests state the properties instead of trusting the implementation: identical → 0 bits, JPEG q>=75 → within the 6-bit threshold, unrelated → never inside it and mean > 16 bits | `env/lock/*`; `tests/test_phash.py` | P1 |
| D-026 | **A threshold is a `Calibration`, not a float**: it carries the split it was fitted on, the rule, the sample count and what it achieved there. Applying one to the set it was fitted on **raises** | A threshold chosen on the set it is reported on is the easiest way to turn a mediocre detector into a publishable table, and `argmax` over a test-set sweep looks like analysis. Making it an error rather than a convention means the mistake cannot be made quietly. The naive 0.5 threshold is reported alongside every calibrated one, because that is what exposed the FIRE failure | `dbp/eval/thresholds.py`, `tests/test_thresholds.py` | P1 |
| D-027 | `p0_verify_integrity.py` writes its report only when the content changes, or when `--out` is given explicitly | It rewrote `generated_at_utc` on every run and left the working tree dirty even when the verdict was CLEAN. That trains the reader to ignore a modified integrity report — the one file that must never be ignored | User instruction; `scripts/p0_verify_integrity.py` | P1 |

| D-028 | **`git restore` / `git checkout --` is permitted only on generated report files under `thesis_dbp/reports/` that the current session produced, and every use must be named in the report.** Forbidden everywhere else | P1 used it to discard a timestamp-only diff on `p0_integrity.json`. No content was lost, but §2 as written did not permit it. A blanket ban that gets quietly broken is worse than a narrow rule that holds; the narrow exemption is the accurate one | User instruction, P1 review; `CLAUDE.md` §2a | P1-R |
| D-029 | **T-ctrl takes 2 non-overlapping crops per DIV2K image (1 600 rows from 800 photographs), every row carries `source_image_id`, and any split must group by that column. T-ctrl stays diagnostic-only** | DIV2K holds 800 images against a planned 1 600 (I-021). Two crops recover the count; the condition is that they are never treated as independent samples. Crops are laid out along the longer axis and centred, deduplication drops whole source groups, and the build logs the grouping requirement on every multi-crop run so the constraint travels with the data. Effective sample size stays 800 photographs, and confidence intervals must say so | User instruction, P1 review; `p1_build_reals.py:crop_positions`, `docs/data_cards/t_ctrl_div2k.md` | P1-R |
| D-030 | **The training-side compression confound is fixed in the data, not documented around it.** (a) Reconstruction pairs stay raw PNG; P2 applies the identical JPEG/resize/blur pass to both members of a pair at training time (Aligned-Sync). (b) `scripts/p1_match_generations.py` writes a second variant of every generation, JPEG-encoded with a `(quality, chroma subsampling)` pair drawn per image from the T-real images it is paired with. (c) P2 trains on `gen_matched` by default and reports raw-vs-matched as an ablation. (d) DIV2K — lossless reals and their reconstructions — is the control that shows whether a model leans on JPEG structure at all, and is reported alongside | T-real is entirely JPEG; the generations were PNG that had never been compressed. That is the [R7] confound sitting in the *training* data, where it is worse than in an evaluation set: a model can learn it there and then appear to generalise. Documenting it would have left the strongest available shortcut in place | User instruction, P1 review; closes I-023 | P1-R |
| D-031 | **The matched pair is drawn jointly, from one named real**, not as two independent draws from a quality pool and a subsampling pool. Both the evaluation matched-view (§6b) and the generation matcher use the same `sample_matched_pair` | Independent draws can produce combinations that occur nowhere in the reals — quality 96 with 4:2:0 chroma when every high-quality real is 4:4:4 — and such a combination is itself a detectable artefact, which is the thing the mechanism exists to remove. Sampling one observed real also makes `source_real_id` / `matched_from` name a specific image, so every row is traceable | `dbp/data/jpeg.py:sample_matched_pair`; `tests/test_match_generations.py` | P1-R |
| D-032 | **E1 withdrawn; E7 DDA-COCO and E8 EvalGEN added.** E7 is the PRIMARY head-to-head set (SEF reports FLUX.1 82.8 / SD3.5 88.2 on it); E8 adds five 2024–2025 generators of which three are not diffusion models. Both `Junwei-Xi/*`, apache-2.0, not gated, pinned | E1 was private and had no paper, so it could not support a claim. E7 and E8 are public, pinned, permissively licensed and directly comparable to the published baseline numbers — which is what the head-to-head actually requires | User instruction, P1 review; `HfApi.dataset_info` 2026-09-20; `docs/data_cards/e7_dda_coco.md`, `e8_evalgen.md` | P1-R |
| D-033 | **Every set requiring a citation has it recorded verbatim in `docs/CITATIONS.md`, read by command from the source, and may not be reported on until that entry is filled in** | A citation reconstructed from memory is the same class of fabrication as a commit SHA reconstructed from memory (D-012). Making the citation a gate on reporting means the requirement cannot be satisfied late and approximately | User instruction, P1 review; `CLAUDE.md` §6c | P1-R |

| D-034 | **SEF fairness rule.** Any E7 (DDA-COCO) number compared against SEF must come from a configuration trained **without** `kl_16ch`, `kl_16ch_qwen` and `kl_32ch_flux2` (run `4ch_only`). The full-bank number is reported separately and always labelled *"deployment configuration — in-bank decoders"*. Every E7 table states, per subset, whether that VAE was in the training bank | SEF trained on the SD2.1 VAE alone; our bank has SD3.5, FLUX, Qwen and FLUX.2. Putting a full-bank number next to SEF's published FLUX.1 82.8 / SD3.5 88.2 would be a better-resourced system beating a smaller one, reported as a win — dishonest by omission. Wired into the schema now rather than remembered later: `excluded_lineages` is validated against the registry, so a typo fails loudly, and `RunConfig.describe()` is the block every evaluation report prints | User instruction, P1-R review; `configs/p2_runs.yaml`, `src/dbp/eval/run_config.py`, `tests/test_run_config.py` | P1-R2 |
| D-035 | **E8 metric policy.** Primary: fake detection rate at thresholds fitted on **our** validation reals (FPR 1% and 5%), per generator. Secondary, always carrying its caveat and never the headline: AUC against COCO `val2017` reals re-encoded to JPEG q96 and matched to E8's uniform q96 | E8 has no real class, so AUC is undefined on it alone. The primary needs no reals inside E8 at all. The secondary crosses a content boundary — GenEval compositional prompts versus COCO photographs — so its AUC can be inflated by content rather than by generative artefacts, and that cannot be cleanly separated. Reals for the secondary come from `val2017` because T-real trains on `train2017` only (I-024) | User instruction, P1-R review; `docs/data_cards/e8_evalgen.md` §2 | P1-R2 |
| D-036 | **Budget control by seeded stratified subsampling.** E8: 1 000 images per generator (5 000 total), drawn by a seeded permutation of each stratum's sorted image ids, salted with the set id. The sampler, seed rule, per-stratum counts and **every kept id** go in the summary. Same flags available for E7 | "The first N files" in a set laid out by generator returns one generator. Seeding on sorted ids rather than on file order makes the subset identical on any machine and independent of how a filesystem enumerates a directory, so the subset is a citable object rather than whatever that run happened to pick | User instruction, P1-R review; `p1_build_eval.py:stratified_subsample`, 8 tests | P1-R2 |
| D-037 | **Chameleon is evaluated at a 320 px crop, single crop** — not the project's standard 384, and not multi-crop | Measured on the archive: every one of the 14 863 reals is exactly 500 px wide and short (median height 359), while the fakes are native up to 5376x6720. At a 384 crop **44.0 % of reals survive against 94.5 % of fakes**, flipping the class balance from 57/43 to 38/62 and keeping only the unusually tall reals — a confound introduced by *our own* pipeline on top of the one in the data. At 320 it is 93.1 % / 95.9 %. Two non-overlapping 320 crops would retain 5 211 reals against 9 200 fakes, so multi-crop is rejected too. The review's premise ("images go up to 4K, use the standard multi-crop policy") holds only for the fakes | Measured this session from the archive's JPEG headers; `configs/p2_runs.yaml` runs `full_bank_chameleon` / `4ch_only_chameleon`; `docs/data_cards/e5_chameleon.md` §5 | P1-R2 |
| D-038 | **E5 Chameleon and E6 MNW terms recorded verbatim and added to §6c**; E5 is *"only used for academic research. Commercial use in any form is prohibited"* and the AIDE repository's MIT licence covers the **code**, not the dataset; E6 is evaluation-only, no training, no commercial use, citation required. Both citations are in `docs/CITATIONS.md`, read from the repositories by direct URL | The AIDE repo being MIT is exactly the kind of detail that gets mis-transcribed into "the dataset is MIT". Recording which licence covers what, verbatim and with the fetch date, is the only way that does not happen | `raw.githubusercontent.com` fetches 2026-09-21; `CLAUDE.md` §6c | P1-R2 |

| D-039 | **Greyscale reals are excluded, by flag, and nothing is deleted.** (a) B4 now rejects greyscale sources at build time: single-channel images, and three-channel images whose per-pixel channel spread is <= 2 (`GREYSCALE_TOLERANCE`). (b) T-real, already built, is re-scanned by `p1_rescan_greyscale.py`, which sets `excluded_reason=greyscale` on the affected rows; the PNGs, the 49,200 B5 pair rows and the 2,488 generations are **not** rebuilt, and downstream manifests filter on that column. (c) The matcher's reference set independently refuses rows that are flagged or whose subsampling is unmappable, and counts both. (d) A generation whose exact partner is excluded **switches to a seeded colour partner** rather than being dropped, recorded as `pairing=sampled_after_exclusion`. No backfill: T-real is 2,500 minus the excluded count | Two distinct harms. First, a greyscale JPEG has no chroma channels, so Pillow reports `subsampling = -1`; there is no honest `(quality, subsampling)` pair to copy onto a colour image, and 13 of 2,488 generations failed on exactly that. Second, and worse, "has no colour" is a shortcut: if one class contains black-and-white photographs and the other does not, a detector scores well without looking at a generative artefact. Flagging rather than deleting keeps the decision reversible and auditable, and costs nothing downstream because every consumer joins on `image_id`. Dropping the generation instead of re-pairing it would shrink the training set for a reason unrelated to the generator | User instruction 2026-09-26; `src/dbp/data/colour.py`, `scripts/p1_build_reals.py`, `scripts/p1_rescan_greyscale.py`, `scripts/p1_match_generations.py`; 16 new tests | P1-K |
| D-040 | **The reconstruction goldens are looked up by `(diffusers, torch, threads)` and the test skips where no entry exists.** No digest was changed; only the lookup. The one captured environment is `(0.40.0, 2.11.0+cpu, 4)` | The digests are not portable, which two measurements showed: on Kaggle (diffusers 0.37.1, torch 2.10.0+cu128) all three differ, and locally, changing only `torch.get_num_threads()` from 4 to 1 moves the `dcae` digest while the model weights are identical. Neither is a protocol change, so failing was the wrong signal. The user approved the keyed lookup on condition that no expected value change -- and none did. Cross-environment cover comes from `p0_bank_smoke_test.py --compare-to`, which passed on Kaggle over 20 (member, crop) pairs on real weights | `tests/test_reconstruct_regression.py`; approved 2026-09-26 | P1-K |
| D-041 | **Black-and-white robustness is handled in P2 by *synchronized* greyscale augmentation**, not by keeping greyscale images in the corpus: both members of a reconstruction pair are converted together, p ~ 0.05-0.1, recorded per run | Symmetric by construction, so it teaches invariance instead of creating a shortcut: if only one side of a pair were converted, "is greyscale" would again predict the label. Keeping natural greyscale reals in the corpus cannot achieve this, because their reconstructions would be the only greyscale fakes | User instruction 2026-09-26; `configs/p2_runs.yaml` note | P1-K |

| D-042 | **Every evaluation set reports whether an encoding *parameter* predicts the label, pooled and per stratum.** The measure is the accuracy of the best rule that looks only at that parameter, against a majority-class baseline; the difference is the *lift*, and a lift above 5 points is recorded as a shortcut. Affected subsets carry the caveat wherever they appear; no matched view is built for them | §6b asks whether an image was compressed, and E7 passed it cleanly — both classes are entirely JPEG. Meanwhile every FLUX.1 reconstruction in it is 4:2:0 and every real is 4:4:4, so `4:2:0 -> fake` scores **100.00%** on that subset, which is one of the two carrying the SEF comparison. Pooled, the lift is +0.03 points; on the FLUX.1 subset it is **+49.85**, which is why the per-stratum split is the whole point. Re-encoding is **not** a fix: chroma subsampling is destructive, so 4:4:4 re-encoding upsamples rather than restores, and equalising the headers would hide the shortcut while leaving the signal in the pixels — worse than leaving it visible | Approved 2026-09-27; `dbp/data/jpeg.py:parameter_separability`, `separability_by_stratum`, `p1_build_eval.py`, `CLAUDE.md` §6d, 7 tests | P1-K |

| D-043 | **The pipeline-matched view.** Where the two classes differ in how they were *published*, the fakes are put through the reals' whole pipeline -- downscale to the reals' width with the aspect ratio preserved, then encode at their setting -- not only its JPEG step. Reals untouched, originals untouched, derived view. LANCZOS by default with BICUBIC as a sensitivity check. **For E5 Chameleon this view is PRIMARY** | The compression-only matched view fixed E5's compression confound and left its larger one in place: every real is exactly 500 px wide because Chameleon thumbnailed them, so `width == 500 -> real` scores **100.00%** even after matching. §6b copies a compression setting; it cannot copy a publishing pipeline. This deliberately resizes, which §7 forbids in the *reconstruction protocol* -- there a resize fabricates a trace; here it **is** the trace being matched, applied to the one class that lacks it. The honest cost is recorded too: downscaling a 4K fake to 500 px destroys the high-frequency detail a decoder fingerprint lives in, so this is the fair comparison, not the favourable one | User instruction 2026-09-28; `p1_build_eval.py:build_pipeline_matched_view`, `width_rule_accuracy`, 6 tests | P1-close |
| D-044 | **Reporting rules, written into the data cards and `configs/p2_runs.yaml`.** E7 FLUX.1 numbers print the trivial-rule baseline in the adjacent column and P2 adds a chroma sensitivity check there; E4 tables carry a red flag naming the +11.39 quality separability; E5 reports from the pipeline-matched view; E3 from the matched view with its +6.47 residual stated; E8 as a detection-rate set | A measured shortcut that lives only in a JSON summary will not reach the thesis table, and a number without its baseline beside it invites exactly the reading it should prevent. Putting the rules in the config the evaluation report reads makes them part of the run rather than part of someone's memory | User instruction 2026-09-28 | P1-close |

| D-045 | **Every P2 run evaluates at crop 320, and `p_a_sd1` exists as the primary protocol.** `eval_crop` was 384 on the three non-Chameleon runs and 320 on the two Chameleon ones; it is now 320 everywhere. Training takes a **synchronized random** 320 crop out of the stored 384 pair; evaluation takes a fixed centre crop. Two new runs, `p_a_sd1` and `p_a_sd1_chameleon`, resolve to exactly {sd15, sd21, sd_ft_ema} | One input size for all three streams, or S3's bank would have to be run twice. 320 is a multiple of 16 (DINOv3 patches), 32 (DC-AE), 8 (KL VAEs) and 4 (the SR model), it is already D-037's Chameleon crop, and it cuts the S3 bank cost to 69.4% of 384. The P-A gap was real: `4ch_only_strict` was the nearest existing run and it retains `kl_4ch_sdxl`, which would have made SDXL an in-bank lineage while the pre-registration lists its E7 subset as unseen. **Deliberate change to existing values**, not an append; a test asserts every run is at 320. Consequence recorded: S3's residual statistics are computed on a different crop from P1's pair PSNRs and will not reproduce them | User approval 2026-09-29; `configs/p2_runs.yaml`, `tests/test_run_config.py` | P2-A |
| D-046 | **The train/validation split: 80/20, seed 2026, grouped by `source_image_id`, stratified by COCO supercategory -- and generations follow the caption.** A real follows its own source image (t_ctrl stores two crops per photograph); a pair row follows the real it was reconstructed from; a generation and its `gen_matched` variant follow the COCO image whose **caption** produced them, never `source_real_id`. The stratum is the **rarest** supercategory present in the image. `verify_no_content_crosses` runs over all sets together and raises | P1 left no `split` column at all -- only `split_grouping_column: source_image_id` -- so the split had to be created, and creating it once and committing it is what makes every later calibration reproducible. The caption rule is the subtle one: `source_real_id` names the real whose JPEG settings the matcher copied, which is usually a different picture, so following it would put a generation in validation while the photograph its content came from sat in training. Ranking supercategories by the *most* frequent instead would label most of the corpus `person` and stratify nothing. Pre-registered consequence: validation holds ~480 reals, so TPR@1%FPR rests on ~5 false positives and is reported but never gated; TPR@5%FPR is the gate metric | User approval 2026-09-29; `src/dbp/data/splits.py`, `scripts/p2_build_splits.py`, 22 tests | P2-B |
| D-047 | **S2's step count comes from a measured step, against a GPU-hour allowance, with a hard floor at 2,000 steps.** Each run times 20 real training steps at batch 16 with gradient checkpointing on, converts a 2.25 GPU-h allowance into a step ceiling, and **stops with `STOPPED-UNDERTRAINED` below 2,000 steps** rather than running | No committed measurement of a training step on this backbone exists: P1's backbone throughput JSON has `"backbones": null`, and P0's LoRA entry is explicitly `"dry-run only: no optimizer, no training step"`. A fixed 8,000-step plan would have been a guess wearing a number. The floor is the user's guard: an undertrained S2 would not be a weak result, it would make the three-stream comparison invalid, and that is a decision to escalate rather than absorb | User approval 2026-09-29; `dbp/streams/s2_artifact.py:steps_within_allowance`, `scripts/p2_train_s2.py` | P2-B |
| D-048 | **The evaluation panel is drawn once, seeded and stratified, and shared by every stream**; S1 and S2 are additionally scored on the full views and those numbers are labelled separately. Caps: E7 500/subset, E8 250/generator, E3 1,500, E4 1,500, E5 2,000/label, T-ctrl 200+200 -- **12,150 images**. Allocation is proportional with largest remainders and a floor of one per non-empty stratum | Complementarity is a statement about *which images* each stream gets wrong, so it is only defined where the streams saw the same images; scoring S1 on 34,969 E7 rows and S3 on a cheaper subset would make every double-fault rate a comparison of two different sets. The floor departs from strict proportionality on purpose: under pure proportional allocation a stratum holding 0.3% of a view rounds to zero, and D-044 requires E3 to be reported per content type, which a stratum absent from the panel makes impossible. `PANEL_CAPS` raises on a set it has no approved cap for, so adding one is a plan change rather than a default | Approved P2 section B1; `src/dbp/eval/panel.py`, 21 tests | P2-B |

| D-049 | **The three P2 image lists are built by a tested script, `p2_build_image_lists.py`, not by a notebook cell.** Paths are resolved against the staged corpus root and **verified to exist** at list-build time, then stored relative to it. The S3 training subset is balanced per protocol over that protocol's active members plus `gen_matched`, seeded, with `--train-reals`, `--train-fakes-per-protocol` and `--val-cap` as the reduction levers. The script also draws `p2_panel_t_ctrl_diagnostic.csv`, which has no evaluation manifest behind it | These three files commit the whole GPU budget: `p2_probe_images.csv` costs twelve reconstructions and twelve LPIPS forwards per row, and the 9.63 GPU-h estimate assumed 18,150 of them. A pasted cell deciding that is the wrong place for it. Verifying paths here is the same argument: a row that cannot be resolved costs one CPU cell now instead of surfacing nine GPU-hours into Session B. Two defects the tests caught before any GPU time was spent: T-ctrl rows are **both** panel rows and training reals, so they were listed twice -- reconstructed twice and weighted twice in every metric -- and the panel files were globbed as `p2_panel_*.csv`, which on a second run matched the script's own `p2_panel_images.csv` output and raised on its missing `panel_set_id`. Panels are now named from `PANEL_CAPS` | User approval 2026-10-02; `scripts/p2_build_image_lists.py`, 29 tests | P2-B |

| D-050 | **The evaluation panel is frozen as a directory, not as a list of paths.** `p2_materialise_panel.py` copies all 12,150 panel images to `<panel_root>/<view>/<image_id><suffix>`, **re-hashes each against the `sha256` its manifest already records**, rewrites the path column relative to the panel root, and writes a panel digest -- SHA-256 over the sorted `image_id<TAB>sha256` lines. Published as the **private** dataset `dbp-p2-panel`. The source suffix is preserved (no re-encoding), an `image_id` that could escape the root is refused rather than sanitised, and a hash mismatch stops the run without overwriting the recorded hash. Image lists now carry a `root` column (`corpus` or `panel`) and the extraction scripts take `--panel-root`. E7's FLUX.1 and real subsets additionally get a greyscale-in-RGB **PNG** companion view, not counted in the 12,150 | Session A froze the panel's *selection* and not its *pixels*. Every panel row recorded a path valid only in the session that built its view -- `/tmp/e7x/DDA-COCO/...`, or an absolute `/kaggle/working/.../eval/e3_raw_jpeg_matched/...` -- so **7,750 of 12,150 rows were unreachable in a fresh session** and the first thing to discover it would have been a GPU run. Freezing it once means no evaluation corpus is ever re-downloaded again and no later session depends on a `/tmp` path from a session that no longer exists. The digest exists so Session B can assert "this is the panel" in a CPU cell instead of inferring it from a crash. PNG for the luma companion because it is lossless and identical for both classes, so the transform adds no label-correlated difference of its own, while taking luma discards the chroma planes where E7's 4:2:0-versus-4:4:4 shortcut lives (pre-registration §8). Private because WildRF is research-only and Chameleon prohibits commercial use in any form (§6c) | User approval 2026-10-02; `scripts/p2_materialise_panel.py`, 30 tests; `kaggle_run_p2_panel.md` | P2-B |

| D-051 | **T-ctrl is withheld from every training list, not merely labelled.** `DIAGNOSTIC_ONLY_SETS = {"t_ctrl"}` in `dbp.eval.panel`; `p2_build_image_lists.py` filters the S1 list, the S3 training subset **and the S3 validation subset** on it, and `p2_train_s2.py` filters the pair rows, logging the count separately from D-039's. The 400-row diagnostic panel is drawn from the same rows and is unchanged | The P2 prompt (B1) makes T-ctrl diagnostic-only, and the implementation did not honour it: `corpus` is the population for both the training lists and the diagnostic panel, and a T-ctrl row's split label reads `train` like any other, so **20,722 rows -- 1,594 control reals and their 19,128 reconstructions -- were in `p2_s1_images.csv`**, 313 of the 400 diagnostic rows among them. The JPEG-reliance check would have been measured on the streams' own training data. Validation is filtered too, because validation fits the temperatures, the thresholds and the stacker: a row used there is still a row the streams were tuned on. The S1 cache drops 54,631 -> 33,909 and the diagnostic becomes fully out-of-sample | User instruction 2026-10-03; `src/dbp/eval/panel.py`, `scripts/p2_build_image_lists.py`, `scripts/p2_train_s2.py`, 5 tests | P2-B1b |
| D-052 | **The luma-only view is implemented as a declared derived view, outside `PANEL_CAPS`.** `LUMA_VIEWS` in `dbp.eval.panel` names its parent, its subsets and the gate slice it displaces; rows carry `<image_id>__luma`; they join the probe list and get their own S1 cache tag; `p2_score.py` scores them as `e7_dda_coco_luma` with metadata joined from the parent row; `p2_analyze.py` substitutes `e7_dda_coco/FLUX.1` -> `e7_dda_coco_luma/FLUX.1` in M1 and M2 and **refuses a verdict** if the luma companion was not scored | Pre-registration §8 says the luma-only number is the one that enters M1 and M2 and that an unchecked FLUX.1 number is not claimable, and `grep -i luma` over the four P2 scripts matched nothing: the gate could not be computed as pre-registered. Not added to `PANEL_CAPS` because that dict is what `p2_materialise_panel.py` iterates and what the 12,150-row assertion and the frozen digest are defined over -- a seventh key would demand a sampler CSV nothing writes and would change the digest. The id suffix is load-bearing: `p2_probe_features.py` writes one `.npz` per bank member keyed by image id, so a shared id would overwrite the standard row's twelve reconstructions with the greyscale ones, and a luma-only second run would invalidate and rebuild the first | User approval 2026-10-03; `src/dbp/eval/panel.py`, `scripts/p2_score.py`, `scripts/p2_analyze.py`, `scripts/p2_build_image_lists.py`, 24 tests | P2-B1b |
| D-053 | **Exact duplicates are reported and quantified, never removed.** `p2_analyze.py` measures them from the frozen panel list's `sha256` column, prints a `caveat` on every affected slice in `metrics.csv`, and writes `metrics_deduplicated.csv` as an analysis-only sensitivity table with each group counted once | An evaluation set is used as distributed (P1), and E4 ships 25 byte-identical groups over 50 of its 1,500 panel rows, so those images are weighted twice in every E4 metric. Dropping them would change the set; saying nothing would misreport it. The counts are computed in the run rather than quoted, so the caveat cannot drift from the panel | User instruction 2026-10-03; `scripts/p2_analyze.py`, 14 tests | P2-B1b |
| D-054 | **A panel row must survive the evaluation crop, and the rule is applied label-blind with no backfill.** `dbp.eval.panel.filter_by_min_side` and `draw(min_side=EVAL_CROP)`; `p2_materialise_panel.py --min-side 320` filters the lists from the manifests' recorded `width`/`height`, re-reading no image. Rows recording no size are kept and counted. Panel 12,150 -> **11,775**, luma 1,000 -> **978**; the panel *directory* is untouched | 375 rows had a shorter side below 320, the smallest 150 px, so they could not be centre-cropped to the evaluation crop and could not be scored by any stream (I-047). Up-scaling or padding them would keep every row but stamp a resampling or border signature on exactly the undersized subset -- in E3 that subset is reals only -- which is a D-042-class shortcut manufactured by us, in the direction that most flatters the result. Lowering `eval_crop` to 256 was rejected because S2 trains on 320 crops. No backfill because a replacement draw would be a second, unseeded draw | User approval 2026-10-03; `src/dbp/eval/panel.py`, `scripts/p2_materialise_panel.py`, pre-registration Amendment 1, 19 tests | P2-C |
| D-055 | **A panel file already present under the panel root that hashes to its recorded `sha256` is the panel image, and no source is resolved for it.** `materialise_view` checks the destination before calling `resolve_source` | Freezing the panel is pointless if re-deriving its *list* needs the evaluation corpora back: re-running materialise to apply D-054 would otherwise have demanded another 10 GB of downloads for files already verified byte-for-byte. It also lets the panel root be read-only, which it is once published as `dbp-p2-panel`. The hash is still checked -- what changes is that a correct hash now ends the search instead of starting it | 2026-10-03; `scripts/p2_materialise_panel.py`, 1 test | P2-C |
| D-056 | **GPU_BUDGET_HOURS raised 32 -> 36, with no cuts to the experiment.** S3 authorised at the measured 12.90 GPU-h with a hard stop at 14.5. If the measured total exceeds 36 the ladder is: (1) full-view S2 scoring, (2) move the `full_bank` S2 run to P3/P4, (3) the second P-A seed. **The S3 training subset is never halved.** Quota exhaustion means continuing next week, not cutting | The 11 GPU-h S3 guard fired on measurement. Two calibration runs at 300 images each gave sd15 0.1739 s/image and movq_k22 0.4230 s/image at crop 320; solving the two for a shared decode scale and a constant LPIPS gives **LPIPS = 8.4 ms per (image, member)** and a decode cost **1.226x** the area-scaled P1 rate. LPIPS was never the risk I-041 feared; the error was mine, in scaling P1's crop-384 cost to 320 by area when per-call overhead does not shrink with pixel count. Halving the S3 subset was refused because it would undertrain one of the three streams the gate compares | User instruction 2026-10-03; `env/kaggle_run_p2_streams.md` | P2-C |
| D-057 | **Runbooks are percent-format `.py` files of thin calls to committed, tested scripts, and each is dry-run locally on the exact cell text before hand-over** (CLAUDE.md section 12). New scripts: `p2_stage_corpus.py` (links with a junction fallback on Windows, counts asserted), `p2_restore_partial.py` (restores a bundle and verifies every member with `shard_is_complete`), `p2_preflight.py` (paths, imports, counts, list digest against every sidecar, every row resolves, regenerated files against committed hashes, 2-image smoke per model), `p2_run_shards.py` (one job per device; the first failure stops the rest), `p2_report.py` (always exits 0), `runbook_dryrun.py` (fixture shaped like `/kaggle/input`) | Every recent Kaggle loss came from inline cell code that was never run before Kaggle ran it. The dry-run is only meaningful if the cells are plain Python (no IPython locally) and the fixture has the real slugs at the real depth | User instruction 2026-10-04; 43 tests | P2-C |
| D-058 | **The code zip is built from git's raw blobs** (`build_code_zip.py`, `git cat-file --batch`) and the dry-run's code fixture is built by the same function | The working tree on Windows had drifted to CRLF in 20 tracked `.sh`/`.py` files while every commit held LF. The old zip read the working tree, so the next upload would have shipped a CRLF `kaggle_bootstrap.sh`, which fails on Kaggle's bash at its first line -- and Git Bash runs it happily, so no local run would have shown it. Earlier zips shipped CRLF `.py` files: harmless, but not byte-identical to their commits | 2026-10-04; 4 tests | P2-C |
| D-059 | **`kaggle_bootstrap.sh` removes a torchao that blocks `peft` itself**, and gains `DBP_DRYRUN=1` (GPU checks skipped, nothing installed or removed, editable install skipped, guardrail still enforced) | It used to print the uninstall command, which in commit mode nobody reads; the operator has uninstalled it every session. No kernel restart is needed because every stage now runs in a fresh subprocess. The dry-run mode is what lets the bootstrap -- now the only environment step -- be dry-run at all without modifying the local venv | 2026-10-04 | P2-C |
| D-060 | **`COMMIT.txt` is untracked and generated into each zip** | A tracked file cannot contain its own commit hash; it was committed by mistake in `da31f85` and was stale from then on. `build_code_zip.py` excludes it by name even from commits that track it | 2026-10-04; `.gitignore` append | P2-C |
| D-061 | **Every P2 output path is built by `dbp.utils.run_paths`, keyed by run id (`<protocol>_seed<seed>`); every log and scratch path also carries the device (`gpu0`/`gpu1`)** | Two parallel jobs must never write one path (I-053). The GPU index is deliberately not in a checkpoint's path: the queue gives a run to one worker at a time, so the run id is already unique, and a resumed run may land on the other GPU -- a device in the path would make it restart from step 0. S2 runs now own `outputs/p2/s2/<run id>/` (checkpoint, selected LoRA, head, scores); heads, fusions, panel scores and analysis are keyed by run id, not protocol | User instruction 2026-10-04 (step 1); `tests/test_output_paths_unique.py` | P2-D |
| D-062 | **Session D is one queue with one worker per T4** (`scripts/p2_run_queue.py`): a worker takes the next run as soon as its GPU is free, trains it and scores it on the same GPU (`scripts/p2_s2_infer.py`). **A failed run is logged and not retried; the queue continues** and the notebook's last cell fails if any run failed | User instruction 2026-10-04. This differs deliberately from CLAUDE.md 12.4 (one failure stops every job): the four runs are independent, and the failures that are systematic -- a missing module, a model that cannot load or save -- are what the per-run, per-GPU smoke pre-flight catches before the queue starts, where the first failure does stop everything | `tests/test_s2_queue.py` | P2-D |
| D-063 | **S2 scoring uses each run's selected model** -- the LoRA and head saved at the best validation AUC -- **on three lists: the panel (11,775), the luma companion (978) and the S1 validation rows (6,883)**, centre crop 320 | The trainer's early stopping exists to select that model; scoring the last step would discard it. The validation rows are needed because `p2_fit_heads.py` calibrates S2 and fits both fusions on validation scores only. The three lists match the S1 caches row for row (checked against the downloaded caches), so the S2 scores join S1 by image id | 2026-10-04 | P2-D |
| D-064 | **Session D inputs not in git travel in one private dataset, `dbp-p2-s2-inputs`**, each file pinned by sha256 in `env/p2_s2_expect.json` and checked by the pre-flight: the pairs manifest (equal to the committed digest), the split, the generation manifest and the three image lists | The S3 output does not hold the split or the pairs manifest, and the runbook previously took them from wherever an `rglob` found them. Inputs are found by dataset name under any owner, so the runbook does not depend on which account owns a dataset | 2026-10-04 | P2-D |
| D-065 | **The S2 trainer gains a smoke mode and two dry-run seams.** `--smoke N` (N steps, save, reload into fresh objects, tensor-by-tensor comparison) is the per-GPU model pre-flight. `--dryrun-steps` and `DBP_FAULT_EXIT_AFTER_STEP` are honoured only with `DBP_DRYRUN=1`; so is a random-init two-layer DINOv3 of the pinned architecture, which refuses CUDA | The runbook must be dry-run end to end, including a kill and a resume, on a machine with neither a GPU nor the gated weights. The real weights and the token are first exercised by the Kaggle pre-flight, and every dry-run artefact says it is one | `tests/test_s2_queue.py`; the dry-run | P2-D |
| D-066 | **Checkpoints are loaded with `weights_only=True`**, and the budget recorded in a checkpoint is restored verbatim on resume rather than recomputed | CLAUDE.md 8 (no pickle for our own artefacts): the checkpoint holds only tensors, numbers, strings and plain containers. Recomputing the budget would also have re-applied the step floor to a dry-run budget and stopped a resumed run as undertrained | 2026-10-04 | P2-D |
| D-067 | **Every Session E join is by image id** (`dbp.eval.join`): duplicates, missing ids, unexpected ids and label disagreement between the S1 list, the probe list and the split raise, naming the first ten ids; fusions fit and score only through `fit_by_id`/`score_by_id`, refuse any stream set other than `FUSION_STREAMS = (aeroblade, s1, s2, s3_lr, s3_set)` -- the five the committed code already fused -- and record the streams they used | I-056 | User instruction 2026-10-05; `tests/test_join.py` (14) | P2-E |
| D-068 | **The protocol decides what every Session E fit may see.** S1 trains on the S1 list's training reals and the protocol's training fakes (+`gen_matched`); S3 on the 1,000 reals and the protocol's own 1,000-fake pick, re-drawn with the list builder's seeded `balanced_take` -- the three protocols' picks reproduce the probe list's 2,861 training fakes and 1,000 reals **exactly**; validation is the probe list's 2,000 validation rows restricted to the protocol's lineages (all 491 validation reals kept: 932 rows for P-A, 1,550 for `4ch_only`, 2,000 for `full_bank`) | The committed `p2_fit_heads.py` trained S1 and S3 on every lineage whatever the protocol, while S2 respected it, and fitted the fusions on every lineage too: under P-A both would have seen the families the gate calls unseen | User GO 2026-10-05; recorded for Amendment 2 | P2-E |
| D-069 | **Pre-registration Amendment 2** (committed before any panel score): the verdict is decided by **F-mean(A2)** = equal-weight mean of S1, S2 and S3-Set, each robust-z-scored on the run's 491 validation reals (median, 1.4826 x MAD, clip +-5; MAD 0 -> std; std 0 -> raise), with no fake label, weight or temperature; **best single = the maximum over all five single heads of the panel macro, per metric, re-maximised in every bootstrap replicate**; GO rule unchanged (>= 2.0 points on M1 or M2, 95% paired CI excludes 0); **P-A GO only if both seeds pass**. The original F-stack vs validation-chosen best single (Analysis A) is run and reported unchanged as secondary; `4ch_only` and `full_bank` gates are secondary | I-060: on P-A validation S3-LR is perfect and holds 97-98% of the F-stack weight, so section 6 would compare F-stack with itself | User decision 2026-10-05 (option D, seed rule ii, with the user's changes to C); `docs/P2_PREREGISTRATION.md` Amendment 2 | P2-E |

| ID | Issue | Impact | Proposed action | Status |
|---|---|---|---|---|
| I-001 | DINOv3 gating request | — | Was 403 GatedRepoError at 21:44; **approved during the session** and confirmed at 22:41 by a real gated download of `config.json` (hidden 768, patch 16, 4 register tokens) | **RESOLVED** |
| I-002 | ~~Local bandwidth 0.12 MB/s~~ — moot: all 10 members measured on Kaggle. Local bandwidth 0.12 MB/s | High for local work only — the full 4.9 GB bank would take ≈ 11.5 h to fetch here | Only `sd15` (335 MB) and `realesrgan_x4` (67 MB) were fetched and measured locally. The other 8 members are fetched on Kaggle | **OPEN (mitigated)** |
| I-003 | `transformers 5.17.0` locally vs Kaggle's 4.x | Medium — diffusers 0.40.0 predates transformers 5.x; DINOv3's class is `DINOv3ViTModel`, not `Dinov3ViTModel` | Code goes through `AutoModel` only, never the concrete class. Both lock files recorded. `pip check`: no broken requirements | **OPEN (monitored)** |
| I-004 | DC-AE repo exposes **no license tag** on the Hub; model card says MIT | Low, but a thesis-compliance item | Recorded in the registry as `"Hub tag: none; MIT per model card"` | **OPEN** |
| I-005 | Python 3.13 locally vs 3.12 on Kaggle | Low — lock files are not interchangeable | Separate `lock_local_*` / `lock_kaggle_*`; `requires-python >= 3.10`; no 3.13-only syntax | **ACCEPTED** |
| I-006 | Protected-asset guardrail currently guards an empty set | Medium — FIRE is not mirrored locally, so it cannot be protected by hash here | Re-baseline when `prior_results/` is added | **OPEN** |
| I-007 | Tools run from `REPO_ROOT` can litter it | Low, but it defeats the scope lock | Caught in P0: `ruff` created `REPO_ROOT/.ruff_cache/`. Removed; cache pinned; docs and `CLAUDE.md` updated (D-011) | **RESOLVED** |
| I-008 | VS Code's selected interpreter is the **system** Python, not `thesis_dbp/.venv` | Cosmetic — the IDE reports `peft`/`spandrel`/`lpips` as missing although the venv has them | Select `thesis_dbp\.venv\Scripts\python.exe` as the workspace interpreter | **OPEN (cosmetic)** |
| I-009 | The first member measured on a GPU absorbs CUDA context initialisation, inflating its `ms/img` | Low, but it makes the first row of every timing table wrong: sd15 read 2 029 ms where comparable members read 744–786 ms, and dinov2 read 385.6 ms against dinov3's 56.9 ms | Discard a warmup forward before timing in both smoke scripts; re-measure in P1 | **RESOLVED (code)** — `--warmups` (default 3) added to `p0_bank_smoke_test.py` and `p0_backbone_smoke_test.py`, and to `forward_once`, which also resets peak-memory stats *after* the warm-ups so the reported figure is the steady-state pass. Re-measurement is PENDING-GPU |
| I-010 | **Pillow ships different libjpeg in the two runtimes** — local 8.0, Kaggle 6.2 | **High for P1.** JPEG encoder differences change compression artefacts, which is the single best-documented confound in this literature [R17]. Building JPEG-augmented data in one runtime and evaluating in the other would inject a spurious real/fake signal | All JPEG augmentation happens in exactly one runtime (Kaggle), recorded in the dataset manifest; never mix. **P1 must additionally run a byte-level JPEG encoder comparison: encode the identical source image at q=75 under local Pillow (libjpeg 8.0) and Kaggle Pillow (libjpeg 6.2), and compare the resulting files byte-for-byte plus their decoded pixel difference.** If the bytes differ, the runtime becomes part of the dataset provenance record | **PARTIALLY RESOLVED.** `scripts/p1_jpeg_equivalence.py` implements it over a 24-cell grid (2 sizes x q in {50,75,90,95} x 3 chroma modes), recording byte SHA-256, decoded-pixel SHA-256 and pixel deltas, and separating "different bytes, same pixels" from "different pixels". The **local** half is measured and committed at `reports/p1_jpeg_equivalence.json` (Pillow 12.3.0, libjpeg 8.0, Windows; quality round-trip exact at every tested q). The Kaggle half and the comparison are **PENDING-GPU** |
| I-011 | Kaggle preinstalls `torchao 0.10.0`, which trips a transformers version guard on the `peft` import path and killed the LoRA dry-run with `ImportError: ... only versions above 0.16.0 are supported` | Medium — blocks all LoRA work in P2 until handled | Resolved for the session by `pip uninstall -y torchao` (we use no quantization); torch/torchvision/numpy verified unchanged afterwards. Folded into `kaggle_bootstrap.sh` step [7/7], which detects `torchao`, tries the `peft` import, and prints the exact uninstall command **without running it** — removing a package from someone else's image is the operator's decision | **RESOLVED** |
| I-012 | `test_t4_has_fp16_but_no_native_bf16` asserted the wrong predicate and failed on the T4 | None on results; the test was wrong, not the hardware | `torch.cuda.is_bf16_supported()` defaults to `including_emulation=True` and returns True on sm_75 because bf16 *tensors* can be created without tensor-core support. Fixed to use `including_emulation=False`. The measurement actually **supports** the fp16 policy | **RESOLVED** |
| I-013 | `p0_results.zip` was placed at `REPO_ROOT/p0_results.zip`, outside the write scope | Low, but it makes the integrity check report an `ADDED` protected file | Its contents are now tracked under `thesis_dbp/reports/kaggle_20260920/`. The zip itself is the user's file; it is neither moved nor deleted. Either delete it or re-baseline the manifest deliberately | **RESOLVED** — user deleted `REPO_ROOT/p0_results.zip` on 2026-09-20; verified absent and the integrity check returns CLEAN. Its contents remain tracked under `reports/kaggle_20260920/`. Convention for future archives recorded as D-017 and in `CLAUDE.md` 7b |
| I-014 | **P1 GPU budget.** Computed from `reports/kaggle_20260920/p0_bank_smoke.csv` at 512 px on a T4, replacing the `sd15` warm-up row (2 029.0 ms) with the peer median 769.5 ms: **7.824 s per real image** across the 7 distinct core decoders. Shares: **dcae 48.7 %** (3 813.8 ms), sd35 10.0 %, sdxl 9.9 %, sd15 9.8 %, sd21 9.8 %, flux 9.6 %, realesrgan_x4 2.2 % (169.1 ms). **5 000 images = 10.87 GPU-hours; 10 000 images = 21.73 GPU-hours** | **High for P1 planning.** DC-AE alone is nearly half the budget. 21.7 GPU-h does not fit a single Kaggle session (12 h cap) and consumes most of a typical weekly GPU quota | Decide in P1: shard across sessions with a resumable manifest; consider whether DC-AE needs the same image count as the others; confirm the peer-median correction after the warm-up fix (I-009) | **OPEN** |
| I-015 | `setup_logging` raised `OSError: [Errno 30] Read-only file system` when the repo was mounted under `/kaggle/input`, killing two P0 scripts before any work started | Low severity, high nuisance: it cost a Kaggle session | **RESOLVED** — an unwritable log directory now degrades to console-only and logs a warning naming the directory and the error. Two tests cover both branches. (This defect was listed without an ID at P0 closure; it is numbered here) | **RESOLVED** |
| I-016 | **Kaggle extracts an uploaded zip when it creates the dataset**, so `kaggle_run_p0.md` Cell 1's `*.zip` glob found nothing, the `copytree` fallback was never reached, and `DBP` silently pointed at the read-only `/kaggle/input` | High — it is what triggered I-015, and it fails *silently* | **RESOLVED** — Cell 1 now looks for a zip **or** an extracted `thesis_dbp/` tree, raises with a directory listing if it finds neither, asserts the result is not under `/kaggle/input`, and writes a probe file to prove the tree is writable before going further | **RESOLVED** |
| I-017 | `pip install --quiet -e` hid the real reason the editable install failed, leaving only `error: subprocess-exited-with-error` | Medium — the visible error named nothing actionable | **RESOLVED** — `--quiet` dropped from both installs in `kaggle_bootstrap.sh`; the editable install is now non-fatal and falls back to `PYTHONPATH` with an explanation, since every script already inserts `src/` on `sys.path` itself | **RESOLVED** |
| I-018 | E1, the private Kaggle zero-shot set | — | **WITHDRAWN** by the user at the P1 review: it is not public and has no paper, so it cannot support a claim. **Replaced by E7 DDA-COCO and E8 EvalGEN** (D-032). No reference to E1 remains in the runbooks, data cards or scripts | **CLOSED (withdrawn)** |
| I-019 | E5 (Chameleon) and E6 (MNW) | Low for P1, medium for P2 breadth | **E5 Chameleon: access RESOLVED.** The first author directed the user to a community mirror linked in AIDE issue #1; the archive is in hand (2 683 359 341 bytes, SHA-256 `a7cdadfe3528…`), counts verified against the paper exactly, terms and citation recorded. Closes on successful Kaggle ingestion. **E6 MNW: repository resolved**, terms and citation recorded, blocked on the download route (I-028) | **PARTIALLY RESOLVED** |
| I-020 | **Whether the FLUX.2-klein-4B VAE is the same tensors as the FLUX.2-dev VAE is UNVERIFIED** | Low for correctness, relevant for the claim "this probes the FLUX.2 lineage". The klein config's `_name_or_path` points at FLUX.2-dev and the weight files differ exactly 2x in size, but neither fact is proof | Cannot be settled without FLUX.2-dev access (403). No identity claim is made in either direction. **Recorded observation:** dev's vae is 336 213 556 bytes and klein's is 168 120 878 — a ratio of exactly 2.00, which is *consistent with* fp32 versus half-precision storage of the same architecture. That is an observation, not evidence: two different models of the same shape would show the same ratio. The preflight (Cell 7b) records klein's stored dtype and bytes-per-parameter, which settles klein's half of it and still says nothing about dev's. If access is later granted, compare `decoder_state_dict_sha256()` for both | **OPEN (BLOCKED-ACCESS)** |
| I-021 | **DIV2K train HR holds 800 images, but the P1 plan budgeted T-ctrl at 1 600** | Medium — the control corpus is half the planned size | Verified without downloading: a range request for the archive's end-of-central-directory reports **801 entries** = 800 PNGs + 1 directory. **RESOLVED by D-029**: `--crops-per-image 2` takes two non-overlapping crops, every row carries `source_image_id`, and any split must group by it. My P1 recommendation was to reject this; the user's instruction supersedes it, and the grouping column plus the diagnostic-only status are what make it sound. Effective sample size remains 800 photographs | **RESOLVED** |
| I-022 | The three new bank members carry `refs: [P1-B3]`, which points at this log entry rather than at a bibliography key | Cosmetic, but `refs` elsewhere means `[R1]`..`[R21]` | Deliberate: inventing a citation for Qwen-Image, Kandinsky 2.2 / MoVQ or FLUX.2 without verifying it this session would be exactly the fabrication D-012 forbids. Assign real keys when the thesis reference list is next revised | **OPEN (cosmetic)** |
| I-023 | **The training side has its own compression confound**: T-real (COCO) is entirely JPEG while the B6 SD1.5 generations are PNG that have never been compressed | **High for P2.** This is the same [R7] confound that D-022 fixes on the evaluation side, sitting in the training data. A detector could learn it there and appear to generalise | **RESOLVED by D-030.** `scripts/p1_match_generations.py` writes a JPEG-matched variant of every generation, drawn jointly from one named T-real image and seeded by the generation id; the PNG variant is kept so P2 can report raw-vs-matched as an ablation. Reconstruction pairs stay raw PNG and get the Aligned-Sync treatment at training time instead. DIV2K is the lossless control. 13 unit tests, including one asserting the confound is gone after matching | **RESOLVED (code); PENDING-GPU to run** |

| I-024 | **E7 DDA-COCO's real images are the MSCOCO *validation* set, and T-real is built from COCO** | **High.** If T-real selected from `val2017`, E7's evaluation reals would be inside the training corpus and the primary head-to-head number would be measured on seen images. The failure is silent: nothing errors, the AUC is just too high | Defended twice: `p1_build_reals.py --root` points at the `train2017` images directory (the build runbook locates it and asserts no `val2017` path is reachable), and the pHash leakage check runs E7 against the T-real manifest anyway. B6's prompts already come from `captions_train2017.json`. If the check reports `LEAKAGE`, fix the *training* selection and rebuild — never drop evaluation images | **MITIGATED; verification PENDING-GPU** |
| I-025 | E6 MNW's repository identifier | Medium | **RESOLVED**: `github.com/microsoft/MNW` (Microsoft-Northwestern-WITNESS; mirror `nsail-lab/MNW`), supplied by the user and verified by **direct URL fetch** of the repo metadata and README — not by search, which is why the earlier attempt failed: I had been searching Hugging Face for a GitHub repository. Terms and the IEEE Intelligent Systems citation are now recorded verbatim. **Still blocked on the download route** (I-028) | **RESOLVED** |
| I-026 | **E8 EvalGEN appears to contain no real images** — its README describes ~55 300 images across five generators and names no real source | Medium — AUC, AP and EER are undefined on a single-class set, and `dbp.eval.metrics` raises rather than returning 0.5 | Confirm from the extracted tree. If it holds, E8 is reported either as a detection rate at a threshold calibrated elsewhere (D-026), or paired with E7's COCO-validation reals. Its README also states every image is JPEG q96, so a pairing with reals of a different quality distribution is itself confounded even though neither set is confounded alone — recorded in its data card | **OPEN (PENDING-GPU to confirm)** |
| I-027 | GenImage was offered as an optional comparison set against older baselines | Low | `github.com/GenImage-Dataset/GenImage`, Google Drive, million-scale. **Only if the budget allows after the main build, and only after asking.** Recorded in `kaggle_run_p1_eval.md` as explicitly not to be started unprompted | **OPEN (optional)** |

| I-028 | **MNW has no download URL in its README** | Medium — everything else for E6 is ready | The repository is resolved and the navigator (`microsoft.github.io/MNW/`) browses but does not serve an archive. The user will supply the access route. **If it carries a token or signed URL, that value goes in Kaggle Secrets only** — never in the repository, a log line or a report (§4). MNW is periodically updated, so whatever revision or release the route identifies must be pinned | **OPEN (BLOCKED — needs the route)** |
| I-029 | **Chameleon carries no category labels** | Medium — the review requires per-category reporting (scene / object / animal / human), and the animal split is 3 998 real : 313 fake, which a pooled metric hides completely | The archive is two flat directories of UUID-named files: no manifest, no CSV, no per-category subdirectory, and the AIDE README describes no mapping. Needs either the **official URL list** (likely grouped by category, since the paper's Table 1 was produced somehow) or a mapping file from the authors. Until then Chameleon is reported **in aggregate only**, with the limitation stated wherever the number appears | **OPEN (BLOCKED)** |
| I-030 | **Chameleon's reals are all exactly 500 px wide; a one-line rule `width == 500 -> real` scores 100.00 %** | **High.** 14 863 / 14 863 reals are 500 px wide and exactly 1 of 11 170 fakes is. The reals have been through a normalising pipeline (also uniformly JPEG q75, 4:2:0, non-progressive); the fakes are native | The fixed square crop is the defence: every image enters the model as `crop x crop`, so width is not observable in the pixels. That removes the cue but not the selection effect, which D-037 addresses by moving the crop to 320. Any Chameleon number must state both facts. No detector result on *raw* Chameleon images means anything unless the pipeline is provably blind to image dimensions | **MITIGATED (crop + D-037); must be stated in every Chameleon table** |
| I-031 | ~~**Chameleon's images came from a third-party mirror**~~ **RESOLVED 2026-09-28.** Read from the GitHub API: issue `shilinyan99/AIDE#1`, mirror posted by **`qwqhjcxczx`** on **2025-01-02T03:45:28Z**, Drive id `1QLYJMhy0CbBVT01BLkkw7KPPL5BpmxnH`. The uploader states they *"downloaded the Chameleon dataset using the image URLs provided by the author"* -- so it is a **re-download from the original hosts**, not the authors' files. That is the likely source of the 500-px uniform width and q75 4:2:0: hosts serve resized derivatives, and Unsplash serves width-parameterised JPEGs. **The width confound may therefore belong to the mirror, not to Chameleon**, which cannot be settled without the authors' copy. Every Chameleon result states the mirror, its SHA-256 and that caveat | **RESOLVED** | | Medium — the official release is a URL list; we have an image archive from a community link in AIDE issue #1, as directed by the first author by email | Counts match the paper exactly (14 863 / 11 170), which is reassuring but is not proof the pixels are the authors'. The uniform q75 across all 14 863 reals is consistent with re-encoding somewhere in the chain. The archive SHA-256 pins exactly what was scanned. **The mirror link, uploader and date are still to be pasted in** — the review said they would be supplied. Every Chameleon number is reported as *"from the community mirror, SHA-256 `a7cdadfe…`"* until it can be checked against the official URL list | **OPEN (provenance incomplete)** |
| I-032 | `Chameleon.zip` was placed at `REPO_ROOT/Chameleon.zip`, outside the write scope | Low, but it made the integrity check report `CRITICAL-PROTECTED-FILES-CHANGED` | Exactly the I-013 / D-017 situation repeating with a different archive. Reported rather than acted on: moving or deleting a file outside `thesis_dbp/` is forbidden (§1, §2). The user moved it to `F:\thesis-2\Chameleon.zip` and the check returned **CLEAN**. Large source archives belong outside `REPO_ROOT`, per §7b | **RESOLVED** |

| I-033 | **The bootstrap's torchao probe tested the wrong thing** | Medium -- it cost a preflight cycle | Step [7/7] decided by `import peft`, which succeeds; the failure is later, inside `get_peft_model` -> `dispatch_torchao` -> `is_torchao_available()`, which *raises* for torchao < 0.16 instead of returning False. So the bootstrap printed "peft imports cleanly, so no action is needed" and both LoRA dry-runs then failed. The probe now calls `is_torchao_available()` itself and prints the uninstall plus a restart instruction. Uninstalling torchao fixed both tests | **RESOLVED** |
| I-034 | **Kaggle now mounts inputs under `/kaggle/input/datasets/<owner>/<slug>/`** | Medium -- every hard-coded `/kaggle/input/<slug>` path in the runbooks is wrong on that layout, and one attempt ran the code straight from the read-only mount, which failed on `outputs/p0` | The runbooks now locate their inputs by searching rather than assuming, and the staging cell keeps its `assert not str(DBP).startswith("/kaggle/input")`, which is what caught the read-only run | **RESOLVED (runbooks updated)** |
| I-035 | **An interactive Kaggle session was lost and took the build with it** | Medium -- T-real was rebuilt twice and a 1.8 h B5 run was lost | A Quick Save taken *during* B5 captured a partial tree, and the work done after it was never saved. T-real rebuilt byte-identically afterwards, which is the reassuring part: the selection is deterministic. Long stages now run through **Save & Run All (Commit)**, which runs headless for up to 12 h and saves its output at the end | **RESOLVED (workflow changed)** |
| I-036 | **83 of T-real's 2,500 images are greyscale — 3.3% of the corpus** | Was high, now measured and contained | Found 2026-09-26 by `p1_rescan_greyscale.py`: **11 single-channel sources** (the ones `unknown(-1)` exposed, which broke 13 matched variants) and **72 `rgb_equal`** — greyscale pictures inside three-channel JPEG containers, which no format inspection can see. The second kind is 87% of the problem, so filtering on the subsampling label alone would have left 72 in place. 82 of the 83 have a channel spread of exactly 0. T-real is now **2,417 usable rows**; 996 of the 49,200 pair rows are filtered by the join. The matched set rebuilt to **2,488 of 2,488, 0 failures**. **T-ctrl has not been scanned** | **RESOLVED for T-real; T-ctrl scan outstanding** |

| I-037 | **The greyscale re-scan marked 1,594 colour DIV2K rows as greyscale** | High if it had gone unnoticed: T-ctrl would have been 6 usable rows out of 1,600, and the control set that proves the model is not leaning on JPEG structure would have been destroyed | My defect. `scan()` asked whether the chroma subsampling label was mappable **without first asking whether the row was a JPEG at all**. Every PNG row carries `not-jpeg`, the expected value, so all 1,600 looked unreadable; `excluded_ids` then took every finding rather than only the greyscale ones. T-real was unaffected — all of its rows are JPEG and its report shows `unmappable_but_colour: 0` — so the 83 exclusions there stand. Fixed: mappability is only considered for JPEG rows, exclusion follows the pixels alone, a colour row with an unreadable chroma setting is reported and kept, and the counters are seeded so a zero cannot read as "not measured". Four tests, one of them end-to-end on a PNG corpus. **The T-ctrl manifest written by the broken run must be discarded and the scan re-run** | **RESOLVED (re-run outstanding)** |

| I-038 | **E7's FLUX.1 subset is separable by chroma subsampling alone** | High for that subset: it is one of two carrying the head-to-head claim against SEF | 4,971 of 4,971 FLUX.1 fakes are 4:2:0; 0 of 5,000 reals are. A one-line rule scores 100.00%, lift +49.85 points. It is how DDA-COCO was published, not a property of FLUX's VAE, and it **cannot be removed** — chroma subsampling is destructive. Handled by caveat (D-042): FLUX.1 numbers are reported as an upper bound, never as the headline, and the other five subsets — all 4:4:4, matching the reals — carry the comparison | **OPEN (mitigated by disclosure; not removable)** |

| I-039 | **Two greyscale reals aborted the whole E4 build** | Medium: E4 produced a manifest but no summary, no matched view and no leakage entry, and E4 is the most confounded set in the suite (+25 lift) | The same defect as the 13 generation failures, in the other matched-view builder. I fixed the reference set in `p1_match_generations.py` under D-039 and **did not carry it into `p1_build_eval.py`**, which has its own. 2 of E4's 1,251 reals are greyscale JPEGs; the seeded draw picked one, `subsampling_code` raised, and the run died *after* writing the manifest. Fixed: the evaluation reference skips unmappable rows and counts them, the images stay in the set, and `MAPPABLE_SUBSAMPLING` now lives in `dbp.data.jpeg` so the two call sites cannot drift apart again. Regression test builds the exact fixture (6 colour reals, 2 greyscale, 6 PNG fakes) | **RESOLVED** |

| I-040 | **Two different fp16 parity numbers for `realesrgan_x4`: 59.862/59.554 dB (P0) and 44.94 dB (P1)** | Low as it turned out -- the conservative number won and the build is correct -- but it cost a review query, which is its own cost | Not a contradiction: different scripts, images and crops. **P0** measures parity in `p0_bank_smoke_test.py` on skimage photographs at crops 512 and 1024. **P1** measures it in `p1_throughput_bench.py` on `synthetic_image(384, seed=2026)`, a smooth sinusoidal field, at the build crop. A GAN super-resolver's fp16 behaviour is content-dependent, so the two disagree by ~15 dB. The **P1 number decided the build**, via `fp16_build_allowed` in `p1_throughput.csv`, and the pair manifests confirm the outcome: `realesrgan_x4`, `movq_k22` and `sdxl` are **fp32 for all 4,100 rows each**. No rebuild needed. The CSV now records `fp16_parity_source` so the two numbers can never again look like the same measurement disagreeing with itself | **RESOLVED (no rebuild; provenance column added)** |

| I-041 | **Two budget inputs are derived, not measured: the S2 training-step time and the S3 LPIPS cost** | Medium. The P2 budget fits inside 32 GPU-h with +1.56 h of headroom, and both numbers sit inside that margin. If either is materially worse than the arithmetic suggests, the reduction ladder has to bite mid-session | The S2 step time is extrapolated from a **batch-1 forward** (56.86 ms at input 336, `p0_backbone_smoke.json`) times an assumed 3.5x for backward plus checkpoint recompute; batching should beat it comfortably, but nothing measured says so. The LPIPS-vgg cost of 12 forwards per image was never measured at all. Both are now measured **before** their full run: `p2_train_s2.py --calibrate` times 20 real steps and D-047's floor stops the run if the allowance buys too few; `p2_probe_features.py --calibrate` projects the full S3 cost from 50 images and the runbook says to stop and report above 11 GPU-h | **OPEN -- measured at the start of each B4/B5 run** |
| I-042 | **The raise to GPU_BUDGET_HOURS = 32 assumes Kaggle charges a T4 x2 session at session time, not at GPU time** | Medium. If the quota meter charges per GPU, the effective budget halves and two of the four S2 runs do not fit | Not verifiable without running a session. The runbook requires the quota meter to be read after the first GPU session and the actual charge reported against the estimate, before the remaining runs are launched | **CLOSED 2026-10-17: Kaggle bills session wall time, not per GPU** (LOLO sessions 1-4: about 6.7 h of measured wall time vs 6h57m on the meter; per-GPU billing would be at least 12.4 h). See the P3-C entry of 2026-10-17 |

| I-043 | **A flagged real's twelve reconstructions were still reaching the training lists** | High, and it would have shown up twice. D-039 exists because a greyscale image in one class and not the other is the cheapest feature in the corpus; keeping 1,068 greyscale reconstructions as training fakes while dropping their 89 greyscale reals reinstates exactly that. Separately, `p2_train_s2.py` would have raised `KeyError` the first time it sampled one, because the real is absent from `reals_index` -- mid-training, after the GPU time was spent | `p2_build_image_lists.py` and `p2_train_s2.py` both filtered the *reals* manifest on `excluded_reason` and neither filtered pair rows by their source real. The arithmetic gave it away: the list reported 55,699 rows against the split's 55,788, a difference of exactly 89 -- the flagged reals and **not** their 1,068 reconstructions. 49,200 - 1,068 = 48,132, which is the usable-pair count P1 recorded, so the right number was already on record. Both scripts now exclude them and report how many; a test reinstates the defect and fails on it | **RESOLVED** |

| I-044 | **The panel's selection was frozen but its pixels were not** | High. 7,750 of the 12,150 panel rows could not be opened in any session other than the one that built their view, and the discovery point would have been Session B's first GPU cell | Each evaluation view's manifest records `relative_path` against the root P1 extracted it to (`/tmp/e7x/DDA-COCO`, `/tmp/e8x`, `/tmp/e3`, `/tmp/wildrf/WildRF`) and, for a derived view, an absolute `view_path` into that session's `/kaggle/working`. None of those corpora was ever saved as a Kaggle dataset. Resolved by D-050: one CPU session re-downloads them, re-derives the two matched views, and materialises the panel into a verified directory published as `dbp-p2-panel`. Consequence recorded: the image lists needed a `root` column, because the corpus and the panel are two directories and a single `--root` resolved only one of them | **RESOLVED (D-050)** |
| I-045 | **The DIV2K lossless diagnostic was 78% in-sample** | High for that secondary report, none for the gate. §7 of the pre-registration attaches no gate to the slice, so M1 and M2 were never affected; but 313 of its 400 rows sat in the training split and all 400 were in the S1 feature cache, so a strong score there would have been partly memorisation | Measured against the manifests: the panel's `view` column records the split each row came from. Resolved by D-051 -- T-ctrl is withheld from every training list, so the same 400 rows are now fully out-of-sample. The panel and its digest are unchanged | **RESOLVED (D-051)** |
| I-046 | **E4 carries 25 byte-identical duplicate groups inside its 1,500 panel rows** | Low. All 25 are within one view *and* one label -- 0 cross-label, 0 cross-view -- so nothing leaks between classes; 50 rows (3.3%) are weighted twice | Already recorded by P1: the full 2,503-row set has 51 exact groups, and these pairs appear in its `near_duplicate_groups_within_set`. Kept as distributed per D-053, with the caveat printed beside every E4 number and a deduplicated sensitivity table alongside | **ACCEPTED (D-053)** |
| I-047 | **375 of the 12,150 panel rows were too small for the pre-registered 320 px evaluation crop** | High, and it would have been higher later: S1, S2 inference and S3 all centre-crop to 320, so the panel could never have been scored. Discovered as a crash 640 images into the S1 panel pass, after the S1 *train* cache had already been built | The panel sampler never checked that a selected image survives the crop. Shorter sides ran down to 150 px: e3 23, e4 8, e5 252, e7 92, e8 0. Resolved by D-054 -- drop them, label-blind, no backfill -- with the per-view x per-label counts and the E3 class shift (23 reals, 0 fakes) declared in pre-registration Amendment 1. `draw` now enforces the rule for every future panel, with a test that fails against the old behaviour | **RESOLVED (D-054)** |
| I-048 | **My crop-384 -> crop-320 cost scaling was wrong by 1.226x** | Medium. It put the P2 S3 estimate at 10.09 GPU-h when the measured figure is 12.90, which triggered the 11 h guard mid-session and forced a budget decision under time pressure | I scaled P1's per-member medians by pixel area, `(320/384)^2 = 0.694`. Decode cost does not scale with pixel count: fixed per-call overhead does not shrink. Two members measured at 300 images agree (sd15 1.288x, movq_k22 1.251x against the area-scaled prediction), so it is systematic, not noise. Any future estimate at a new crop must be measured, not scaled | **ACCEPTED -- measure, do not scale** |
| I-049 | **`spandrel` missing on Kaggle: one shard died, the other ran 3.5 h alone** | High: 3.5 GPU-hours of one T4 idle, and three members unfinished | My recovery cell pip-installed `lpips` alone instead of running `kaggle_bootstrap.sh`, which installs all of `requirements/base.txt`. Shard 0 failed on `realesrgan_x4` at 23:05 and the cell had no fail-fast. Resolved by D-057/D-059: the bootstrap is the only environment step, pre-flight smoke-tests every member the run uses, the runner stops every job at the first failure | **RESOLVED (D-057)** |
| I-050 | **The persist cell failed a ten-hour commit** | High: the run was marked failed; the output survived only because Kaggle happened to keep it | My Cell 8 imported `dbp` on the line before inserting `src/` on `sys.path`. Resolved by D-057: cells never import `dbp`, status and logs are written by the work step itself, the report step always exits 0 | **RESOLVED (D-057)** |
| I-051 | **The working tree drifted to CRLF; the zip would have shipped it** | High, caught before reaching Kaggle | Python `write_text` on Windows writes CRLF; my patch scripts used it. Caught while building the dry-run, because the bootstrap is bash. Resolved by D-058 | **RESOLVED (D-058)** |
| I-052 | **The recovery bundle would have stopped Cell 3 on Kaggle** | Medium, caught locally | The first bundle I staged contained `p1_gen_matched_manifest.csv`, which is committed and differs (the failed run's matcher rewrote its paths); the restore refuses to overwrite a committed report with different content. Caught by running the restore and the digest pre-flight against the real bundle with the real expectations, which the fixture alone does not exercise. The bundle now carries `p1_gen_manifest.csv` instead | **RESOLVED** |
| I-053 | **Two S3 jobs wrote one timing file** | Low in effect, high in kind: no feature was touched, but the measured cost of `qwen_image` and `realesrgan_x4` survived only in `probe_job1.log` | Both recovery jobs ran `p2_probe_features.py` as `shard 0/1`, so both wrote `reports/p2_probe_timing_shard0.json`; GPU 0 finished last and replaced GPU 1's record. The audit found the same defect waiting in Session E: `p2_fit_heads.py`, `p2_score.py` and `p2_analyze.py` keyed their outputs by protocol only, so `p_a_sd1_seed1` would have overwritten `p_a_sd1_seed0`'s heads and fusions; and `setup_logging` named files by the second, which two jobs started together share. Fix: D-061 (`--tag job<i>_gpu<k>` from the runner; run-id keyed paths; pid in log names). Paths checked: `p2_probe_features.py` (timing, caches, sidecars), `p2_run_shards.py` (job logs, status), `p2_train_s2.py` (checkpoint, LoRA, head, report, tmp files), `p2_fit_heads.py` (S1 probe, S3-LR, S3-Set, heads report), `p2_score.py` (scores, summary), `p2_analyze.py` (metrics, gate, headline, complementarity), `p2_preflight.py` (per-stage report), `p2_report.py` (archive name from the caller), `dbp.utils.logging` (log files) | **RESOLVED (D-061)**; regression test `tests/test_output_paths_unique.py` (would have failed on the old runner: no `--tag`, and both jobs' timing names equal) |
| I-054 | **Two numpy scalars in the S2 checkpoint would have stopped every resume** | High, caught locally | The LR lambda returned `np.float64`, which reaches the optimiser's and scheduler's state, and the validation AUC was a numpy scalar; `weights_only=True` refuses both. The per-GPU smoke pre-flight (save, reload, compare) caught it on its first local run, before any Kaggle time. Both are now plain floats | **RESOLVED**; caught and re-checked by the smoke pre-flight in every dry-run |
| I-055 | **Validation crops were seeded with `hash(pair_id)`** | Medium: Python randomises `str` hashes per process, so every run, every seed and every resumed session validated on different crops, and the curves -- and the model selection made on them -- were not comparable | Found in the step 3 audit; no S2 run had trained. Replaced by a blake2b seed | **RESOLVED**; `test_the_validation_crop_seed_is_the_same_in_every_process` |
| I-056 | **Session E: `p2_fit_heads.py` aligns streams by length, not by image id; `p2_score.py` reads per-view panel files that do not exist** | High for Session E, none for D | S1's validation set is the 6,883 S1-list rows, S3's is the 2,000 probe rows, and S2's is restricted to probe ids; streams whose validation arrays differ in length from S1's are silently dropped from both fusions, so as written the fusions would exclude S3 and S2 -- and two arrays of equal length in different orders would be fused row-misaligned. `p2_score.py` looks for `p2_panel_<set>.csv`; the panel is one list, `p2_panel_images.csv`, with a `view` column | **OPEN -- fix before Session E**; no pre-registration change is needed to align by id |
| I-057 | **The code upload's slug was `dbp-code-ed7a8cb`; Cell 1 looked for `dbp-code`** | None in cost: found by listing `/kaggle/input` on a CPU session before any GPU time | A new dataset per commit gets its own slug. Cell 1 now matches `dbp-code*` and still requires exactly one match; the S2 fixture uses a suffixed slug so the dry-run exercises the pattern | **RESOLVED**; `test_cell_1_finds_the_code_dataset_under_either_slug` |
| I-058 | **The corpus dataset titled `p1-data` has the slug `report`** | None in cost: found by running Cell 1 on a CPU session | Cell 1 looked for `datasets/*/p1-data/...`. It now finds `p1_reals/` and `p1_generated/` under any dataset slug, still requiring exactly one match; the corpus counts and every `gen_matched` hash in pre-flight 1 prove it is the right data | **RESOLVED**; `test_cell_1_finds_the_corpus_under_any_slug` |
| I-059 | **S2's in-training validation saturates, so early stopping selects on differences of 1e-5** | Medium, for the thesis's account of S2, not for the gate | The trainer validates on at most 512 val pairs of the protocol's own lineages; all four runs pass 0.999 by step 1,000 and stop on the patience rule between steps 4,500 and 5,500. The selected step is therefore close to arbitrary within a flat plateau. The gate does not use this number (it uses the panel), and every run is far above the 2,000-step floor, but the thesis must not present the in-training AUC as a measure of S2 | **OPEN -- report as a limitation**; no change to a finished run |
| I-060 | **The fusion check stops P-A: on the protocol's own validation set S3-LR separates real from fake perfectly and takes 97-98% of the F-stack weight, while on every unseen decoder its validation AUC is below 0.5** | High: the pre-registered F-stack and best-single choice are both fitted on data where one head is perfect, so the gate would compare F-stack with the stream it is made of | P-A validation is one lineage (SD1) by design. S3-LR fits it exactly and anti-generalises (dcae 0.44, movq 0.42, sdxl 0.16, flux 0.22, flux2 0.30 on the full validation set); S3-Set generalises (0.84-0.999). `4ch_only` and `full_bank` pass the check (S2 holds 56% and 53%) | **RESOLVED (procedure) by Amendment 2 / D-069**, committed before any panel score; the finding itself is reported in P2 as exploratory lineage-transfer evidence |
| I-061 | **Two analysis defects found in the I-056 audit**: E7 and E3 slices hold no reals (their reals are a slice of their own), so their AUC is undefined and they would silently leave M2; and the gate's bootstrap resamples a *pooled* TPR/AUC, not the macro M1/M2 the point estimate is | High for the verdict | Found reading `p2_analyze.py`; no panel score exists | **FIXED (P2-E part 3)**: shared real class per view; `macro_bootstrap` resamples slice fakes and view real pools and recomputes the macros; `tests/test_session_e_analysis.py` |
| I-062 | **Thresholds were not fitted on validation reals alone**: `calibrate_at_fpr` takes the threshold with the best validation TPR under the FPR budget, so where a head separates validation perfectly the threshold is the lowest validation *fake*. P-A S2's "5% FPR" threshold ran at 0.20% FPR on the validation reals (seed 1: 0.61%), S3-LR's at 0.00% | High for M1: single heads would be judged at stricter operating points than the fusion, which flatters the fusion | Found re-reading the trial calibrations (`achieved_fpr`) while writing Amendment 2; no panel score exists | **FIXED (P2-E part 3)**: `calibrate_on_reals`; every threshold now sits at 4.89% / 0.81% FPR on the 491 validation reals; `p2_analyze` refuses any other rule; `tests/test_amendment2.py` |
| I-063 | **The fusions had no thresholds**: `p2_fit_heads.py` calibrated the five single heads only, and the old `p2_analyze.py` read `thresholds.get("f_stack", {})`, so F-stack's TPR@5%FPR -- and with it the M1 half of the gate -- would have been `None` and printed as n/a | High: M1 could never have passed | Found wiring F-mean(A2) into the fit | **FIXED (P2-E part 3)**: every fusion is calibrated on the validation reals like the heads |
| I-064 | **The validation thresholds do not hold on the panel's real images**: at the 5%-FPR threshold fixed on the 491 validation reals (4.89% there), the share of panel reals called fake is ~4-6% on E7's COCO reals but 49-84% on E3, E4 and E5 (P-A seed 0: S2 68.8 / 83.0 / 83.9, F-mean(A2) 74.2 / 75.9 / 79.3) | High for interpretation: M1 is TPR at the validation threshold, not TPR at 5% FPR on the panel; on views whose reals differ from the training reals it largely measures the shift. The verdict is unaffected (M2 fails by more) | Measured in the one-time analysis (`metrics.csv`, `fpr_at_5pct_threshold`) | **OPEN -- the calibration problem the P3 reliability gate must solve; reported in P2 section 3** |


## 6. Prompt entries (append-only)

### P0 — Environment setup & scaffold — 2026-09-19

- **Goal:** prove that every bank member and the backbone load and run correctly, deterministically and within T4 limits, before any data is built; and establish the scaffold, guardrails and reproducibility record.

- **Files created:** 36 files under `thesis_dbp/`, plus `REPO_ROOT/.gitignore` and `REPO_ROOT/CLAUDE.md`. Full tree in the P0 REPORT.

- **Files modified outside the scope lock:** none. `.gitignore` and `CLAUDE.md` did not previously exist and were created under the two approved exceptions.

- **Dependency changes:** clean venv, so everything was a fresh install. `torch 2.11.0+cpu` and `torchvision 0.26.0+cpu` installed first by exact version from the official PyTorch CPU index; a `pip --dry-run` report then confirmed the remaining resolution would not touch `torch`/`torchvision`/`numpy`, and a before/after comparison confirmed it did not. `pip check`: no broken requirements. The **system** Python was never written to.

- **Commands run (key ones):**
  - `git init -b main`; `git config --local user.name/user.email` (global config untouched)
  - `python -m venv thesis_dbp/.venv`
  - `pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cpu`
  - `pip install --dry-run --report` guard → `pip install -r requirements/dev.txt` → `pip install -e thesis_dbp`
  - `scripts/p0_env_check.py`
  - `scripts/p0_bank_smoke_test.py --device cpu --crops 512,1024 --members sd15,realesrgan_x4 --no-lpips`
  - `scripts/p0_backbone_smoke_test.py --device cpu --backbones dinov2-base`
  - `scripts/p0_verify_integrity.py --self-test`
  - `pytest -q -rs thesis_dbp/tests`; `cd thesis_dbp && ruff check .`

- **Results (real numbers only):**
  - diffusers API check against the installed 0.40.0: `AutoencoderKLOutput.latent_dist` ✓, `DiagonalGaussianDistribution.mode` ✓, `DecoderOutput.sample` ✓, `EncoderOutput.latent` ✓ — **no deviation from §6.4 was required**.
  - Bank smoke, 4/4 rows OK — see §3.
  - Residual PNG triplets written for both members at both crops (12 files). Visual check of `sd15_512_residual_x10.png`: energy concentrated on hair, fabric and edges, near-zero on flat background, and **no tiling grid seams**, independently confirming tiling is off.
  - Backbone smoke, real `dinov2-base` weights: 86.6 M parameters, 577 tokens at 336×336 (576 patch + 1 CLS, no registers), attention projections discovered as `['key', 'query', 'value']`, LoRA r=8 giving **442 368 trainable of 87 022 848 (0.5083 %)**, one forward pass after attachment. Status OK.
  - Integrity: 0 files checked, 0 changed, verdict CLEAN, self-test byte-identical.

- **Tests:** `85 passed, 7 skipped, 0 failed` (92 collected) after the P0 review additions; `65 passed, 7 skipped` at the end of the original P0 build. All 7 skips are in `test_gpu_smoke.py` with the explicit reason *"no CUDA device in this runtime; run on Kaggle T4 (PENDING-GPU)"*. Both network tests passed, including the real sd15 VAE reconstruction and the pinned-revision resolution guard. `ruff check .`: all checks passed.

- **Deviations from the prompt (what & why):**
  1. **B0 split into two commits.** `P0-B0a` puts `.gitignore` and `.gitattributes` on `main` before branching, so `main` has a commit to diff against in B9. The §5 ordering would have left `main` empty.
  2. **`thesis_dbp/.gitattributes` added.** The system-level `core.autocrlf=true` would otherwise give `kaggle_bootstrap.sh` CRLF endings and break it on Kaggle. Approved in place of touching global git config.
  3. **`__init__.py` × 4, `.gitkeep` × 3.** Required for the src layout to import and for git to track empty directories. Approved.
  4. **`outputs/.gitkeep` force-added**, since `outputs/` is git-ignored. Keeps `.gitignore` strictly append-only rather than adding a negation rule.
  5. **`center_crop` and `residual_int16` added to `utils/io.py`** beyond the three functions named in §5. §6.4 needs exact 512/1024 crops and an `int16` residual; putting them elsewhere would split the protocol across modules.
  6. **`reports/p0_decoder_identity.json`** added as a second output of the bank smoke test, holding the §6.3 byte-identical-decoder finding, which does not fit a one-row-per-member-per-crop CSV.
  7. **`tests/test_backbone_cpu.py` added** (9 tests) beyond the four test files in §5. The LoRA attach is the most fragile part of P2 and it can be regression-tested offline against a tiny randomly-initialised DINOv2; leaving it untested until a T4 session was the worse option.
  8. **`[tool.ruff] cache-dir`** added to `pyproject.toml` after the integrity check caught a stray `REPO_ROOT/.ruff_cache/` (see I-007).
  9. **The measured disk estimate replaces the §6.3 figure**: ≈ **4.9 GB** of weights, not 2–3 GB (DC-AE alone is 1.25 GB), and ≈ **8 GB** local peak including the venv.
  10. **Only 2 of 10 bank members were measured locally**, because of the 0.12 MB/s link (I-002). This is a reduction in local coverage, not in deliverables: the remaining 8 are measured on Kaggle, which is where every reported number was always going to come from (D-008).

- **Pending items:**
  - **PENDING-GPU:** fp16 parity for all members (and the expected SDXL fp16 overflow), `peak_gpu_mb`, T4 `ms/img`, any `OOM@1024`, the fp16-autocast backbone forward, and 7 skipped GPU tests.
  - **NOT-RUN locally (bandwidth):** `sd21`, `sd_ft_ema`, `sd_ft_mse`, `sdxl`, `sdxl_fp16fix`, `sd35`, `flux`, `dcae`; LPIPS for every member (the 553 MB VGG16 was not downloaded locally); the byte-identical-decoder comparison; `dinov3-vitb16` (access granted, weights not fetched).
  - **BLOCKED-ACCESS:** none. All three gated repositories are accessible.
  - Runner: `env/kaggle_run_p0.md`.

- **Next step:** run `env/kaggle_run_p0.md` on a Kaggle T4 and send back `reports/*` plus the Cell 5 pin output and the pytest and ruff output, so every `PENDING-GPU` and `NOT-RUN` marker in §3 can be replaced with a measurement. Then P1 (dataset construction).

- **Merge gate (user instruction, P0 review):** `p0-env-setup` **must not be merged yet**. The order is: (1) Kaggle P0 results pasted back, (2) §1 and §3 of this log updated with the real measurements, (3) *then* fast-forward `main` to `p0-env-setup` and tag `p0-complete`. `main` is deliberately left at `62d33a2` until then, so the tag marks a P0 whose numbers are actually filled in rather than one full of `PENDING-GPU`.

#### P0 correction record

A genuine error was made and caught during P0, recorded because it is exactly the class of fault these guardrails exist to prevent:

> The Phase A audit printed only **12-character** commit-SHA prefixes. When writing `configs/decoder_bank.yaml`, the remaining 28 characters of each 40-character revision were filled in from memory rather than fetched. **Five of nine** pins (`sd21`, `sd_ft_ema`, `sdxl`, `flux`, `dcae`) were therefore wrong and pointed at commits that do not exist. It was caught by comparing the `flux` prefix in the YAML against the audit output, then corrected by re-fetching all nine from the Hub and re-verifying that each resolves.
>
> Permanent guard: `tests/test_registry.py::test_every_pinned_revision_actually_resolves_on_the_hub` asserts each pin *exists* — not that it equals HEAD, since the point of a pin is not to follow upstream. Recorded as decision **D-007**.

A second, smaller scope violation was also caught and fixed: running `ruff` from `REPO_ROOT` created `REPO_ROOT/.ruff_cache/`, which `p0_verify_integrity.py` flagged as `CRITICAL-PROTECTED-FILES-CHANGED`. Removed, cache pinned inside `thesis_dbp/`, documented in `CLAUDE.md` (I-007, D-011). The integrity check working as designed on a real violation is itself a useful result.

### P0-K — Kaggle T4 measurement run — 2026-09-20

- **Goal:** replace every `PENDING-GPU` and `NOT-RUN` marker from the P0 build with a real T4 measurement.

- **Provenance:** run from commit `758e4c1fb636fb1947226c16734e183a916c87e3`, recovered on Kaggle from the embedded `COMMIT.txt` and recorded as `git_commit` with `git_commit_source: "COMMIT.txt"` in `env_report.json`. The chain built in the P0 review worked in production on its first real use.

- **Environment:** Kaggle, Python 3.12.13, 2x Tesla T4 (sm_75, 14 912 MB, driver 580.159.04), CUDA 12.8, cuDNN 91002, torch 2.10.0+cu128, 33.7 GB RAM. Caches on `/tmp` (1 100 GB free) leaving `/kaggle/working` at 20.9 of 21.0 GB free, as designed.

- **Guardrail #4 held.** The bootstrap dry-run determined only `lpips` and `spandrel` were missing, and before/after comparison confirmed `torch==2.10.0+cu128 torchvision==0.25.0+cu128 numpy==2.0.2` unchanged.

- **Results:** all in §3 above. 20/20 bank rows OK, both backbones OK, no OOM at 1024, every row bit-identical on re-run. Raw artefacts tracked under `reports/kaggle_20260920/`.

- **Tests:** `91 passed, 1 failed` of 92 on the T4 — every previously skipped GPU test ran. The single failure was `test_t4_has_fp16_but_no_native_bf16`, a wrong assertion of mine, not an environment problem (I-012). `ruff`: all checks passed. Integrity: CLEAN, self-test byte-identical.

- **Four defects found by running for real** (none of which local CPU testing could have surfaced):
  1. `kaggle_run_p0.md` assumed an uploaded `.zip` stays a `.zip`. **Kaggle auto-extracts dataset zips**, so the glob found nothing and the run started from the read-only `/kaggle/input` mount.
  2. `setup_logging` hard-crashes on a read-only log directory instead of falling back to console-only, which is how defect 1 surfaced — twice.
  3. `kaggle_bootstrap.sh` passes `--quiet` to `pip install -e`, swallowing the real error and costing a diagnostic round-trip.
  4. Timing has no warmup pass, so the first measured member on a GPU absorbs CUDA initialisation (I-009).

- **Next step:** P1 (dataset construction), with the leave-one-lineage-out split treating `sd21` and `sd_ft_mse` as a single lineage, `sdxl` restricted to fp32, and the JPEG-augmentation runtime fixed to Kaggle (I-010).

### P0 closure — 2026-09-20

Merge gate satisfied and P0 closed.

- `REPO_ROOT/p0_results.zip` deleted by the user; verified absent (I-013).
- Gate at closure: `ruff` all checks passed; `85 passed, 7 skipped, 0 failed` (92 collected), every skip carrying the explicit reason *"no CUDA device in this runtime; run on Kaggle T4"*; protected integrity **0 files checked, 0 changed, CLEAN**, algorithm self-test byte-identical.
- `main` fast-forwarded from `62d33a2` to `p0-env-setup`; tag **`p0-complete`** applied.
- Carried into P1 as open issues: I-002 (local bandwidth, mitigated), I-003 (transformers major-version split), I-004 (DC-AE licence tag), I-005 (Python 3.13 vs 3.12), I-006 (empty protected set until `prior_results/` lands), I-009 (timing warm-up), I-010 (libjpeg 8.0 vs 6.2 plus the q=75 byte comparison), I-011 (Kaggle `torchao` conflict, fix pending in the bootstrap), I-014 (P1 GPU budget).
- Four defects found by running on Kaggle remain to be patched in P1's first commit: zip auto-extraction in `kaggle_run_p0.md`, `setup_logging` on a read-only directory, `--quiet` hiding the editable-install error, and the missing timing warm-up.

### P1 — Data Foundation & Evaluation Suite — 2026-09-20

Branch `p1-data-foundation`, from `main` @ `bd3f61c`. **Local build complete; every
GPU and dataset number is `PENDING-GPU`** — P1 writes the machinery, the Kaggle runbooks
execute it. Phase A was delivered and approved; the six blocking questions were answered
and the answers are implemented as recorded below.

**Answers as implemented.** Issue numbering kept as-is (the prompt's "I-013" = this log's
I-014, not renumbered). E1 excluded and listed pending — `KAGGLE_ZS_DATASET` was left as
the literal placeholder (I-018). E5 and E6 excluded on the same rule (I-019). FLUX.2
represented by `FLUX.2-klein-4B`, not `-dev` (D-018, I-020). Network allow-list and its
four conditions appended to `CLAUDE.md` as §6a. `alias_of` implemented (D-020). E2
deferred to P2 with its own audit. The JPEG-matched-view rule generalised to every E-set
(D-022, §6b). Commercial-use constraints recorded (D-023, §6c).

**B3 — bank extension.** 10 → 13 members, 9 → 12 distinct, 5 → 8 lineages. Schema
version 2. Every identifier below was produced by a command in this session:

| key | repo | revision | licence | gated |
|---|---|---|---|---|
| flux2_klein | `black-forest-labs/FLUX.2-klein-4B` (`vae`) | `e7b7dc27f91deacad38e78976d1f2b499d76a294` | apache-2.0 | no |
| qwen_image | `Qwen/Qwen-Image` (`vae`) | `75e0b4be04f60ec59a75f475837eced720f823b6` | apache-2.0 | no |
| movq_k22 | `kandinsky-community/kandinsky-2-2-decoder` (`movq`) | `9ae140d347fed8ce6e8bb3005dcc1f48543bb8e3` | apache-2.0 | no |

`scripts/p0_verify_pins.py`: **15 pins, 0 failed, 0 blocked.**

**The regression guarantee, in two layers.** The refactor touched
`reconstruct.py`/`loaders.py`/`registry.py`, so "nothing moved" had to be demonstrated
rather than asserted:

1. *Offline, committed.* `tests/test_reconstruct_regression.py` pins SHA-256 digests of
   reconstructions captured from the **pre-change** code (commit `bd3f61c`) through
   seeded tiny models, for all three pre-existing kinds plus the model-free helpers.
   Re-run after every edit: **identical**. It also pins each pre-P1 member's lineage and
   revision, so adding kinds cannot reshuffle the leave-one-lineage-out groups.
2. *Real weights, on the T4.* `p0_bank_smoke_test.py --compare-to` re-measures the ten P0
   members and requires `psnr_db`, `lpips_vgg`, `mean_abs_residual` and
   `max_abs_delta_rerun` to match `reports/kaggle_20260920/p0_bank_smoke.csv` exactly.
   Timing and memory columns are excluded — they measure the machine, not the protocol,
   and they are *expected* to move now that warm-ups are discarded. **PENDING-GPU**, and
   it is the gate on the whole build run.

**B8 — libraries, zero new dependencies.** `dbp.data.{phash,jpeg,manifest}` and
`dbp.eval.{metrics,thresholds}`. The pHash is scipy-DCT rather than `imagehash` (D-025).
The JPEG quality estimator was validated rather than trusted: it recovers Pillow's own
setting **exactly** at q ∈ {30,50,60,75,85,90,95} × every chroma mode. Metrics are
checked against scikit-learn including heavy ties, a constant score and a single
positive; `summarise()` always reports ranking and a committed operating point together,
with a test asserting the FIRE pattern is visible (AUC 1.0 while FPR = 1.0 at 0.5).

**B1/B2/B4/B5/B6/B7 — six scripts.** `p0_verify_pins.py` gained `require_pins()`, so
CLAUDE.md §7a is enforced by a call every build script makes rather than by convention.
B5 iterates `distinct_members()`, is member-major, chunked and resumable, and shards
members round-robin across two GPUs. B6 generates from the `sd15` member's own pin with
COCO captions, per-image seeds and the safety checker enabled. B7 indexes a set, builds
the `label × compression` table, decides the confound and builds the matched view.

**B2, the half that could run here.** `reports/p1_jpeg_equivalence.json`: 24 cells,
Pillow 12.3.0 / libjpeg 8.0 / Windows-11, quality round-trip exact at every tested
quality. The Kaggle half and the verdict are PENDING-GPU (I-010).

**Measured this session, by command, without downloading image data:**

- E3 (`1498eead24292a9a2d134476b2c559193b68b9de`, apache-2.0, not gated): 4 356 files =
  2 178 images × 2 views; 726 per label; Texture 648 / Structure 939 / Hybrid 591. `Raw`
  format × label: real **726 JPEG, 0 PNG**; fake **1 361 PNG, 91 JPEG**. The prompt's
  "2 904 BMP 320×320" is wrong in every particular (D-021).
- E4 WildRF: `barcavia/RealTime-DeepfakeDetection-in-the-RealWorld`, licence
  `NOASSERTION`, Drive id `1A0xoL44Yg68ixd-FuIJn2VC4vdZ6M2gn`, layout
  `test/{reddit,twitter,facebook}/{0_real,1_fake}`. **Test split only.**
- DIV2K `DIV2K_train_HR.zip`: HTTP 200, **3 530 603 713 bytes**, last modified
  2017-02-14, **801 zip entries = 800 PNGs + 1 directory** (I-021).
- COCO Kaggle mirror `awsaf49/coco-2017-dataset`: page returns 200.

**Deferred P0 defects, all patched** (I-015 read-only logging, I-016 zip
auto-extraction, I-017 `--quiet`, I-009 warm-ups, I-011 torchao in the bootstrap), plus
D-027 (the integrity report no longer dirties the tree).

**Gate before the Kaggle build run:** the preflight runbook's checklist. In particular
the build must not start until the `--compare-to` regression passes and
`generation.median_ms_per_image` is a real number — B6's cost has never been measured.

- **Next step:** run `env/kaggle_run_p1_preflight.md`, then the build and eval runbooks.

### P1-R — review decisions folded in — 2026-09-20

Branch `p1-data-foundation`, continuing. Still **local-only**: every GPU and image-corpus
number remains `PENDING-GPU`.

**Decisions 1–4 of the review, implemented.**

- **D-028** — `CLAUDE.md` §2a appended: `git restore` / `git checkout --` is permitted
  only on generated reports under `thesis_dbp/reports/` that this session produced, and
  every use must be named in the report.
- **D-029 (I-021)** — `p1_build_reals.py --crops-per-image N` takes non-overlapping
  crops laid out along the longer axis and centred as a block. Verified on the DIV2K
  geometry: `1404x2040, n=2 -> [(510, 636), (510, 1020)]`, a 384-pixel gap, no overlap.
  Manifest gains `source_image_id`, `crop_index`, `crop_top`, `crop_left`; the summary
  records `crops_per_image`, `distinct_source_images` and
  `split_grouping_column: source_image_id`; a warning is logged on every multi-crop run.
  Deduplication now drops **whole source groups**, so a `source_image_id` always
  identifies a complete set of crops. T-ctrl stays diagnostic-only, effective *n* = 800.
- **D-030 (I-023)** — new `scripts/p1_match_generations.py`, inserted into
  `kaggle_run_p1_build.md` as Step 6b. Columns exactly as specified — `gen_id`,
  `source_real_id`, `matched_quality`, `subsampling`, `seed`, `png_sha256`, `jpg_sha256`
  — plus `pairing`, `png_path`, `jpg_path`, `width`, `height`. The PNG variant is kept;
  nothing is resized; the reals are never re-encoded. Where a generation and a T-real row
  came from the same COCO image the exact partner is used (`pairing: exact`), otherwise a
  real is drawn by seed (`pairing: sampled`). 13 tests, including the two the review
  asked for by name — sampling deterministic under its seed, quality round-trips exactly
  — and one asserting the confound is gone after matching.
- **D-031** — the draw is now **joint**: one observed `(quality, subsampling)` pair from
  a single named real, via `sample_matched_pair`, used by both the evaluation
  matched-view and the generation matcher. Independent draws could produce combinations
  present in no real image, which would be their own artefact. `matched_from` and
  `source_real_id` now name that specific real.
- **I-020** — the 2.00× size ratio (dev 336 213 556 bytes, klein 168 120 878) recorded as
  an *observation consistent with* fp32-versus-half storage, explicitly not evidence of
  identity. Preflight Cell 7b records klein's stored dtype and bytes-per-parameter. The
  claim stays UNVERIFIED and the issue stays open.

**Evaluation suite, per the review (D-032).**

E1 **withdrawn** and every reference removed; I-018 closed. Added, both audited by
command on 2026-09-20 and both `apache-2.0`, not gated:

| set | repo | revision | archive | note |
|---|---|---|---|---|
| **E7 DDA-COCO** | `Junwei-Xi/DDA-COCO` | `8c9330a3b374bcac46a8045a0e3c09ebcf7868fb` | 4 301 456 312 B | **PRIMARY**; reals are MSCOCO **val**; fakes are VAE reconstructions |
| **E8 EvalGEN** | `Junwei-Xi/EvalGEN` | `86c719a9bfc72b7e5d91bfbb04432774feb4e3b7` | 14 538 202 808 B | 553 GenEval prompts, ~55 300 images, 11 060 per generator, all JPEG q96 |

Two findings from those audits that change how the sets are used:

1. **I-024 — E7's reals are COCO *validation*, and T-real is built from COCO.** T-real is
   now restricted to `train2017` at the source, the build runbook asserts no `val2017`
   path is reachable, and the leakage check verifies the outcome. Silent if missed.
2. **I-026 — E8 appears to have no real class**, so AUC/AP/EER are undefined on it alone;
   and being uniformly JPEG q96 it can confound a *pairing* with reals of a different
   quality distribution even though it is not confounded internally.

Also recorded: E7's fakes are VAE reconstructions of COCO, which is the same operation
B5 performs. The novel part of the claim is the cross-lineage result, so E7 is reported
per source VAE with in-bank and out-of-bank subsets separated.

**E6 MNW — access granted, ingestion blocked on one string (I-025).** Terms recorded
(evaluation-only, no training, no commercial use, citation required) and added to
`CLAUDE.md` §6c; data card and ingestion command written. The repository identifier was
**not** produced by any command in this session — HF dataset search and GitHub repository
search both returned nothing relevant — so it is marked `NOT-VERIFIED` rather than
written from memory (D-012). E5 Chameleon stays excluded pending access. GenImage
recorded as optional and not to be started unprompted (I-027).

**New:** `docs/CITATIONS.md` (D-033), with the DDA/EvalGEN BibTeX reproduced verbatim
from both READMEs; MNW, WildRF, COCO and DIV2K entries marked `PENDING` or
`NOT-VERIFIED` rather than filled in from memory.

**Gate unchanged.** 223 passed / 7 skipped, ruff clean, protected integrity CLEAN.

- **Next step:** P1-K — run the preflight, build and eval runbooks in order, honour both
  gates (the `--compare-to` regression, and a real B6 timing), and return the reports.
  `p1-data-foundation` is **not** merged until those numbers are folded in.

### P1-R2 — fairness rule, E5/E6 resolved, Chameleon audited — 2026-09-21

Branch `p1-data-foundation`, continuing. Still local-only; Kaggle runs remain
`PENDING-GPU`.

**D-034, the SEF fairness rule, is now schema rather than intention.**
`configs/p2_runs.yaml` defines five runs; `src/dbp/eval/run_config.py` validates every
`excluded_lineages` entry against the registry, so a typo like `kl16ch` fails loudly
instead of silently excluding nothing. `RunConfig.describe()` / `report_header()` are the
block every evaluation report prints, and `subset_status()` answers the per-subset
in-bank / out-of-bank question an E7 table has to state per row. 13 tests, including one
asserting that exactly one run may carry `comparable_to: SEF` and that it excludes all
three 16/32-channel lineages.

**Chameleon (E5) — the archive is in hand and was audited read-only.** Nothing was
extracted; the central directory and each member's JPEG header were read from the zip.

| check | result |
|---|---|
| counts | **14,863 real / 11,170 fake** — matches the paper's Table 1 **exactly** |
| archive | 2,683,359,341 bytes, SHA-256 `a7cdadfe35287dd3e81b5c35e7cce5437e68ba3b4cdbb833ac0686b990c2fba6` |
| layout | flat `Chameleon/test/{0_real,1_fake}/`, UUID filenames, all `.jpg` |

Three findings, in descending order of how much they matter:

1. **The resolution confound (I-030).** Every one of the 14,863 reals is **exactly 500 px
   wide**; the fakes span 327–5,376 px across 158 distinct widths. **`width == 500 → real`
   scores 100.00 %** (14,863/14,863 reals, 1/11,170 fakes). The reals are evidently
   normalised — also uniformly JPEG **q75, 4:2:0, non-progressive**, all 14,863 of them.
2. **The crop threshold is a class-dependent filter (D-037).** Retention: at 384,
   **44.0 % of reals vs 94.5 % of fakes**; at 320, 93.1 % / 95.9 %; at 512, **zero reals
   survive**. The project default would have flipped the class balance and kept only the
   tall reals. Chameleon therefore runs at **320, single crop** — two 320 crops would
   retain 5,211 reals against 9,200 fakes.
3. **No category labels (I-029).** The paper reports scene/object/animal/human; the
   archive has none of it. Per-category reporting is blocked until the official URL list
   or a mapping file arrives. Aggregate only, limitation stated — the animal split
   (3,998 : 313) is exactly what a pooled number hides.

Compression: fakes are 6.8 % never-compressed (642 PNG + 115 WEBP) against 0.0 % of
reals, so the set is **CONFOUNDED** and the matched view applies. The reference
distribution is a single point (q75, 4:2:0), which the joint sampler handles correctly.

**E6 MNW resolved by direct fetch (I-025).** `github.com/microsoft/MNW` —
*Microsoft-Northwestern-WITNESS*. The earlier failure was my own method: I searched
Hugging Face for a GitHub repository. Terms verbatim (*"intended for evaluation purposes
and cannot be used for training or for commercial purposes"*) and the IEEE Intelligent
Systems 41(02):15-23 citation are now in `docs/CITATIONS.md`; the review's volume, pages
and DOI all check out against the README. Still blocked on the **download route**, which
the README does not contain (I-028) — and any token in it goes to Kaggle Secrets only.

**E8 (D-035, D-036).** Metric policy written into its data card: primary is the fake
detection rate at FPR 1 %/5 % from thresholds fitted on our own validation reals, per
generator; the COCO-`val2017`-q96 AUC is secondary and never the headline, with its
content caveat attached. `p1_build_eval.py` gained `--subsample N --stratify-by FIELD`:
seeded permutation of each stratum's sorted ids, every kept id recorded, short strata
kept whole and named. 8 tests, including one asserting the result is not "the first N".

**A scope-lock repeat (I-032).** `Chameleon.zip` arrived at `REPO_ROOT` and the integrity
check went `CRITICAL-PROTECTED-FILES-CHANGED`. Reported rather than acted on — moving or
deleting anything outside `thesis_dbp/` is forbidden. The user moved it to
`F:\thesis-2\Chameleon.zip`; the check is **CLEAN** again.

- **Next step:** P1-K. Upload the rebuilt archive, run preflight → build → eval, honour
  both gates. Chameleon uploads as its own private dataset and its SHA-256 is verified
  after upload before anything else runs.

### P1-K — the Kaggle build, and the greyscale defect it exposed — 2026-09-26

**Gates, both honoured.** `p0_bank_smoke_test.py --compare-to reports/kaggle_20260920/p0_bank_smoke.csv`
printed `regression check PASSED: every deterministic column is unchanged` over **20
(key, crop) pairs**, with 26/26 rows OK across all 13 members, on diffusers 0.37.1 /
torch 2.10.0+cu128 / Tesla T4 — the same stack that produced the P0 baseline. The only
byte-identical decoder pair is `['sd21', 'sd_ft_mse']`, confirming `alias_of`; none of
the three new members duplicates an existing one. The B6 timing gate is a measured
`median_ms_per_image = 4237.0` (1.177 GPU-h per 1,000), against which the corpus cost
2.94 GPU-h.

**Measured throughput (T4, crop 384, batch 1, at the build precision).** fp16 refused
for exactly three members: `sdxl` (5.73 dB — its fp16 output contains NaN, which is the
`invalid value encountered in cast` warning), `movq_k22` (54.86 dB, just under the 55 dB
rule) and `realesrgan_x4` (44.94 dB). The nine others passed at 56.70–64.83 dB. Batch
size makes no difference to ms/img at any size tested, so the build runs at batch 1.
The runbook's prediction that `sdxl_fp16fix` would be refused was **wrong**: it measured
56.70 dB and is allowed. Measurement wins; the runbook text is corrected.

**JPEG encoder equivalence: `IDENTICAL`.** Kaggle (Pillow 11.3.0, libjpeg 6.2) and the
local machine (Pillow 12.3.0, libjpeg 8.0) produce byte-identical JPEGs at all 24 cells.
Matched views are therefore reproducible on either machine; they are still built on
Kaggle, because that is where the data is.

**The corpus.**

| stage | result |
|---|---|
| T-real | 2,500 kept, 498 skipped as under 384 px, 0 unreadable, 0 near-duplicate groups; all JPEG, estimated quality 80/96/96; 0.62 GB |
| T-ctrl | 1,600 rows from **800** distinct DIV2K sources (2 non-overlapping crops each), all PNG, 0 skipped; 0.39 GB |
| B5 pairs | **49,200 rows** = 4,100 reals × 12 distinct members, `sd_ft_mse` absent by alias; both shards exited 0 |
| B6 generations | **2,488** written, **12** discarded by the safety checker (`enabled`), 0 failed |
| gen_matched | 2,475 of 2,488 — **13 failures**, all `unknown chroma subsampling label 'unknown(-1)'` |

**The defect those 13 failures exposed.** 11 of the 2,500 T-real images are greyscale
JPEGs. A greyscale JPEG has no chroma channels, so its subsampling reads as `-1` and
there is no pair to copy onto a colour generation. The matcher refusing to guess was
correct. The real fault was upstream: B4 should never have admitted them. Fixed per
**D-039** at source and by flag in the built corpus, with the matcher's guard kept as
defence in depth. **D-041** records how black-and-white robustness is handled instead:
synchronized augmentation in P2.

Verified locally this session: a greyscale JPEG reports `subsampling='unknown(-1)'`
while a colour one reports `'4:4:4'`, and `subsampling_code('unknown(-1)')` raises
`ValueError` — so the failure path is understood, not inferred.

**Still to run on Kaggle** (one cell, in the eval session): `p1_rescan_greyscale.py`
with `--instances` for the per-supercategory effect, then `p1_match_generations.py` to
rebuild the whole matched set from the colour-only reference. Expected: every eligible
generation matched, 0 failures. Until that runs, the exact excluded count and the
stratification effect are **PENDING-KAGGLE**.

**Also this round:** the bootstrap's torchao probe (I-033), Kaggle's new input layout
(I-034), the lost interactive session (I-035), and the goldens keyed by environment
(D-040, no expected value changed).

### P1-K fix run — greyscale exclusion measured — 2026-09-26

Ran on Kaggle against the built corpus, reading the images in place from the mount:
`p1_rescan_greyscale.py --set-id t_real --instances instances_train2017.json`, then
`p1_match_generations.py`.

**The pixel test found 7.5× more than the header test.** 83 greyscale images, of which
only **11** were visible as `subsampling = unknown(-1)`; the other **72** are `rgb_equal`
— greyscale photographs stored in three-channel JPEGs, indistinguishable from colour by
format, mode or channel count. The 13 matcher failures pointed at the 11 and would have
led to a fix that left 72 in place. The reviewer's instruction to add the R≈G≈B test is
what caught them.

| | |
|---|---|
| scanned | 2,500 |
| greyscale | **83** (11 `single_channel_source`, 72 `rgb_equal`) |
| channel spread | 82 rows at exactly 0, 1 row at 2 — the tolerance decides one image |
| T-real after | **2,417** usable rows; no backfill |
| pair rows filtered | 83 × 12 = **996** of 49,200; nothing deleted or rebuilt |
| gen_matched | **2,488 of 2,488, 0 failures** (was 2,475 with 13 failures) |
| pairing | exact 2,002, sampled 418, **sampled_after_exclusion 68** |
| reference | 2,417 reals, quality **90–96**, subsampling 4:4:4 only |

**The reference range moved from 80–96 to 90–96 for a trivial reason, recorded so it is
not over-read:** exactly one image in the corpus was q80 and it happened to be greyscale.
The excluded 83 are otherwise ordinary — 67 at q96, 15 at q90.

**Stratification:** the exclusion costs between 0.30% (`food`) and 5.60% (`accessory`) of
a supercategory's rows, against a 3.3% average; no supercategory falls below 147 rows.
`accessory`, `vehicle` and `person` lose most, `food` and `appliance` least, which is the
pattern black-and-white photography would predict. The full table is in the T-real data
card §9.

**Artifacts committed** under `reports/kaggle_20260926/`: the re-scan and matched-set
summaries, the two reals manifests, the generation manifest, both B5 shard summaries,
`p1_throughput.csv` (72 rows) and `p1_backbone_throughput.json`. **The 13-member gate's
`p0_bank_smoke.csv` is not among them** — that session's copy was not preserved, so the
`regression check PASSED` result over 20 (key, crop) pairs exists only as console output
quoted in the entry above. The number is real and was observed three times; the CSV
artifact behind it is not in the repository, and this entry says so rather than
attaching the committed 4-row CPU file and letting it pass for the Kaggle run.

**Outstanding:** the same scan on **T-ctrl**. DIV2K is expected to be entirely colour,
but that is an expectation, not a measurement.

**Operational note (I-035 again).** The build notebook's Save Version failed with a
Kaggle editor error (`ConcurrencyViolation … ExpectedSequence=469, ActualSequence=467`)
after about an hour queued. The reals, generations and reports were recovered by zipping
them from the live session and re-uploading them as a dataset, which is how this run had
its inputs. The 13 GB of B5 pairs may still be unsaved; they are rebuildable in ~1.8 h,
but **the per-member `p1_pairs_manifest_<key>.csv` files must be deleted first** or the
resume logic will skip every image it believes is already written.

### P1-K — T-ctrl re-scan, and the defect it exposed — 2026-09-26

Running the same scan on T-ctrl, as recommended, immediately broke it: `scanned 1600:
6 greyscale, 1594 colour with unreadable subsampling`, and the script marked all 1,600
rows excluded. The DIV2K crops are lossless PNG, so their subsampling label is
`not-jpeg` — the correct value for a format that has no chroma subsampling — and the
scan treated that as "unreadable" (I-037).

The lesson is narrow and worth keeping: the check was written against the corpus that
motivated it. T-real is entirely JPEG, so `is_jpeg` never had to be consulted, and the
bug could not appear there. It took a set with a different format to expose it, which is
also the argument for having T-ctrl at all.

**T-real's 83 exclusions are unaffected**: every row is a JPEG and its report records
`unmappable_but_colour: 0`, so no row reached the faulty branch.

**Re-run with the fixed script, same session:** `scanned 1600: 6 greyscale (0
single-channel source, 6 rgb_equal), 0 colour with unreadable subsampling`. The count of
six happens to match the broken run, but the classification does not — the earlier run
called them `single_channel_source` on the strength of the same faulty flag — and
`unmappable_but_colour` fell from 1,594 to 0, which is the fix confirming itself on the
data that exposed it.

**T-ctrl: 1,600 -> 1,594 usable rows**, from 3 source photographs (`0230`, `0769`,
`0800`), both crops of each, all `rgb_equal` at spread 0. Both crops of every affected
source were excluded together, so no source is left contributing a partial group and the
`source_image_id` rule is unaffected. 0.375% of DIV2K's 800 sources, against 3.3% of
T-real — plausible, since DIV2K is a curated high-resolution set rather than a scrape.

**The corpus, final:** 4,011 usable reals (2,417 + 1,594), **48,132 usable pair rows** of
49,200, 2,488 generations each with a matched variant, 0 failures. Nothing deleted,
nothing rebuilt.

### P1-K eval — E7 ingested, and a shortcut the §6b check could not see — 2026-09-27

**E7 DDA-COCO built: 34,969 rows**, 5,000 real / 29,969 fake over six source VAEs,
**leakage CLEAN** against both training manifests. The reals are not shipped with the
set — DDA-COCO contains reconstructions only — so COCO `val2017` was mounted as the real
class, which is exactly the split T-real was kept away from (I-024). That discipline is
now confirmed rather than assumed.

**The finding.** The compression check reported "not confounded", correctly: both classes
are entirely JPEG at q80/96/96. But every FLUX.1 reconstruction is **4:2:0** and every
real is **4:4:4**, so on that subset chroma subsampling alone is a perfect classifier.
The rule that caught nothing was asking whether an image had been compressed; the
question that mattered was how.

Two details worth keeping:

1. **Pooled, it is invisible.** Across all six subsets the lift is +0.03 points, because
   five of them match the reals. On the FLUX.1 subset alone it is +49.85. A single
   summary number for the set would have hidden it completely, which is why D-042
   measures per stratum against the shared real class.
2. **It cannot be fixed, only disclosed.** My first proposal — re-encode both classes at
   one setting — was wrong, and I withdrew it: chroma subsampling is destructive, so
   re-encoding at 4:4:4 upsamples the chroma rather than restoring it and leaves the blur
   and 2x2 block structure in the pixels. That change would have equalised the headers
   and hidden the shortcut while leaving the signal, which is worse than leaving it
   visible.

**Also recorded:** the 5,000 near-duplicate groups inside E7 are its structure — each
val2017 photo and its six reconstructions — not contamination, so perceptual hashing says
nothing about duplication within the set; and 10 of the val2017 reals are greyscale,
which is harmless here because their reconstructions are greyscale too, so it is not a
class shortcut.

E8, E3, E4 and E5 have not yet been ingested. All of them now get the D-042 check.

### P1-K eval — the suite ingested, and what the parameter check found — 2026-09-28

All six evaluation views are built, and **every one is CLEAN against both training
manifests**. That is the result the whole `train2017`-only discipline was for (I-024),
and it now holds by measurement rather than by construction.

| view | rows | labels | compression | separability |
|---|---|---|---|---|
| E7 DDA-COCO | 34,969 | 5,000 / 29,969 | not confounded | **FLUX.1: +49.85 subsampling, +14.35 quality** |
| E8 EvalGEN | 5,000 | fake only | n/a — one label | undefined, correctly |
| E3 raw | 2,178 | 726 / 1,452 | **CONFOUNDED** 93.7% vs 0.4% | **+28.97 / +29.75 pooled** |
| E3 matched | 2,178 | 726 / 1,452 | fixed | +0.14 / +0.23, **residual +6.47 on Nano-Banana2** |
| E3 preprocessing | 2,178 | 726 / 1,452 | not confounded | +0.00 |
| E4 WildRF test | 2,503 | 1,251 / 1,252 | **CONFOUNDED** 39.5% vs 2.2% | **+19.18 / +25.41 pooled** |
| E4 matched | 2,503 | 1,251 / 1,252 | fixed | subsampling +1.24, **quality still +11.39** |
| E5 Chameleon | 26,033 | 14,863 / 11,170 | **CONFOUNDED** 6.8% vs 0.0% | +2.95 / +3.13, under threshold |
| E5 matched | 26,033 | 14,863 / 11,170 | fixed | +0.05 / +0.22 |

**What D-042 bought.** Three of the six sets carry an encoding shortcut, and **two of
them were invisible to the §6b check**: E7 passed "not confounded" while its FLUX.1
subset was perfectly separable by chroma subsampling, and E4's matched view passed
"confounded: False" while quality still lifted +11.4 points. A rule that asks only
whether an image was compressed cannot see either.

**What the matched views can and cannot do.** They work: E3 falls from +29 to +0.2, E5
from +3.1 to +0.2, E4's subsampling from +19.2 to +1.2. What survives is structural, and
both residuals have the same two causes — never-compressed **reals** are left alone by
design, so `quality = None` becomes a real-only value, and fakes that were **already**
JPEG keep their own settings. E4 keeps +11.39 pooled on quality; E3 keeps +6.47 on the
Nano-Banana2 subset. Neither is a defect to fix; both are caveats to report.

**The honest summary for the thesis**, per set:

- **E7** is the primary head-to-head set and is clean on five of six subsets. FLUX.1
  numbers are an upper bound and never the headline (D-042, I-038).
- **E3** is reported from the matched view, with the Nano-Banana2 residual stated.
- **E4** carries its caveat everywhere, naming reddit as the affected platform.
- **E5** is reported from the matched view, and its dominant problem remains the 500-px
  real width, not compression.
- **E8** is a detection-rate set by construction, exactly as D-035 specifies.

Artefacts under `reports/kaggle_20260928/`: six summaries and `p1_leakage.json`.

### Where the P1 data lives — 2026-09-28

The image data is ~19.5 GB and is never committed (CLAUDE.md §2). It lives in Kaggle
notebook outputs, attached as inputs to whatever runs next. **User-supplied location, not
verified by any command in this session**: kernel `muhammadadeelshah/notebook3a585d800a`,
committed output **version ID 353334139**, attached through *Add Input -> Your Work* and
read at `/kaggle/input/notebook3a585d800a/`. Verifying it locally is
declined on purpose — it would need API credentials on disk and a dataset-scale download
to this machine, and §6a(a) allows neither.

| artefact | where | expected |
|---|---|---|
| `outputs/p1/reals` | build notebook output | 4,100 PNG |
| `outputs/p1/pairs` | build notebook output | 49,200 PNG |
| `outputs/p1/generated` | build notebook output | 2,488 PNG |
| `outputs/p1/gen_matched` | **fix** notebook output | **2,488 JPEG** |
| evaluation manifests | eval notebook output | six views |

**Two `gen_matched` sets exist and they are not interchangeable.** The build notebook's
holds **2,475** files, drawn from a reference that still contained the 83 greyscale reals;
the fix notebook's holds **2,488**, drawn from the colour-only reference with 0 failures.
Removing 83 rows from the reference shifts the sampled index for *every* generation, so
the two sets differ image by image, not only in the 13. **P2 uses the 2,488 set.** A count
of 2,475 identifies the stale one.

Manifests carrying `excluded_reason` (T-real 83 flagged, T-ctrl 6) are likewise the fix
notebook's copies, not the build's.

### P1-close — review items, before merge — 2026-09-28

**D-037 confirmed** by the user: Chameleon evaluates at 320 px, single crop, for the
as-distributed view.

**Item 1, the fp16 discrepancy: no rebuild needed.** The audit is in I-040. The short
version is that P0 and P1 measure parity on different images at different crops, the
conservative P1 number decided the build, and the pair manifests record **fp32 for all
4,100 rows** of `realesrgan_x4`, `movq_k22` and `sdxl`. The one real defect was that the
CSV did not say which image its parity came from; it does now.

**Item 3, Chameleon: the pipeline-matched view did not exist and now does** (D-043). The
compression-only view left `width == 500 -> real` at 100.00%, which is the set's dominant
confound and a far larger one than its compression. The new view downscales the fakes to
the reals' width and encodes at their setting, with a BICUBIC sensitivity check, and the
summary reports the trivial rule's accuracy before and after so the claim is checkable
rather than asserted.

**Items 2 and 4** are `env/kaggle_run_p1_close.md` and D-044 respectively.

**Still unverified until that session runs:** the 49,200 pairs on disk, and the
13-member gate's CSV as a saved artefact rather than console output.

### P1-close — the session ran; P1 is complete — 2026-09-28

All four review items are closed with artefacts rather than console text.

**The pairs are intact: 49,200 images, every one of the 12 distinct members at 4,100.**
Counted on the files, not inferred from the manifests.

**The fp16 question is settled, and no rebuild was needed.** The merged manifest records
the precision per member: `movq_k22`, `realesrgan_x4` and `sdxl` are **fp32 for all 4,100
rows each**; the other nine are fp16, matching their measured verdicts. The 44.94 dB that
prompted the query was the P1 bench on a synthetic image at crop 384; P0's 59.86 dB was a
photograph at 512/1024. The conservative number decided the build (I-040).

**The 13-member gate now exists as an artefact.** `reports/kaggle_20260928/p0_bank_smoke.csv`,
26 rows, all `OK`. I re-ran the comparison **independently** from the two committed CSVs
rather than trusting the console: over the 20 (key, crop) pairs shared with the P0
baseline, `psnr_db`, `lpips_vgg`, `mean_abs_residual` and `max_abs_delta_rerun` differ in
**zero** cells. The three P1-only members are deterministic with `max_abs_delta_rerun = 0`:
flux2_klein 34.891/43.458 dB, qwen_image 33.039/47.138, movq_k22 27.882/37.940.

**Chameleon's pipeline-matched view works** (D-043). The trivial rule
`width == 500 -> real` falls from **99.996% (lift +42.90)** to **57.093%, lift exactly
0.00** — worth precisely the majority-class baseline. 11,169 fakes downscaled, 14,863
reals untouched, no parameter separability, compression no longer confounded. The BICUBIC
sensitivity check reproduces it to the digit, so nothing rests on one filter.

**`gen_matched` was regenerated and is now 2,488** — exact 2,002, sampled 418,
sampled-after-exclusion 68, quality 90-96, 0 failures. The stale 2,475 set from before the
greyscale fix is superseded; the count is how you tell them apart.

**Loose end for P2, not for P1:** the Kaggle dataset copy of `outputs/p1/reals` holds
1,314 PNGs rather than 4,100 — a truncated upload, not a data loss, since `p1_reals.zip`
(1.09 GB) is intact locally and the manifests are complete. P2 must attach a complete
copy; the count is the check.

### P2-A -- Phase A audit, read-only -- 2026-09-29

Preconditions all met: `p1-complete` on `main` at `100fe1e2`, integrity **CLEAN**
(0 protected files, empty by design), 286 tests passing, ruff clean, and D-034, D-037,
D-041, D-043, D-044 all present.

**Two gaps found in `p2_runs.yaml`, both fixed under D-045.** There was no P-A protocol
at all -- `4ch_only_strict` retains `kl_4ch_sdxl` and would have made SDXL in-bank while
the pre-registration lists its E7 subset as unseen -- and `eval_crop` was 384 on three
runs against the plan's 320.

**Backbone: DINOv3 ViT-B/16**, and the reason is the absence of evidence rather than a
preference. Phase A was not allowed to measure, and the committed record licenses
nothing else: `reports/kaggle_20260926/p1_backbone_throughput.json` has
`"backbones": null`, and P0 smoke-tested only `dinov2-base` and `dinov3-vitb16`, with a
LoRA entry marked `"dry-run only: no optimizer, no training step"`. So there is no
training-step memory figure for *any* backbone, let alone the under-13 GB evidence
ViT-L would have needed. The user added the matching sensitivity analysis to the
pre-registration: repeat the go/no-go with S1 on frozen ViT-L/16 in P4, because a weaker
semantic stream could inflate the fusion gain.

**Budget.** Derived from the realised P1 cost -- the sum of per-member medians of
`ms_per_img` across the twelve `p1_pairs_manifest_<key>.csv` files is **2,750.4 ms/image
at crop 384**, so **1,910.0 ms/image at 320**. At the approved caps the total is
**30.44 GPU-h** against a raised cap of 32.

### P2-B -- implementation -- 2026-09-29

Branch `p2-streams` off `main`. New libraries: `dbp/data/splits.py`,
`dbp/train/augment.py`, `dbp/eval/panel.py`, `dbp/eval/fusion.py`,
`dbp/eval/complementarity.py`, `dbp/streams/{s1_semantic,s2_artifact,s3_probe}.py`.
Nine `p2_*` scripts, three Kaggle runbooks, and `docs/P2_PREREGISTRATION.md`.

**The manifests are now partly tracked.** `p1_reals_manifest_t_real.csv`,
`_t_ctrl.csv` and `p1_gen_matched_manifest.csv` are committed (each under 1 MB) and
their SHA-256s match the values P1 recorded in the rescan and matcher summaries exactly,
which closes the chain from the build to the committed copy. `p1_pairs_manifest.csv` is
11.3 MB and section 2 forbids committing it, so
`reports/p1_pairs_manifest_digest.json` holds the hash, row count, members and
precisions of all thirteen pair manifests instead, and B0 verifies the Kaggle copy
against it.

**Two bugs found and fixed while building, both mine.** The radial spectrum normalised
its frequency radius by the **corner** rather than by the axis Nyquist, which left the
outermost four bins reachable only by diagonal content and exactly zero for a purely
horizontal signal -- a quarter of every member's feature vector would have been a
constant. And `p2_build_splits.py` assigned orphan generations -- those whose caption
has no T-real partner -- one at a time *after* the draw, and a stratum of one always
goes to train, so every one of them would have landed in training. Both are now covered
by tests that fail on the old behaviour.

**Not yet run:** everything that needs a GPU. The three Kaggle sessions are
`env/kaggle_run_p2_inputs.md` (CPU), `kaggle_run_p2_streams.md` (T4 x2) and
`kaggle_run_p2_score.md` (CPU). Until they run, P2 has no measured results, and this
entry claims none.

### P2-B0 — the Kaggle input session, measured — 2026-10-02

**B0 PASS.** `scripts/p2_verify_inputs.py` reports
`VERDICT: PASS -- the attached corpus is the P1 corpus`, with reals 4,100 / pairs 49,200
(all twelve members at 4,100) / generations 2,488 / `gen_matched` 2,488, 83 and 6 rows
flagged, 0 differing hashes anywhere, and all thirteen pair manifests matching
`reports/p1_pairs_manifest_digest.json`.

**`gen_matched` was rebuilt rather than recovered, and the rebuild is byte-identical.**
The attached notebook output carried the build notebook's **stale 2,475** set.
`p1_match_generations.py` re-ran against the post-D-039 reals manifest -- `reference:
2417 real JPEGs ... reference excluded 83 row(s)` -- and produced 2,488 rows with
**0 of 2,488 `jpg_sha256` values differing** from the committed manifest. The 2,475 set
is therefore retired on evidence. The console's `2002 exact, 486 sampled` is P1's
`2002 / 418 / 68` summed; the manifest keeps the two sampled categories apart.

**Three defects of mine surfaced in this session**, all in material the user was about to
run and none of them in the corpus:

1. I invented four flag names for `p1_build_reals.py` (`--set-id`, `--source-root`,
   `--count`, `--seed`) against its real `--set` / `--root` / `--limit`, and gave
   t_ctrl `--count 800` where `--limit` counts **rows**, so 1,600. Fixed, and all eleven
   scripts' invocations across the three P2 runbooks are now checked against their
   argparse: 0 bad flags.
2. The runbook claimed DIV2K was attached. It never was -- P1 fetched `DIV2K_train_HR`
   into `/tmp` in-session, which is what `p1_reals_manifest_t_ctrl.json` records as its
   source root. The download cell is now Step 1a. In the event it was not needed: the
   user had uploaded the built reals.
3. The staging cell I gave in chat copied every `p1_*manifest*.csv` it found, which
   overwrote the tracked post-D-039 manifests with pre-D-039 copies from an uploaded
   `p1_reports` folder -- 0 rows flagged, 2,475 `gen_matched` rows. The gate caught it
   and named both symptoms, which is what it is for; Step 0b now refuses to overwrite
   the three tracked names and asserts their contents.

**One defect in committed code:** `p1_gen_matched_manifest.csv` records `jpg_path` as an
**absolute** path from the session that built it, including a `gen_matched/gen_matched`
nesting. `Path(root) / "/abs"` discards the root silently, so the verifier would have
reported all 2,488 rows missing on a corpus that was fine. `resolve_row_path` now
recovers it and the report counts how many rows needed recovering (2,488). The manifest
is **not** rewritten: its SHA-256 is part of the P1 chain.

**Still PENDING-GPU:** everything in Sessions B and C.

### P2-B1 — Session A complete, measured — 2026-10-02

The split, the Chameleon pipeline-matched view, all six panels and the three image
lists are built. Measured outputs:

| artefact | value |
|---|---|
| split groups | **3,718** -> 2,974 train / 744 val (2,500 T-real + 800 DIV2K + 418 orphan generations) |
| content crossing the split | **0**, across 15 sets checked together |
| strata | 13, from `person` 9 to `none` 832 |
| Chameleon `width == 500 -> real` | **100.00% -> 57.09%**, equal to the 57.09% baseline |
| panel | **12,150** rows across six views |
| S3 probe list | **17,945** rows, against a budget of 18,150 |

Two cross-checks that came out exactly right. The **418 orphan generations** reconcile
with P1's matcher: 2,488 - 2,002 exact partners = 486 without one, of which 68 are
`sampled_after_exclusion` -- their caption's COCO image *is* in T-real, merely flagged --
and 486 - 68 = 418. And `person: 9` is the rarest-present stratum rule behaving as
designed: `person` occurs in 1,361 images but is the most common supercategory, so
almost nothing has it as its rarest; ranking by the most frequent instead would have
labelled over half the corpus `person` and stratified nothing.

**Two defects of mine, both found from the numbers rather than from a crash.**

`PANEL_CAPS` said E3 was stratified by `("generator", "content_type")`. Its manifest
calls them `source_label` and `subset`. `stratum_of` raising rather than defaulting to
`"unknown"` is why the E3 panel refused to build instead of silently pooling nine strata
into one. The same guess had reached `p2_score.py`, which would have written empty
columns for every E3 row and left D-044's per-content-type report with nothing to group
by, and `p2_analyze.py`, whose baseline lookup keyed on `e3_matched/Nano-Banana2` while
the slice is `e3_matched/Nano-Banana2-fake`. A test now pins all six views' column names
against the committed manifests.

I-043 is the second and the worse one: the flagged reals' reconstructions were still in
the training lists. The 55,699-against-55,788 difference is what exposed it.

**Still PENDING-GPU:** Sessions B and C.

### P2-B1b — Freeze the panel; T-ctrl out of training; wire the luma view — 2026-10-03

The panel is frozen. `p2_materialise_panel.py` copied **12,150** images into
`/kaggle/working/panel`, re-hashed every one against the `sha256` its manifest already
recorded, and found **0 mismatches**. Digest:

```
2e1c192024ec52e42c4e1761afa6fb2e37c0f7a6983f5ff75d1174305550b769
```

Recomputed offline from `p2_panel_images_frozen.csv` in the downloaded reports archive:
identical. Per view: e3 1,500, e4 1,500, e5 4,000, e7 3,500, e8 1,250, t_ctrl 400. The
luma companion is 1,000 rows (FLUX.1 500 + real 500), not counted in the 12,150, so
13,150 files on disk. The P1 chain behind it verifies end to end: t_real 2,500 with 83
flagged, t_ctrl 1,600 with 6, gen_matched 2,488, and **13 of 13** pair manifests matching
their committed hashes.

Three defects surfaced while getting there, and two of them were mine.

**I-044, the one the session existed for**, had a sting in its tail: E5's
pipeline-matched view leaves `real` *and already-JPEG* rows in place with
`view_path = source_path`, pointing into the raw Chameleon dataset
(`p1_build_eval.py:486-490`). I told the user the Chameleon input was safe to drop
because I had reasoned about the re-encoded fakes only. It was not: 2,000 of E5's 4,000
panel rows resolve to those originals. The run stopped on a missing file rather than on a
wrong one, because D-050's resolver refuses to fall back from a derived row to its
pre-derivation source — but the cost was a session's worth of re-downloads, about four
hours, and the user said so. Recorded here because "the hash check would have caught it"
is not a defence when the cheaper check was reading the code that writes the column.

**The E7 source root in the runbook was wrong.** `--source e7_dda_coco=/tmp/e7x/DDA-COCO`
assumed the archive's wrapper directory survives extraction; `extract_members` measures
each archive's prefix and strips it, because DDA-COCO wraps its tree in a directory the
manifest does not record while EvalGEN's top-level directory *is* part of the recorded
path. Corrected to `/tmp/e7x`.

**`p2_panel_images.csv` is written twice, by two scripts, with different columns.**
`p2_materialise_panel.py` writes `image_id, relative_path, label, view, sha256`; running
`p2_build_image_lists.py --panel-list` over it rewrites the same filename with `root` in
place of `sha256`. Session C needs both — `root=panel` to resolve a path, `sha256` to
verify the digest — and no single script writes both, so the hash-bearing copy is kept as
`p2_panel_images_frozen.csv` and the digest is defined over that. One extra report file
beyond the approved plan, flagged at the time and recorded here.

Then the verification of the downloaded archive turned up two things worth more than the
panel itself.

**I-045.** 313 of the 400 T-ctrl diagnostic rows were in the *training* split and all 400
were in `p2_s1_images.csv`. The P2 prompt makes T-ctrl diagnostic-only; the code did not,
because `corpus` is the population for both the training lists and the diagnostic panel
and a T-ctrl row's split label reads `train` like any other. **20,722 rows** — 1,594
control reals and their 19,128 reconstructions — were in every training list. D-051
withholds them, validation included, since validation fits the thresholds and the
stacker. The S1 cache drops 54,631 → 33,909 and the diagnostic is now fully
out-of-sample. The panel and its digest are untouched.

**The gate could not be computed as pre-registered.** `grep -i luma` over
`p2_score.py`, `p2_analyze.py`, `p2_extract_s1.py` and `p2_probe_features.py` matched
nothing, while §8 says the luma-only FLUX.1 number is the one that enters M1 and M2 and
that an unchecked FLUX.1 number is not claimable. D-052 wires it: a declared `LUMA_VIEWS`
entry outside `PANEL_CAPS` (adding a seventh key there would change the frozen digest),
rows under `<id>__luma` because `p2_probe_features.py` keys one `.npz` per member by
image id, a second S1 cache tag, and a substitution in the analysis that **refuses a
verdict** if the luma rows were never scored. Two of my own bugs here were caught by the
new tests before anything ran: `slice_key` did not subdivide the luma view, so the slice
would have been named `e7_dda_coco_luma` and the substitution would have matched nothing;
and `baseline_for` did not reach it, so the trivial-rule baseline D-044 requires would
have been missing from the one number that exists to be read against it.

**Budget.** Computed from the committed table's rates and the measured counts: S1 cache
0.49, S1 scoring 1.20, S2 training 9.00, S2 inference 4.81, S3 probe 10.09 → subtotal
25.59, +20% → **30.70 GPU-h** against the 32-hour cap, **+1.30** headroom. The two
changes pull against each other: withholding T-ctrl removes 20,722 S1 images, while the
luma view adds 1,000 probe rows and the S3 subset, no longer able to draw T-ctrl, redraws
from `t_real` and reaches its full validation cap — so the probe list goes 17,941 →
**19,011**, which is +861 over the 18,150 the original estimate assumed. S3 at 10.09 is
still under the 11 GPU-h stop-and-report line.

Tests: **609 passed, 7 skipped**, up from 571. Ruff clean. The panel is published as the
private dataset `dbp-p2-panel`; it holds WildRF (research-only) and Chameleon
("Commercial use in any form is prohibited"), so private is a licence condition and not a
preference (§6c).

**Still PENDING-GPU:** Session C (S1 cache, four S2 runs, S3 probe features) and
Session D (fit heads, score the panel once, the go/no-go).

### P2-C (part 1) — the panel will not fit the crop, and the budget had to be remeasured — 2026-10-03

Session C opened with the corpus staged and verified and ended, for now, with two
findings that both came out of measurement rather than reasoning.

**The corpus had a gap.** `gen_matched` on the attached P1 dataset is the stale
pre-D-039 set of 2,475, while the committed manifest has 2,488; Session A's rebuild into
`/kaggle/working/gm` was never saved. Re-running `p1_match_generations.py` reproduced it
exactly — 2,488 rows, `2002 exact partners, 486 sampled`, **0 of 2,488 `jpg_sha256`
differing** from the committed manifest. The first attempt failed because I passed
`--gen-root` the corpus root: the script joins `gen_root.parent / relative_path`, so the
flag wants the `generated` directory itself. I had read its help text rather than
line 246.

**The lists were rebuilt under both approved changes** and landed exactly on the counts
measured locally the night before: S1 33,909, panel 12,150, probe 19,011, luma 1,000,
with **0** T-ctrl rows in the S1 list and **0** in the probe training subset.

**The S3 guard fired.** A 50-image calibration read 0.2224 s/image for `sd15`; at 300
images it read 0.1739, so the first figure was 28% warm-up — the same trap as I-009 in
P1. Measuring a second member (`movq_k22`, 0.4230 s/image) gave two equations for two
unknowns: **LPIPS costs 8.4 ms per (image, member)**, and decode costs **1.226x** the
area-scaled P1 rate. LPIPS was never the risk I-041 flagged. The error was mine (I-048):
I scaled P1's crop-384 medians to 320 by pixel area, and per-call overhead does not
shrink with pixels. S3 is **12.90 GPU-h**, not 10.09, and the full plan 34.08 against a
32 h cap. The user raised the cap to 36 and refused every cut, fixing a ladder for later
(D-056) with the S3 training subset explicitly off it.

**Then S1 crashed on the panel.** `image 422x317 is smaller than the 320px centre crop`,
640 images in. **375 of the 12,150 panel rows have a shorter side below 320**, down to
150 px (I-047). The panel sampler had never checked that a drawn image survives the
pre-registered crop, so the panel contained rows no stream could score. The S1 *train*
cache was unaffected and is complete: 33,909 features of dim 1536 in 907.1 s
(0.0267 s/image), with 4 register tokens and 400 patch tokens confirmed in the log.

The fix is the one that invents nothing: drop them (D-054). Keeping them would have
required up-scaling or padding exactly the undersized subset — and in E3 that subset is
**reals only, 23 of 500, against 0 of 1,000 fakes**. A resampling signature present in
4.6% of one class and none of the other is a D-042 shortcut we would have manufactured
ourselves, pointing the way that flatters the result. Pre-registration Amendment 1
records the rule, both digests, the per-view x per-label counts and the E3 class shift,
and states plainly that no panel score existed when it was written — because the pass
that would have produced one is the pass that crashed.

Panel **12,150 -> 11,775**, luma **1,000 -> 978**. New digests, computed here over the
surviving rows (the per-row hashes are unchanged, only rows were removed), so the Kaggle
run must reproduce them rather than announce them:

```
panel  bff608c3d8e00ea834532cd541849b4f21de803c8f8e27a55ca5e70424c51116
luma   190279cc889da9c4d879c4d7ffd18afae42949c58896a983a94f1bd6ec9c2303
```

D-055 falls out of this: applying the filter means re-running materialise, and
`materialise_view` resolved a source for every row *before* checking whether the
destination already existed. That would have demanded the evaluation corpora back —
about 10 GB and 40 minutes — to rewrite a list of files already verified byte-for-byte.
A file under the panel root that hashes to its recorded value **is** the panel image, so
the hash check now ends the search instead of starting it, and the panel root may be
read-only, which it is once published.

Tests **629 passed, 7 skipped** (from 609). Ruff clean.

**Still PENDING-GPU:** the panel and luma S1 passes, S3 probe features (12.90 GPU-h
authorised, hard stop 14.5), the four S2 runs, and Session D.

### P2-C (part 2) — the first S3 commit, and a runbook process that does not need luck — 2026-10-04

The first committed S3 run ran 10 h 8 min and was marked **failed**. Kaggle kept its
output anyway, and it held **9 of 12** members complete with valid sidecars, all three S1
caches, and the image lists. Two of my mistakes caused it. Shard 0 died at 23:05 on
`realesrgan_x4` with `ModuleNotFoundError: spandrel`, because my first cell pip-installed
`lpips` alone instead of running the bootstrap (I-049); the second shard then ran on alone
until the 7.25 h hard stop fired at 02:33, mid-way through `movq_k22`. The hard stop worked
as designed. My persist cell then imported `dbp` before putting it on the path (I-050).

Measured per member, s/image at crop 320 including LPIPS: sd15 0.1813, sd21 0.1813,
sd_ft_ema 0.1842, sdxl_fp16fix 0.1844, flux 0.1804, sd35 0.1789, flux2_klein 0.1793,
dcae 0.1959, **sdxl 0.3648**, movq_k22 0.393 (in flight). Full S3 projects to 13.01 GPU-h
against the 12.63 estimated after I-048 and the 14.5 hard stop.

The quota reading -- 7 h 49 min left of 30 after a 10 h 8 min session -- is consistent with
Kaggle charging **session time** on T4 x2, which is what the raise to 32 and then 36
assumed (I-042). One reading, so recorded as consistent rather than measured.

The user then set seven standing rules (CLAUDE.md section 12), and the recovery runbook
was rebuilt to satisfy them before it was handed over: cells are thin calls; the bootstrap
is the only environment step; two pre-flights -- models, then data -- run before the long
job; the runner fails fast; the report cannot fail; and the whole runbook is dry-run
locally on its exact text.

Building the dry-run turned up two more defects before either reached Kaggle. The working
tree had drifted to CRLF in 20 files while the commits held LF, and the zip was built from
the working tree -- so the next upload would have shipped a CRLF bootstrap that Kaggle's
bash cannot run and Git Bash runs happily (I-051, D-058). And the recovery bundle I first
staged would have stopped Cell 3, because it carried a committed manifest with different
content (I-052). The second was caught only by also running the restore and the digest
pre-flight against the real bundle with the real expectations: 9 members verified by
`shard_is_complete`, 18,614 rows, digest `522115667bc4...`, 9 sidecars agreeing.

### P2-D (step 1) — the timing-file collision, and the same bug waiting in Session E — 2026-10-04

S3 is complete: 12 of 12 members, recovered in 1.99 h wall (the full record is the step 2
entry). Its one defect was mine: both recovery jobs ran as `shard 0/1`, so both wrote
`reports/p2_probe_timing_shard0.json`, and GPU 0, finishing last, replaced GPU 1's record
(I-053). Nothing computed was lost -- every member writes its own cache and sidecar -- but
two members' measured cost now exists only in a log.

The user asked for the rule behind the fix rather than the fix alone: no two parallel jobs
may write one path, and every script Sessions D and E call is audited for it. The audit
found the same defect three more times, none of which had run yet. `p2_fit_heads.py`,
`p2_score.py` and `p2_analyze.py` keyed their outputs by protocol, so the second P-A seed
would have silently replaced the first seed's fitted fusions and verdict; and
`setup_logging` named a log file by the second, which two jobs started together share.

Every output path now comes from one module, `dbp.utils.run_paths` (D-061): results are
keyed by run id, and logs and scratch paths also carry the device. One deviation from the
instruction as written, stated here because it is deliberate: the GPU index is **not** in
a checkpoint's path. The queue gives a run to exactly one worker, so the run id is already
unique; and a run resumed in a later session may be picked up by the other GPU, which a
device-keyed checkpoint path would turn into a restart from step 0.

`tests/test_output_paths_unique.py` builds the paths of parallel jobs the way the scripts
build them and asserts none is shared -- the exact S3 layout first (it fails on the old
runner), then every pairing of the four Session D runs over both devices, and a lint that
no script hand-builds a protocol-keyed output name. Tests **694 passed, 7 skipped** (from
672). Ruff clean.

CLAUDE.md gains section 13: plan only on the user's own Kaggle account; if quota is short,
run what fits and continue after the weekly reset.

### P2-D (step 2) — S3 is complete: 12 of 12 members — 2026-10-04

**Result.** All twelve probe caches exist, each **18,614 rows x 20 features** on the probe
list with digest `522115667bc498b745cf6a562a8d5b0e1a17cc491dcfda9f71cc97ef60788010`.
Nine came from the first (failed) commit and were restored byte-identical; `movq_k22`,
`qwen_image` and `realesrgan_x4` came from the recovery run, `COMPLETE` in **1.99 h wall**
on two T4s with both jobs exiting 0 (`reports/kaggle_20261004/p2_s3_status.json`). The
recovery's pre-flights passed before the long job started: 9 restored members verified,
imports, a 2-image smoke test per remaining member, every list row resolving, and **all
2,488 `gen_matched` files** matching the committed `jpg_sha256`
(`p2_preflight_s3_models.json`, `p2_preflight_s3_data.json`). The regenerated
`gen_matched` manifest differs from the committed one only in its two path columns.

**Verification** (`p2_preflight.py --verify-caches`, run locally on the downloaded
output; record `reports/kaggle_20261004/p2_preflight_s3_verification.json`): every cache
matches its sidecar's sha256 and size, holds the list's rows in the list's order, is
built on the list's digest, is finite, and carries the stage's 20 feature names. No two
members' features are identical. The feature files themselves are **not committed**; they
stay in the private Kaggle output and in `F:\thesis-2\kaggle_raw\results_S3_latest.zip`.

| member | mean PSNR (dB) | mean LPIPS-VGG | bytes | sha256 |
|---|---|---|---|---|
| sd15 | 26.58 | 0.1256 | 1519203 | `900e193c6d45a8a8a0da3b239cddf0d30d3c263d968d083be14423ad6196ae6a` |
| sd21 | 28.00 | 0.1070 | 1520426 | `958f353c56d3ee8519663d4fe1cc87f9d9b2ce9402c748ac5efbbc0f159786ed` |
| sd_ft_ema | 27.27 | 0.1110 | 1519242 | `97cfe68c5de06c1c2844ad76e063d9a7aede6e3837d10db39de1a4fe492031f2` |
| sdxl | 28.20 | 0.1066 | 1521577 | `d072c1bca46f36d1b772badf9acc0e2e9ec05c483a37766bab1cd31d3316e48c` |
| sdxl_fp16fix | 28.18 | 0.1084 | 1521318 | `89eb0bb99ddfb746183c1ee5d4b74297f4a779d8933fc9522d2915ce138ae3e9` |
| sd35 | 31.63 | 0.0578 | 1533916 | `26624bfcabf64c564531d88a3d7a8d1f07bfb1691e99c0611f5ffb3bd5f0f7a2` |
| flux | 33.58 | 0.0375 | 1537677 | `cfa2fc5a99337920834c46429f0e3e65a86a8f24da060c259c5c0b097cd0bd52` |
| dcae | 25.39 | 0.1517 | 1517228 | `b6bc8927cfc70b176f4c1f3283a3adcdc827f7c6a18a86d584804e3fd38881e9` |
| realesrgan_x4 | 24.32 | 0.3007 | 1514706 | `b698b4c6ff0648f1bc0f5c208821c1cd8367b6b7e608f9987abd9a970eaba3f6` |
| flux2_klein | 34.31 | 0.0296 | 1537256 | `5c7b30aa8bd54ebeee8936e3fe24513cf177da073a45f961877644703637f611` |
| qwen_image | 32.18 | 0.0562 | 1535447 | `f51883fea22339205c6ba049040fedcca082be64cbc00a7f6d0849a50f97b1ef` |
| movq_k22 | 27.59 | 0.1333 | 1519164 | `231bcec7de6f7916c64ef897fa30b15a77774886358bfe8347d8fc2b3ce2f132` |

The S1 caches, unchanged since P2-C and byte-identical to the recovery bundle (`sha256sum`
on the downloaded output):

| cache | rows x dim | sha256 |
|---|---|---|
| `p2_s1_dinov3-vitb16_train.npz` | 33,909 x 1536 | `d8f7360320a82478f14096c7df6916ab82d2572407e056fef4e5172c5367ad72` |
| `p2_s1_dinov3-vitb16_panel.npz` | 11,775 x 1536 | `06360ed9b7d48d2571db039e04c6642198121421402cb62fdfdf17ffdf7c7093` |
| `p2_s1_dinov3-vitb16_panel_luma.npz` | 978 x 1536 | `99e52c3c08e70f0fb849aa112cc027f9cbdc9f21ba17314ff97af28d26c4e5a0` |

**Cost: 13.35 GPU-h actual against 12.90 authorised (D-056), 13.01 projected after the
first run, and the 14.5 hard stop.** Per member, s/image at crop 320 including LPIPS, from
the `<member>: 18614 image(s) in X s = Y s/image` line of each log
(`first_run_probe_shard{0,1}.log`, `recovery_probe_job{0,1}_gpu{0,1}.log`):
movq_k22 0.3832 (1.98 GPU-h), sdxl 0.3648 (1.89), qwen_image 0.2307 (1.19), dcae 0.1959
(1.01), sdxl_fp16fix 0.1844, sd_ft_ema 0.1842, sd15 0.1813, sd21 0.1813, flux 0.1804,
flux2_klein 0.1793, sd35 0.1789 (0.93-0.95 each), realesrgan_x4 0.1376 (0.71). The overrun
is `qwen_image`, which had no S3 measurement before it ran. This counts the twelve
complete member passes only: the first run's partial `movq_k22` and its idle GPU are lost
time on top, and the length of the recovery *session* is not in the archive.

**Precision check.** S3 must reconstruct each member at the precision its B5 pairs were
built at (`_precision_from_throughput`). Probe precision from each log's `loading
<member> (<kind>, <precision>)` line, against the `precision` column of
`p1_pairs_manifest.csv` (sha256 `b19b7f5a...`, from `kaggle_raw\p1_build_reports.zip`):

| member | probe | B5 pairs | match |
|---|---|---|---|
| dcae | fp16 | fp16 x 4100 | yes |
| flux | fp16 | fp16 x 4100 | yes |
| flux2_klein | fp16 | fp16 x 4100 | yes |
| movq_k22 | fp32 | fp32 x 4100 | yes |
| qwen_image | fp16 | fp16 x 4100 | yes |
| realesrgan_x4 | fp32 | fp32 x 4100 | yes |
| sd15 | fp16 | fp16 x 4100 | yes |
| sd21 | fp16 | fp16 x 4100 | yes |
| sd35 | fp16 | fp16 x 4100 | yes |
| sd_ft_ema | fp16 | fp16 x 4100 | yes |
| sdxl | fp32 | fp32 x 4100 | yes |
| sdxl_fp16fix | fp16 | fp16 x 4100 | yes |

12 of 12 match (9 fp16, 3 fp32), so Step 3 was not blocked.

**Incidents, each with the regression test committed with its fix:**

| incident | what it cost | test |
|---|---|---|
| I-049: `spandrel` missing; one shard died, the other ran 3.5 h alone | 3.5 h of one idle T4, three members unfinished | `test_a_member_that_cannot_run_fails_the_smoke_test`, `test_one_failing_job_stops_the_others_at_once`, `test_runbook_tooling.py` lint `bootstrap only` |
| I-050: the persist cell imported `dbp` before its `sys.path` insert | a ten-hour commit marked failed | `test_no_cell_imports_dbp`, `test_the_report_cell_cannot_raise`, `test_the_report_cannot_fail_even_with_nothing_to_report` |
| I-053: two jobs wrote one timing file | two members' timing survives only in a log | `tests/test_output_paths_unique.py` |

Committed under `reports/kaggle_20261004/`: the status, restore and three pre-flight
records, the recovery timing file (renamed `..._gpu0_only` because that is what it holds),
both runs' probe logs, the 12 sidecars, the regenerated `gen_matched` summary, and the S1
extraction, image-list and split summaries from the first run. Text only, 176 KB.

### P2-D (step 3) — the Session D runbook: four S2 runs, trained and scored — 2026-10-04

**What runs.** `env/kaggle_run_p2_s2.py`, seven thin cells. The four runs from
`configs/p2_runs.yaml`, in queue order: `p_a_sd1` seed 0 (the primary protocol), `p_a_sd1`
seed 1, `4ch_only` seed 0, `full_bank` seed 0, each with the 2.25 GPU-h allowance, the
8,000-step ceiling and a 20-step calibration. One queue, one worker per T4 (D-062); each
worker trains a run, then scores it with that run's selected model on the panel, the luma
companion and the validation rows (D-063). Checkpoints every 500 steps; an attached
previous version's output is restored and unfinished runs resume; complete runs, verified
by the sha256 of their score files, are skipped.

**Pre-flights, both before anything long.** Data: GPU count = 2, `HF_TOKEN` present (the
value never printed), imports, the sha256 and row count of every input in
`dbp-p2-s2-inputs` (D-064), every row of the panel, luma, S1 and pairs lists resolving,
and the regenerated `gen_matched` against the committed hashes. Models: every run on every
GPU -- two real training steps, a checkpoint save, a reload into fresh objects compared
tensor by tensor, and two rows of each list scored -- with the first failure stopping
everything.

**Two defects found before Kaggle.** The smoke pre-flight's first local run failed on
reload: two numpy scalars in the checkpoint, which `weights_only=True` refuses (I-054).
Every resume would have failed in the same way. And the step 3 audit found validation crops
seeded with Python's per-process `hash()`, so no two runs validated on the same crops
(I-055).

**Local dry-run: PASSED**, in two sessions on the exact cell text: in session 1 every run
was killed right after its step-4 checkpoint, the queue logged both failures without
retrying, the report cell ran and the last cell failed the notebook; in session 2, with
session 1's output attached as an input, both runs were restored, resumed at step 5,
finished at step 6, and scored panel 3 / luma 1 / val 6 rows. The backbone in the dry-run
is a random-init two-layer DINOv3 of the pinned architecture (D-065).

**Session E can run on CPU.** It reads only cached features and saved scores: logistic
regressions, a 200-epoch full-batch set head on 12 x 20 features, temperature and fusion
fits, single-shot panel scoring and the paired bootstrap. None of it needs a GPU, and all
of its inputs are, or after Session D will be, in `kaggle_raw`. It is **not ready to run**:
I-056 must be fixed first, because as written the fusions would silently drop S3 and S2.

**Open for the user:** which account owns `adeelshah29`; the meter readings before and after
Session D, which close I-042.

### P2-D (step 3, addendum) — the code dataset's slug — 2026-10-04

Before committing the Session D notebook, the user listed `/kaggle/input` on a CPU session.
The code had been uploaded as a new dataset, `dbp-code-ed7a8cb`, so Cell 1's
`datasets/*/dbp-code/thesis_dbp` would have stopped the run at its first line (I-057).
Cell 1 now matches `dbp-code*`. Only the runbook, the dry-run fixture and a test changed;
no script the notebook executes differs from `ed7a8cb`, so the uploaded code dataset stays
valid. The listing also showed `p1-data` not mounted although attached -- open with the
user.

Cell 1 then found the corpus reals and generations under
`muhammadadeelshah/report`, the slug of the dataset titled `p1-data` (I-058); the user had
edited the pattern by hand. Cell 1 now matches `p1_reals/` and `p1_generated/` under any
slug, so the handed-over text is again the tested text.

### P2-D (step 4) — Session D result: four S2 runs trained and scored — 2026-10-05

**Result.** Commit `P2_training` version 1 (code `ed7a8cb`, cells `8d6571b`) ran in
**12,615.7 s (3.50 h)**; the queue itself took **2.98 h wall, 5.79 device-hours**, and
every run exited 0 in training and in scoring (`reports/kaggle_20261005/p2_s2_status.json`).
Pre-flight 1 passed 10 checks (2 x Tesla T4, `HF_TOKEN present: true`, every input hash,
every list resolving, all 2,488 `gen_matched` hashes); pre-flight 2 passed all 8 smoke
tests (4 runs x 2 GPUs). No run was resumed.

| run | GPU | measured s/step | budget | steps run | best step | in-training val AUC | GPU-h (train) | queue h |
|---|---|---|---|---|---|---|---|---|
| p_a_sd1_seed0 | 0 | 1.0657 | 7,600 | 5,500 | 4,500 | 0.99998 | 1.377 | 1.54 |
| p_a_sd1_seed1 | 1 | 1.0684 | 7,581 | 4,750 | 3,750 | 0.99989 | 1.169 | 1.50 |
| 4ch_only_seed0 | 1 | 0.9399 | 8,000 | 4,500 | 3,500 | 1.00000 | 1.152 | 1.31 |
| full_bank_seed0 | 0 | 0.8566 | 8,000 | 5,000 | 4,000 | 0.99897 | 1.289 | 1.44 |

Every run **stopped early** (1,000 steps without improvement) and is **above the 2,000-step
floor**. The two P-A runs, calibrated while both GPUs were busy, measured 1.07 s/step
against the 0.8361 measured alone, so their budgets fell to 7,600 and 7,581 steps; it did not
bind, since both stopped earlier. The in-training validation saturates (I-059).

**Scores verified** (each list's sha256 equals its summary, image ids and order equal the
input list, every value finite): panel 11,775, luma 978, val 6,883 per run; the LoRA and
head files match the hashes in the score summaries; the backbone revision is the P0 pin
`5931719e67bb`, not the dry-run model. **No panel metric was computed** -- the panel is
analysed once, in Session E.

**Validation, all 6,883 S1 validation rows** (491 reals; 491 reconstructions per member;
500 `gen_matched`), AUC of reals against each member's fakes -- validation only:

| fakes | p_a_sd1 s0 | p_a_sd1 s1 | 4ch_only | full_bank |
|---|---|---|---|---|
| sd15 / sd21 / sd_ft_ema (P-A's own lineage) | 1.0000 / 0.9999 / 0.9999 | 0.9999 / 0.9998 / 0.9998 | 0.9999 / 0.9998 / 0.9998 | 0.9999 / 0.9995 / 0.9996 |
| sdxl / sdxl_fp16fix | 0.9982 / 0.9974 | 0.9979 / 0.9970 | 0.9996 / 0.9994 | 0.9992 / 0.9991 |
| dcae / movq_k22 / realesrgan_x4 | 0.9974 / 0.9995 / 1.0000 | 0.9974 / 0.9992 / 1.0000 | 0.9995 / 0.9999 / 1.0000 | 0.9981 / 0.9995 / 1.0000 |
| sd35 | 0.9736 | 0.9708 | 0.9771 | 0.9990 |
| flux | 0.9291 | 0.9249 | 0.9340 | 0.9950 |
| qwen_image | 0.8550 | 0.8317 | 0.8747 | 0.9989 |
| **flux2_klein** | **0.5978** | **0.6107** | **0.5950** | 0.9819 |
| gen_matched | 1.0000 | 0.9999 | 0.9999 | 0.9994 |
| **all val rows** | 0.94989 | 0.94845 | 0.95226 | 0.99763 |

Read with care: these fakes are *reconstructions of the same COCO reals*, so they measure
whether a decoder's trace transfers, not detection of generated images. Within that: an S2
trained on the SD1 lineage alone transfers almost perfectly to the other 4-channel and
compressing decoders, and degrades in order SD3.5 > FLUX > Qwen > FLUX.2, reaching near
chance on FLUX.2's 32-channel decoder. `4ch_only`, which never saw the 16/32-channel
families either, shows the same pattern; `full_bank`, which did, does not. The two P-A seeds
agree closely (Spearman 0.981 on validation scores).

Score files, sha256 (not committed; `kaggle_raw\P2_Training-results.zip`):

| run | panel.csv | luma.csv | val.csv | lora adapter |
|---|---|---|---|---|
| p_a_sd1_seed0 | `ad4011accd2969c617caec34cf5dbe64667ceb9f6c1f47fbf028a2889302641b` | `b5c5c176565365baecf32d390f97939c62ab897b6dfc6d4b9cf0e7f0b2370d98` | `17347f1eb3646e08e9ea850c8d145721ae91b98f0f8d328ac48eb3a7a9529487` | `f411fba0cd83b24e7fddb7ac3d61cb29440ea9eb06f229f591bfa5680dbf7270` |
| p_a_sd1_seed1 | `9454a184f002f2efab5c0ab0af5ecc369e1134e644cc3a9ecc30beb199604760` | `6cbf069835da99c8109bd687606455c085147f55b07786c22b30a849a6448aa5` | `4274e0c841fadbae8dbac62b5f8780c18f4d537e557235821329515a08233a0f` | `1b0a6c1383d6e2800014ee30e71bc01bf97e01fadfd8683ad4b966f7c48d06d0` |
| 4ch_only_seed0 | `c90a1654408905d45ea8fc12d37125aa726e0c34a0e55d9d342647904531a541` | `272bd4c144bb27e9da164d95bf80decec738b18ff94cb8b993e238bfbb118484` | `71bff1ddafeacc36d76b5dfbf35dd1cfb18aa7e61fbbc7d6f79044e8058b5af6` | `84d99dca8e91a85113d7542413dec538e0bfd257263e72966a6bed3154753d40` |
| full_bank_seed0 | `926427b9a9aed4dc2e2ca3019bd5a13b3706047915578a8aa6b6b8c79f0e8710` | `4053014c4a45fca612e08359c92d002f5821063b1bb59d473a9f88bbab2e43d0` | `90c43049d6c9fbee1a24c48b8ba1eb6ed507ce81b904d3ad3d47d7ae900c958f` | `4cd74c92a2c1ca4f387a44554205fd2821fa09fbaa429c65d05635c7961a6082` |

**Cost.** 3.50 h session against the 5.1 h estimate; the over-estimate is early stopping.
The quota-meter reading after the run has not been reported yet, so **I-042 stays open**.
P2's GPU work is complete; Session E runs on CPU, after I-056 is fixed.

### P2-E (part 1) — I-056 fixed in the fits; the fusion check stops P-A — 2026-10-05

Quota after Session D: **18 h left of 30**, so 12 h were charged against a 3.50 h commit.
The difference is not explained by either billing hypothesis alone (3.5 h or 7.0 h); the
editor session was on GPU for an unknown part of the time ("Running for 41,458 s"), so
**I-042 still cannot be closed**. From now on all runs use `muhammadadeelshah` only, and
the meter is read with the editor on CPU (CLAUDE.md section 13).

**Queue timeout (I-053 follow-up).** After a kill, the queue waited for the process with no
limit; a child stuck in an uninterruptible read could have held the notebook past its own
deadline. The wait is now bounded and such a process is abandoned and logged (5 minutes,
one test).

**I-056.** `dbp.eval.join` and the id-keyed fusion API (D-067), 14 regression tests
including the four required: a shuffled stream gives an identical fit and identical
scores, and a missing id, a duplicate id or a missing stream raises. `p2_fit_heads.py` is
rewritten on it, and the rewrite found the protocol leak (D-068).

**The fusion check (validation only; no panel row was read).**

| run | validation rows | S1 | S2 | S3-LR | S3-Set | AEROBLADE | F-stack standardised weight | check |
|---|---|---|---|---|---|---|---|---|
| p_a_sd1 s0 | 932 | 0.95899 | 0.99997 | **1.00000** | 0.97783 | 0.63033 | S3-LR 97.1%, S2 2.7%, S1 0.2% | **STOP** |
| p_a_sd1 s1 | 932 | 0.95899 | 0.99990 | **1.00000** | 0.98061 | 0.63033 | S3-LR 98.5%, S2 1.3%, S1 0.2% | **STOP** |
| 4ch_only | 1,550 | 0.95027 | 0.99973 | 0.99733 | 0.99350 | 0.66060 | S2 56.5%, S3-Set 23.0%, S3-LR 20.2% | pass |
| full_bank | 2,000 | 0.89837 | 0.99757 | 0.97268 | 0.98435 | 0.72473 | S2 52.6%, S3-Set 28.9%, S3-LR 10.7%, S1 7.9% | pass |

For P-A, AUC on the full validation set by decoder (validation only; P-A heads):

| decoder | S1 | S2 | S3-LR | S3-Set | AEROBLADE |
|---|---|---|---|---|---|
| sd15 / sd21 / sd_ft_ema (in protocol) | 0.96 / 0.95 / 0.93 | 1.00 | 1.00 | 0.98 | 0.53-0.59 |
| sdxl / sdxl_fp16fix | 0.90 / 0.92 | 0.999 / 0.998 | **0.16 / 0.17** | 0.99 | 0.66 / 0.68 |
| dcae / movq_k22 / realesrgan_x4 | 0.92 / 0.96 / 0.999 | 0.998 / 0.999 / 1.00 | **0.44 / 0.42 / 0.68** | 0.99 / 0.999 / 0.99 | 0.50 / 0.60 / 0.93 |
| sd35 / flux | 0.83 / 0.82 | 0.97 / 0.95 | **0.41 / 0.22** | 0.92 / 0.90 | 0.72 / 0.96 |
| qwen_image / flux2_klein | 0.86 / 0.54 | 0.87 / 0.60 | **0.38 / 0.30** | 0.84 / 0.95 | 0.84 / 0.98 |

The panel was not scored. Stopped and reported per the user's rule (I-060).

### P2-E (part 2) — Amendment 2, committed before any panel score — 2026-10-05

The user chose option D with changes (2026-10-05): run the pre-registered analysis
unchanged as a secondary result (Analysis A), and let a fixed-weight fusion decide the
gate. Recorded verbatim in `docs/P2_PREREGISTRATION.md`, Amendment 2 (D-069):

* **Governing fusion F-mean(A2)** over S1, S2 and S3-Set, the members fixed by design
  role (S3-Set is the designated S3 head, B5; S3-LR and AEROBLADE were always ablation
  and baseline heads). No exclusion rule: S2 is also ~1.000 on P-A validation, so one
  would have been arbitrary.
* **Normalisation on validation reals only**: robust z (median, 1.4826 x MAD), clipped to
  +-5; MAD 0 falls back to std, std 0 raises. No fake label, weight or temperature.
* **Best single = maximum over all five heads on the panel, per metric, re-maximised in
  every bootstrap replicate.** Gate statistic F-mean(A2) - max(single heads); rule
  unchanged.
* **Seed rule (ii): P-A is GO only if both seeds pass**, each reported separately.
* D-068, the validation-set definition (932 / 1,550 / 2,000 rows), I-056 and the I-061
  fixes (shared real class per view; macro bootstrap resampling slice fakes and view
  real pools; Jaccard of misses; no silent drop) are written into it.

**One more defect found while writing it (I-062).** Section 4 fixes every threshold "on
validation reals", but `calibrate_at_fpr` maximises validation TPR under the FPR budget;
on saturated heads that puts the threshold at the lowest validation fake. Trial
calibrations (`fit_trial/reports/p2_heads_<run id>.json`, `achieved_fpr`): P-A S2 at
0.0020 (seed 0) and 0.0061 (seed 1) instead of ~0.049, S3-LR at 0.0000 on both seeds;
every non-saturated head sits at 0.0489 / 0.0081 as intended. Amendment 2 point 3
defines the threshold from the reals alone, which reproduces those 0.0489 / 0.0081
operating points and removes the dependence on fakes.

**Disclosure.** The amendment states that the per-decoder validation numbers, including
S3-LR's below-chance AUC on every out-of-protocol decoder, were seen before it was
written. No panel metric exists: the only panel-side files are Session D's S2 score
CSVs (`image_id, score`, no label), checked for counts and sha256 only.

### P2-E (part 3) — Session E code fixed, fitted, dry-run; real panel checked without labels — 2026-10-05

All CPU, local. Order as instructed: Amendment 2 committed first (`5a0b250`), then the
validation report, then the code.

**Fit (validation only).** `p2_fit_heads.py` now fits F-mean(A2) (robust z on the 491
validation reals; all twelve scales took the MAD rule, none fell back), calibrates every
head **and fusion** on the validation reals alone (I-062, I-063), and reports the
per-decoder validation AUC of F-mean(A2) and the exploratory PSNR correlation. Run for all
four runs into `kaggle_raw/session_e/` (heads, not committed: weights) and
`reports/p2_session_e/validation/p2_heads_<run id>.json` (committed). The single-stream
validation AUCs reproduce the trial fit exactly. Every threshold's FPR on the validation
reals is now 0.0489 (5%) and 0.0081 (1%) for all eight scores in all four runs.

Validation AUC, protocol validation set / per decoder on the full validation set (source:
`reports/p2_session_e/validation/p2_heads_<run id>.json`, keys `streams` and
`per_group_val_auc_full_validation`; * = in protocol). P-A seed 0:

| decoder | S1 | S2 | S3-LR | S3-Set | AEROBLADE | F-mean(A2) |
|---|---|---|---|---|---|---|
| protocol validation (932) | 0.9590 | 1.0000 | 1.0000 | 0.9778 | 0.6303 | 0.9955 |
| sd15* / sd21* / sd_ft_ema* | 0.965 / 0.950 / 0.932 | 1.000 | 1.000 | 0.980 / 0.985 / 0.984 | 0.59 / 0.59 / 0.53 | 0.996 / 0.995 / 0.994 |
| sdxl / sdxl_fp16fix | 0.905 / 0.921 | 0.999 / 0.998 | 0.163 / 0.167 | 0.992 / 0.989 | 0.661 / 0.682 | 0.996 / 0.997 |
| dcae / movq_k22 / realesrgan_x4 | 0.921 / 0.957 / 0.999 | 0.997 / 0.999 / 1.000 | 0.436 / 0.424 / 0.683 | 0.993 / 0.999 / 0.994 | 0.503 / 0.601 / 0.933 | 0.998 / 1.000 / 1.000 |
| sd35 / flux | 0.830 / 0.818 | 0.970 / 0.945 | 0.407 / 0.222 | 0.923 / 0.900 | 0.717 / 0.955 | 0.967 / 0.952 |
| qwen_image / flux2_klein | 0.859 / 0.541 | 0.874 / 0.600 | 0.378 / 0.296 | 0.839 / 0.952 | 0.839 / 0.979 | 0.913 / 0.771 |

Seed 1 is within 0.03 of seed 0 everywhere. `4ch_only` protocol validation: F-mean(A2)
0.9993; `full_bank`: 0.9983. On validation F-mean(A2) is above every member on most
decoders and **below S3-Set and AEROBLADE on FLUX.2-klein** (0.771 vs 0.952 / 0.979). Recorded for
the report; it changes nothing in the amendment.

**Exploratory (not part of the gate).** Spearman across the 12 bank decoders between mean
reconstruction PSNR on the validation reals (S3 `psnr_db`) and P-A S2's validation AUC:
**rho = -0.785 (p = 0.0025)** for seed 0 and -0.788 (p = 0.0023) for seed 1; over the 9
unseen decoders only, -0.817 (p = 0.0072). Decoders whose reconstructions are closer to
the input are harder for S2 to transfer to.

**I-056 audit, final list.**

| code | combines arrays by | status |
|---|---|---|
| `p2_fit_heads.py` | was array length | fixed (P2-E part 1): `join_streams` |
| `dbp.eval.fusion` (F-mean, F-stack, **F-mean(A2)**) | was position | fixed: `fit_by_id` / `score_by_id` / `robust_score_by_id`, exact stream sets |
| `p2_score.py` | id dicts, but NaN on a miss and a silent `usable` mask in the fusions | **rewritten**: frozen lists with Amendment 1's digests, per-set metadata joined by id with sha256 agreement, exact S1/S2 joins, superset S3 joins, label agreement, luma parents = E7 FLUX.1+real; `--check-only` reads no label |
| `p2_analyze.py` | rows carry every stream (aligned by construction) | **rewritten**: no silent drop (non-finite raises), slices by shared real pool, macro bootstrap, Analysis A beside the governing gate, P-A slices asserted |
| `complementarity.pairwise` / McNemar / oracle | position, on aligned slice arrays | aligned by construction; non-finite now raises; Jaccard of misses added |
| `macro_bootstrap` | row indices per slice and pool | units joined by parent id; members of one group must hold the same units |

**Tests.** New: `tests/test_amendment2.py` (14), `tests/test_session_e_analysis.py` (17),
`tests/test_p2_score_checks.py` (11). Full suite **795 passed, 7 skipped** (the GPU smoke
tests, PENDING-GPU); ruff clean; integrity CLEAN.

**Synthetic-panel dry-run: PASSED.** `scripts/p2_session_e.py synthetic` builds a
2,000-row panel from the probe validation rows (491 reals in five view pools, 210 luma
rows, every real slice name), scores and analyses all four runs and combines P-A:
12 M1 slices, 7 M2 slices, the section 8 substitution applied, no empty AUC on any slice
with reals, every output stamped SYNTHETIC. Its numbers measure nothing (the heads were
fitted on these rows) and are not reported.

**Real panel, check only: PASSED** for all four runs (`reports/p2_session_e/check_real/`):
panel 11,775 rows digest `bff608c3...`, luma 978 digest `190279cc...` (both = Amendment
1), metadata joined 1,477 / 1,492 / 3,748 / 3,408 / 1,250 / 400 by id with sha256
agreement, 978 luma rows = E7's FLUX.1 and real rows, S1 [12753, 1536], S2 [12753], S3
[12753, 12, 20]. `labels_read: false`; nothing scored.

### P2-E (part 4) — the panel, scored and analysed once: P-A NO-GO — 2026-10-05

`scripts/p2_session_e.py real` at commit `5e76fed` (pre-registration with Amendment 2
committed; tree clean; marker `SESSION_E_REAL_STARTED.json` 09:28:41, `crash_rerun`
null). Scoring 5 s and analysis 9-11 s per run, on CPU. No crash, no re-run. Outputs
copied verbatim to `reports/p2_session_e/real/` (8.6 MB, no file > 10 MB). The full
write-up is `docs/P2_REPORT.md`.

**Verdict (governing, Amendment 2)** -- `real/analysis/<run id>/go_no_go.json`:

| run | M1: F-mean(A2) vs best single, Δ [95% CI] | M2: F-mean(A2) vs best single, Δ [95% CI] | verdict |
|---|---|---|---|
| p_a_sd1 seed 0 | 87.63 vs S2 88.52, -0.89 [-1.77, -0.08] | 74.46 vs S2 80.36, -5.89 [-6.96, -4.80] | NO-GO |
| p_a_sd1 seed 1 | 91.58 vs S2 89.92, +1.66 [+1.00, +2.32] | 75.10 vs S2 80.13, -5.03 [-6.14, -3.98] | NO-GO |
| **P-A** (both seeds required) | | | **NO-GO** (`p2_verdict_p_a_sd1.json`) |
| 4ch_only seed 0 (secondary) | 90.28 vs S2 84.39, +5.89 [+5.11, +6.64] | 71.11 vs S2 76.99, -5.88 [-6.97, -4.88] | "GO" on M1 only, no gate role |
| full_bank seed 0 (secondary) | 95.33 vs S2 89.14, +6.19 [+5.41, +6.94] | 73.14 vs S2 77.40, -4.26 [-5.93, -2.63] | "GO" on M1 only, no gate role |

The best single head is S2 in every run, on both metrics, in 100% of replicates.
**Analysis A** (original section 6, secondary): seed 0 F-stack 35.88 vs S3-LR 31.97 on
M1, +3.91 [+3.35, +4.47], meets the original criterion; seed 1 +1.71 [+1.33, +2.11] does
not; M2 +1.52 and +0.79. F-stack and S3-LR are both 53-57 points below S2 on M1, the
outcome I-060 predicted for an F-stack fitted on saturated validation.

**I-064 (new).** The 5%-FPR validation threshold calls 49-84% of the E3/E4/E5 panel
reals fake for S1, S2 and F-mean(A2) (~4-6% on E7's COCO reals). M1 is therefore TPR at
the validation threshold, not at 5% panel FPR; in the secondary protocols F-mean(A2)'s
M1 "gain" coincides with a higher real FPR than S2's in all six E3/E4/E5 cases. Reported
in P2 section 3.

**Complementarity.** Pooled over the 12 P-A gate slices (seed 0), S2 · S3-Set double
fault 0.165, Jaccard of misses 0.31, McNemar p = 5.2e-13; F-oracle headroom over S2
+0.207 (S1/S2/S3-Set), +0.273 (all five). Errors fall on different images, but
equal-weight fusion does not capture it.

**Not run (P2 section 8):** E3 per-content-type table (needs a go-ahead, since the
panel was "analysed exactly once"), the DIV2K lossless diagnostic, the `gen_matched`
ablation (GPU, budget permitting). Because of these, the tag is **`p2-verdict`**, not
`p2-complete`.

### P2-X1 — EXPLORATORY: is frozen DINOv3 blind to VAE-only images? — 2026-10-05

User request (2026-10-05), CPU only, no P2 result changes. Replicates the question of
*Simplicity Prevails* (arXiv 2602.01738, Table 9) at ViT-B/16. Written up as section
"X1" of `docs/P2_REPORT.md`.

**Setup.**

- `scripts/x1_gen_only_probe.py`: frozen DINOv3-B features from the existing S1 caches
  (none missing) and a logistic regression with S1's settings (`fit_probe`).
- Trained on train-split reals vs train-split `gen_matched` only: 3,101 fit images.
  C = 0.1 was chosen on an 813-image COCO-group hold-out of the train split.
- Tested on the val split: `gen_matched`, plus each decoder's reconstructions of the
  491 val reals. Also E7 on the panel, from existing scores.
- S2 (P-A seed 0, full_bank seed 0) side by side.
- Thresholds at 5% FPR on the val reals; 1,000-resample bootstrap CIs, paired by source
  image for reconstructions.
- A re-run reproduced `x1_results.csv` byte for byte.

**Result (AUC, gen-only probe).** Source: `reports/p2_x1/x1_results.json`.

- `gen_matched`: 0.999. Reconstructions, by decoder:

  | decoder | AUC |
  |---|---|
  | sd15 | 0.893 |
  | sd21 | 0.813 |
  | sd_ft_ema | 0.819 |
  | sdxl | 0.796 |
  | sdxl_fp16fix | 0.796 |
  | sd35 | 0.721 |
  | flux | 0.708 |
  | dcae | 0.783 |
  | realesrgan_x4 | 0.996 |
  | **flux2_klein** | **0.546** |
  | qwen_image | 0.854 |
  | movq_k22 | 0.887 |

- At 5% FPR it catches 16–46% of the reconstructions for ten decoders, 5.9% for
  FLUX.2-klein and 97.4% for Real-ESRGAN (an upscaler, not a VAE).
- E7 on the panel: 0.689–0.881. The FLUX.1 figure is an upper bound because of the
  chroma shortcut.
- S2: P-A seed 0 is ≥ 0.997 on every decoder except sd35 0.974, flux 0.929,
  qwen 0.855 and flux2_klein 0.598. full_bank is ≥ 0.982 everywhere.

**Reading.** Frozen DINOv3-B is not blind to VAE-only images here, but it is weak on
them.

**Files.**

- New: `scripts/x1_gen_only_probe.py`, `tests/test_x1_split.py` (6 tests: no image id
  and no COCO group in both training and test; reconstructions paired with their own
  val real), `reports/p2_x1/x1_results.{json,csv}`.
- The figure (`x1_auc_by_decoder.png` / `.svg`) is in `F:\thesis-2\kaggle_raw\x1\`,
  **not committed** (CLAUDE.md §2, image files), pending the user's decision.

### P2-X — EXPLORATORY (post-verdict) X1–X3 for the supervisor report — 2026-10-06

This came from the user's instruction of 2026-10-06. Everything ran on CPU. The verdict
is unchanged: no P2 model was retrained and no P2 output was rewritten. The write-up is
section "X" of `docs/P2_REPORT.md`, which includes a 10-point plain-language summary
with the results that go against the hypothesis.

**Code.**
- `scripts/p2_exploratory_x.py` with subcommands `x1`, `x2`, `x3a`, `x3b`, `x3c` and
  `all`.
- Tests in `tests/test_exploratory_x.py` (10). They check:
  - the vectorised slice-calibrated threshold is exactly `calibrate_on_reals`;
  - the vectorised AUC is exactly `roc_auc`;
  - paired resampling pairs;
  - the X3b orders are fixed and add one lineage per step;
  - LaTeX escaping.

**Outputs.**
- `reports/exploratory_x/{x1,x2,x3a,x3b,x3c}/`: CSV, LaTeX booktabs and JSON; 29 files,
  668 KB.
- Figures `x3a_min_lpips_by_decoder` and `x3b_auc_vs_k` (png and svg) are in
  `F:\thesis-2\kaggle_raw\exploratory_x\figures\`. They are **not committed**
  (CLAUDE.md §2), pending the user's decision.

**Run.** A 20-resample smoke run took 18 min. The full run used 1,000 resamples, five
analyses in parallel, the longest 560 s. Point estimates are identical between the two.

**Hard checks passed.**
- F-mean(A2) is reproduced from the S1/S2/S3-Set columns (≤ 1e-4).
- The validation thresholds of S1/S2/S3-Set are reproduced from the frozen heads
  (≤ 1e-3) before the pairs' thresholds are derived.
- X3b at k = 12 reproduces the P2 full_bank S3-LR (0.9727) and S3-Set (0.9844) validation
  AUC.
- All four runs' panel rows are identical (X3c).

**Design choices, fixed before any number existed.**
- Slice-calibrated thresholds are re-fitted in each bootstrap replicate.
- Gains are taken against the best of S1/S2/S3-Set, re-maximised per replicate.
- Pairs use Amendment 2's robust z with each run's validation-real parameters. Their
  validation thresholds come from re-scored validation reals.
- F-oracle takes the max z over modules for fakes and the min z for reals. It is an
  upper bound only, and too loose to be informative.
- E3 content-type slices pit that type's fakes against that type's reals.
- X3b orders are "near-first" and "far-first", both starting at sd15. Training images
  are full_bank's S3 subset; the settings are C = 1, 200 epochs and no validation
  selection.
- X3a variants: sd15; the SD1 family; all 11 VAE members (Real-ESRGAN excluded, as in
  AEROBLADE).

**Headline (exploratory).**

| question | result | source |
|---|---|---|
| X2: does any pair or fusion beat S2 on unseen-slice macro AUC? | No, in any run. S2 0.770–0.804; best pair S2+S3-Set −3.64 to −6.89 points; F-mean(A2) −4.26 to −5.89; F-stack −2.85 to −25.16; every CI below 0 | `x2/x2a_macro.csv` |
| X2: S2 TPR at a slice-calibrated 5% FPR | 0.36–0.48 (I-064 quantified) | `x2/x2a_macro.csv` |
| X2c: do the modules' errors overlap? | Largely: Jaccard of misses 0.54–0.77; oracle accuracy headroom 0.04–0.07 | `x2/x2c_complementarity.csv` |
| X2b: E3 by content type | Structure is caught only by S3-LR (0.75) and AEROBLADE (0.72); texture by S2 (0.84); hybrid by nothing | `x2/x2b_per_slice.csv` |
| X3a: min-LPIPS over all 11 VAEs | Worse than one decoder on SD1 reconstructions (0.53–0.59 vs 0.98–0.99) and on E4 (0.47); better only on FLUX / FLUX.2 / Qwen | `x3a/x3a_auc.csv` |
| X3b: learned heads, k = 1 → 12 | Validation 0.81 → 0.97 and P-A gate slices 0.64 → 0.72 rise; full_bank-unseen E3/E4/E5 does not (S3-LR 0.648 → 0.638, S3-Set 0.650 → 0.604) | `x3b/x3b_decoder_count.csv` |
| X3c: S2 by protocol | full_bank beats P-A on unseen E3 (GPT-Image2 0.663 vs 0.492; Nano-Banana2 0.825 vs 0.723), is slightly worse on E4 (0.927 vs 0.955) and equal on E5 | `x3c/x3c_s2_auc.csv` |

**Not run.** GPU S2 trained on sd15 only: estimated about 1.5 GPU-h, 2.6 h worst case,
from Session D's measured 1.15–1.38 GPU-h per run. It needs a new protocol config and a
runbook, both pending approval.

### P2-close (part 1) — charts may be committed (CLAUDE.md §2b) — 2026-10-06

User decision (2026-10-06): charts generated by our scripts from results (SVG, PDF or
PNG, ≤ 1 MB each) may be committed under `reports/**/figures/`. Dataset images stay
forbidden. Recorded as CLAUDE.md §2b, appended; §2 is unchanged.

- Enforced by `tests/test_repo_figures.py`. Every tracked image must be a chart type under
  `thesis_dbp/reports/**/figures/` and ≤ 1 MB; 7 tests.
- Committed in `8534378`: `reports/p2_x1/figures/x1_auc_by_decoder.{png,svg}` (82 / 87
  KB), and `reports/exploratory_x/figures/x3a_min_lpips_by_decoder.{png,svg}` (71 / 78
  KB) and `x3b_auc_vs_k.{png,svg}` (104 / 95 KB). They are byte copies of what the
  scripts wrote to `F:\thesis-2\kaggle_raw\`.
- The two P2 report passages that said "not committed" were updated in the follow-up
  commit. The report and log edits meant for `8534378` failed in a shell patch, and the
  commit went ahead without them.

**Task 1 (fill the `\TBD` markers in `docs/report/thesis_report_updated_sections.tex`)
is blocked.** The file does not exist in the repository, under `F:\thesis-2` or under
the user profile, so no marker was filled.

### P2-close (part 2) — DIV2K lossless diagnostic run; gen_matched ablation deferred — 2026-10-06

**DIV2K lossless diagnostic** (pre-registration §7, CPU).

- `scripts/p2_div2k_diagnostic.py` reads the committed Session E scores of the
  `t_ctrl_diagnostic` view (200 never-JPEG DIV2K reals, 200 lossless DIV2K
  reconstructions; out-of-sample since D-051). Nothing was re-scored.
- **Correction.** The P2 report's section 8 had said DIV2K was not in the panel; it
  is the T-ctrl view.

Results are in `reports/p2_div2k/`, at the 5%-FPR validation threshold:

- **False alarms on DIV2K reals:** S1 52–68% and S2 31–54%, against 4–6% on E7's
  COCO reals. Every gap CI excludes 0.
- **S3-Set** flags 9.5–22% of them.
- **Ranking:** S2's DIV2K real-vs-reconstruction AUC is 0.91–0.96 and S3-Set's
  0.92–0.94, so the ranking survives and the failure is calibration (as in I-064).
- **Caveat:** DIV2K differs in content and resolution as well as compression.

**`gen_matched` vs raw ablation: DEFERRED to P3** (user decision 2026-10-06; needs GPU).

**E3 per content type:** DONE, exploratory (X2b).

### P2-X1c — X1 format control: runbook ready, Kaggle run pending — 2026-10-06

This comes from the user's instruction of 2026-10-06: re-encode all X1 images with
identical JPEG settings (q95, 4:4:4), re-run the probe, and report the result beside
the original. **It needs pixels, which exist only on Kaggle** (CLAUDE.md §6a(a); there
are 0 images under `F:\thesis-2\kaggle_raw`). So it is a Kaggle session, prepared and
dry-run here.

**Code.**
- `dbp.utils.io.reencode_jpeg(arr, quality)`: an in-memory JPEG round-trip with
  `subsampling=0` (4:4:4), no file written.
- `p2_extract_s1.py --reencode-jpeg Q`: applied to the full image before the centre
  crop. Off by default, so P2's caches never used it. It is recorded in the extraction
  report, along with `--limit N` (pre-flight) and `--device auto`.
- `scripts/x1_format_control.py` with three subcommands:
  - `build-list`: the X1 image list;
  - `summarise`: the report cell, which cannot fail;
  - `analyse`: local, after the session. It compares three arms — the original caches,
    the same images re-extracted as-is in the session (reproducibility), and the images
    re-encoded — with paired CIs.
- `env/kaggle_run_x1_format.py` and `env/x1_format_expect.json`.
- `runbook_dryrun.py` fixture `x1_format`.
- `tests/test_runbook_tooling.py`: the report-cell and pre-flight-order lints now find
  the report cell by its title and recognise the extraction as a long job. Both rules
  still apply to every runbook.

**List.** `reports/p2_x1/x1_format_control_images.csv` has 14,205 rows, sha256
`889bc1290fd3c778…`:
- train: 3,101 fit + 813 tune;
- val: 491 reals, 500 `gen_matched` and 5,892 reconstructions;
- E7 panel: 3,408.

**Tests.** `tests/test_x1_format_control.py` (7) checks that the re-encode is
deterministic and really 4:4:4 (compared with a 4:2:0 encode on 1-pixel chroma
edges), and that the committed list matches the expect file's sha256 and rows on LF
bytes.

**local dry-run: PASSED** (89 s, index; repeated at `--ref HEAD` after the commit). On
21 fixture rows covering every root and kind, both arms hold every row, and re-encoding
changed the features (max |Δ| = 5.54).

**Expected cost** (`env/x1_format_expect.json`):
- **GPU T4:** about 0.76 h wall. That is 2 × 14,205 images × 0.0301 s/image (measured
  in Session C), about 14 min, plus 0.52 h of setup (measured in Session D). The quota
  charged depends on I-042.
- **CPU:** possible and free of GPU quota, but NOT-MEASURED.

**Caveat.** Re-encoding at 4:4:4 cannot restore chroma a 4:2:0 source already lost
(§6d), so E7 FLUX.1 keeps its shortcut.

### P3-A — Phase A: plan and audit (no training code) — 2026-10-08

This comes from the user's instruction of 2026-10-08. It produced `docs/P3_PLAN.md` and
a **DRAFT** `docs/P3_PREREGISTRATION.md`. Its thresholds are marked PROPOSED, and it
binds only when the user sets them and commits it as FROZEN, before any P3 model is
trained. Nothing was run on GPU and no data was downloaded.

**Audit facts this session established, with sources.**
- **Unused pools after the P2 draw:**
  - E3: 226 real and 226 + 226 fake;
  - E4: 502 real, 501 fake;
  - E5: 12,863 real, 9,170 fake;
  - E8: 3,750 fake.

  Sources: `reports/kaggle_20260928/p1_eval_*_summary.json` against Amendment 1's draw.
- **E7 shares COCO val2017 source images across all six VAE subsets and the reals.** 745
  of the 2,634 COCO ids in the P2 panel appear in more than one subset (from
  `p2_panel_sets/p2_panel_e7_dda_coco.csv`). So Panel B's disjointness is enforced on
  the COCO id, and each id appears at most once in Panel B.
- **MNW** is still blocked on its download route (`docs/data_cards/e6_mnw.md`).
- **`sd21` ≡ `sd_ft_mse`** is byte-identical (P1-K). LOLO has 8 lineage folds, and the
  SD1 fold also holds out `gen_matched`.
- **Measured costs used in the estimates:**
  - bank reconstruction: 1.177 GPU-h per 1,000 reals (P1-K);
  - S3 features: 13.35 GPU-h per 18,614 rows (Session C);
  - S2 step: 0.86–1.07 s;
  - S2 scoring: about 0.028 s per image (`reports/kaggle_20261005/`).
- **The pHash duplicate threshold** is 6 bits (`dbp.data.phash.DUPLICATE_THRESHOLD_BITS`).

**Plan in one line.**
- **Panel B:** a once-only test panel, disjoint from the P2 panel, frozen on a Kaggle
  CPU session before any training.
- **Training:** 8 LOLO folds, whose nested held-out-lineage AUC gives `N*`, the
  non-saturating selection signal (I-059); a bank-size curve (7 new runs); the
  deployable model and its ablations (6 runs).
- **New reals:** 3 candidate sources, with licences NOT-VERIFIED until approved.
- **Calibration and robustness:** synchronised degradation chains, few-shot calibration
  (CPU), and robustness curves.
- **Budget:** core **57.7 device-h** (derived), optional gate +7.0. That is about 2.5
  weeks at ≤ 30 device-h per week. Quota billing stays unknown (I-042); the first P3 GPU
  session measures it.

### P3-A (part 2) — review decisions applied; licences checked (metadata only) — 2026-10-08

The user's decisions on the P3-A review are written into `docs/P3_PLAN.md` and
`docs/P3_PREREGISTRATION.md`, which is still **DRAFT**. It becomes FROZEN only after the
user approves the diff.

**Decisions applied.**
- **Panel B:** E8 is in it as a secondary fakes-only view. MNW is not; it becomes a
  separate once-only Panel C under the same frozen analysis.
- **Real sources:** Open Images V7 2,000 + Unsplash Lite 2,000, with 10% of each held out
  as validation reals. RAISE is optional.
- **Seeds:** two for S2-B and S2-D only.
- **Thresholds**, all set by the user:
  - `N_max` 6,000;
  - H1-a 0.90 / 0.85 / 0.75; H1-b 0.85;
  - H2 −2;
  - H3 15% / 25% / 50% relative (97.5% CI), with ≤ 10% reported descriptively;
  - H4 +2; H5 5.
- **A:** H1-c added (LOLO-f vs BANK-1 on folds 2–8, +2 points, paired 95% CI).
- **B:** the deployed k = 0 threshold is set on the union of the 491 COCO validation
  reals and each new source's 10% hold-out, with the COCO-only threshold reported beside
  it.
- **C:** H3-k now uses (Q), a binomial-bound order statistic, and (S), a logit shift.
  Success: FPR ≤ 7.5% in ≥ 80% of draws at k = 50.
- **D:** E3's small sample is stated.

**Exact (Q) rule** (`scipy.stats.binom`, this session):
- **k = 50 → j = 1**, the highest real (P(FPR > 7.5%) = 0.0203). j = 2 gives 0.1025, just
  over the 10% bound, so it is not "about the 2nd-highest" as first suggested.
- **k = 100 → j = 4** (0.0526).
- **k = 10:** not attainable (j = 1 gives 0.4586).

**Licence check (metadata only, raw fetch 2026-10-08 07:54 UTC; no image fetched).**

| source | host contacted | page sha256 |
|---|---|---|
| Open Images V7 | `storage.googleapis.com/openimages/web/factsfigures_v7.html` | `03f731edf0a6524ae0403253fb3d805b51a42bbaf45a436bfc4491a650fa4fb0` |
| Unsplash Lite | `raw.githubusercontent.com/unsplash/datasets/master/README.md` | `94230898482042473196be7679f0c21526964076fcd975e60eb6b921820d58dc` |
| Unsplash Lite | `raw.githubusercontent.com/unsplash/datasets/master/TERMS.md` | `79bec96fe07431e1c40efcdf9f9753da24338a3cb97d7d22ac29dc830e3e2437` |
| RAISE | `loki.disi.unitn.it/RAISE/download.html` | `171657e028b45b2b2b101b9529f927472b445a10fd914cc4fec89c8ccc15b86f` |

The verbatim licence sentences are in `P3_PLAN.md` §4a. What they mean for P3:
- **Open Images:** images are "listed as having a CC BY 2.0 license", with no warranty,
  so the licence must be checked per image.
- **Unsplash Lite:** training use is permitted, but the terms forbid disclosing or
  publishing "any portion of the Licensed Data". So the images and their reconstructions
  stay private and never appear in the thesis.
- **RAISE:** non-commercial research use. RAISE-All is about 350 GB, so RAISE-1k is
  about 43 GB (derived), in camera raw. Recommended: skip.

The hosts these pages name for the image fetch are **not** approved yet.

**Report files.**
- `docs/report/thesis_report_updated_sections.tex` (28 `\TBD` markers) moved to
  `docs/report/archive/`, left untracked.
- `docs/report/thesis_report_v3.tex` (0 `\TBD`; it references six PNGs under `images/`
  that are not in the repository) is committed.

### P3-A (part 3) — Changes 1–4 applied; platform check; NOT frozen — 2026-10-08

**What was applied (`docs/P3_PLAN.md`, `docs/P3_PREREGISTRATION.md`, still DRAFT).**
- **Change 1:** Unsplash Lite dropped (it is E5's real source), and RAISE dropped. Open
  Images V7 only: 4,000 = 3,600 train + 400 held out, CC BY 2.0 per image, attribution
  file. The k = 0 union is 491 COCO validation + 400 Open Images reals. Unsplash and any
  set built from it are excluded as training sources. The platform rule is in
  pre-registration §2.
- **Change 2:** FPR macros count each real pool once (E3, E4, E5). AUC and AP macros
  stay over the four unseen slices.
- **Change 3:** E5 calibration pool 500. Single-draw cells (E3 k = 50, E4 k = 100) carry a
  bootstrap CI over test reals. E3 k = 25 is added, descriptive. H3-k is judged on E4 and
  E5. The E4 overlapping-draw limitation is stated. (Q) at k = 25: j = 1 gives 0.1424,
  bound not met (`scipy.stats.binom`).
- **Change 4:**
  - the macro wording ("MNW is Panel C only; E8 is in no macro");
  - k = 1 is sd15 alone, with k ≥ 2 adding whole lineages and the decoder count shown
    beside k;
  - the S2-D row (3,600 + 400);
  - the budget is unchanged at 55.3 device-h core.
- **User decision 2 (accepted):** H3 also reports the three-step table (P2 COCO-only →
  P2 union → S2-D union).

**Platform check: test real-image sources vs Open Images (Flickr).** Raw fetches on
2026-10-08, 07:54 and 08:46 UTC; documents only, no image fetched.

| view | real-image source | verbatim quote | link (page sha256) |
|---|---|---|---|
| E3 (MLLM, GPT Image2 / Nano Banana2) | **NOT STATED** | README: "This dataset contains real and AI-generated image samples organized for binary MLLM-generated image detection." Paper: "a unified preprocessing procedure for all real and generated images … across images from different sources". Neither names a platform. A search of the paper for Flickr, Open Images, Unsplash, Pexels, COCO, ImageNet and LAION finds 0 hits | `huggingface.co/datasets/zr-zhang/MLLM-Generated-Image-Detection-Dataset/raw/main/README.md` (`1c7e4f0b…`); `arxiv.org/html/2608.01258` (`7ba471f3…`) |
| E4 (WildRF) | Reddit, Twitter, Facebook | "a realistic benchmark consisting of images sourced from popular social platforms. Specifically, we *manually* collected real images and fake images using keywords and hashtags associated with the suitable content." The platforms are named by its directory layout (`reddit/`, `twitter/`, `facebook/`) and its protocol ("train on one platform (e.g., Reddit) and test … on … Twitter and Facebook") | `raw.githubusercontent.com/barcavia/RealTime-DeepfakeDetection-in-the-RealWorld/main/README.md` (`0e644001…`) |
| E5 (Chameleon) | Unsplash ("platforms like Unsplash") | "Eventually, we collect over 20K images from platforms like Unsplash ( uns, ) , which is an online community providing high-quality, free-to-use images contributed by photographers worldwide." Table 1 caption: "a similar scale of real-world photographs from uns ()" (reference 5: Unsplash). 0 hits for Flickr or Open Images | `arxiv.org/html/2406.19435` (`aefc794b…`) |

**Verdict: NOT FROZEN.**
- **E4 and E5** name no Flickr or Open Images source. E5's "platforms like Unsplash"
  leaves room for unnamed other platforms.
- **E3's real-image source is not stated at all**, so the platform rule cannot be
  confirmed for E3.
- Per the user's instruction, this is **reported, not decided**. Freezing waits for the
  user.
- **A second item also needs a decision not written in the instruction:** whether
  BANK-2 also adds the rest of the SD1 lineage (`sd21` ≡ `sd_ft_mse`, `sd_ft_ema`) to
  sd15. The decoder counts depend on it.
- Phase B has not started.
- **E3, third document checked:** the code repository named by the paper,
  `raw.githubusercontent.com/xbrainnet/SAP-DSP/main/README.md` (sha256 `6cd7f535…`). It
  says only "The MLLM-Generated Image Dataset contains real images and generated images
  from GPT-Image2 and Nano-Banana2." No source is named.
- **E3's real-image source is therefore stated in none of its three primary documents**
  (Hugging Face README, paper, code README).

### P3-A (part 4) — decisions 1 and 2 applied; pre-registration frozen — 2026-10-08

**Decision 1 (E3, option b).**
- **The source-statement principle** is written in pre-registration §2:
  - a view whose real-image source is stated nowhere is excluded from every FPR
    decision rule (E3);
  - a view with a stated source is kept, with unnamed extra platforms recorded as a
    limitation (E5).
- **H3** decides on the E4 and E5 real pools only. E3 is descriptive and stays in the
  three-step table.
- **H2** is kept as pre-decided (four unseen slices) and also reported without E3, with
  both verdicts stated if they differ.
- **Dedup:** the Open Images draw is checked against **all** real images of E3, E4 and
  E5 (full datasets), with matches removed and counted per view.
- **Descriptive only:** E3 reals that pHash-match COCO train2017 + val2017.

**Decision 2 (BANK-k).**
- SD1 is complete from k = 2, and k = 8 = the 12-decoder bank = S2-B.
- **Decoder counts verified this session** against `configs/decoder_bank.yaml` and the
  committed `X3B_ORDERS` (lineage sizes SD1 3, SDXL 2, 16ch 2, the other five 1 each).
  They are **identical to the user's numbers**:
  - near-first 1 / 5 / 8 / 10 / 12;
  - far-first 1 / 4 / 6 / 9 / 12.
- The two orders share no decoder count between k = 2 and 6, so equal-count comparison
  is descriptive (AUC vs decoder count).
- **`BANK-1L`** (whole SD1, 3 decoders) is added, descriptive, at 2.2 device-h. The core
  budget is 55.3 → **57.5 device-h**, under the user's 60 limit, so it is added without
  a further question.

**Consistency edits (no new decision):**
- the plan's FPR-macro line;
- the claim-scope line ("+ MNW, + E8" replaced by: MNW via Panel C, E8 descriptive).

**FROZEN (2026-10-08).** `docs/P3_PREREGISTRATION.md` is frozen from draft commit
`b562216ea1793e530ba9f55e68ca8e1a5b555634`. Only the status block changed; the plan's
status line now says "approved". Phase B begins with the Panel B draw on a Kaggle CPU
session.

### P3-B (step 2 done, step 1 paused) — Open Images metadata; an E3 source-identity question — 2026-10-08

**Step 2: Open Images V7, metadata only; no image fetched.** Raw fetches on 2026-10-08,
stored in `F:\thesis-2\kaggle_raw\p3_open_images_metadata\`.

| file | from | sha256 |
|---|---|---|
| `oi_download.html` | `storage.googleapis.com/openimages/web/download_v7.html` | `6d71544e85e5205878573bf489736927e9cc2c552e56878b5c8ff3a24803954a` |
| `oi_validation_meta.csv` (15,245,485 bytes) | `storage.googleapis.com/openimages/2018_04/validation/validation-images-with-rotation.csv` | `ed93a0e121fe345effdfc7359b848dbc64a1ff6778c8c73563157cb500b33a17` |
| `oi_downloader.py` (official downloader) | `raw.githubusercontent.com/openimages/dataset/master/downloader.py` | `614947a4f8d8063292a5e5583c30b29e8b636de283949014b7a13ea5535cdb0b` |

- **Validation split:** 41,620 images. **All 41,620 list `https://creativecommons.org/licenses/by/2.0/`**, so the CC
  BY 2.0 filter leaves 41,620. Of those, 41,144 have rotation 0 or unrecorded.
- **Columns:** every row carries `Author`, `OriginalURL`, `OriginalLandingURL`, `License` and `OriginalMD5`, which
  is enough for the attribution file.
- **Candidate image hosts (not contacted):**
  - **(a)** the official downloader's AWS S3 bucket: `BUCKET_NAME = 'open-images-dataset'`, unsigned `boto3`
    access, objects `<split>/<image_id>.jpg`;
  - **(b)** the original Flickr URLs, `farm*.staticflickr.com` (all 41,620 `OriginalURL`s are there).
- **Pending:** the user approves a host before any image fetch.

**Step 1 (Panel B) facts gathered.**
- **E7:** P1's E7 build has 5,000 reals and 5,000 per VAE subset (FLUX.1 4,971; SD3.5 4,998;
  `reports/kaggle_20260928/p1_eval_e7_dda_coco_summary.json`), confirming one COCO val2017 id set across subsets.
- **Recorded P1 build arguments:** the globs and fields of every view are in the committed P1 summaries. E5's
  pipeline-matched view: width 500, q75, 4:2:0, lanczos.
- **E4 needs `gdown`:** the bootstrap has no `gdown` option, so one must be added (session-only, pinned 5.2.0,
  §6a(c)).

**E3 source identity: a decision is needed.**
- **The numbering pattern:** E3's files are numbered per content folder, and the same number recurs across the
  real, GPT-Image2 and Nano-Banana2 folders.
- **If (folder, number) is a source image,** the P2 panel already uses 707 of E3's 726 source ids (Texture 210/216,
  Structure 304/313, Hybrid 193/197), leaving about 19, so E3 would essentially leave Panel B.
- **Under the frozen plan's rule** (E3 source = the distributed file path), about 226 per label remain.
- **The pHash test does not confirm pairing.** Same-number real/fake pairs in the P2 panel have median Hamming
  distance 32 (GPT-Image2, n = 343) and 30 (Nano-Banana2, n = 341), against 32 for different-number pairs; none
  is within 6 bits.
- **But semantic pairing is possible.** The E3 paper's image-to-text-to-image and image+prompt protocols start
  from a real image.
- **Nothing was drawn.** The draw waits for the user's decision.

### P3-B (step 1 built) — the Panel B runbook, dry-run PASSED — 2026-10-11

**User decisions applied (2026-10-11).**
- **E3:** option 1, the frozen rule; the limitation text and the `e3_number_in_p2` flag
  are in `P3_PLAN.md` §1.
- **Open Images:** Flickr originals, with the fetch rules (a)–(f) in `P3_PLAN.md` §4a. The
  fetch runs after the Panel B digest.
- **gdown:** P1's pin (`gdown==5.2.0`), `--no-deps`, behind the bootstrap's opt-in, with
  torch, torchvision and numpy printed before and after (they must be unchanged) and
  `import gdown` verified.
- **Tag:** `p3-prereg-frozen` kept; nothing pushed.

**Built.**
- **Rules:** `src/dbp/eval/panel_b.py`. The draw is a seeded walk:
  - P2 id and source exclusion (E7 by COCO id, others by file path);
  - crop eligibility ≥ 320;
  - sha256 and pHash ≤ 6 dedup against the references and against rows already in Panel B;
  - each E7 COCO id used once;
  - calibration pools first;
  - every rejection counted by reason;
  - shortfalls reported, never backfilled.
- **Scripts:**
  - `scripts/p3_fetch_eval_sets.py` — P1's pinned HF revisions and Drive id, P1's roots, zip
    members that would escape the target refused, archives deleted after extraction;
  - `scripts/p3_build_eval_views.py` — `p1_build_eval.py` with the arguments each committed
    P1 summary records. The run stops unless every one of the 12,150 P2 rows is reproduced
    byte-identically and the label counts equal P1's;
  - `scripts/p3_build_panel_b.py`;
  - `scripts/p3_materialise_panel_b.py` — reuses P2's verified copy, luma companion and
    digest functions, and removes the temporary derived images afterwards.
- **Bootstrap:** `env/kaggle_bootstrap.sh` gains `DBP_CPU_SESSION` (skips the GPU checks
  only) and `DBP_WITH_GDOWN`. Both are off by default, so every existing runbook is
  unchanged.
- **Runbook:** `env/kaggle_run_p3_panel_b.py`, with `env/p3_panel_b_expect.json`.
- **P2 panel rows committed** as `reports/p2_panel_sets/` (12,150 rows, 4.2 MB, metadata
  only). Checked this session: every one of the 11,775 frozen P2 rows is in them with an
  identical sha256.
- **Tests:** `tests/test_panel_b.py` (10) and `tests/test_p3_panel_b_tooling.py` (7).
- **Runbook lint generalised:**
  - the report cell is found by title;
  - the long-job markers include the fetch;
  - the `gen_matched` reference check applies only to runbooks that regenerate it.

**local dry-run: PASSED** (63 s, index):
- 75-row Panel B: 67 test, 8 calibration;
- luma 4;
- `e3_number_in_p2` exercised;
- no P2 row in Panel B;
- every listed image materialised;
- the derived images cleaned up.

**Checks:** full suite 854 passed, 7 skipped, plus the lint fix; integrity CLEAN.
- **local dry-run at `--ref HEAD` (`a26b9e8`): PASSED** in 63 s, with the same digest as the index run
  (`4c4e141a…`; fixture only), so the draw is deterministic.
- **Code zip** (from git's bytes): `dbp-code-a26b9e8.zip`, 6,540,617 bytes, 354 entries, sha256
  `9052aa4b1978887df110cedfc25ad3698089685ea816c268a809bc9e5f0cef5e`. It is stored in `kaggle_raw`.
- **Status:** handed over for the Kaggle CPU run (PENDING-KAGGLE).

### P3-B (step 1, Kaggle run 1) — stopped at the E7 view check; fixed — 2026-10-11

**What happened.** The first Panel B session (code `a26b9e8`, CPU) ran pre-flight, the
fetch and four view rebuilds, then stopped in Cell 5 as designed, before any draw:
`e7_dda_coco: 500 P2 row(s) missing ... 0 with a different sha256`. The rebuilt E7 had
29,969 fakes and **no reals** (`matched 29969 file(s) labelled fake`, no real line).

**Verified in that run (user's pasted output):**

| view | rows | P2 rows reproduced byte-identically | manifest identical to P1's | build time (log stamps) |
|---|---|---|---|---|
| e3_matched | 2,178 | 1,500 of 1,500 | True | 15:34:37 → 15:38:55 |
| e4_wildrf_matched | 2,503 | 1,500 of 1,500 | True | 15:38:55 → 15:45:33 |
| e5_chameleon_pipeline | 26,033 | 4,000 of 4,000 | True | 15:45:33 → 16:11:58 |
| e7_dda_coco | 29,969 (0 real) | **FAILED**: 500 missing | — | 16:11:58 → 16:25:25 |

So `gdown` reached Google Drive and WildRF fetched correctly. E3, E4 and E5 reproduce P1
exactly, down to the manifest hash.

**Cause.** DDA-COCO ships only the six VAE reconstructions. In P1, MSCOCO `val2017`
from `awsaf49/coco-2017-dataset` was placed under `DDA-COCO/real/` (E7 data card,
"Measured on ingestion"; log P1-K). `p3_fetch_eval_sets.py` copied P1's archive steps
but not that one. The dry-run fixture hid the gap: it wrote the reals straight into
`real/`, so the fetch step never needed to.

**Fix (commit `692ab48`).**
- **Fetch script:** `p3_fetch_eval_sets.py --coco-val DIR`
  - copies every `*.jpg` byte for byte into `<root>/real/`;
  - requires exactly 5,000 (P1's E7 real count);
  - refuses a `real/` that already holds files;
  - is required whenever E7 is in `--sets`.
- **Runbook:**
  - attaches `awsaf49/coco-2017-dataset`;
  - Cell 1 finds `coco2017/val2017`;
  - pre-flight requires it;
  - Cell 4 passes it.
- **Fixture:** E7's reals now live in a separate COCO input, so a fetch step without the
  copy fails the dry-run.
- **Regression tests (§12.6):**
  - the copy and its refusals;
  - E7 without `--coco-val` refused;
  - the runbook attaches val2017.

Byte identity is not assumed: Cell 5 still checks all 500 E7 P2 reals by sha256.

**Checks:**
- full suite 858 passed, 7 skipped (GPU smoke);
- ruff clean;
- local dry-run PASSED at the index (57 s) and at HEAD `692ab48` (60 s);
- integrity CLEAN.

**Code zip:** `F:\thesis-2\kaggle_raw\dbp-code-692ab48.zip`
- 6,542,393 bytes;
- sha256 `7f59bf5817b0b5650714f7350691a459cd4008cfb60e44e37d95c0cac9fc4676`.

Panel B is still undrawn and no P3 model has trained. Nothing pushed.

### P3-B (step 1 done) — Panel B drawn, verified and FROZEN — 2026-10-12

**Run.** Kaggle CPU, account `muhammadadeelshah`, code `692ab48` (`COMMIT.txt` in the
output). Output archive `F:\thesis-2\results-phase-3.zip` (1,835,167,314 bytes).
Reports extracted to `reports/kaggle_20261011/`.

**Panel B digest: `267d9fde75d76a23640674fa2fff7e454de23971b8e296323ca4762507c2a551`**
- 9,405 rows: 8,655 test, 750 calibration;
- test digest `af544cc1…`, calibration digest `fd5fb95e…`;
- 700 luma rows, luma digest `41445030…`;
- 1,732,167,236 bytes of images.

This is committed **before any P3 model trains**, as pre-registration §2 requires.

**View rebuild (`p3_eval_views_rebuild.json`).** All five views reproduce P1 exactly:
every manifest sha256 equals P1's, and every one of the 12,150 P2 rows is byte-identical.

| view | rows | real / fake | P2 rows reproduced |
|---|---|---|---|
| e3_matched | 2,178 | 726 / 1,452 | 1,500 |
| e4_wildrf_matched | 2,503 | 1,251 / 1,252 | 1,500 |
| e5_chameleon_pipeline | 26,033 | 14,863 / 11,170 | 4,000 |
| e7_dda_coco | 34,969 | 5,000 / 29,969 | 3,500 |
| e8_evalgen | 5,000 | 0 / 5,000 | 1,250 |

**Panel B composition (`p3_panel_b_summary.json`; no shortfall against any slot).**

| view | test fakes | test reals | calibration reals | plan (`P3_PLAN.md` §1) |
|---|---|---|---|---|
| E3 matched | GPT-Image2 223, Nano-Banana2 **104** | 121 | 50 | ≤ 226 each / ≤ 176 / 50 |
| E4 matched | 465 | 392 | 100 | ≤ 501 / ≤ 402 / 100 |
| E5 pipeline | 2,000 | 2,000 | 500 | 2,000 / 2,000 / 500 |
| E7 | 300 × 6 VAEs | 300 | 100 | 300 × 6 / 300 / 100 |
| E8 | 250 × 5 generators | — | — | 250 × 5 |

The plan's "about 9,750" was an upper estimate. The difference comes from E3 and E4,
whose "all eligible" slots lost rows to dedup and the crop rule. Nothing was backfilled.

**Rejections by reason.**

| view | in P2 (id) | source in P2 | source already in B | < 320 | pHash/sha dup of P2 | dup of a B row |
|---|---|---|---|---|---|---|
| E5 | 898 | — | — | 323 | 26 | 12 |
| E7 | 1,098 | 4,771 | 3,136 | 114 | — | — |
| E4 | 1,500 | — | — | 3 | 32 | 11 |
| E3 | 1,500 | — | — | 16 | 144 | 20 |
| E8 | 407 | — | — | 0 | 23 | 8 |

"Slot full" walks are not rejections: E5 20,274, E7 23,650 and E8 3,312.

**`e3_number_in_p2`, descriptive (user decision option 1):**

| label | flagged true / false |
|---|---|
| GPT-Image2 | 204 / 19 |
| Nano-Banana2 | 94 / 10 |
| real (calibration) | 45 / 5 |
| real (test) | 110 / 11 |

**Independent checks this session** (`results-phase-3.zip` read directly; scratchpad
script):
- **Digests:** all four recompute from the lists.
- **Images:** all 10,105 (9,405 + 700 luma) re-hash to their listed sha256. None is
  missing and none is unlisted, and the byte total equals the digest's.
- **Disjoint from the P2 panel:** 0 image_ids and 0 sha256s shared with the 12,150 P2
  rows.
- **E7:** 2,200 distinct COCO ids, each used once; 0 shared with P2's E7.
- **Size:** 0 rows under 320 px on the shorter side.

**Finding: why E3 Nano-Banana2 is short (104 of 226 eligible).**
- 114 of the 122 lost rows are pHash ≤ 6 matches to P2-panel rows with a *different*
  file number, in the same folder: 79 match P2 Nano-Banana2 fakes, 35 match P2 *reals*.
- The other direction also occurs: 31 eligible E3 reals match P2 Nano-Banana2 fakes.
- So Nano-Banana2's near-copies follow content, not file number. The pHash rule caught
  them, which is what it is for, and it adds to the E3 limitation in `P3_PLAN.md` §1.
- E3 stays descriptive under the source-statement principle, so no decision rule
  changes.

**Timings (log stamps, UTC).**
- **Fetch:** 18:20:59 → 18:38:28, but that session already held part of E3, so this is
  not a clean measure.
- **View rebuild:** 19:19:57 → 20:49:21, about 89 min in all:
  - E3 4 min;
  - E4 6.5 min;
  - E5 25 min;
  - E7 17 min;
  - E8 36.5 min.
- **Draw and copy:** not logged.

**Not committed:**
- the rebuilt `p1_eval_*` manifests: identical to P1's by sha256, two over 10 MB, kept in
  `kaggle_raw/results-phase-3_x/`;
- the images.

**Next:** Phase B step 3, the Open Images fetch from the Flickr originals, rules (a)–(f).
It dedups against Panel B (`p3_panel_b_selection.csv` carries pHash), the P2 panel, the
training corpus, COCO, and all E3/E4/E5 reals.

### P3-B (step 1, follow-up) — the 375-row gap, overlap against all 12,150, disk, E3 — 2026-10-12

User: Panel B **accepted**; E3 finding accepted with no rule change.

**Correction to the previous entry.** It said "every one of the 12,150 P2 rows is
byte-identical". The view rebuild covers the **11,750** rows of the five evaluation
views (1,500 + 1,500 + 4,000 + 3,500 + 1,250; `p3_eval_views_rebuild.json`,
`p2_rows_reproduced`), and all 11,750 reproduced. The other **400** of the 12,150 are
`p2_panel_t_ctrl_diagnostic.csv`, DIV2K T-ctrl training images (D-049) with no evaluation
manifest behind them, so no view rebuild can reproduce them. They are kept out of Panel B
by the draw's T-ctrl reference: the full 1,600-row T-ctrl manifest, by sha256 and pHash.

**1. The 375-row gap: 12,150 drawn vs 11,775 frozen.**
- **Source of the frozen list:** `kaggle_raw\p2_s3_recovery_reports\p2_panel_images_frozen.csv`,
  11,775 rows, sha256 `5def431e…f6`.
- **Matching:** the frozen list is a strict subset of the drawn rows. Drawn but not
  frozen: **375**; frozen but not drawn: 0.
- **Why they were dropped:** all 375 have a recorded shorter side below 320 px (range
  150–319). They are the rows D-054 / I-047 removed, label-blind with no backfill, so
  they could be centre-cropped to the pre-registered 320 evaluation crop (Amendment 1).
- **Check the other way:** no drawn row under 320 px is in the frozen list.

| view | real | fake | total |
|---|---|---|---|
| e3_matched | 23 | 0 | 23 |
| e4_wildrf_matched | 6 | 2 | 8 |
| e5_chameleon_pipeline | 136 | 116 | 252 |
| e7_dda_coco | 14 | 78 | 92 |
| e8_evalgen, t_ctrl_diagnostic | — | — | 0 |

These per-view totals equal I-047's.

**2. "0 overlap with P2" was run against all 12,150 drawn rows.** The script globbed every
committed `reports/p2_panel_sets/p2_panel_*.csv` (printed `P2 rows 12150`), not the
11,775. Re-run this session from the committed row files, so the 375 dropped rows and the
400 T-ctrl rows are included:
- **Panel B (9,405 + 700 luma) against all 12,150:**
  - **0** shared image_ids and **0** shared sha256 (all 12,150 rows carry a sha256);
  - 0 with the 375 gap rows;
  - 0 with the 400 T-ctrl diagnostic rows.
- **pHash ≤ 6:** 0 of the 9,405 selection rows match any of the 11,750 P2 view rows that
  carry a pHash. The T-ctrl diagnostic file has no pHash column; the draw covered it
  through the T-ctrl manifest's pHash.

**3. Disk in the Panel B session: NOT-MEASURED.**
- No step logged `df`.
- The `env_report.json` in the output is the committed local P0 report (generated
  2026-09-19, Windows), not the session's.
- The only sizes the session recorded:
  - WildRF archive 6,403,695,558 bytes, in `/tmp`, deleted after extraction;
  - Panel B images 1,732,167,236 bytes, in `/kaggle/working`.
- Neither is a peak.
- From now on every runbook logs `df -h` at each stage (user rule, 2026-10-12; first used
  by the Open Images fetch).

**4. E3 generation protocol: no per-image labels.**
- **What E3 records per image:** only its regime (folder: Texture / Structure / Hybrid)
  and its generator (folder: real / GPT-Image2-fake / Nano-Banana2-fake). The rebuilt
  manifest's columns are `subset` and `source_label`.
- **What is absent:** no file or field records whether an image is text-to-image,
  image-to-text-to-image or an edit. The data card lists the distributed tree as 4,356
  image files = 2,178 images × 2 views.
- **Consequence:** E3 results cannot be reported per protocol. They are reported per
  regime and per generator (data card §6), as well as pooled.

Panel B E3, descriptive:

| regime | GPT-Image2 | Nano-Banana2 | real test | real calibration |
|---|---|---|---|---|
| Structure | 95 | 30 | 58 | 22 |
| Texture | 67 | 39 | 38 | 13 |
| Hybrid | 61 | 35 | 25 | 15 |
| total | 223 | 104 | 121 | 50 |

**5. Panel B Nano-Banana2 fakes with a pHash ≤ 6 match among E3 reals (all 726, any
split): 9 of 104.**
- For contrast, Panel B GPT-Image2: 0 of 223.
- Count only, descriptive. These are probably edits of those reals.

### P3-B (step 3 built) — Open Images fetch runbook, dry-run PASSED — 2026-10-12

**User instruction (2026-10-12).** Build the fetch under rules (a)–(f), and add three
things:
- log `df -h` at the start and at each stage, and stop before any download if free space
  is short;
- attach the Panel B output dataset (`dbp-p3-panel-b`);
- dry-run until it PASSES, then state the download size, output size, run time and peak
  disk, each marked measured or estimated.

**Correction to the step-2 entry (2026-10-08).** It said every `OriginalURL` was on
`farm*.staticflickr.com`. Measured this session from `oi_validation_meta.csv` (sha256
`ed93a0e1…`, unchanged):

| hosts | rows |
|---|---|
| `farm1–8.staticflickr.com` (approved) | 20,674 |
| `c1–c8.staticflickr.com` (not approved) | 20,946 |
| total | 41,620 |

Rule (a) applies: the `c*` hosts are listed and wait for approval. The runbook never
contacts them; their rows are rejected as "host not approved (not contacted)" and counted.

**Header-only probe (local, 2026-10-12).**
- **What:** HEAD requests to the first 50 approved-host rows in seed order. Sequential,
  0.5 s apart, redirects not followed, no image bytes. Headers only, which §6a(a)
  allows locally.
- **Saved as:** `kaggle_raw\p3_open_images_metadata\head_probe_2026-10-12.json`.
- **Status codes:** 200 × 44, 404 × 5, 410 × 1.
- **Sizes:** all 44 `Content-Length` values equal the metadata's `OriginalSize`.
- **Redirects:** none.

**Interpretation of rules (c) and (d), flagged for the user.**
- (c) says crop eligibility ≥ 320; (d) says crop to the pipeline's COCO size, which is
  `p1_build_reals.DEFAULT_CROP = 384`.
- A 320–383 px original cannot give a 384 crop without resizing, which (d) forbids. So the
  effective eligibility is ≥ 384, and the 320–383 band is counted separately.
- Greyscale originals are rejected, as `build_rows` does for COCO (D-039).

**Built (commit `26775f6`).**
- **`src/dbp/data/open_images.py`:**
  - CC BY 2.0 filter;
  - seed-2026 walk order (`blake2b("open_images:2026:<id>")`) and a separate split key;
  - approved-host check: `farm\d+\.staticflickr\.com` exactly; a redirect may land on
    `live.staticflickr.com` only; lookalike hosts are refused;
  - `Fetcher`: redirects checked before they are followed, at most 8 in parallel, 3
    retries with backoff on 429/5xx/network errors, 404/410 recorded as "gone";
  - base64-MD5 comparison;
  - `LocalFetcher`, for the dry-run only.
- **`src/dbp/utils/disk.py`:** `df -h` (when present) plus `shutil.disk_usage` per stage;
  `require_free`.
- **`scripts/p3_coco_index.py`:**
  - sha256 and pHash of all COCO train2017 + val2017 (expects 118,287 + 5,000), using
    `phash_file`, the function the evaluation manifests used;
  - the descriptive E3-reals-in-COCO count;
  - the index goes to `/tmp`, not the output.
- **`scripts/p3_fetch_open_images.py`:**
  - batches are downloaded in parallel but decided strictly in walk order;
  - MD5 check, then `build_rows` (centre crop 384, PNG, no resize);
  - pHash and sha256 of the original **and** the crop, checked against Panel B, the P2
    panel (12,150), T-real and T-ctrl (crops and sources), the SD1.5 generations,
    `gen_matched`, the bank pairs, all of COCO, and every E3/E4/E5 real;
  - each rejection names the reference set it matched;
  - seeded 3,600 / 400 split;
  - outputs: manifest, attribution (image id, author, original URL, licence), walk log and
    summary (rejections, unapproved hosts, size distributions, EXIF camera count,
    rotation, bytes, disk per stage);
  - exits non-zero on any shortfall.
- **Runbook:**
  - `env/kaggle_run_p3_open_images.py` (7 cells, CPU) and `env/p3_open_images_expect.json`;
  - the pre-flight checks the three E3/E4/E5 manifests from `dbp-p3-panel-b` against the
    sha256 and row counts in the committed `p3_eval_views_rebuild.json`.
- **Input committed:** `reports/p3_open_images/oi_validation_cc_by_2.csv`
  - the 41,620 CC BY 2.0 rows of the step-2 file, 7 columns (ImageID, OriginalURL,
    License, Author, OriginalSize, OriginalMD5, Rotation);
  - 7,309,877 bytes, sha256 `5eabde3b…`;
  - the raw file (15.2 MB) is over the 10 MB limit.
- **Tests:** `tests/test_open_images.py` (13). The runbook lint now knows the COCO-index
  long job.
- **Dry-run fixture `p3_open_images`:** walks every rejection path once (unapproved
  host, 404, MD5 mismatch, greyscale, < 320, 320–383, COCO copy, E5-real copy, pHash
  re-encode of an accepted image) plus one row after the target. The harness now exposes
  the sessions' `/tmp` to the checks, so leftover originals are detected.

**local dry-run: PASSED**
- at the index (14 s) and at HEAD `26775f6` (21 s);
- 6 accepted (4 train / 2 held out), 11 rejected, each with the expected reason;
- `c3.staticflickr.com` listed as not contacted;
- crops 384 × 384 PNG;
- no original left in `/tmp`;
- `df` logged at every stage.

**Checks:** full suite 878 passed, 7 skipped (GPU smoke); ruff clean; integrity CLEAN.
**Code zip:** `F:\thesis-2\kaggle_raw\dbp-code-26775f6.zip`, 10,825,100 bytes, sha256
`99385d0a4c5fee377fd931159134f344d9262d4d4adb16411f6beb836fdbe6a1`.

**Estimates for the hand-over** (all ESTIMATED, with their basis; nothing here was
measured on Kaggle):

| item | estimate | basis |
|---|---|---|
| download | 8–12 GB | metadata `OriginalSize` over the first 4,545–6,667 approved-host rows in seed order × 0.88 (probe availability) |
| peak `/tmp` | 0.2–1.1 GB | one 64-file batch in flight (mean 2.0 MB, max 15.7 MB) plus the COCO index |
| output | about 1.1 GB | 4,000 × 268 KB, the measured mean of P1's 2,500 T-real PNG crops |
| wall time | about 1–1.5 h | local single-thread costs: COCO 17 ms per image (123,287 images); Open Images 0.43 s (1.1 MB) to 1.5 s (4 MB) per image. Kaggle CPU speed and Flickr bandwidth not measured |

### P3-B (step 3, revision) — c1–c8 approved, politeness rules, dry-run PASSED — 2026-10-13

**User decisions (2026-10-13).**
- **Accepted:** the 375-row gap, the overlap re-check, disk NOT-MEASURED, the E3 regime
  counts and the 12,150-wording correction.
- **(f) Hosts: `c1–c8.staticflickr.com` APPROVED**, in addition to `farm1–8`.
  - Reason: the same Flickr CDN and terms, and the roughly 50/50 split is undocumented;
    it may follow upload era or server batch, so a farm-only draw could be biased.
  - All 41,620 CC BY 2.0 rows are walked in the one seed-2026 order, with no host filter.
  - The summary reports each host group (`farm*` / `c*`), descriptively: rows walked,
    accepted, gone (404/410), checksum mismatches, and the median original width × height.
  - `live.staticflickr.com` stays approved for redirects only. Any other host is not
    contacted; it is counted and reported.
- **(h) Eligibility ACCEPTED: the resolution of rules (c) and (d).**
  - The shorter side must be ≥ 384, the COCO crop (`p1_build_reals.DEFAULT_CROP`), with
    no resizing.
  - The 320–383 band is counted separately.
  - Greyscale originals are rejected, as in the COCO pipeline (D-039).
- **Concurrency:** at most 8 HTTP connections open at once; a 64-file batch is fine.
  Retries keep their backoff. On repeated 429/503 the run must slow down and log it, not
  hammer the server.

**Changes (commit `8618a24`).**
- **Hosts:** `APPROVED_HOST_PATTERNS` = `farm[1-8]\.staticflickr\.com`,
  `c[1-8]\.staticflickr\.com`, both exact. `farm9`, `c9`, `c0` and lookalikes are refused
  (tested). `host_group()` maps them to `farm*` / `c*` / `other`. The expectations file
  lists the same two patterns.
- **Connection cap:** a `BoundedSemaphore(8)` inside `Fetcher`, held from the request until
  the body has been read. So no more than 8 connections are open whatever the thread count
  (tested with 24 threads: peak ≤ 8). A `Fetcher` with more than 8 slots is refused. The
  walk's thread pool is also 8.
- **Slow-down:** a `Throttle` shared by every request.
  - Each 429/503 doubles a delay applied before every request (base 1 s, ceiling 60 s),
    or sets it to `Retry-After` when that is longer.
  - Each change is logged as a warning and recorded in the summary.
  - 50 successes in a row halve the delay.
  - 10 refusals in a row at the ceiling trip it: the walk stops, and no row after the
    first unfetched one is decided, so the walk order stays strict. The run then exits
    non-zero.
  - Per-request retries (3, backoff 2^n s) are unchanged.
- **Summary:** `by_host_group` with `rows_walked`, `accepted`, `gone`, `checksum_mismatch`,
  `download_failed`, `not_contacted`, `not_needed`, `originals_verified`, `median_width`
  and `median_height`; `throttle`; and `fetch_policy.max_parallel_connections`.
- **Fixture:**
  - good rows on both `farm*` and `c*`;
  - the unapproved rows moved to `c9` (still not contacted);
  - one row answers 429 twice before serving its file.
  - The checks cover the group counts, the throttle refusals (2, not tripped) and the cap
    (8).
- **Tests:** 6 new, 19 in `test_open_images.py` in total.

**Header-only probe of `c1–c8` (local, 2026-10-13).**
- **What:** the first 50 `c*` rows in seed order, run the same way as the 2026-10-12
  `farm*` probe.
- **Results:** 200 × 43, 404 × 5, 410 × 2. All 43 `Content-Length` values equal
  `OriginalSize`. No redirects.
- **Saved as:** `kaggle_raw\p3_open_images_metadata\head_probe_c_hosts_2026-10-13.json`.
- **Note:** a first attempt at this probe, with a broken host filter, re-probed the same 50
  `farm*` URLs (identical result) under that file name. It was overwritten by the correct
  `c*` probe.
- **Combined:** 87 of 100 online.
- **Original sizes from the metadata:**

  | group | rows | mean | median |
  |---|---|---|---|
  | `farm*` | 20,674 | 2.02 MB | 1.08 MB |
  | `c*` | 20,946 | 1.98 MB | 1.04 MB |

- The first 4,598 rows in seed order are 2,335 `farm*` and 2,263 `c*`.

**local dry-run: PASSED**
- at the index (21 s) and at HEAD `8618a24` (22 s);
- 6 accepted (3 `farm*` + 3 `c*`; 4 train / 2 held out);
- 11 rejected with the expected reasons;
- 3 `c9` rows not contacted;
- 2 × 429 slowed the shared delay, and the run went on.

**Checks:** full suite 883 passed, 7 skipped (GPU smoke); ruff clean; integrity CLEAN.
**Code zip:** `F:\thesis-2\kaggle_raw\dbp-code-8618a24.zip`, 10,831,469 bytes, sha256
`3031bf86768a7a1fab38faf1f9eebaf8a80c09caf3e24ccafbe7fce32e384fc6`. It supersedes
`dbp-code-26775f6.zip`, which was never run.

**Cost table (all ESTIMATED; nothing measured on Kaggle):**

| item | estimate | basis |
|---|---|---|
| download | 8–12 GB | `OriginalSize` over the first 4,598–6,667 rows in seed order (all hosts) × 0.87 measured availability |
| peak `/tmp` | 0.2–1.1 GB | one 64-file batch in flight (mean 2.0 MB, max 15.7 MB) plus the COCO index |
| output | about 1.1 GB | 4,000 × 268 KB, the measured mean of P1's 2,500 T-real PNG crops |
| wall time | about 1–1.5 h | local single-thread costs: COCO 17 ms per image; Open Images 0.43–1.5 s per image. Kaggle CPU and Flickr bandwidth not measured; any 429/503 slow-down adds to it |

### P3-B (step 3, Kaggle run 1) — Flickr refused every request; the walk stopped — 2026-10-13

**Run:** Kaggle CPU, code `8618a24`. Source: the user's pasted log.
- **Inputs:** after the first attempt failed in Cell 1 (Panel B was attached as a notebook
  output, not a dataset), `dbp-p3-panel-b` was attached as a dataset.
- **Pre-flight and the COCO index:** passed. Their summary is not yet retrieved.
- **The fetch step:** stopped. 0 accepted, 0 downloaded.

**What happened** (log stamps, session time):
- **Refusals:** all 8 first GETs, sent in parallel at 08:46:16, answered **429**. The waves
  at 08:47:18 and 08:48:22 answered 429 again: refusals 9–16 and 17–24, so 24 of 24
  requests refused.
- **Stop:** the throttle tripped and the walk stopped at 08:48:30. Nothing was decided
  (64 rows "not walked"), and the run exited non-zero as designed.
- **Load sent to Flickr:** 24 requests in about 2 minutes, at most 8 open at once.

**Measured (first disk figures from a Kaggle CPU session):**
- **`/kaggle/working`:** 20 GB, 19.45 GB free (`/dev/loop2`).
- **`/tmp`:** overlay, 1,069.85 GB free of 8,156.88 GB.
- **Time before the fetch:** 1,490 s into the session, covering staging, the bootstrap,
  pre-flight and the COCO index.

**Interpretation (not verified).**
- 100% refusals from the very first request look like Flickr refusing this client or IP
  range, not like a rate limit we triggered.
- The local header probes (2026-10-12/13, home connection, same User-Agent, HEAD not GET)
  got 200 × 87 of 100. They are not a like-for-like comparison.
- No `Retry-After` value or response body was recorded, so the cause cannot be told apart
  from this run alone.

**Code flaw found (ours).**
- `Throttle.refused` doubled the shared delay once per refusal. 8 simultaneous refusals
  from one parallel wave therefore escalated 1 s → 60 s in 0.2 s, and "10 refusals at the
  ceiling" was reached after 3 waves, about 2 minutes.
- A wave of concurrent refusals should count once, and the stop rule should need refusals
  sustained over minutes.
- The response headers of refusals were not recorded.
- To be fixed with a regression test (§12.6) once the user chooses how to proceed.

### P3-B (step 3) — run 1 recorded as FAILED; throttle fixed; Flickr probe built — 2026-10-13

**Open Images Kaggle run 1 (code `8618a24`) is a FAILED run** (user decision 2026-10-13).
- **Reason:** Flickr answered HTTP 429 to all 24 requests; the walk stopped.
- **Its rows decide nothing:** 0 accepted, 0 rejected; 64 rows logged "not walked".
- **Output:** `F:\thesis-2\results-Phase-3_b.zip` (15,514,356 bytes; `testzip` finds no
  bad member; reports extracted to `kaggle_raw\results-Phase-3_b_x\`). It holds no
  `oi/` image.

**Measured in that run, still valid (Cells 3 and 4 passed):**
- **COCO index:** 123,287 images (train2017 118,287 + val2017 5,000) hashed in **886 s**
  with 4 workers, 7.19 ms per image (`p3_coco_index_summary.json`).
- **E3 reals with a pHash ≤ 6 match in COCO train2017 + val2017: 1 of 726.**
  Descriptive, not a gate (`P3_PLAN.md` §4a).
- **Disk:** `/tmp` 1,069.86 GB free; `/kaggle/working` 19.45 GB free of 19.52 GB.

**Step 1: throttle fix (user-approved; commit `2f84225`).** This is the regression
fix, with tests, for the flaw recorded in the run-1 entry.
- **One raise per wave:** a refusal raises the shared delay only when the last raise is
  at least one current delay old, or when `Retry-After` is longer.
  Regression test: 8 simultaneous 429s give delay 1 s and 1 raise (run 1 went to 60 s).
- **Stop only on sustained refusals:** the walk stops only after **600 s of refusals at
  the 60 s ceiling** with no success in between; a success resets the clock. Tested at
  599 s (no stop), after a reset, and at 600 s (stop).
- **Evidence:** the first 20 refusals are kept in the summary: UTC time, host, status,
  headers without `Set-Cookie`/`Cookie`, and the first 200 body bytes (tested).
- **403 counts as a refusal**, alongside 429 and 503. Otherwise a blanket block would
  send one request per row across 41,620 rows.

**Step 2: Flickr probe (user-approved, with 6 GETs instead of 5).**
- **Files:** `scripts/p3_probe_flickr.py`, `env/kaggle_run_p3_flickr_probe.py` (CPU,
  6 cells), `env/p3_flickr_probe_expect.json`.
- **Requests:** the first 3 `farm*` and the first 3 `c*` rows in seed order, sent in
  that combined order, **one at a time** (never more than one connection open), 30 s
  apart. Approved hosts only; redirects recorded, not followed.
- **No image kept:** each body is streamed through MD5 and compared with `OriginalMD5`.
- **Recorded per request:** UTC time, host group, status, seconds, headers without
  cookies, and the first 200 body bytes of a non-200.
- **Egress IP range:** first two octets only, from one GET to `api.ipify.org`. The user
  approved that host for this purpose on 2026-10-13. It is the only echo host the script
  accepts, and the full address is never written.
- **Verdict:**
  - **A:** more than half 200 in every group, i.e. a rate limit;
  - **B:** every request refused in both groups, i.e. blocked;
  - otherwise **mixed**.
- **Pre-decided follow-ups (user, 2026-10-13):**
  - **A** → rerun on Kaggle with ≤ 2 connections and a fixed gap; report the wall time
    first, and ask if it is over 9 h.
  - **B** → do not use the S3 copies and do not change the source; prepare the local
    route (ii) and wait for the user's OK on sizes before any local download.
- **Tests:** 3 for the probe (IP truncation and echo-host refusal, the per-group pick, the
  verdicts); the runbook lint knows the probe as a long job.

**local dry-run: PASSED** at HEAD `2f84225`, for both runbooks:
- **probe** (5 s): 3 `farm*` + 3 `c*`; the unapproved `c9` row skipped; one 429 recorded
  without its cookie; MD5s match; no image written; the egress check correctly skipped in
  the dry-run.
- **Open Images** (9 s): unchanged results, with the throttle now logging "raise n".

**Checks:** full suite 896 passed, 7 skipped; ruff clean; integrity CLEAN.
**Code zip:** `F:\thesis-2\kaggle_raw\dbp-code-2f84225.zip`, 10,843,784 bytes, sha256
`a060623af1d7a85371503010c8a6f46d358ac80041bea98fae0ca7717d43b9d0`.

### P3-B (step 3) — Flickr probe: verdict B (Kaggle IPs blocked); route (ii) chosen — 2026-10-14

**Probe run** (Kaggle CPU, code `2f84225`). Output `F:\thesis-2\results_phase-3C.zip`
(15,426,751 bytes; `testzip` finds no bad member; no image inside). Report extracted to
`kaggle_raw\results_phase-3C_x\Implementation\thesis_dbp\reports\p3_flickr_probe.json`.
- **Egress IP range:** `34.12.x.x`, from `api.ipify.org`; first two octets only.
- **Requests:** 6 GETs, one at a time, 30 s apart, 2026-10-13 10:00:45 → 10:03:16 UTC.

  | # | UTC | host | group | status | seconds |
  |---|---|---|---|---|---|
  | 1 | 10:00:45 | c3 | c* | 429 | 0.16 |
  | 2 | 10:01:15 | c1 | c* | 429 | 0.16 |
  | 3 | 10:01:46 | farm5 | farm* | 429 | 0.15 |
  | 4 | 10:02:16 | c4 | c* | 429 | 0.35 |
  | 5 | 10:02:46 | farm3 | farm* | 429 | 0.17 |
  | 6 | 10:03:16 | farm2 | farm* | 429 | 0.17 |

- **What every refusal carried:** `Server: Jubilee`, `X-Cache: Error from cloudfront`,
  `Via: … (CloudFront)`, `X-Amz-Cf-Pop: AMS56-P1`, **no `Retry-After`**, and an empty
  body head.
- **Verdict: B**, refused on both host groups. At one request every 30 s, from the very
  first request, the CDN refuses this client range; this is not a rate limit we
  triggered. No image was kept.

**Pre-decided consequence (user, 2026-10-13): route (ii).**
- **Not used:** the S3 copies; the source is not changed.
- **Local stage, on the user's PC:**
  - the seeded walk, the `OriginalMD5` check, the ≥ 384 and greyscale checks, the
    native-resolution PNG crop, and sha256 + pHash of each crop;
  - raw originals go only into `F:\thesis-2\kaggle_raw\p3_oi_originals\`, never into
    REPO_ROOT.
- **Kaggle CPU stage:**
  - dedup against COCO, Panel B, P2, the training corpus and all E3/E4/E5 reals;
  - final acceptance in the same seed order up to 4,000, then the 3,600 / 400 split;
  - a shortfall means the next rows are fetched locally in the same order.

**DEVIATION RECORD (2026-10-14).**
- **What changed:** for this one source (Open Images V7 originals from Flickr), the
  fetch location changes from Kaggle to local.
- **Why:** Flickr's CDN refuses Kaggle's IPs (Open Images run 1: 24/24 × 429; this probe:
  6/6 × 429 at 30 s spacing).
- **What kind of change:** a CLAUDE.md §6a(a) runbook deviation approved by the user. It
  is **not** a pre-registration change: no data rule changes. Same metadata, seed order,
  checksum, eligibility, crop, dedup references and split.
- **What still holds:** every other dataset is still downloaded on Kaggle only.
- **Gate:** before any local download, the user is given the expected download size,
  local disk use, time and upload size, and must reply OK.

### P3-B (step 3) — route (ii) built, dry-run PASSED; WAITING for the user's OK to download — 2026-10-14

**Built (commit `1d0be77`).**
- **`scripts/p3_oi_local_fetch.py`** (LOCAL stage, the user's PC):
  - the same metadata, seed order, approved hosts, politeness (≤ `--workers` connections,
    default 4; backoff; the shared slow-down; the 10-minute stop), `OriginalMD5` check,
    ≥ 384 and greyscale rules, and `build_rows` crop as on Kaggle;
  - no dedup;
  - keeps the first `--target-candidates` passing rows ("candidates"), and the originals
    of candidates only, in `F:\thesis-2\kaggle_raw\p3_oi_originals\`;
  - crops, candidate manifest, walk log and summary go to `F:\thesis-2\kaggle_raw\p3_oi_candidates\`;
  - refuses any path inside REPO_ROOT;
  - stops before the first download if the disk has less than 20 GB free;
  - logs disk use per 500 candidates;
  - resumes after the last decided row: a resumed walk equals an uninterrupted one
    (tested).
- **`scripts/p3_oi_dedup_accept.py`** (Kaggle CPU stage):
  - re-hashes every uploaded crop against the **committed** candidate manifest, and stops
    on any mismatch;
  - dedups original and crop (sha256, pHash ≤ 6) against COCO, Panel B, the P2 panel,
    the training corpus, all E3/E4/E5 reals and the rows already accepted;
  - accepts in walk order up to 4,000, then the seeded 3,600 / 400 split;
  - writes the manifest, attribution, acceptance log and summary;
  - a shortfall prints the `--target-candidates` to rerun the local walk with, and the
    position it resumes after.
- **`env/kaggle_run_p3_oi_accept.py`** (7 cells) and **`env/p3_oi_accept_expect.json`**.
  The candidate manifest's sha256 and row count are `PENDING-LOCAL-RUN`; they are filled
  in, committed and zipped after the local stage.
- **Dry-run fixture `p3_oi_accept`:** runs the local stage on the Open Images fixture,
  then the Kaggle runbook.
  - **Local stage:** 10 candidates, 10 originals kept, 3 unapproved-host rows refused.
  - **Kaggle stage:** all 10 crops verified; exactly 3 dedup rejections (COCO val2017
    sha, E5-real sha, pHash of an accepted row); 6 accepted (4 / 2); the 10th candidate
    logged "not needed".
- **Tests:** `tests/test_oi_local_route.py` (4). `process()` gained `keep_original`.

**local dry-run: PASSED** at HEAD `1d0be77`:
- `p3_oi_accept` (16 s);
- `p3_open_images` re-run (18 s).

**Checks:** full suite 907 passed, 7 skipped; ruff clean; integrity CLEAN.

**Estimates handed to the user before any download** (nothing downloaded yet):
- **Margin:** target 4,200 candidates, i.e. 5% over 4,000.
  - **Basis:** no measurement of Open Images' dedup loss exists yet. The only related
    measurement is E3-in-COCO, 1 of 726 (0.14%).
  - **Backstop:** a shortfall triggers a resumed local walk in the same order.
- **Rows walked:** about 4,941–6,000. Basis: a local pass rate of 0.85–0.70, assumed; the
  measured availability is 0.87.
- **Download:** about 8.7–10.6 GB. Basis: metadata `OriginalSize` over those rows × 0.87
  measured availability (HEAD probes, 87 of 100).
- **Local disk:** about 9.5 GB in total.
  - originals kept: 4,200 × 2.03 MB mean (metadata) ≈ 8.5 GB;
  - crops: 4,200 × 268 KB (measured P1 crop mean) ≈ 1.1 GB;
  - F: had 98.4 GB free (measured 2026-10-14).
- **Upload:** about 1.1 GB (the crops plus small CSVs).
- **Time:** about 1–2 h. Basis: local processing measured at 0.43–1.5 s per image per
  thread with 4 workers; home bandwidth not measured. Flickr may also slow a home IP at
  this volume, and the slow-down would then add time.

### P3-B (step 3) — route (ii) APPROVED with conditions; decoding rules applied — 2026-10-14

**User approval (2026-10-14):** start the local fetch with `--target-candidates 4200
--workers 4`, under four conditions.
1. **Decoding.** It must match the COCO pipeline exactly.
   - EXIF orientation and ICC colour profiles are handled as P1 handled COCO.
   - No colour-mode conversion: CMYK, 16-bit, palette and alpha originals are rejected
     as well as greyscale, each counted separately.
   - The Python, Pillow and libjpeg(-turbo) versions are recorded in the candidate
     manifest header and in this log.
2. **Home-IP politeness.**
   - At most 4 connections.
   - 10 minutes of refusals (429/403) in a row → STOP, save the state and report. No
     aggressive retries; the user says when to resume.
3. **Originals.** The raw originals stay in `F:\thesis-2\kaggle_raw\p3_oi_originals\` until the Kaggle accept step has
   passed. They are never committed or uploaded; only the crops go to Kaggle.
4. **Reporting.** On finishing, report MEASURED values; then commit the candidate
   manifest, dry-run, build the zip and hand over.

**How P1 decoded COCO** (read from the code this session; this is what the local route
now does too):
- `p1_build_reals.build_rows` → `dbp.utils.io.load_rgb_uint8` →
  `Image.open(path).convert("RGB")`.
- **EXIF orientation is NOT applied.** Pixels are kept in stored order; there is no
  `exif_transpose` in `src/` or `scripts/`.
- **ICC profiles are NOT applied.** There is no `ImageCms`; values stay as encoded.
- A test now pins both.

**Applied (commit `dfa1b7a`).**
- `colour_mode_reason()` keeps only 8-bit `RGB` and rejects, without converting:
  - `CMYK`;
  - 16-bit or float (`I;16`, `I`, `F`);
  - palette (`P`, `PA`);
  - alpha (`RGBA`, `LA`, …, and `RGB` with a transparency key);
  - greyscale by mode (`L`, `1`).
- Each mode has its own rejection reason. Colour-mode images whose three channels are
  equal stay rejected by the COCO greyscale rule (D-039), under their own reason.
- **Decoder versions** go in every candidate row (columns `decoder_python`,
  `decoder_pillow`, `decoder_libjpeg`, i.e. in the manifest's header) and in the summary.
- **Local versions** (measured this session): Python **3.13.1**, Pillow **12.3.0**,
  libjpeg-turbo **3.1.4.1** (jpeglib API 8.0).
- **Fixture:** one CMYK, palette, alpha and 16-bit original, each rejected under its own
  reason. Dry-runs PASSED (`p3_open_images`: 15 rejections as designed; `p3_oi_accept`
  unchanged). 30 tests in the two Open Images test files pass.
- **Politeness check:**
  - `--workers 4` gives at most 4 connections (the `Fetcher` semaphore).
  - The `Throttle` trips after 600 s of refusals at the 60 s ceiling, counting 403 as well
    as 429 and 503. The walk then stops, saves its state and exits non-zero, and is
    resumed only on the user's word.
  - Per-request retries: 3, with 2^n s backoff, on top of the shared delay.

### P3-B (step 3) — LOCAL run 1: Flickr refused the home IP after about 6 minutes; STOPPED — 2026-10-14

**Run:**
- `p3_oi_local_fetch.py --target-candidates 4200 --workers 4`, code `22b4f2d`, on the
  user's PC.
- Started 2026-10-14 05:33:51Z; stopped 05:52:23Z (wall 18 min 32 s; 1,104.4 s in the
  script).
- Console log `F:\thesis-2\kaggle_raw\p3_oi_local_run.log`; state in `F:\thesis-2\kaggle_raw\p3_oi_candidates\`.
- Exit 1, as required by condition 2. **Not resumed:** the user says when.

**What happened:**
- The first 96 rows were decided in about 6 minutes, at about 0.67 MB/s, with no
  refusal.
- At 05:40:02Z Flickr began answering 429. The response signature is the same as on
  Kaggle: `X-Cache: Error from cloudfront`, no `Retry-After`, an empty body.
- The shared delay went 1 → 2 → 4 → 8 → 16 → 32 → 60 s, one raise per wave, then held at
  the ceiling.
- After **648 s at the ceiling** (refusal 60), the walk stopped at 05:52:18Z and saved
  its state.

**MEASURED** (`p3_oi_local_summary.json`, walk log, candidate manifest, `du`):

| item | value |
|---|---|
| rows reached | 122 (positions 1–122) |
| rows decided | **107** (positions 1–107, no gap) |
| rows undecided | 15 (positions 108–122: refused by 429, see the fix below) |
| candidates | **88** (c* 45 of 56 decided; farm* 43 of 51) |
| rejected: gone, http 404 | 11 |
| rejected: gone, http 410 | 3 |
| rejected: shorter side 320–383 | 2 |
| rejected: shorter side < 320 | 1 |
| rejected: greyscale (rgb_equal, D-039) | 2 |
| rejected: colour mode (CMYK / palette / alpha / 16-bit / L) | 0 |
| rejected: checksum mismatch | 0 |
| bytes downloaded | 228,098,376 |
| originals kept | 222,358,156 bytes (88 files, in `p3_oi_originals`) |
| crops | 16,499,918 bytes (88 PNG; mean 187,499 B) |
| local disk used | 238,964,634 bytes (originals 222,358,156 + candidates folder 16,606,478) |
| slow-down events | **17** throttle raises; **63** refusals (all 429); tripped after 648 s at 60 s |
| pass rate (decided) | 88 / 107 = 0.82 |
| median original size | farm* 2,048 × 1,662; c* 2,816 × 2,244 |
| EXIF camera make or model | 72 of 88 candidates |
| decoder | Python 3.13.1, Pillow 12.3.0, libjpeg-turbo 3.1.4.1 (jpeglib 8.0) |

**Bug found and fixed (commit `653f7c2`, regression tests).**
- **The bug:** the 15 rows refused beyond their per-request retries were logged as
  rejected ("download failed (http 429)"). A refusal says nothing about the image, so a
  resume would have skipped them and broken the seed order.
- **Rule now:** refused and failed downloads are **undecided**.
  - The walk log is the source of truth, and the earlier version's "download failed"
    rows are dropped on load and fetched again.
  - Earlier gaps are retried first, at most 2 failed attempts per row per session.
  - The target counts only with no undecided row before the last candidate; any extra
    candidate beyond it is trimmed.
  - A redirect to an unapproved host stays a decided rejection.
  - The Kaggle walk script halts on a failed row instead of rejecting it.
- **Effect on this run's data:** no crop or original exists for the 15 rows, and
  positions 1–107 are a clean decided prefix. Nothing already decided changes.
- **Tests:** resume after 429s equals an uninterrupted walk; old "download failed" rows
  are refetched.
- **Checks:** full suite 911 passed, 7 skipped; both route dry-runs PASSED.

**Consequence for the plan (for the user):** at about 16 rows a minute with 4 connections,
the home IP is refused after about 100 rows (about 200 MB). The cooldown is unknown. At
this pattern, 4,200 candidates (about 5,100 rows) need many bursts. A steadier, slower
pace is untested.

### P3-B (step 3) — local run 1 accepted; resume plan; pacing options built — 2026-10-14

**User decisions (2026-10-14).**
- LOCAL run 1 accepted, including the 429-as-rejected fix.
- `--min-interval` APPROVED.
- **Resume plan:**
  - Not before **2026-10-15 06:00 UTC** (24 h after the trip).
  - **Trial:** 1 connection, `--min-interval 6`, 30 minutes.
  - **If the trial is clean:** continue at the same pace, unattended, in resumable
    sessions.
  - **If it trips the 10-minute stop:** wait 24 h, then ONE trial at `--min-interval 15`.
  - **If that trips too:** STOP and report, with no third pace. The user decides (a
    smaller target by dated amendment, or another option).
  - After every session the log records rows decided, candidates so far, refusals and
    the pace.
- **Who runs it:** the user, in their own PowerShell window.

**Built (commit `47d22a9`; dry-run and tests PASSED).**
- **`Fetcher(min_interval)`:** a fixed minimum gap between the starts of any two
  requests, retries and redirects included, across all threads (tested with a fake
  clock: 3 requests give 2 gaps of 6 s).
- **Session controls**, which the plan needs to run safely from the user's window. These
  are small additions beyond `--min-interval`, flagged to the user:
  - `--max-minutes`: the session ends cleanly after the current batch;
  - `--not-before`: refuses to start before a UTC time (exit 2);
  - `--status`: rows decided, candidates and the last session, from the saved state,
    without network;
  - a session log: one row per session in `p3_oi_local_sessions.csv` (start and end UTC,
    connections, minimum interval, rows decided before and after, candidates before and
    after, requests started, refusals, slow-down raises, tripped, how it ended, exit
    code).
- **Ctrl+C:**
  - every earlier batch stays saved; the batch in progress is undecided and its files
    are swept; exit 130;
  - results are collected in 1 s waits, so Ctrl+C is honoured on Windows;
  - files left by a hard kill are swept at the next start;
  - tested in-process: an interrupt inside the second batch, then a resume, equals an
    uninterrupted walk.
- **Pace check on the fixture:** `--workers 1 --min-interval 1 --max-minutes 0.25`
  decided 20 rows in about 16 s (about 1 request per second) and ended "session time
  limit"; `--status` printed the state.
- **Checks:** full suite **915 passed, 7 skipped**; the route dry-run `p3_oi_accept`
  PASSED.

**MISTAKE (recorded).**
- **What happened:** commit `47d22a9` was staged with `git add thesis_dbp`, which picked
  up two untracked user files: `docs/Claude outputs/thesis_report_v3_preview.pdf`
  (303,723 B) and `docs/report/archive/thesis_report_updated_sections.tex` (72,946 B).
- **Fix:** commit `557e187` removes them from the index (`git rm --cached`); the working
  files are untouched and untracked again.
- **What remains:** they stay in the local history of `47d22a9`. History is not
  rewritten (§2) and nothing was pushed. Purging them would need the user's explicit
  approval.
- **Rule reaffirmed:** stage files by name only.

### P3-C (Phase C) — LOLO runbook built, dry-run PASSED — 2026-10-14/15

**User instruction (2026-10-14).**
- Start Phase C, which does not need the Open Images reals.
- **The runbook:** the LOLO-fold runbook for a T4, one fold per session, following the
  frozen pre-registration §3 (the P3 recipe, `N_max` = 6,000, held-out-lineage AUC every
  500 steps). It must be platform-neutral, because an AWS quota exists (8 vCPU of G
  instances in us-east-2), but the AWS version is not built yet.
- **The hand-over** gives a short AWS plan: instance, getting the private Kaggle datasets
  onto it with the API key from an env var only, auto-shutdown, and cost per fold.
- **`N*`** runs only after all 8 folds, exactly as frozen.

**Interpretation, stated:** "the P3 recipe" is `S2-B`'s, as the frozen §3 table defines
it: "the P3 recipe (fixed `N*` steps), no new reals, no degradation". So a LOLO fold is
P2's S2 trainer and hyper-parameters (batch 16, LR 1e-4, warm-up 300, cosine, LoRA as in
P2), trained for a fixed 6,000 steps with no early stopping, on the fold's lineages.

**Built (commit `19f6d5d`).** These are additions the plan's §9 approved ("P3 run configs
… in a new `configs/p3_runs.yaml`"; "fixed-step training with held-out-lineage evaluation
in `p2_train_s2.py`"), plus the session tooling around them.
- **`configs/p3_runs.yaml`:**
  - `lolo_f1`…`lolo_f8`, each excluding exactly the lineage of the frozen fold table;
  - fold 1 also has `gen_matched: false`, a new optional run-config key whose default
    `true` leaves every P2 configuration unchanged;
  - crop 320.
- **`p2_train_s2.py --fixed-steps 6000 --heldout-every 500 --runs-config`:**
  - **Training:** exactly N steps, with no calibration steps (those are real, uncounted
    training steps in P2's mode) and no early stop; cosine over N.
  - **Every 500 steps:**
    - the **held-out-lineage AUC**: validation reals vs their reconstructions by the
      held-out members, each scored once on a fixed seeded crop, pooled and per member;
    - beside it, in-protocol validation AUC, log-loss and TPR@1%FPR (the plan's
      secondary signals);
    - a snapshot `snapshots/step_NNNNN/` (LoRA + head) for scoring at `N*(-f)` later.
  - **No selection inside a fold.**
  - **The fold-1 batch:** 8 pairs, with the `gen_matched` share 0 by the run config; the
    batch size is unchanged at 16.
- **`p2_run_queue.py --train-only`:** no scoring step (LOLO folds are scored on Panel B
  only at `N*(-f)`); a run is complete when its training report says `OK`, mode `train`.
  Resume now follows `--out-dir` instead of a hard-coded `outputs/p2/s2`.
- **`scripts/p3_lolo_next.py`:**
  - picks the lowest-numbered unfinished fold, so an interrupted fold comes first and
    resumes from its 500-step checkpoint;
  - **carries every complete fold forward** into the session's tree, so only the latest
    output ever needs attaching;
  - `DBP_FOLD` overrides; exit 3 when all eight are complete.
- **`env/kaggle_run_p3_lolo.py`** (8 cells) and **`env/p3_lolo_expect.json`**:
  - two pre-flights (data; then a 2-step smoke with held-out evaluation, save and
    reload);
  - every path from the environment (`DBP_INPUT_ROOT`, `DBP_WORKING_ROOT`,
    `DBP_TMP_ROOT`, `DBP_RESUME_DIRS`, `DBP_GPUS`, `DBP_FOLD`), Kaggle's by default;
  - inputs laid out as `datasets/<owner>/<slug>/` on any platform.
- **Dry-run harness:** a session now sees only the latest previous output, as Kaggle
  attaches one version.
- **Measured sizes** (P2 run directories in `kaggle_raw`):
  - checkpoint 7,219,345 B; LoRA 2,378,085 B; head 8,109 B;
  - so a fold with 12 snapshots is about 38 MB, and all 8 folds about 0.3 GB.

**local dry-run: PASSED**
- at the index (536 s) and at HEAD `19f6d5d` (632 s);
- **session 1:** fold 1 picked, killed after its step-4 checkpoint;
- **session 2:** fold 1 resumed at step 5 and completed; held-out curve at steps 2/4/6;
  3 snapshots; no `gen_matched`;
- **session 3:** with only session 2's output attached, fold 1 was carried forward and
  fold 2 trained.

The P2 S2 dry-run was re-run as a regression check (the trainer and queue are shared) and
PASSED. Full suite **930 passed, 7 skipped**; ruff clean; integrity CLEAN.

**A first HEAD dry-run FAILED, and it was not the code.**
- **What happened:** fold 2 froze after its step-4 checkpoint (12:25:51), and the queue
  stopped it at its 1-hour wall-clock deadline.
- **Cause:** Windows' power log shows the **PC slept from 12:26:06 to 14:07:58** (local);
  the deadline fired on wake.
- **Two follow-up start-up errors were mine:** my shell's working directory was inside
  the work folder, which Windows locks; and a wrong `--runbook` path.
- **The re-run PASSED.**
- **Lesson for the local Open Images fetch:** this PC does go to sleep unless "Sleep:
  Never" is set.

**Code zip:** `F:\thesis-2\kaggle_raw\dbp-code-19f6d5d.zip`, 10,888,181 bytes, sha256
`50b3900edb0608cac7ddb627d8fca816fa30434cbffb434ccb41225ee1e1c3d1`.

**Estimates (ESTIMATED from MEASURED rates):**
- **Per fold:** about 2.0–2.2 h. Basis:
  - training 6,000 × 0.886–0.928 s/step, the measured all-in rate of P2's four S2 runs
    on a T4 (`gpu_hours` × 3600 / `steps_run`, `reports/kaggle_20261005/p2_s2_*.json`),
    = 1.48–1.55 h;
  - held-out evaluation 12 × 491 × (1 + 1–3 members) images × 0.0267 s = 0.09–0.17 h;
  - setup about 0.4 h.
- **All 8 folds:** about 16–17.5 device-hours (the plan said 17.6).
- **Quota:** one fold uses one T4 of Kaggle's T4 × 2. How the idle second GPU is billed is
  I-042, still open.

**Not built yet:**
- the AWS runner (as instructed);
- the nested `N*` selector (CPU), to be built before the 8th fold finishes and run only
  after it;
- Panel B scoring at `N*(-f)`.

### P3-C — two LOLO folds per session; WHERE WE ARE AND WHAT COMES NEXT — 2026-10-15

**User decisions (2026-10-15).**
- **Two folds per session, one per T4** (the queue runs one worker per GPU; Session D's
  step times were measured with two runs in parallel). The two lowest unfinished folds
  are picked, an interrupted one first. "One fold per session" left a T4 idle; the user
  withdrew it.
- **LOLO session 1 is running now**, on the earlier `dbp-code-19f6d5d` (fold 1 only). The
  user reads the quota meter before and after (I-042).
- **Sessions 2 onward** use the two-fold version. `p2_train_s2.py` must be byte-identical
  to `19f6d5d`.
- **AWS prices, verified by the user** (on-demand, us-east-2):
  - g4dn.xlarge: 4 vCPU, 16 GiB, 1× T4, **$0.526/h**;
  - g4dn.2xlarge: 8 vCPU, 32 GiB, 1× T4, $0.752/h.
  - So two g4dn.xlarge fit the 8-vCPU quota, at about $1.05–1.16 per fold (2.0–2.2 h
    estimated).
  - **Spot is not available** (no quota), so plan on on-demand only.
  - **Do not build AWS yet**; the user decides after LOLO, on Kaggle quota.
- **`47d22a9`:** no purge; history left as it is.
- **The earlier HEAD dry-run failure** was the PC sleeping. Sleep is now off.

**Built (commit `33f43d8`).**
- **`env/p3_lolo_expect.json`:** `folds_per_session` 2, devices `0,1`. Pre-flight 2
  smoke-tests each fold on each T4.
- **Runbook text** updated for two folds.
- **Fixture:** `sd35` and `dcae` added; folds 1–2 are killed and resumed, then folds 3–4
  run with only the latest output attached. One new test.
- **Training code unchanged since `19f6d5d`:** `git diff 19f6d5d 33f43d8` is empty for
  `p2_train_s2.py`, `p2_run_queue.py`, `p3_lolo_next.py` and `p3_runs.yaml`.
- **The trainer's hash inside both zips:** `scripts/p2_train_s2.py` sha256 =
  `e60ad4b5aae06306fd6c76f52f3d389777afedd6c06ff7c01d0375f0e801e228` in both
  `dbp-code-19f6d5d.zip` and `dbp-code-33f43d8.zip`.

**local dry-run: PASSED at HEAD `33f43d8`** (610 s):
- **session 1:** picked `lolo_f1` and `lolo_f2`; both killed after step 4;
- **session 2:** both resumed at step 5 and finished;
- **session 3:** with only session 2's output attached, it carried folds 1–2 and trained
  folds 3–4;
- every fold has curve steps 2/4/6 with held-out AUC and 3 snapshots;
- fold 1 trained without `gen_matched`.

Lint, LOLO and run-config tests: 125 passed. Integrity CLEAN.

**Code zip for LOLO sessions 2 onward:** `F:\thesis-2\kaggle_raw\dbp-code-33f43d8.zip`, 10,891,268 bytes,
sha256 `e91587a3fe9700c29dd0a55a03dc3a9f8f1e33c0c0c0cccb06426691c382f637`.

**Estimates (ESTIMATED from MEASURED rates):**
- **Per session:** about 2.0–2.3 h wall for two folds in parallel. Basis:
  - training 6,000 × 0.886–0.928 s/step, the measured all-in rate of Session D's runs,
    which ran two at a time on T4 × 2, = 1.48–1.55 h;
  - held-out evaluation 0.09–0.17 h;
  - setup about 0.4 h;
  - pre-flight 2 about 0.1 h.
- **Remaining 7 folds:** 4 sessions (2 + 2 + 2 + 1), about 8–9 h wall.

---

**WHERE WE ARE (2026-10-15), for the next session to pick up.**
- **Branch:** `p3-plan`. HEAD after this entry is the log commit following `33f43d8`.
  Nothing is pushed; the tag `p3-prereg-frozen` is kept; `main` = `498817b`.
- **Panel B:** FROZEN and committed (`0f099ac`): digest
  `267d9fde75d76a23640674fa2fff7e454de23971b8e296323ca4762507c2a551`, 9,405 rows.
  Accepted by the user.
- **Open Images, route (ii), local:**
  - **Local run 1** stopped after Flickr refused the home IP: 107 rows decided, **88
    candidates**, state in `F:\thesis-2\kaggle_raw\p3_oi_candidates\`, originals in
    `F:\thesis-2\kaggle_raw\p3_oi_originals\` (kept until the Kaggle accept step passes; never committed
    or uploaded).
  - **The user's resume plan:**
    - trial not before 2026-10-15 06:00 UTC, 1 connection, `--min-interval 6`, 30 min;
    - if clean, continue at that pace;
    - if it trips, wait 24 h, then ONE trial at `--min-interval 15`;
    - if that trips too, STOP and report.
  - **Who runs it:** the user, from their own PowerShell window. The commands are in the
    hand-over of 2026-10-14.
  - **No session is recorded yet:** `p3_oi_local_sessions.csv` does not exist.
  - **After each session I log** rows decided, candidates, refusals and the pace (from
    `p3_oi_local_sessions.csv`, or `--status`).
- **When the local stage reaches 4,200 candidates:**
  - commit `p3_oi_candidates_manifest.csv` to `reports/p3_open_images/`;
  - fill `env/p3_oi_accept_expect.json` (`candidates.sha256` / `rows`, now
    `PENDING-LOCAL-RUN`);
  - dry-run `p3_oi_accept`; build a zip;
  - the user uploads the crops as the private `dbp-p3-oi-candidates` and runs
    `env/kaggle_run_p3_oi_accept.py`;
  - then commit the attribution file to `docs/`.
- **LOLO (Phase C):**
  - **session 1:** running on `dbp-code-19f6d5d` (fold 1);
  - **sessions 2–5:** `dbp-code-33f43d8`, two folds each (f2+f3, f4+f5, f6+f7, f8), with
    the previous version's output attached;
  - **after each session:** log the held-out curve (Cell 7) and the quota reading
    (I-042 from session 1).
- **Next code to build, before LOLO's last session finishes:**
  - **(1)** the nested `N*` selector (CPU), run only after all 8 folds:
    - `N*(-f)` maximises the mean held-out AUC over the other 7 folds;
    - `N*` maximises the mean over all 8;
    - with H1-a's statistics (`G`, the 97.5% CI, no fold < 0.75);
  - **(2)** Panel B scoring of each fold at `N*(-f)` from its snapshot (GPU), for H1-a/b
    and H1-c (with `BANK-1`).
  - Both get tests and a dry-run.
- **Still open:**
  - X1 format control on Kaggle (meter readings);
  - I-042 (quota billing).

### P3-C — correction: no LOLO session ran on `19f6d5d`; all 8 folds use `33f43d8` — 2026-10-15

- **The user clarified:** no Kaggle session was ever run with `dbp-code-19f6d5d`. The
  previous entry's "session 1 is running now on `19f6d5d` (fold 1)" is therefore wrong.
- **Corrected plan:** every LOLO fold is trained with **`dbp-code-33f43d8`**, two folds per
  session, in **4 sessions**: f1+f2, f3+f4, f5+f6, f7+f8.
  - **Session 1** attaches only the four datasets: `dbp-code-33f43d8`,
    `dbp-p2-s2-inputs`, `report` (p1-data) and `thesis-dbp-1-dataset`. There is no
    previous output.
  - **Sessions 2–4** also attach the notebook's latest completed version.
- **Trainer:** `p2_train_s2.py` sha256 `e60ad4b5…e228`, identical to `19f6d5d` regardless.
- **I-042:** to be read from the quota meter around session 1.

### P3-C — LOLO session 1: folds 1 and 2 COMPLETE — 2026-10-15

**Run:** Kaggle GPU T4 × 2, code `dbp-code-33f43d8` (trainer sha256 `e60ad4b5…e228`).
- **Archive:** `F:\thesis-2\p3_lolo_reports.zip` (10,055,763 bytes; `testzip` finds no bad
  member). Committed under `reports/kaggle_20261015/`: both fold reports, queue and smoke
  status, plan, pre-flight, logs.
- **Pre-flight 1:** PASSED, 6 checks:
  - 2 × Tesla T4;
  - `HF_TOKEN` present;
  - 4 input hashes match;
  - imports: peft 0.20.0, transformers 5.16.1;
  - all 2,488 `gen_matched` files match the committed reference.
- **Pre-flight 2 (smoke):** each fold on each GPU, COMPLETE in 0.0583 h.
- **Queue:** COMPLETE.
  - wall 1.575 h; device-hours 3.0834;
  - `lolo_f1` on GPU 0, 1.575 h; `lolo_f2` on GPU 1, 1.508 h;
  - training 2026-10-15 11:13:44 → 12:47:57 (f1) and → 12:43:58 (f2), from the logs;
  - no warning, error or NaN in either training log.

**MEASURED per fold (fold reports):**

| fold | held out | reals / recons | gen_matched | steps | GPU-h | train-loop s/step |
|---|---|---|---|---|---|---|
| lolo_f1 | sd15, sd21, sd_ft_ema | 491 / 1,473 | **no** (batch: 8 pairs, 0 gen) | 6,000 | 1.568 | 0.81464 |
| lolo_f2 | sdxl, sdxl_fp16fix | 491 / 982 | yes (6 pairs + 2 gen) | 6,000 | 1.502 | 0.79676 |

**Held-out-lineage AUC every 500 steps** (validation reals vs held-out reconstructions):

| step | 500 | 1000 | 1500 | 2000 | 2500 | 3000 | 3500 | 4000 | 4500 | 5000 | 5500 | 6000 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| f1 | 0.99459 | 0.99788 | 0.99857 | 0.99901 | 0.99916 | 0.99934 | 0.99931 | 0.99934 | 0.99944 | 0.99943 | 0.99941 | 0.99940 |
| f2 | 0.97948 | 0.99391 | 0.99680 | 0.99735 | 0.99812 | 0.99814 | 0.99810 | 0.99839 | 0.99844 | 0.99827 | 0.99822 | 0.99821 |

- **In-protocol validation at step 6000:**
  - f1: AUC 0.99892, log-loss 0.04623, TPR@1%FPR 0.97852;
  - f2: AUC 0.99888, log-loss 0.04619, TPR@1%FPR 0.97852.
- **Descriptive only.** Nothing is selected from these curves. `N*` and `N*(-f)` are
  computed only after all 8 folds, by the frozen nested rule, and H1 is judged at
  `N*(-f)`.
- Both folds held out a 4-channel KL lineage while another 4-channel KL lineage stayed in
  training (SDXL for f1, SD1 for f2). The lineages that differ more (GAN-SR, DC-AE, MoVQ,
  FLUX.2, Qwen, 16ch) are folds 3–8.

**Timing against the estimate:** training plus evaluation per fold MEASURED 1.50–1.58 h,
inside the estimated 1.57–1.72 h. Setup before 11:13:44 is not in the reports.

**Next:**
- **LOLO session 2:** the same notebook and cells, plus this version's output attached;
  it should pick `lolo_f3` + `lolo_f4`.
- **Then sessions 3 and 4:** f5+f6, then f7+f8.
- **Pending from the user:** the quota meter readings before and after session 1 (I-042).

### P3-B (step 3) — Flickr route CLOSED; the CVDF S3 copies, one Kaggle CPU session — 2026-10-15

**Local trial sessions** (state in `F:\thesis-2\kaggle_raw\p3_oi_candidates\`, MEASURED):

| session | UTC | pace | rows decided | candidates | requests | refusals | ended |
|---|---|---|---|---|---|---|---|
| A | 10:32:34 → 10:47:36 | 1 connection, `--min-interval 6` | 107 → 120 | 88 → 98 | 30 | 17 (all 429) | throttle stop: 10 min at the 60 s ceiling (exit 1) |
| B | from 18:22:20 | same | 120 → 128 | 98 → 105 | — | ≥ 10 (8 by the last saved batch) | Ctrl+C by the user |

- **Session B's record:** no session row was written. Ctrl+C through PowerShell's
  `Tee-Object` pipe likely ended the process before it could; the state is consistent as
  of its last saved batch.
- **Final local state:** 128 rows decided, **105 candidates**, positions 121–122
  undecided, 106 originals on disk, 261,897,889 bytes downloaded in total.

**USER DECISION (2026-10-15): the Flickr route is STOPPED permanently.**
- All three tries were refused: Kaggle IPs; the home IP at 4 connections; the home IP at
  1 request per 6 s, about 25 h after the first block.
- Flickr does not allow bulk downloads of the originals, and that is respected.
- There is no 15 s retry.

**DEVIATION (dated 2026-10-15), reversing the user's 2026-10-11 host decision.**
- **New source:** the Open Images V7 copies in the official CVDF S3 bucket
  `open-images-dataset` (no credentials). This is option (a) of step 2.
- **What stays the same:** the source (Open Images V7 validation), the CC BY 2.0 rows,
  the seed-2026 walk order and the attribution file. Only the file copy changes: the CVDF
  README says the copies are "rescaled to have at most 1024 pixels on their longest
  side".
- **Reason:**
  - the test reals are web-processed, not camera originals (E5 Chameleon reals are all
    exactly 500 px wide; E4 is social-media images);
  - S2-D's degradation chains supply the varied compression.
- **Rules kept:**
  - shorter side ≥ 384, no resizing, native 384 crop to PNG;
  - greyscale, CMYK, 16-bit, palette and alpha rejected;
  - dedup (sha256 + pHash ≤ 6) against COCO, Panel B, P2, the training corpus and all
    E3/E4/E5 reals;
  - the first 4,000 in seed order; a seeded 3,600 / 400 split.
- **Checksum:** `OriginalMD5` does not apply to a rescaled copy; each file's sha256 is
  recorded instead.
- **Where it runs:** ONE Kaggle CPU session. **The local-route (ii) deviation of
  2026-10-14 is closed.**
- **Hosts:** `open-images-dataset.s3.amazonaws.com`, plus the bucket's regional endpoint;
  up to 8 connections.
- **The Flickr originals already downloaded** (105 candidates, 106 files) are NOT used,
  so there is one processing history per source. They stay in `kaggle_raw` and are not
  uploaded.

**Pre-registration check (the user asked).**
- The frozen text says "**The new real source:** Open Images V7 only (Flickr)" (§2, the
  platform rule) and "the new real source: Open Images V7 only, 3,600 train + 400 held
  out". It never says "originals".
- "(Flickr)" names the platform for the platform rule, and the CVDF copies are the same
  Flickr-platform photos.
- **So no amendment is needed; this log entry is enough.**
- The non-frozen `P3_PLAN.md` §4a ("Flickr originals, not the S3 copies") now carries a
  dated reversal note.

**S3 header probe (local, 2026-10-15; HEAD only, 0.2 s apart).**
- The first 50 rows in seed order, at `.../validation/<ImageID>.jpg`.
- **Results:** 200 × 50 (all `image/jpeg`); no redirects.
- **Content-Length:** mean **371,264 B**, median 347,284, min 46,598, max 1,294,820.
- **Saved as:** `F:\thesis-2\kaggle_raw\p3_open_images_metadata\s3_head_probe_2026-10-15.json`.

**Built (commit `69cc811`).**
- **`open_images.py`:** `S3_HOST_PATTERNS`, exact for the bucket host plus the same
  bucket's regional endpoint, so lookalikes are refused (tested); `s3_url()`, whose
  image-id format is validated; `SOURCE_COPY`.
- **`p3_fetch_open_images.py --source s3`:**
  - downloads the CVDF copy and skips the `OriginalMD5` comparison (the sha256 is
    recorded);
  - allows no redirect off the bucket;
  - each manifest row records `fetch_url`, `source_copy` (`cvdf_s3_rescaled_max1024px`)
    and `md5_checked` (False);
  - the attribution keeps the Flickr page URL, which identifies the photo;
  - every other rule is unchanged.
- **Runbook:** `env/kaggle_run_p3_open_images.py` (header, plus Cell 5 passes `--source`).
- **Expect file:** `source s3`, the S3 hosts, 8 workers, estimates.
- **Fixture:** parametrised; S3 by default, and the route-(ii) fixture keeps Flickr.
- **Tests:** 2 new; the runbook test updated.

**Checks:**
- **local dry-run: PASSED** at the index (20 s) and at HEAD `69cc811` (19 s): 6 accepted
  (4 / 2), 11 rejections each on its own path, no host or checksum rejection in S3 mode,
  manifest rows record the S3 copy;
- the route-(ii) accept fixture still PASSES;
- full suite **933 passed, 7 skipped**; integrity CLEAN.
- `p2_train_s2.py` is unchanged: sha256 `e60ad4b5…` in this zip too. LOLO keeps
  `dbp-code-33f43d8`.

**Code zip:** `F:\thesis-2\kaggle_raw\dbp-code-69cc811.zip`, 10,917,640 bytes, sha256
`fc2d97e8332ac45577e9d67b13d45f0392089583c5b7e0be7ced648f1d570aaf`.

**Estimates for the hand-over:**

| item | value | basis |
|---|---|---|
| download | about 1.6–1.75 GB | ESTIMATED: about 4,300–4,700 rows walked × 371,264 B, the **measured** S3 mean |
| peak `/tmp` | about 0.1 GB | ESTIMATED |
| output | about 0.75–1.1 GB | ESTIMATED: 4,000 × 187,499–268,079 B, the **measured** crop means (local Flickr run; P1 T-real) |
| wall time | about 0.5–0.7 h | bootstrap about 212 s (**measured**, LOLO session 2); COCO index 886 s (**measured**, Open Images run 1); the walk ESTIMATED |

**Next:**
1. The user runs `env/kaggle_run_p3_open_images.py` on `dbp-code-69cc811` (one CPU
   session).
2. Then commit the manifest, summary, walk log and coco summary, and the attribution file
   to `docs/`.
3. Then the reconstruction of the 4,000 new reals through the bank (step 4a, GPU) can be
   built.

### P3-C — LOLO session 2: folds 3 and 4 COMPLETE — 2026-10-15/16

**Run:** Kaggle GPU T4 × 2, `dbp-code-33f43d8`, with session 1's output attached.
- **Archive:** `F:\thesis-2\kaggle_raw\p3_lolo_reports_Phase-2.zip` (10,058,406 bytes; `testzip`
  finds no bad member).
- **Committed:** under `reports/kaggle_20261015/lolo_session2/` (the fold-3/4 reports, the
  session's status, plan, smoke status, pre-flight and logs).
- **Setup:** MEASURED about 405 s from the session start to the smoke (user's pasted log).
- **Pre-flight 1:** PASSED, 6 checks.
- **Carry-forward:** `lolo_f1` and `lolo_f2` were carried from session 1; their reports in
  this output are **byte-identical** to session 1's.
- **Pre-flight 2 (smoke):** COMPLETE in 0.0417 h.
- **Queue:** COMPLETE, wall 1.4833 h, device-hours 2.9584.
  - training 2026-10-15 18:36:13 → 20:04:47 (f3, GPU 0) and → 20:04:37 (f4, GPU 1);
  - no warning, error or NaN in either log.

**MEASURED per fold:**

| fold | held out | reals / recons | gen_matched | steps | GPU-h | train-loop s/step |
|---|---|---|---|---|---|---|
| lolo_f3 | flux, sd35 | 491 / 982 | yes | 6,000 | 1.474 | 0.78145 |
| lolo_f4 | dcae | 491 / 491 | yes | 6,000 | 1.471 | 0.80035 |

**Held-out-lineage AUC** (per member in brackets for f3):

| step | 500 | 1000 | 1500 | 2000 | 2500 | 3000 | 3500 | 4000 | 4500 | 5000 | 5500 | 6000 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| f3 | 0.93592 | 0.96325 | 0.98092 | 0.97628 | 0.98369 | 0.98732 | 0.98694 | 0.98703 | 0.98872 | 0.98798 | 0.98803 | 0.98800 |
| f3 flux | 0.90986 | 0.94102 | 0.96814 | 0.96036 | 0.97258 | 0.97848 | 0.97798 | 0.97816 | 0.98075 | 0.97953 | 0.97964 | 0.97957 |
| f3 sd35 | 0.96198 | 0.98547 | 0.99370 | 0.99219 | 0.99480 | 0.99617 | 0.99590 | 0.99590 | 0.99669 | 0.99643 | 0.99642 | 0.99643 |
| f4 dcae | 0.98197 | 0.99421 | 0.99573 | 0.99484 | 0.99402 | 0.99427 | 0.99303 | 0.99339 | 0.99356 | 0.99340 | 0.99352 | 0.99353 |

- **In-protocol at step 6000:**
  - f3: AUC 0.99865, log-loss 0.04522, TPR@1%FPR 0.98047;
  - f4: AUC 0.99805, log-loss 0.0559, TPR@1%FPR 0.96875.
- **Descriptive only.** Nothing is selected; `N*` and `N*(-f)` come after all 8 folds by
  the frozen rule.
- Fold 3 (16-channel KL) is the first fold whose held-out AUC starts clearly lower (0.936
  at step 500; FLUX.1 0.910). Fold 4 (DC-AE) peaks early (0.99573 at 1500) and settles at
  about 0.9935.

**LOLO status:** folds 1–4 complete. Remaining: sessions 3 (f5 + f6) and 4 (f7 + f8), the
same notebook and cells, with this version's output attached.

**Timing:** the per-session wall is about 405 s setup + 0.04 h smoke + 1.48 h training ≈
1.65 h (MEASURED components), below the 2.0–2.3 h estimate.

### P3-C — LOLO session 3: folds 5 and 6 COMPLETE; fold 6 (FLUX.2) is the first hard fold — 2026-10-16

**Run:** Kaggle GPU T4 × 2, `dbp-code-33f43d8`, with session 2's output attached.
- **Archive:** `F:\thesis-2\kaggle_raw\p3_lolo_reports_Phase-3.zip` (10,061,008 bytes; `testzip`
  finds no bad member).
- **Committed:** under `reports/kaggle_20261016/lolo_session3/`.
- **Checks:**
  - pre-flight 1 PASSED (6 checks);
  - folds 1–4 carried forward, each report **byte-identical** to the session that trained
    it;
  - smoke COMPLETE (0.0417 h);
  - queue COMPLETE, wall 1.4917 h, device-hours 2.9584;
  - training 2026-10-16 07:10:43 → 08:38:21 (f5, GPU 0) and → 08:39:47 (f6, GPU 1);
  - no warning, error or NaN in either log.

**MEASURED per fold:**

| fold | held out | reals / recons | steps | GPU-h | train-loop s/step |
|---|---|---|---|---|---|
| lolo_f5 | realesrgan_x4 (GAN-SR) | 491 / 491 | 6,000 | 1.458 | 0.79277 |
| lolo_f6 | flux2_klein (kl_32ch_flux2) | 491 / 491 | 6,000 | 1.482 | 0.80648 |

**Held-out-lineage AUC:**

| step | 500 | 1000 | 1500 | 2000 | 2500 | 3000 | 3500 | 4000 | 4500 | 5000 | 5500 | 6000 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| f5 | 0.99963 | 0.99992 | 0.99996 | 0.99997 | 0.99999 | 0.99995 | 0.99998 | 0.99999 | 0.99998 | 0.99999 | 0.99998 | 0.99998 |
| f6 | **0.60187** | 0.65471 | 0.71733 | 0.72713 | 0.73214 | 0.76231 | 0.76615 | 0.75554 | 0.76551 | 0.76872 | 0.76789 | 0.76798 |

- **In-protocol at step 6000:**
  - f5: AUC 0.99826, log-loss 0.05466, TPR@1%FPR 0.96484;
  - f6: AUC 0.99976, log-loss 0.02945, TPR@1%FPR 0.99219.
- **Descriptive only; nothing is selected.**
- **Fold 6 is the first fold far below the others.** Its held-out AUC (FLUX.2's 32-channel
  KL VAE) rises from 0.602 to 0.769. It is below 0.75 for every step up to 2500 and above
  0.75 from 3000 on (4000: 0.756).
- **Why this matters for the frozen rule:** H1-a's "no fold below 0.75" is judged at
  `N*(-f)`, which is computed from the *other seven* folds. So whether fold 6 passes
  depends on the step those folds select. That is decided only after all 8 folds, by the
  frozen rule, and is not inferred here.
- **Fold 5 (GAN-SR) is effectively separable at every step** (≥ 0.99963).

**LOLO status:** folds 1–6 complete. Remaining: session 4 (f7 Qwen-Image + f8 MoVQ), with
this version's output attached.

**Timing (MEASURED):** about 1.49 h of training per session (sessions 2 and 3 both under
1.5 h), plus about 7 min of setup and 2.5 min of smoke.

### P3-C — LOLO session 4: folds 7 and 8 COMPLETE; ALL EIGHT FOLDS DONE — 2026-10-16/17

**Run:** Kaggle GPU T4 × 2, `dbp-code-33f43d8`, with session 3's output attached.
- **Archive:** `F:\thesis-2\kaggle_raw\p3_lolo_reports_phase-4.zip` (10,063,746 bytes; `testzip`
  finds no bad member).
- **Committed:** under `reports/kaggle_20261016/lolo_session4/`.
- **Checks:**
  - pre-flight 1 PASSED (6 checks);
  - folds 1–6 carried forward, each report byte-identical to the session that trained it;
  - smoke COMPLETE (0.0417 h);
  - queue COMPLETE, wall 1.5 h, device-hours 3.0;
  - training 2026-10-16 09:53:22 → 11:23:02 (both folds);
  - no warning, error or NaN.

**MEASURED per fold:**

| fold | held out | steps | GPU-h | train-loop s/step |
|---|---|---|---|---|
| lolo_f7 | qwen_image (kl_16ch_qwen) | 6,000 | 1.492 | 0.81177 |
| lolo_f8 | movq_k22 (vq_movq_4ch) | 6,000 | 1.492 | 0.81204 |

**Held-out-lineage AUC:**

| step | 500 | 1000 | 1500 | 2000 | 2500 | 3000 | 3500 | 4000 | 4500 | 5000 | 5500 | 6000 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| f7 | 0.93659 | 0.95862 | 0.96913 | 0.97690 | 0.98112 | 0.97960 | 0.98446 | 0.98577 | 0.98499 | 0.98586 | 0.98602 | 0.98609 |
| f8 | 0.99405 | 0.99873 | 0.99902 | 0.99890 | 0.99958 | 0.99940 | 0.99955 | 0.99937 | 0.99947 | 0.99938 | 0.99937 | 0.99935 |

- **In-protocol at step 6000:**
  - f7: AUC 0.99851, log-loss 0.04927, TPR@1%FPR 0.97852;
  - f8: AUC 0.9984, log-loss 0.0526, TPR@1%FPR 0.9668.
- **Descriptive only.**

**LOLO COMPLETE: 8 / 8 folds**, all at 6,000 steps with 12 held-out evaluations and 12
snapshots each.
- **Training device-hours (MEASURED):** 3.0834 + 2.9584 + 2.9584 + 3.0 = **12.0 h** over 4
  sessions. The plan estimated 17.6.
- **Every fold** used trainer sha256 `e60ad4b5…e228` (`dbp-code-33f43d8`).
- **The latest Kaggle version's output holds all 8 folds** (run directories with
  checkpoints and snapshots, plus reports). Panel B scoring at `N*(-f)` needs that output;
  it should be kept, ideally published as a private dataset.

**What comes next, as frozen (pre-registration §3; nothing selected yet):**
1. **The nested `N*` selector (CPU), to be built with tests and a dry-run, then run once
   on the 8 committed fold reports:**
   - `N*(-f)` = the step maximising the mean held-out AUC over the other seven folds;
   - `N*` = the step maximising the mean over all eight;
   - H1-a: `G` = the mean of each fold's held-out AUC at `N*(-f)`; success if `G ≥ 0.90`,
     the 97.5% CI lower bound ≥ 0.85, and no fold below 0.75.
2. **Panel B scoring of each fold at `N*(-f)` (GPU),** for H1-b (E7 subsets, folds 1–3)
   and H1-c (folds 2–8 against `BANK-1`), and the training of every non-LOLO model for
   exactly `N*` steps.

**Still pending:**
- the Open Images S3 session (`dbp-code-69cc811`, CPU);
- the quota readings (I-042).

### P3-C — the nested `N*` selector: built and tested; decisions fixed BEFORE it runs — 2026-10-17

**Written and committed before the selector touched any real fold report.**

**Decisions (user, 2026-10-17):**
1. **Tie-break.** Pre-registration §3 defines `N*` and `N*(-f)` as an argmax but names no
   tie-break (checked: no tie rule anywhere in `P3_PREREGISTRATION.md` or `P3_PLAN.md` §2).
   **The earliest tied step is taken**, and every tie is listed in the output. This is
   fixed now, before the run.
2. **Panel B is NOT scored yet, not even for the LOLO folds.**
   - **Why:** pre-registration §8 invalidates the analysis if any §5 value is chosen while
     Panel B scores exist. The degradation-chain strengths (§5, `P3_PLAN.md` §4b) are not
     fixed yet.
   - **The plan:** `P3_PLAN.md` §7.3 puts all Panel B scoring in the final week.
   - **So:** Panel B is scored **once, at the end, for every frozen model together**. H1-b
     (E7 subsets, folds 1–3) waits for that. H1-a and H1-c use validation data only.
   - This replaces "Panel B scoring at `N*(-f)` next" in the session-4 entry above.

**Built** (approved item, `P3_PLAN.md` §9: "the nested `N*` selector"):
`scripts/p3_select_nstar.py` + `tests/test_p3_nstar.py` (15 tests).
- **`select`:** reads the 8 fold training reports and validates each one:
  - status OK, mode train, seed 0;
  - `fixed_steps` 6000, `heldout_every` 500, the held-out steps exactly 500…6000;
  - held-out lineages as in `configs/p3_runs.yaml`;
  - each fold exactly once.
- **`select` then writes `reports/p3_nstar.json`** with `N*(-f)` for every fold, `N*`,
  every tie, the mean curves, each fold's AUC at `N*(-f)` against 0.75, `G`, and the
  H1-a verdict as far as it can be decided.
  - Means are compared **exactly**: the reports carry 5-decimal AUCs, which become integer
    1e-5 units.
  - **It refuses to overwrite its output, so the real selection happens once.**
- **`ci`:** computes the H1-a CI from per-image scores, reading the frozen selection; it
  never re-selects.
  - **Inputs:** per-image scores at each fold's `N*(-f)`.
  - **Refusals:**
    - a score at any other step;
    - folds scored on different reals;
    - a rescored fold AUC more than 0.001 from its training-time value.
  - **What is resampled** (pre-registration §4: 2,000 resamples, seed 2026, 97.5% for a
    primary hypothesis):
    - the 491 validation reals, once per resample and shared by every fold (one real
      pool, the same images in every fold, so paired);
    - each fold's held-out reconstructions within its own slice, independently, in fold
      order;
    - each fold's AUC is recomputed, and `G` is their mean.
  - **Held fixed:** the 8 folds (every lineage, not a sample) and `N*(-f)` (a §5 value,
    chosen once).
  - **Interval:** the 1.25th–98.75th percentile.
- **H1-a verdict logic:**
  - FAIL as soon as `G < 0.90` or a fold is below 0.75, whatever the CI;
  - otherwise PENDING-CI until the CI exists;
  - PASS only with all three criteria met.

**Gap found while building (reported to the user).** The CI cannot be computed from the
committed reports: `p2_train_s2.py` records each held-out AUC rounded to 5 decimals, not
the per-image scores the frozen bootstrap resamples.
- **Proposed:** rescore each fold's snapshot at `N*(-f)` on its held-out validation set in
  the next GPU session (validation data, not Panel B), writing per-image scores, then run
  `ci` locally.
- **This is an addition** (a per-image score writer); it waits for approval (§9b).

**Tests:** `tests/test_p3_nstar.py`, 15 passed; ruff clean.

### P3-C — `N*` SELECTED (once): `N*` = 5000; H1-a `G` = 0.966228, CI pending — 2026-10-17

**Dry-run first** (CLAUDE.md §12.3; local, CPU), from `git archive 86e76a1` and a
synthetic 8-fold fixture (random values, not the real reports):
- `select` wrote its output, and a second `select` was refused (exit 2);
- `ci` refused scores whose AUC did not reproduce the report (fixture 1, jittered on
  purpose);
- on matching scores `ci` computed the CI: 2,000 resamples, seed 2026, percentiles
  1.25/98.75, rescore differences ≤ 4.4e-6.
- **local dry-run: PASSED.**

**The single real run.**
- **Command:** `python thesis_dbp/scripts/p3_select_nstar.py select` with `--report` on
  the 8 committed fold reports (`reports/kaggle_20261015/p2_s2_lolo_f{1,2}_seed0.json`,
  `…/lolo_session2/…f{3,4}…`, `reports/kaggle_20261016/lolo_session3/…f{5,6}…`,
  `…/lolo_session4/…f{7,8}…`).
- **Code:** HEAD `86e76a1`.
- **Output:** `reports/p3_nstar.json` (sha256
  `b24b160ef34324928367bc4acb1318ef54fb9c37ab025abfc83e70d61ac64174`). It records each
  input's sha256.

**`N*` = 5000** (mean held-out AUC over all 8 folds 0.966629; no tie).

The mean over 8 folds (from `p3_nstar.json`):

| step | 500 | 1000 | 1500 | 2000 | 2500 | 3000 | 3500 | 4000 | 4500 | 5000 | 5500 | 6000 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mean | 0.928013 | 0.945154 | 0.957183 | 0.958797 | 0.960978 | 0.965041 | 0.965940 | 0.964853 | 0.966264 | **0.966629** | 0.966555 | 0.966568 |

**H1-a table at `N*(-f)`:**

| fold | held out | `N*(-f)` | ties | runner-up (margin) | held-out AUC | ≥ 0.75 |
|---|---|---|---|---|---|---|
| 1 | SD1 (sd15 0.99980, sd21 0.99907, sd_ft_ema 0.99943) | 5000 | none | 6000 (6.6e-5) | 0.99943 | yes |
| 2 | SDXL (sdxl 0.99869, fp16fix 0.99786) | 5000 | none | 6000 (6.1e-5) | 0.99827 | yes |
| 3 | 16ch (flux 0.97953, sd35 0.99643) | 5000 | none | 6000 (7.3e-5) | 0.98798 | yes |
| 4 | DC-AE | 5000 | none | 6000 (8.9e-5) | 0.99340 | yes |
| 5 | GAN-SR | 5000 | none | 6000 (6.9e-5) | 0.99999 | yes |
| 6 | **FLUX.2** | **4500** | none | 6000 (6e-6) | **0.76551** | **yes** (by 0.01551) |
| 7 | Qwen-Image | 5000 | none | 6000 (1.03e-4) | 0.98586 | yes |
| 8 | MoVQ | 5000 | none | 6000 (6.6e-5) | 0.99938 | yes |

- **`G` = 0.966228** (exact: the sum of the 8 AUCs in 1e-5 units ÷ 8 = 0.9662275).
- **H1-a so far:**
  - `G ≥ 0.90`: **met**;
  - no fold below 0.75: **met**; fold 6 is the lowest at 0.76551;
  - the 97.5% CI lower bound ≥ 0.85: **NOT-RUN**.
  - **Verdict: PENDING-CI.** It cannot be PASS until the CI exists; it is not FAIL.
- **Secondary (in-protocol, at `N*(-f)`):**
  - validation AUC 0.99809–0.99987;
  - log-loss 0.0245–0.05632;
  - TPR@1%FPR 0.96484–0.99609.

**Recorded observations:**
- **The curve is flat from step 3000.** The 8-fold mean stays within 0.965–0.967, and every
  `N*(-f)` beats its runner-up by at most 1.03e-4.
- **The inputs are 5-decimal AUCs**, so margins this small sit near the reports'
  rounding. The rule was applied exactly as frozen. This is recorded so no claim reads
  more into 5000 versus 5500 or 6000 than the data hold.
- **Fold 6 passes the floor at its `N*(-f)` of 4500.** At steps ≤ 2500 it would not
  have, but no fold picks its own step.

**Frozen from here (§5):**
- `N*` = 5000 for every non-LOLO model: BANK-k, BANK-1L, S2-B, S2-D and the ablations;
- `N_max` (6,000), the grid and the thresholds unchanged.

**Pending:**
- the H1-a CI, which needs per-image held-out scores at `N*(-f)` (see the gap in the
  previous entry);
- H1-b, after Panel B is scored at the end;
- H1-c, after `BANK-1` trains.

**Checks:** full suite 948 passed, 7 skipped; ruff clean; integrity CLEAN.

### P3-B (step 3) — Open Images S3 session COMPLETE: 4,000 reals (3,600 / 400), reviewed — 2026-10-17

**Run:** one Kaggle CPU session on `dbp-code-69cc811`.
- **Archive:** `F:\thesis-2\dbp-p3-open-images.zip`, 937,899,551 bytes; `testzip` finds no bad
  member.
  - It sits in `F:\thesis-2\`, not `kaggle_raw` (CLAUDE.md §7b); it is outside REPO_ROOT, so
    integrity is unaffected.
  - Extracted to `F:\thesis-2\kaggle_raw\p3_oi_s3_x\`.
- **Code:** `COMMIT.txt` = `69cc81148522c657774087cec34d4ea0daef23fd`. The fetch script,
  `open_images.py` and the runbook are byte-identical to the commit.
- **Committed:** under `reports/kaggle_20261017/p3_open_images/` (manifest, summary, walk,
  COCO index summary, pre-flight). The attribution file is at
  `docs/p3_open_images_attribution.csv`.
- **The images are NOT committed** (4,000 PNGs, 872.5 MB). They stay in the Kaggle output
  and `kaggle_raw`.

**MEASURED (from `p3_open_images_summary.json`):**
- **Pre-flight PASSED:** 4 checks, 6 input hashes matching.
- **COCO index:** 123,287 images in 685.6 s.
- **Walk:**
  - **4,096 rows walked** in seed order: **4,000 accepted, shortfall 0**, plus 87
    rejected and 9 not needed;
  - S3 only: 0 refusals, 0 failed downloads, no redirects, no unapproved host;
  - 1,261,956,971 bytes downloaded in 973.6 s;
  - crops 872,531,042 bytes.
- **Rejections:**
  - greyscale mode L/1: 51;
  - greyscale `rgb_equal`: 8;
  - shorter side < 320: 8;
  - shorter side 320–383: 6;
  - CMYK: 3;
  - pHash duplicates: COCO train2017 6 (+2 at crop level), E5 reals 2 (crop level),
    Panel B 1.
- **References deduplicated against:** Panel B 9,405; the P2 panel (all views); T-real;
  T-ctrl; `gen_matched`; 49,200 bank reconstructions; E3 726, E4 1,251 and E5 14,863
  reals; COCO train2017 + val2017.
- **Split:** 3,600 train / 400 held out.
- **Descriptive (not a gate):** 1 of 726 E3 reals pHash-matches COCO.
- **Against the estimates:**
  - download 1.26 GB vs about 1.6–1.75 GB estimated (4,096 rows walked, not 4,300–4,700);
  - output 0.87 GB, within 0.75–1.1.

**Independent local review** (scratchpad scripts; repo code imported, the output read as
data):
- **Files:** 4,000 PNGs on disk = 4,000 manifest rows = 4,000 attribution rows; none
  extra, none missing.
- **Per image:**
  - every sha256 matches;
  - every pHash, recomputed, matches;
  - every file is PNG, RGB, no transparency, 384 × 384.
- **Uniqueness:** 4,000 distinct Open Images ids, sha256 values and image ids.
- **Within the set:** 0 crop pairs at pHash ≤ 6 bits.
- **Licence:** every accepted id is in the committed CC BY 2.0 metadata (41,620 rows,
  identical to the output's copy). Every attribution licence is CC BY 2.0, and none has
  an empty author.
- **The rules, re-derived:**
  - the walk is exactly the first 4,096 rows of the seed-2026 order;
  - the accepted ids equal the manifest;
  - the 3,600 / 400 split is reproduced exactly by `assign_split`;
  - every `fetch_url` is the bucket's validation copy, `source_copy` =
    `cvdf_s3_rescaled_max1024px`, `md5_checked` False;
  - every crop lies inside its source, and every source's shorter side is ≥ 384.
- **Visual check:** two random 48-image contact sheets (seed 7, not committed) show
  ordinary, varied photographs, with no blank, corrupt, rendered or synthetic-looking
  crops.

**Observations, recorded (none breaks a rule):**
1. **4 crops are greyscale or near-greyscale** (channel spread ≤ 2; one exactly 0): a
   chrome car, a black-and-white portrait, hair, a zebra.
   - D-039 judges the **whole source**, by design (`p1_build_reals.py`: "a crop of a
     colour photograph can be locally colourless"). These sources passed it.
   - They are kept, the same rule as T-real.
2. **5 accepted sources are larger than 1,024 px** (up to 8,100 × 3,888), despite the
   CVDF README's "at most 1024 pixels". Their crops are still native 384 crops. The
   `source_copy` label describes the bucket, not each file.
3. **Source JPEG quality:**
   - minimum 26; 1 below 50, 40 below 70; median 94;
   - subsampling 4:2:0 1,778, 4:4:4 1,129, 4:2:2 1,013, unknown(-1) 80;
   - the 80 are colour images with a sampling layout Pillow does not name.
   - Every crop is PNG; the real/reconstruction pairs share this history, and S2-D's
     degradation chains add compression on top.
4. **Rotation:** 39 accepted rows carry a non-zero Open Images `Rotation` (90/180/270)
   and 493 have none. The pixels are used as stored and the value is recorded, as the
   code intends.
5. **Author names:** 68 of 4,000 contain HTML entities (e.g. `O&#x27;Brien`), as
   distributed: 657 of the 41,620 metadata rows. Unchanged; to be unescaped only for
   display if the attribution is typeset.
6. **EXIF:** a camera make/model on only 4 of 4,000, as expected of rescaled copies.

**Verdict: the 4,000 reals meet every approved rule and are fit for step 4a.**

**Still to do for this source (approved in the plan, `P3_PLAN.md` §9):**
- the data card and the `docs/CITATIONS.md` entry. They must be written from the source
  pages fetched in a session, not from memory (§9a).

### P3-C — decisions of 2026-10-17: recipe, additions, sessions, degradation tuning RULE (fixed before any candidate trains); I-042 CLOSED; Open Images data card

**Accepted by the user:**
- Open Images: the six observations stay as logged; no rule changes.
- `N*` = 5000; `N*(-6)` = 4500.

**Open Images data card and citation** (written from pages fetched this session):
- `docs/data_cards/t_oi_open_images.md`.
- `docs/CITATIONS.md`, entry "Open Images V7".
- **Sources, fetched 2026-10-17:**
  - `factsfigures_v7.html` (sha256 `03f731ed…4fa4fb0`, identical to the P3-A fetch): the
    licence and the citation request, verbatim;
  - `download_v7.html` (sha256 `6d71544e…3954a`, identical): the meaning of `Rotation`,
    verbatim;
  - the CVDF `README.md` at commit `16c6bae89b22dddc7194618bbb0e24852a911d94` (sha256
    `6ed423bf…0f48`, from `api.github.com` + `raw.githubusercontent.com`): "rescaled to
    have at most 1024 pixels".
- **Citation:** Kuznetsova et al., IJCV 2020, as the page's BibTeX. The page asks for
  extra citations only for the segmentations, Localized Narratives and point-labels, none
  of which is used.

**I-042 — CLOSED: Kaggle bills a session's wall time, not each GPU.**
- **Meter (user, 2026-10-17):** "GPU 23h 3m available of 30h", i.e. **6h57m used** this
  quota week.
- **Weekly reset.** Kaggle's docs pages render client-side: a direct fetch of
  `kaggle.com/docs/efficient-gpu-usage` and of the staff post `kaggle.com/product-feedback/173129`
  returned no body text. The text comes from the search tool's quotation of that post,
  not from a page I read:
  - "It will reset weekly on Saturday morning (midnight UTC) the same way it always has";
  - an older staff note gives midnight CET Friday→Saturday.
  - **The day and hour are therefore NOT-VERIFIED from a page read this session.**
- **Why the conclusion does not depend on the reset reading:**
  - every reading puts all four LOLO sessions (2026-10-15 11:10 → 2026-10-16 11:23 UTC,
    log timestamps) in one quota week;
  - no other GPU session ran from 2026-10-10 on: Panel B, the Flickr probe and every
    Open Images session were CPU;
  - no GPU session has run since the last LOLO one, so a meter showing 6h57m is the LOLO
    week.
- **MEASURED, from the session outputs in `kaggle_raw/p3_lolo_session{1..4}_x`** (first
  smoke-log line → last training-log line):

  | session | logged span | of which queue wall (status) |
  |---|---|---|
  | 1 | 11:10:13 → 12:47:57 = 1.629 h | 1.575 h |
  | 2 | 18:33:43 → 20:04:47 = 1.518 h | 1.4833 h |
  | 3 | 07:08:12 → 08:39:47 = 1.526 h | 1.4917 h |
  | 4 | 09:50:52 → 11:23:02 = 1.536 h | 1.5 h |
  | total | **6.209 h** | |

- **Not in the logs:** the setup before the smoke (about 405 s, measured once in session
  2; × 4 ≈ 0.45 h) and the output save after the last cell. That gives **≈ 6.7 h**,
  against **6.95 h** on the meter. Per-GPU billing would read at least 2 × 6.2 = 12.4 h.
- **Budget in wall hours:**
  - each quota week gives **30 session-hours**, and a T4 × 2 session runs two GPUs, so
    about 60 device-hours;
  - this week: 23h03m left at the reading;
  - the reset day and hour are shown on the user's own GPU page.

**D1 — the P3 recipe: the reading of "same recipe" in pre-registration §3 (user,
2026-10-17).**
- Every non-LOLO model (BANK-k, BANK-1L, S2-B, S2-B+deg, S2-B+reals, S2-D) trains on the
  **6,000-step cosine schedule, stopped at `N*` = 5000**.
- This is exactly the model `N*` was selected on: LOLO fold f at step 5000 of a 6000-step
  schedule. It also keeps H1-c the same recipe on both sides.
- **Implementation:** a trainer flag (`--stop-at`), with a test that a stopped run's
  learning rate matches the 6,000-step schedule at every step.

**D2 — approved additions (§9b):**
- (i) A member-level run-config key, validated against the registry like
  `excluded_lineages`. A test checks that the P2 and LOLO configs load unchanged.
- (ii) `scripts/p3_score_heldout.py`: per-image CSV (image id, member, label, score),
  plus the sha256 of the snapshot scored.
- (iii) Every non-LOLO run writes per-image held-out scores at its final step, so no
  rescoring is needed later.

**D3 — approved.** 10 runs, 2 per session.
- **Session 1:** `BANK-1` + `S2-B` seed 0, plus the LOLO rescoring at `N*(-f)` (validation
  only), so the H1-a CI and H1-c can be computed straight after it.

**Degradation tuning — the RULE, committed BEFORE either candidate is trained** (user,
2026-10-17; no Panel B score exists).
- **Model:** `S2-B+deg`, seed 0, no new reals, P3 recipe. The selected candidate's run
  **is** the pre-registered `S2-B+deg` ablation; there is no extra run.
- **Tuned:** only the chain probability p. The operations and ranges stay as
  `P3_PLAN.md` §4b fixes them.
- **Candidates:**
  - §4b names a default as "none | p = 0.3", i.e. no chain with probability 0.3. **So the
    plan's default chain probability is p = 1 − 0.3 = 0.7, and the two candidates are
    p = 0.7 and p = 0.35.**
  - This is my reading of the plan's table, flagged to the user before anything trains.
  - **Not fixed by the plan:** how one of the four operations is chosen once a chain is
    applied. That share is proposed with the degradation module and fixed before
    either candidate trains.
- **Objective (development data only, the P2 panel):** the macro achieved FPR over the
  **E4 and E5 dev real pools** (749 and 2,000 reals, `reports/p2_panel_sets/`), each pool
  counted once.
  - **Threshold:** k = 0, fixed at 5% FPR on the **491 COCO validation reals only** (P2's
    reals-only rule, I-062).
  - **The COCO-only threshold is used for this tuning step** (user instruction); the
    union threshold of pre-registration §4 remains the deployed one.
- **Guard:** a candidate's dev macro AUC over the 4 unseen slices (E3 GPT-Image2 and E3
  Nano-Banana2, each against E3's 500 dev reals; E4; E5) may not fall more than **1
  point** (0.01) below `S2-B` seed 0's dev macro AUC on the same slices.
- **Choice:**
  - exactly one candidate passes the guard → that candidate;
  - both or neither pass → the lower macro FPR, **except** that if the two FPRs are
    within 1 point (|Δ| ≤ 0.01) the milder p (0.35) is taken.
- **Schedule:** after `S2-B` seed 0, which supplies the guard's baseline.

**Session priority (critical path to S2-D first):**
1. `BANK-1` + `S2-B` s0 + the LOLO rescoring;
2. the 4a reconstruction;
3. the two tuning candidates;
4. the remaining bank-size runs, `BANK-1L` and `S2-B` s1;
5. `S2-D` s0/s1 + `S2-B+reals`;
6. robustness.

**Panel B stays untouched until every model is trained.**

### P3-C — built: the P3 recipe, BANK-k, per-image held-out scores, the session-1 runbook, the step-4a runbook, H1-c — dry-runs PASSED at `28df98e` — 2026-10-17

**Code commit `28df98e`** (approved D1, D2 (i)–(iii), D3; the 4a runbook; H1-c).
- **`p2_train_s2.py`** (git-bytes sha256 `d4d7269b…1a32`; LOLO used `e60ad4b5…e228`):
  - `--stop-at N` with `--fixed-steps 6000`: the LR follows the 6,000-step schedule
    (`lr_factor`, the LOLO formula, unchanged) and training stops at `N*`;
  - every curve point records the LR it trained with;
  - in-protocol validation every 500 steps;
  - the held-out members (every distinct member the run does not train on) are scored
    once, at the final step, written per image to
    `reports/p3_heldout_scores/<run>_step_NNNNN.csv` with a sidecar (snapshot sha256);
  - a resume with a different `--stop-at` is refused;
  - in smoke mode only 4 held-out reals are scored.
- **`src/dbp/eval/heldout_scores.py`:** the one held-out scoring path, shared by the
  trainer and the scorer, plus the D2 file format (`image_id, member, label, score` +
  sidecar).
  - `stable_seed` and `heldout_rows` moved here unchanged.
- **`run_config.py`:** `excluded_members` (D2 (i)).
  - Validated against the registry: an alias is refused (name its distinct twin), a
    member already excluded by its lineage is refused, and so is an empty bank.
  - P2 and LOLO configs load unchanged (tested).
- **`configs/p3_runs.yaml`:** `bank_1` (sd15 alone), `bank_{2,4,6}_{near,far}`,
  `bank_1l`, `s2b`.
  - Decoder counts tested: near 1/5/8/10/12, far 1/4/6/9/12.
- **`scripts/p3_score_heldout.py`** (D2 (ii)): re-scores a snapshot; the LOLO mode reads
  `N*(-f)` from `reports/p3_nstar.json` and records each rescored AUC against the report.
- **`scripts/p3_select_nstar.py ci`:** reads the D2 per-fold files and refuses a smoke
  (`--limit`) file or a tampered CSV.
- **`scripts/p3_train_next.py`:** the run chooser.
  - It takes the first two unfinished runs of the committed queue, carries complete runs
    and their score files forward, and says whether the LOLO rescoring is due.
- **Runbook `env/kaggle_run_p3_train.py` + `env/p3_train_expect.json`:**
  - the queue `bank_1:0, s2b:0, bank_2_near:0, bank_2_far:0, bank_4_near:0, bank_4_far:0,
    bank_6_near:0, bank_6_far:0, bank_1l:0, s2b:1`;
  - `--fixed-steps 6000 --stop-at 5000`;
  - the LOLO rescoring on GPU 0 after training, in session 1 only.
- **Step 4a:**
  - **`p1_build_pairs.py`:** `--name` (its manifests become `p3_oi_pairs_*`, never
    `p1_pairs_*`), `--cache-dir` (weights on `/tmp`), `--limit`, and bytes written per
    member. P1's names and defaults are unchanged.
  - **`p2_run_shards.py`:** a generic `--worker-arg` mode, so the reconstruction jobs fail
    fast; the probe command is unchanged (tested).
  - **`scripts/p3_oi_pairs_check.py`:**
    - `inputs`: every committed manifest row's PNG and sha256;
    - `gate`: the output projected from the MEASURED sample, against free space minus a
      margin;
    - `final`: every (real, member) pair once, its sha256, crop 384, and each member at
      P1's committed precision.
  - **Runbook `env/kaggle_run_p3_reconstruct.py` + `env/p3_reconstruct_expect.json`.**
- **`scripts/p3_h1c.py`:** H1-c exactly as pre-registration §6.1/§4.
  - The paired bootstrap: 2,000 resamples, seed 2026; reals drawn once and shared, each
    fold's reconstructions within the fold, the same draw for both models.
- **Tests:** `test_p3_recipe.py` 19, `test_p3_reconstruct.py` 10, `test_p3_h1c.py` 4,
  `test_p3_nstar.py` updated.
  - Full suite **993 passed, 7 skipped**; ruff clean.

**Local dry-runs, both PASSED at HEAD `28df98e`:**
- **`p3_train`, 3 sessions, 1,059 s:**
  - the fixture trains 8 tiny LOLO folds with the committed trainer;
  - session 1 is killed after the step-4 checkpoints (the rescoring still runs);
  - session 2 resumes to step 5;
  - session 3 carries everything and trains the next two runs.
  - **Checked:**
    - every run OK, schedule 6 / stop 5;
    - the LR at every step equals the 6-step schedule;
    - held-out scored at the last step only (none for S2-B);
    - the sidecar snapshot sha256 matches;
    - BANK-1 trains on sd15 alone;
    - **every rescored LOLO fold reproduces its training AUC exactly (diff 0.0)**;
    - **BANK-1 re-scored from its snapshot equals the trainer's per-image file (max
      |diff| 0.00e+00)**;
    - `p3_select_nstar.py ci` runs on the rescored files.
- **`p3_reconstruct`, 306 s:** 6 synthetic reals through `sd15` and `realesrgan_x4` on CPU
  with their pinned weights.
  - Checked: inputs; the gate; the merge; the final check; fp32 per the fixture bench;
    no `p1_pairs_*` file written.

**Defects the dry-runs caught before hand-over:**
1. **The rescored snapshots did not reproduce** (fixture diffs up to 1.0).
   - **Cause:** the scorer built the backbone without the trainer's seed. The dry-run's
     random-init backbone then differed; the pretrained DINOv3 would not.
   - **Fixed:** `seed_everything(seed)` before `load_backbone`, as the trainer does.
   - **Guarded by:** the per-image reproduction check in the fixture.
2. **The two 4a jobs overwrote each other's summary** (both ran as shard 0), and the size
   gate refused, correctly ("realesrgan_x4: no sample").
   - **Fixed:** member-list jobs name their summary by their members. Regression test
     added.

**Recorded:**
- **Line endings.** `reports/p3_nstar.json` was written with CRLF in the working tree:
  sha256 `b24b160e…4174`, as logged at selection. Git's bytes, which every code zip
  carries, are LF: `de30996d05bc0e4fb0408808cc9d4b97b27132d125298d09c991623a6c9ab678`.
  - The content is identical; the expect file checks the git-bytes hash.
  - The new JSON writers write LF explicitly.
- **Staging slip.** `git add thesis_dbp` swept the user's untracked PDF and `.tex` into the
  index again.
  - Unstaged with `git rm --cached` (index only, files untouched) before any commit.
  - Never committed.

**Code zip:** `F:\thesis-2\kaggle_raw\dbp-code-28df98e.zip`, 12,174,412 bytes, sha256
`d858717b5ee8038afbbf76b208e4b3d8ad8696a1a8fea278e4b5fd762eeff9c1` (git bytes; `testzip`
clean; 452 members).

**Step-4a estimates:**
- **Decoder time (MEASURED basis):** P1's per-image `ms_per_img` for the 2,500 T-real
  reals (`dbp-p2-s2-inputs/p1_pairs_manifest.csv`) is 0.762 GPU-h per 1,000 reals over
  the 12 members.
  - ⇒ 3.05 GPU-h for 4,000; 1.54 h + 1.51 h for the two balanced jobs.
  - The plan's 4.7 came from the bench median.
- **Output (ESTIMATED):** about 10.5 GB = 48,000 × 218,133 B, the measured mean of the
  Open Images crops, on the assumption that a reconstruction's PNG is about its real's
  size.
  - **No reconstruction PNG size has ever been measured** (P1's "13 GB" is a remark), so
    the gate measures it on Kaggle before the long job.
  - The `/kaggle/working` filesystem: 19.52 GB total (MEASURED, Open Images session).
- **The merged 4a manifest:** about 48,000 rows, more than 10 MB, so it cannot be
  committed. Its digest will be, as for P1.

### P3-C — integrity incident closed; degradation candidates and operation shares FIXED before training — 2026-10-17

**Integrity (user approval 2026-10-17).**
- **Listed first:** `F:\thesis-2\Implementation\.claude\` held exactly one file,
  `scheduled_tasks.lock` (124 bytes, 17:48), and no subfolder.
- **Removed:** the file was deleted; the folder was then empty and was removed (`rmdir`,
  which refuses a non-empty folder).
- **Re-checked:** `p0_verify_integrity.py`, run on its own: **0 changed, CLEAN**.
- **Restored:** `reports/p0_integrity.json`, a generated report this session rewrote.
  `git restore` under §2a; the diff was the timestamp only, and the committed version says
  CLEAN.
- **Cause:** the agent harness wrote the lock file when this session scheduled fallback
  wake-ups. The log commit `d31e2f4` was chained after the check, so it landed before the
  verdict was read. It holds only the log entry and nothing outside scope.
- **New rule, CLAUDE.md §2c:**
  - read the integrity verdict before every commit, in a command of its own;
  - never chain a commit after the check;
  - no scheduled wake-ups or other tooling that writes into the repository tree.

**Degradation tuning (completes the rule committed in `6b8746a`; user, 2026-10-17;
committed before either candidate trains):**
- **p:** "none | p = 0.3" means a chain is applied with **p = 0.7**, the plan default.
  **Candidates: p = 0.7 and p = 0.35. Confirmed.**
- **Operation shares, fixed and not tuned:**
  - when a chain applies, one of the 4 operations (JPEG, WebP, resize-and-restore,
    social-media style) is picked **with equal probability, 1/4 each**;
  - **each parameter is uniform over its stated range** (JPEG/WebP quality 50–95; resize
    scale 0.5–1.0; social-media resize 0.5–1.0, then JPEG 70–85);
  - **JPEG chroma 4:2:0 vs 4:4:4 at 50/50**;
  - one draw per pair, applied identically to both members.
- **Recorded** in `P3_PLAN.md` §4b as a dated note.

**I-042:** closed as written. The reset time is NOT-VERIFIED, accepted by the user.

**On Kaggle (user):** uploading `dbp-code-28df98e`; running session 1 (train) and session
2 (4a reconstruct).

### P3-B (step 4a) — Kaggle Cell 1 found no reals: the dataset's slug is not its title; lookup by layout — 2026-10-18

**Kaggle (user):** the reconstruct notebook stopped in Cell 1 with `expected exactly one
input matching 'datasets/*/dbp-p3-open-images/oi/reals', found []`.
- **Cause:** the Open Images output was published as a dataset first titled
  `dbp-p3-lolo_data`, then retitled `dbp-p3-open-images`. A Kaggle dataset's **slug**,
  which is its mount path, is fixed at creation; retitling does not change it. So the
  name-based lookup could not match.
- **A side point:** an earlier attempt edited the cell's text to the old title. Runbook
  cells are not edited; the fix goes into the code.

**Fix (`9a70325`):**
- The 4a runbook finds the reals **by layout**: exactly one attached dataset holding
  `oi/reals` (`find("datasets/*/*/oi/reals")`). That cuts the dependency on a name the
  user cannot change.
- Correctness does not rest on the name: Cell 3 checks all 4,000 PNGs' sha256 against the
  committed manifest.
- **Regression test:** `test_the_4a_runbook_finds_the_reals_by_layout_not_by_name`.
- **The dry-run fixture** now mounts the reals under another slug (`dbp-p3-lolo-data`).
- The training runbook is unchanged, so session 1 continues on `dbp-code-28df98e`.

**Local dry-run: PASSED** at HEAD `9a70325` (234 s). The reals were found under
`…/dbp-p3-lolo-data/oi/reals`; 12 pairs were built, merged and verified.

**Code zip:** `F:\thesis-2\kaggle_raw\dbp-code-9a70325.zip`, 12,178,532 bytes, sha256
`1280601ceb28ac9b9c0c3e5b34be7d1de8c29aa98da91be6cb89b5cb29aa75a7` (git bytes; `testzip`
clean; 452 members).

### P3-B (step 4a) — COMPLETE: 48,000 Open Images pairs (4,000 × 12), verified — 2026-10-18

**Run:** Kaggle GPU T4 × 2, `dbp-code-9a70325`.
- **Inputs, as the session's Cell 1 printed them:**
  - code at `/kaggle/input/datasets/muhammadadeelshah/dbp-code-9a70325/thesis_dbp`;
  - reals at `/kaggle/input/datasets/muhammadadeelshah/dbp-p3-open-images/oi/reals`;
  - nothing to resume.
- **The four reports** came as `.txt` (`F:\thesis-2\p3_oi_*.txt`). Each parses as the
  JSON the script writes. They are committed as `.json` under
  `reports/kaggle_20261018/p3_oi_pairs/`.
  - Source sha256: inputs `97cc97556600…`, gate
    `5ba526e4f281…`, status `95064965e1b5…`, final
    `f4270af36773…`.

**Correction to the 2026-10-18 entry above.** The mount path is
`…/dbp-p3-open-images/…`, so this dataset's slug **is** `dbp-p3-open-images`. The cause
logged there ("the slug is fixed at creation and differs from the title") is **not
confirmed** for this dataset; the old name-based lookup would have matched this path. The
first failure's true cause is unknown; possibly the dataset was renamed or attached while
that session ran. The layout lookup stays: it is correct either way.

**MEASURED:**

| check | result |
|---|---|
| inputs (`p3_oi_pairs_inputs`) | **OK**: 4,000 / 4,000 rows; 0 missing; **0 sha256 mismatches** against the committed manifest; split 3,600 train / 400 held-out |
| output-size gate (`p3_oi_size_gate`) | **OK**: 12 members × 32 sample reals. Projected 10,954,216,125 B (10.2 GiB) against a limit of 17.46 GiB (19.46 free − 2.0 margin) |
| run (`p3_oi_pairs_status`) | **COMPLETE**. Job 0 (movq_k22, qwen_image, sd_ft_ema, flux, sd35, realesrgan_x4) and job 1 (sdxl, dcae, sdxl_fp16fix, sd15, flux2_klein, sd21) both exited 0. **Wall 2.0083 h, device-hours 4.0167** |
| final (`p3_oi_pairs_final`) | **OK, no problems**: **48,000 / 48,000 rows**, 4,000 per member; every PNG present with its manifest sha256; crop 384; precision = P1's committed bench (fp32 for sdxl, realesrgan_x4, movq_k22; fp16 for the other nine) |
| output bytes | **10,737,317,606 B** (10.0 GiB). Per member: realesrgan_x4 800,924,333 (the smallest) … dcae 935,838,568 (the largest) |

**Against the estimates:**
- **Bytes:** the gate's projection was 2.0% high. The pre-run estimate (10.47e9 B, from
  the real crops' mean) was 2.5% low.
- **The first measured reconstruction PNG mean:** **223,694 B**, against the real crops'
  218,133 B (+2.5%).
- **Wall time:** 2.01 h, inside the 2.0–2.6 h estimate. It was billed as one session
  (I-042).

**Not yet in the repo:**
- **The merged manifest** `reports/p3_oi_pairs_manifest.csv` (48,000 rows, more than
  10 MB). Its sha256 digest and per-member counts are to be committed, as for P1, once the
  file is downloaded to `kaggle_raw`.
- **The output** is to be published as a private dataset (e.g. `dbp-p3-oi-pairs`); S2-D and
  `S2-B+reals` train from it.

### P3-B (step 4a) — the merged pairs manifest reviewed; digest committed — 2026-10-18

**File:** `F:\thesis-2\p3_oi_pairs_manifest.txt`, the Kaggle output
`reports/p3_oi_pairs_manifest.csv`.
- **Size:** 10,906,886 bytes, CRLF; sha256 as downloaded
  `b1681633b090a5195f7c253822f5af63fa9a5b414e2c7c2aa53a810f184e171e`.
- **Committed instead of the file:** it is over 10 MB, so only its digest is committed:
  `reports/kaggle_20261018/p3_oi_pairs/p3_oi_pairs_manifest_digest.json`.
  - It holds the as-downloaded and LF sha256, and a line-ending-independent **content
    sha256** `0a4e9a3cb428ecf17d161c9c51905c90bf816e266d69c42f3e10507399cd1dc4` over the sorted `pair_id<TAB>sha256` lines.
  - It also holds the rows per member and the precision and revision per member.
- **The file** stays in `kaggle_raw`, alongside the output dataset.

**Review (local; the committed Open Images manifest and P1's
`dbp-p2-s2-inputs/p1_pairs_manifest.csv` read as data). No problems:**
- **Structure:** 48,000 rows, with the same 15 columns as P1's pairs manifest; 12 members
  × 4,000.
- **Images:** each member's image ids are **exactly the committed 4,000** Open Images ids.
- **Uniqueness:** no duplicate (image, member) and no duplicate file sha256.
- **Format:** every `pair_id` is `<image_id>__<member>`; every path is
  `oi_pairs/<member>/<image_id>.png`; `set_id` t_oi; crop 384.
- **Against P1:** lineage, kind, pinned revision and precision are **identical to P1's
  T-real pairs for every member**.
- **Decoder time:** the sum of `ms_per_img` is 2.881 GPU-h (estimated 3.05). Per-member
  medians are within 0–20% of P1's.
- **PSNR:** every value finite; range 12.13–50.25 dB. The lowest: dcae on
  `5bac9b963a87d225` (12.13), the same image also lowest-ranked for sd15.

**Recorded property (descriptive; cause NOT established).** Reconstruction is markedly
**easier on the Open Images reals than on P1's COCO reals, for every member**.
- **Median PSNR is 2.7–6.8 dB higher** (2.71 for realesrgan_x4 to 6.78 for qwen_image), e.g. sd15 28.40 vs 23.56, flux 36.09 vs 29.38,
  realesrgan_x4 25.63 vs 22.92.
- **The median mean-absolute residual is roughly halved.**
- **The whole distribution shifts:** every decile moves up, and 77–89% of Open Images
  pairs lie above P1's median, by member.
- **What it means for S2-D:** its new pairs differ less between real and reconstruction
  than the corpus pairs do. Within a pair both members share the source, so this is not a
  label cue; it is a property of the training mix to keep in mind when S2-D is
  interpreted.
- **A candidate explanation, untested:** the CVDF rescaling and re-encoding of the source
  images.

### P3-C — training session 1 LOST on Kaggle (no log, no output); rescoring watchdog added — 2026-10-18

**What the user saw:**
- `dbp-p3-train` Version #1 (`dbp-code-28df98e`, T4 × 2) showed "Running for 64514.0 s"
  (about 18 h).
- Then "Output Data — An error occurred: Permission 'kernelSessions.get' was denied".
- No log and no output are available. The session was stopped by the user.
- It was started **at the same time as the step-4a session** (two T4 × 2 sessions in
  parallel).

**Quota readings (user, MEASURED on the meter):**

| when | reading | implied use |
|---|---|---|
| 2026-10-17 | 23h03m of 30 h available | 6h57m = the 4 LOLO sessions (I-042) |
| 2026-10-18, while "running 18 h" | about 18 h available | about 12 h used = 6h57m (LOLO) + 2.01 h (4a, MEASURED) + **about 3 h for this session** |
| 2026-10-18, after it was stopped | 29h46m available | **the weekly quota had reset** between the two readings; 14 min used since |

**What this implies (inference, NOT confirmed without a log):**
- The session was charged about 3 h, not 18. That is roughly a full run (1.7–1.9 h)
  plus the rescoring.
- So it most likely ran, and Kaggle then failed to save or show it.
- **The trained models and rescoring files are lost; session 1 is redone.**
- **Reset time:** a reset fell between the two readings of 2026-10-18. With the 2026-10-17
  reading still showing the LOLO week, this does not fit a reset at Saturday 00:00 UTC.
  I-042's reset time stays **NOT-VERIFIED**.

**The gap on our side, closed (`8171516`):**
- Every training step was time-bounded except the LOLO rescoring (Cell 7) and its smoke.
- **`p3_score_heldout.py --deadline-hours`:** a daemon watchdog ends the process with exit 2
  once the wall time passes, even if a load or a forward pass hangs.
  - No summary is written then, so the rescoring stays due for the next session.
- **The runbook passes it on both calls:** smoke 0.5 h, rescoring 1.0 h (expected 0.1–0.15
  h).
- **The header now says to run the session alone,** with no other GPU session at the same
  time.
- **Regression tests:**
  - `test_the_rescoring_watchdog_ends_a_hung_process` (a 60 s "hang" is killed after
    about 1 s with exit 2);
  - `test_every_rescoring_call_in_the_runbook_has_a_deadline`.

**Local dry-run: PASSED** at HEAD `8171516` (1,133 s, 3 sessions; the same checks as at
`28df98e`, including exact LOLO and per-image reproduction).

**Code zip:** `F:\thesis-2\kaggle_raw\dbp-code-8171516.zip`, 12,185,864 bytes, sha256
`89f27f8a24cadf4695cbea618f905a1aa2f15c0a73bcbebcf223cffa66bf35c1`. It is for the training session; the 4a runbook is unchanged and done.

### P3-C — training session 1 COMPLETE; LOLO rescored exactly; H1-a PASS; H1-c PASS — 2026-10-19

**Run:** Kaggle GPU T4 × 2, `dbp-code-8171516`, run alone. The editor session was stopped
before the commit started.
- **Inputs (Cell 1):** `dbp-code-8171516`, `dbp-p2-s2-inputs`, `report` (`p1_reals`,
  `p1_generated`), `thesis-dbp-1-dataset`, and the LOLO output
  `datasets/muhammadadeelshah/dbp-p3-lolo-output`. Nothing to resume.
- **Archive:** `F:\thesis-2\p3_train_reports.zip`, 11,254,781 bytes, sha256
  `cf32556a4cd815e3ea16450da429803c21aadc40aaf45439d2de0d00d7e899e6`; `testzip` clean.
  - It is the session's whole `reports/` tree. Extracted to
    `F:\thesis-2\kaggle_raw\p3_train_session1_x\`.
- **Committed:** this session's files under `reports/kaggle_20261019/p3_train_session1/`
  (plan, status, smoke status, pre-flight, the two training reports, their logs). The 17
  per-image files are under `reports/p3_heldout_scores/`.
- **`thesis_dbp/.gitattributes`** gains `reports/p3_heldout_scores/** -text`. Each sidecar
  hashes its CSV's exact bytes, and a CRLF checkout on Windows would otherwise break that
  check.

**MEASURED:**
- **Pre-flight PASSED:** 6 checks; 2 × Tesla T4; `HF_TOKEN present: true`; 5 input hashes;
  2,488 `gen_matched` files match.
- **Smoke:** COMPLETE, 0.0333 h wall (each run on each GPU).
- **Queue:** COMPLETE, **wall 1.1583 h**, device-hours 2.3083.

| run | members trained | steps | schedule / stop | GPU-h | train-loop s/step | in-protocol val AUC at 5000 | log-loss | TPR@1%FPR |
|---|---|---|---|---|---|---|---|---|
| `bank_1_seed0` | sd15 only | 5,000 | 6,000 / 5,000 | 1.151 | 0.75942 | 0.99998 | 0.00671 | 1.0 |
| `s2b_seed0` | all 12 | 5,000 | 6,000 / 5,000 | 1.140 | 0.78002 | 0.99818 | 0.05359 | 0.9707 |

- **The LR recorded at step 5000** is 6.712e-06 in both runs. That is
  1e-4 × `lr_factor(4999, 6000)`, the 6,000-step schedule (D1).
- **Held-out scoring:** BANK-1 scored its 11 held-out members once, at step 5000. 5,892
  rows (491 reals, 5,401 reconstructions); snapshot sha256 `e4e8210a…b93a`; pooled
  held-out AUC 0.92711.
  - **flux2_klein 0.57907** and flux 0.85481 are the lowest. S2-B has no held-out member.
- **Warnings:** the logs carry peft's "401 … Unable to fetch remote file" for the gated
  DINOv3 config at save time. It is the same warning as in every LOLO log (24 in fold 7's),
  and harmless.

**LOLO rescoring at `N*(-f)`:** 8 folds; **every rescored AUC equals the training
report's at 5 decimals** (diff 0.0 in `p3_lolo_rescore_summary.json`). The unrounded
differences are within 3.8e-6.

**H1-a — PASS** (`python thesis_dbp/scripts/p3_select_nstar.py ci`, run once; output
`reports/p3_h1a_ci.json`).
- **`G` = 0.966228; 97.5% CI [0.96123, 0.97095]**.
  - Basis: 2,000 resamples, seed 2026. The reals are drawn once per resample and shared;
    each fold's reconstructions are drawn within the fold; the folds and `N*(-f)` are held
    fixed.
- **Criteria, as frozen:**
  - `G ≥ 0.90`: met;
  - CI lower bound ≥ 0.85: met;
  - no fold below 0.75: met. Fold 6 (FLUX.2) is the lowest at 0.76551.

**H1-c — PASS** (`python thesis_dbp/scripts/p3_h1c.py`, run once; output
`reports/p3_h1c.json`).
- **Mean (LOLO-f − BANK-1) over folds 2–8 = +5.33 AUC
  points; 95% paired CI [+4.66, +6.00]**.
- **Criteria:**
  - mean ≥ +2 points: met;
  - CI excludes 0: met.

| fold | held-out members | LOLO-f at `N*(-f)` | BANK-1 at `N*` | difference | reconstructions |
|---|---|---|---|---|---|
| 2 | sdxl, sdxl_fp16fix | 0.99827 | 0.95515 | +0.04313 | 982 |
| 3 | flux, sd35 | 0.98798 | 0.90451 | +0.08347 | 982 |
| 4 | dcae | 0.99340 | 0.99347 | -0.00007 | 491 |
| 5 | realesrgan_x4 | 0.99999 | 0.99993 | +0.00006 | 491 |
| 6 | flux2_klein | 0.76551 | 0.57907 | +0.18644 | 491 |
| 7 | qwen_image | 0.98586 | 0.92615 | +0.05971 | 491 |
| 8 | movq_k22 | 0.99938 | 0.99881 | +0.00057 | 491 |

- **Every fold is reported, as frozen.**
  - The gain comes from SDXL (+4.3), the 16-channel KL lineage (+8.3), FLUX.2 (+18.6) and
    Qwen (+6.0).
  - On DC-AE, GAN-SR and MoVQ, BANK-1 is already at 0.993–0.9999 and the difference is
    within ±0.0006.

**Not yet:**
- **H1-b** waits for the single Panel B scoring at the end (§8).
- **Panel B is untouched.**

**Next in the queue:** the two degradation candidates. Build the degradation module
first; then `bank_2_near` + `bank_2_far` …; then S2-D.

### P3-C — H1-a and H1-c FINAL; the degradation module and the tuning runbook built, dry-run PASSED — 2026-10-19

**H1-a and H1-c are FINAL** (user, 2026-10-19):
- **H1-a PASS:** `G` = 0.966228, 97.5% CI [0.96123, 0.97095], no fold below 0.75.
- **H1-c PASS:** +5.33 AUC points, 95% paired CI [+4.66, +6.00].
- Both as computed once from the committed per-image files (`reports/p3_h1a_ci.json`,
  `reports/p3_h1c.json`). Neither is recomputed.
- **H1-b** waits for the single Panel B scoring at the end.

**Next `dbp-p3-train` sessions (user runs them now):**
- the queue as committed in `dbp-code-8171516`: `bank_2_near` + `bank_2_far`, then
  `bank_4_*`, `bank_6_*`, then `bank_1l` + `s2b` seed 1;
- inputs: those of session 1 plus the previous version's output of `dbp-p3-train`,
  attached as a **notebook** output (the runbook looks under `notebooks/`);
- **code stays `dbp-code-8171516`** for all four, so every bank-size run shares the
  trainer of BANK-1 and S2-B.

**Built (code `a7f4256`), as fixed on 2026-10-17:**
- **`src/dbp/train/degrade.py`:**
  - **operations:** with probability p a chain applies, one of 4 operations at 1/4 each;
  - **parameters, uniform:** JPEG and WebP quality 50–95; resize-and-restore scale
    0.5–1.0, bicubic; social = resize-and-restore 0.5–1.0, then JPEG 70–85;
  - **JPEG chroma 4:2:0 / 4:4:4 at 50/50.**
    - **A reading, flagged:** the chroma rule is applied to every JPEG encode, the
      social-media step's included. The rule names no operation.
  - **Synchronised by construction:** `sample_chain` draws one chain per pair, the pure
    `apply_chain` applies it, and `degrade_pair` hands the same chain to both members.
  - **Stream:** each draw consumes the same seven numbers whichever branch it takes.
  - **Codecs:** WebP support is checked at start-up (`check_codecs`).
- **The trainer (`p2_train_s2.py`):**
  - **What is degraded:** each pair (real + reconstruction) and each `gen_matched` row
    (real + generation) gets one shared chain on the full 384 px images, before the crop
    and the existing augmentation.
  - **Its own random stream,** `default_rng([seed, 417])`: a degraded run sees exactly
    the batches of the same seed without degradation. The stream's state is in the
    checkpoint.
  - **Every report** records the chain counts, the measured ms per pair and the
    batch-build ms: the data-loader cost of §4b, measured first by the smoke.
  - **With `degrade_p` = 0** the batches are unchanged; the bank-size runs keep
    `8171516` regardless.
- **The run config:** key `degrade_p` (a number in [0, 1], default 0; every P2, LOLO and
  bank config is unchanged). Runs `s2b_deg_p70` (0.7) and `s2b_deg_p35` (0.35) are S2-B
  plus degradation and nothing else (tested).
- **`scripts/p3_tune_degradation.py`:** the committed rule, unchanged.
  - The k = 0 threshold is `calibrate_on_reals` (I-062) on the 491 COCO validation reals.
  - The dev FPR pools are E4 and E5 only.
  - The four unseen slices are the first four of `p2_analyze.P_A_UNSEEN_SLICES`, sliced by
    `p2_analyze.slice_key`. Nothing is redefined.
  - The guard is inclusive at 1 point; the tie is |delta| ≤ 0.01, going to the milder p.
  - **Development data only.**
- **`p2_s2_infer.py`** gains `--runs-config` (P3 runs) and `--deadline-hours`.
- **`src/dbp/utils/watchdog.py`:** the watchdog, shared by both scorers.
- **Runbook `env/kaggle_run_p3_tune.py` + `env/p3_tune_expect.json`:**
  1. S2-B seed 0 from the `dbp-p3-train` output (not this notebook's own earlier copy),
     scored on the dev panel and the val reals unless a full scoring summary (no
     `--limit`) already exists;
  2. a smoke of each candidate on each GPU, measuring the degradation cost;
  3. both candidates trained and scored, one per T4;
  4. the rule applied.
  - The dev panel root is found by layout (`…/e5_chameleon_pipeline`).
  - Every scoring step is time-bounded.
- **Tests:** `test_p3_degrade.py`, 21 tests. They include **the requested
  synchronisation test**: over 200 pairs, the real and its reconstruction receive
  byte-identical operations and parameters (the chain JSON, compared byte for byte).

**Correction to the committed rule's counts (the rule itself is unchanged).** The dev
panel the session scores is the frozen scoreable list `p2_panel_images.csv` (11,775 rows,
after I-047's crop filter).
- **Its real pools:** **E4 743, E5 1,864** (and E3 477).
- The rule entry of 2026-10-17 quoted 749 and 2,000 from the pre-filter view files in
  `reports/p2_panel_sets/`.
- The E3 slices join every one of the 1,477 E3 rows: 500 GPT-Image2, 500 Nano-Banana2, 477
  reals.

**Local dry-run: PASSED** at HEAD `a7f4256` (509 s, 2 sessions).
- **Session 1:** S2-B seed 0 was scored; both candidates were killed after step 4.
- **Session 2:** both resumed to step 5 (the degradation counts carried across the
  resume), were scored, and the rule was applied.
- **Smoke:** it recorded the cost per pair.

**Full suite:** 1,024 passed, 7 skipped; ruff clean.

**Code zip:** `F:\thesis-2\kaggle_raw\dbp-code-a7f4256.zip`, 12,386,935 bytes, sha256
`ff3c2fba4c92365083a209c98028cfc05e83c0ef8e2d6193ea57ea089ee979c4`.

**Estimate (the expect file):**
- about 1.6–2.0 h wall: training 5,000 × 0.759–0.780 s/step (MEASURED in session 1),
  plus the degradation cost (NOT-MEASURED on Kaggle until the smoke; planned +25%), plus
  three dev scorings of about 0.14 h.

### P3-C — degradation decisions of 2026-10-19 (before any candidate trains); TrueFake metadata; dev-panel ownership — 2026-10-19

**Decisions (user, 2026-10-19). No candidate has trained, and no Panel B score exists.**
1. **Chroma, resolving the first build's ambiguity:**
   - the plain JPEG operation stays at 4:2:0 vs 4:4:4, 50/50;
   - **the social-media operation's JPEG is 4:2:0 only**, because the platforms
     subsample chroma. The first build (`a7f4256`) had applied 50/50 there too.
2. **Re-share variant:** with probability **0.5** (fixed, not tuned) the social-media
   operation is applied **twice in a row**, the second round with independently drawn
   parameters: resize-and-restore 0.5–1.0, then JPEG 70–85 at 4:2:0, twice.
   - Everything else is unchanged: 4 operations at 1/4 each, candidates p = 0.7 / 0.35, the
     same rule and guard.

**§5 judgement: no pre-registration amendment.**
- §5 lists "Degradation chain strengths and probabilities (`P3_PLAN.md` §4b)" among the
  values **fixed before Panel B is scored** and **recorded in the log**; the frozen text
  leaves the chain's composition to the plan.
- These decisions are made before any candidate trained and with no Panel B score in
  existence, so §8 is not touched.
- The pre-registration text does not change. Recorded here and as a dated note under
  `P3_PLAN.md` §4b.

**Built (code `1bc18e2`):**
- **`degrade.py`:**
  - `SOCIAL_SUBSAMPLING = "4:2:0"`, `RESHARE_P = 0.5`;
  - a re-share chain carries `quality2`/`scale2`, and `apply_chain` applies both rounds
    from the one shared chain;
  - every draw now consumes ten numbers whatever the branch.
- **The trainer** counts re-shares separately (`social_reshare`).
- **Tests, `test_p3_degrade.py` (22):**
  - **the byte-identical synchronisation test now covers the re-share path.** Over 300
    pairs every path, re-share included, is exercised, and both members receive
    byte-identical chains;
  - the social JPEG is always 4:2:0;
  - a re-share happens half the time (±0.02);
  - the second round is independent of the first (|r| within 4 standard errors);
  - a re-share equals the social operation applied twice;
  - a half-specified re-share is refused.
- **Local dry-run: PASSED** at HEAD `1bc18e2` (708 s, 2 sessions). The re-share path
  ran inside training (`social_reshare` in both candidates' counts).
- **Full suite:** 1,025 passed, 7 skipped; ruff clean.
- **Code zip:** `F:\thesis-2\kaggle_raw\dbp-code-1bc18e2.zip`, 12,389,978 bytes,
  sha256 `3b2fc8b1f72e5222bfc9c7e331c2078d07c168ddbf1d49e7ddd6a4f001ab555c`. **It replaces `dbp-code-a7f4256` for the tuning session.**

**TrueFake: metadata only, nothing downloaded.**
- **Read on 2026-10-19 through the agent's web tool:** `arxiv.org/abs/2504.20658`,
  `arxiv.org/html/2504.20658`, `github.com/MMLab-unitn/TrueFake-IJCNN25`, and its
  `README.md` and `detector/README.md` on `raw.githubusercontent.com`. arXiv is not in
  CLAUDE.md §6's host list; it was read only because the user asked for metadata.

| item | what the sources say |
|---|---|
| paper | Dell'Anna, Montibeller, Boato et al., "TrueFake: A Real World Case Dataset of Last Generation Fake Images also Shared on Social Networks", IJCNN 2025, DOI 10.1109/IJCNN64981.2025.11228911; arXiv v1 29 Apr 2025 |
| licence | **CC BY 4.0** (the arXiv licence link and the repository's LICENSE). The component sets' own terms (FFHQ, FORLAB) were **NOT-VERIFIED** this session |
| hosting / size | Google Drive: one main link plus 13 partition links (FLUX.1, SD1.5, SD2, SD3, SDXL, StyleGAN 1/2/3, FFHQ, FORLAB; Facebook, Telegram, Twitter). **Download size NOT-STATED** in the paper or README. **600,000 images** (paper) |
| structure | `Dataset_root/<share>/{Fake/<generator>[/<sub>], Real/<source>}`, shares `pre` (PreSocial), `fb`, `tl`, `tw` (detector README). The README names FORLAB as a real source, the detector README keys `realRAISE`: an inconsistency in the sources |
| pre-share | real 100,000 (FFHQ 70,000; FORLAB 30,000: mobile and DSLR camera images); GAN 120,000 (StyleGAN 1/2/3, 40,000 each, FFHQ-trained); diffusion 200,000 (SD1.5, SD2.1, SD3 medium, SDXL base-1.0, FLUX.1 dev; 40,000 prompts) |
| **pre- and post-share versions** | a **60,000-image subset** (20,000 real, 40,000 fake: 15,000 GAN, 25,000 diffusion) was shared through Facebook, X and Telegram, giving 180,000 shared images. It is "sampled from" the pre-share images, so **these 60,000 have both versions**. **Whether files keep the same ids (a one-to-one mapping) is NOT stated**; only a file listing could settle it |
| what the networks do (paper) | Facebook: downscale to 720 px wide, QF 61–92 (adaptive); X: downscale to 1200 px wide, untouched at ≤ 768 × 768, QF 87; Telegram: height ≤ 800, QF 85 |
| resolutions | FFHQ and StyleGAN 1024²; SD1.5 512²; SD2.1 768²; SD3, SDXL, FLUX.1 1024²; FORLAB variable. Formats not stated |

**Overlap with our bank and data (my analysis):**
- **Generators:** every TrueFake diffusion generator decodes through a lineage **in our
  bank**:
  - SD1.5 and SD2.1 → `kl_4ch_sd1` (sd15, sd21);
  - SDXL → `kl_4ch_sdxl`;
  - SD3 medium → `kl_16ch` (our member is the SD3.5 VAE; whether SD3-medium's VAE is
    identical is **NOT-VERIFIED**);
  - FLUX.1 dev → `kl_16ch` (`flux`).
  - So for full-bank models these are in-bank, not unseen. **StyleGAN 1/2/3 have no
    decoder in the bank** and would be truly unseen.
- **Reals:**
  - **FFHQ is Flickr**, the platform of our training reals (COCO, Open Images). Under
    pre-registration §2's platform rule it could not serve as a test real pool beside
    them.
  - FORLAB is camera-original.
  - No image-level overlap was checked (a dedup would need the files).
- **Pre/post-share pairs:** the 60,000 shared images are the part most relevant to the
  re-share decision. Their post-share processing (downscale + JPEG 61–92) is close to our
  social-media operation.

**The dev panel: who owns it (item 4).**
- **The record:** CLAUDE.md §13 says **`adeelshah29` owns the `phase-2` notebook output**.
  Every P2 session's pre-flight read the panel from
  `/kaggle/input/notebooks/adeelshah29/phase-2` (16 references in
  `reports/kaggle_20261004`/`kaggle_20261005`).
- **`dbp-p2-panel`:** no session record reads from it, so **its owner is not recorded**
  and must be checked on Kaggle.
- **Proposal, if the panel exists only on `adeelshah29` (not built; awaits the user):**
  rebuild it on `muhammadadeelshah` in one CPU session, as for Panel B.
  - **Re-derive the E3, E4, E5, E7 and E8 views.** The Panel B session of 2026-10-11 did
    this on `muhammadadeelshah`, `identical_to_p1: true`, reproducing every P2-panel row
    of those views. T-ctrl comes from the `report` corpus.
  - **Re-materialise** with `p2_materialise_panel.py` from the committed
    `reports/p2_panel_sets/*.csv`: 12,150 rows, each with its sha256, every file
    re-hashed.
  - **Byte-identical check:** require the 12,150-row digest
    `2e1c192024ec52e42c4e1761afa6fb2e37c0f7a6983f5ff75d1174305550b769` (P2
    `p2_panel_digest.json`) and the 11,775-row digest
    `bff608c3d8e00ea834532cd541849b4f21de803c8f8e27a55ca5e70424c51116` (Amendment 1).
  - **Publish** as a private dataset on `muhammadadeelshah`.

### P3-C — the P2 dev panel rebuilt on `muhammadadeelshah` (runbook ready); TrueFake listing read — 2026-10-20

**Dev panel: where P2 read it, and what P3 uses (user decision 2026-10-19).**
- **What P2 used:** P2 read its development panel from the **`phase-2` notebook output of
  `adeelshah29`**. Every P2 session's pre-flight names
  `/kaggle/input/notebooks/adeelshah29/phase-2` (16 references in
  `reports/kaggle_20261004`/`kaggle_20261005`).
- **What the user found:** no `dbp-p2-panel` dataset exists on `muhammadadeelshah` (the
  user checked). That account holds only `dbp-p2-manifests` (P1 evaluation manifests),
  `dbp_p2_V4` (an old code copy) and `dbp-p2-s3-partial` (S3 probe feature caches). None
  holds panel images.
- **P3 uses only a copy rebuilt on `muhammadadeelshah`. The `adeelshah29` copy is never
  attached again.** Reasons: CLAUDE.md §13 (one account), and §6c (WildRF and Chameleon
  must not cross accounts).

**Built (code `6782139`): `scripts/p3_rebuild_dev_panel.py` +
`env/kaggle_run_p3_dev_panel.py` + `env/p3_dev_panel_expect.json`.** One CPU session.
- **The source steps are the Panel B session's:** `p3_fetch_eval_sets.py` +
  `p3_build_eval_views.py`. On 2026-10-11 these rebuilt E3, E4, E5, E7 and E8 on
  `muhammadadeelshah` with manifests byte-identical to P1's (`identical_to_p1`), absolute
  paths included.
  - So the P2 rows' recorded `view_path`s (E3 and E5 under
    `/kaggle/working/Implementation/thesis_dbp/outputs/p1/eval/…`, E4 under
    `/tmp/wildrf/…`) point at files rebuilt in the session.
- **T-ctrl:** the diagnostic view comes from the staged corpus (`report` reals +
  `thesis-dbp-1-dataset` pairs) in `/tmp`.
- **Copying:** every one of the **12,150** rows is copied with
  `p2_materialise_panel.materialise_view`, **unchanged**. Each file is re-hashed against
  the sha256 in the committed `reports/p2_panel_sets/*.csv`, and any mismatch stops the
  run.
- **Both digests required:**
  - **all 12,150 rows = `2e1c192024ec52e42c4e1761afa6fb2e37c0f7a6983f5ff75d1174305550b769`**
    (P2 `p2_panel_digest.json`, D-050);
  - **the 11,775 scoreable rows of `p2_panel_images.csv` =
    `bff608c3d8e00ea834532cd541849b4f21de803c8f8e27a55ca5e70424c51116`** (Amendment 1).
  - Every listed scoreable path must exist under the panel root.
  - **Verified locally first:** the committed sets reproduce both digests exactly
    (12,150 → `2e1c1920…`; the 11,775 ids of `dbp-p2-s2-inputs/p2_panel_images.csv` →
    `bff608c3…`).
- **Afterwards:** the derived view images are deleted from the output. The record
  `reports/p3_dev_panel_digest.json` holds the digests, counts by view and label, and the
  measured `panel_bytes`.
- **Publish** `dev_panel/` as the **PRIVATE** dataset `dbp-p2-panel` on
  `muhammadadeelshah`.
- **Tests:** `test_p3_dev_panel.py` (3): the committed sets give `2e1c1920…`; a
  byte-identical rebuild passes; a wrong digest, a changed file or a listed row missing
  from the sets is refused.

**Local dry-run: PASSED** at HEAD `6782139` (64 s; the Panel B fixture with no network).
- The fixture deletes its own derived copies and points the P2 rows at the session's code
  tree. Every derived file copied was therefore rebuilt in the session and re-hashed, as
  on Kaggle.
- Result: 22 rows over all six views (T-ctrl included); both digests met; derived images
  cleaned.

**Full suite:** 1,035 passed, 7 skipped; ruff clean.

**One code zip for both remaining sessions:**
- `F:\thesis-2\kaggle_raw\dbp-code-6782139.zip`, 12,405,668 bytes, sha256
  `5e1b89a4c21bda322c0c01d3bad0cb8409f23dfeb4530e16b8cca17f2c3aa4dd`.
- Between `1bc18e2` and `6782139` only new files, the dry-run harness, tests and the log
  changed (`git diff --stat`). Nothing the tuning runbook executes changed, so its dry-run
  at `1bc18e2` stands for this zip.

**Estimates (dev-panel session, CPU, no quota):**
- **Wall:** about 2.5–3 h, the Panel B session's own estimate (NOT-MEASURED there) plus
  copying 12,150 files.
- **Downloads:** about 19 GB (E8 13.54 GiB + E7 4.01 GiB + E3, E4).
- **Peak `/tmp`:** about 30 GB.
- **Output:** about 2.2–2.4 GB, from 12,150 × the MEASURED Panel B mean of 184,177
  B/image. The session measures it.

**TrueFake: the file listing (metadata only; no image downloaded).**
- **How it was read** (local, 2026-10-19/20):
  - the README's Google Drive links;
  - each link's virus-scan warning page (file name and size);
  - then **HTTP Range reads of `TrueFake.zip`'s ZIP64 end record and central directory
    only**: 65,536 + 123,346,958 bytes, the archive's index, with no image data.
- **Recorded:** `reports/truefake/truefake_listing_summary.json`. The 600,876-entry
  listing (50,027,561 bytes, sha256 `448001839df2…31d0`) stays in
  `F:\thesis-2\kaggle_raw\truefake_metadata\`. Central directory sha256 `869a102f6550…dc0b`.
- **As published:**
  - `TrueFake.zip` **647 GB** (694,732,361,838 bytes);
  - 10 pre-share `.tar.gz` partitions, 17–153 GB each (**622 GB**);
  - Facebook 3.9, Telegram 6.1 and Twitter 14 GB.
  - **The `.tar.gz` files have no index, so they cannot be listed without a full
    download.**
- **Layout:** `TrueFake/<share>/<Fake|Real>/<source>/[<subfolder>/]<file>`, with shares
  PreSocial, Facebook, Telegram, Twitter.
  - Pre-share: 8 generators × 40,000, FFHQ 70,000, FORLAB 30,719.
  - Each platform: 8 × 5,000 fakes + 10,000 FFHQ + 10,000 FORLAB = 60,000.
- **(a) One-to-one: CONFIRMED.**
  - All 60,000 shared files have a pre-share twin with the same path stem.
  - **The same 60,000 ids are on all three platforms.**
  - Post-share files are all JPEG; the pre-share twins are 50,000 PNG + 10,000 JPEG.
  - Subfolders are balanced: 1,250 per prompt category per diffusion model; 2,500 per ψ
    setting per StyleGAN.
- **(b) Download size** of the shared subset in all four versions, from the zip: Facebook
  3.91, Telegram 6.06, Twitter 14.18 and pre-share twins 88.24 GiB, **112.4 GiB in all**.
  - As whole partitions this would mean all 10 pre-share `.tar.gz` (622 GB), because the
    twins are spread across them.
  - The zip allows per-member range reads, so a sample can be fetched without either.

**TrueFake evaluation: PROPOSED, awaiting the user's approval. Not adopted, nothing
downloaded.**
- **Sample:** 4,000 ids, seed 2026, stratified by source and subfolder, × 4 versions =
  16,000 files, **7.99 GiB**.
  - Fakes: 250 per generator, spread evenly over its prompt categories or ψ settings.
  - Reals: 1,000 FFHQ and 1,000 FORLAB.
  - Fetched member by member from `TrueFake.zip`, on Kaggle only.
- **Scoring budget:** 4 models × 16,000 images × 0.064 s (the slowest MEASURED S2 scoring
  rate, P2 Session D) = **1.14 GPU-h**, within the 1.5 GPU-h cap. Decoding of the large
  pre-share originals is NOT-MEASURED.

### P3-C — TrueFake evaluation rule FIXED (before Panel B is scored); data card; FORLAB is Flickr — 2026-10-20

**User approval (2026-10-20)** of the proposed TrueFake evaluation, with additions:
- the fourth model is **`S2-B+deg`** (the tuned winner);
- the StyleGAN / FFHQ content caveat;
- fetching only in a Kaggle CPU session after a hand-over the user approves, stopping on
  any Drive refusal;
- images private, aggregate numbers only;
- a data card with the component terms from fetched pages.
- **The rule is in `P3_PLAN.md`**, section "TrueFake — a secondary, descriptive
  social-media test". It is committed here, before any Panel B score and before any
  TrueFake image exists.

**Sample frozen:** `reports/truefake/truefake_sample.csv`.
- 16,000 rows = 4,000 ids × 4 versions; 1,869,405 bytes; sha256
  `a8954c2f63e81d95173cbec99e3f7c1ee2dd3d49527fac5ff7075de5e1ac8fd6`; 7.985 GiB of members.
- Built from the archive listing with the stratified, seed-2026 draw of the proposal.

**Component terms, read 2026-10-20** (`docs/data_cards/truefake.md`;
`docs/CITATIONS.md`, entry "TrueFake"):
- **TrueFake repository:** `LICENSE` = **CC BY 4.0** legal code (commit `806f848a…`,
  sha256 `9ba9550a…9411`).
  - **Correction:** the CC BY 4.0 shown on arXiv is the **paper's** licence; the paper
    states no dataset licence.
- **FFHQ** (`NVlabs/ffhq-dataset` `LICENSE.txt`, commit `4826aa6e…`, sha256 `620105e3…33d9`):
  - images under CC BY 2.0, CC BY-NC 2.0, PDM 1.0, CC0 1.0 or U.S. Government Works, "for
    non-commercial purposes";
  - the dataset itself under **CC BY-NC-SA 4.0**; citation required.
- **FORLAB:** terms **NOT-VERIFIED**; neither paper states any.

**Finding: FORLAB is Flickr-sourced.**
- The source paper (arXiv 2009.04878, abstract): "over 33.000 Flickr images belonging to
  45 smartphone and 25 DSLR camera models".
- So **both TrueFake real pools share the Flickr platform with our training reals** (COCO,
  Open Images).
- The rule therefore keeps FORLAB's FPR **descriptive only, never a claim**, like FFHQ's.
  The approved proposal had called FORLAB the platform-disjoint camera pool. The rule was
  descriptive throughout, so nothing else changes. **Flagged to the user.**

**Fetch: not built yet.** It awaits its own hand-over, with a download-time estimate, and
the user's approval.

### P3-C — TrueFake fetch runbook built, dry-run PASSED (`408c1f5`); a test-tooling gap fixed — 2026-10-20

**User (2026-10-20):**
- FORLAB keeps its FPR, descriptive only.
- The caveats now state that both TrueFake real pools are Flickr, so neither supports an
  FPR claim (`P3_PLAN.md`, committed in `408c1f5`).

**Built: `scripts/p3_fetch_truefake.py` + `env/kaggle_run_p3_truefake.py` +
`env/p3_truefake_expect.json`.** One Kaggle CPU session.
1. **Archive check:** Drive's virus-scan page gives the confirm form. A range read of the
   last 64 KiB gives the ZIP64 end record; the archive size must be 694,732,361,838 bytes.
   The central directory is read and its sha256 must equal `869a102f…dc0b`.
2. **Members:** every member of the frozen sample (`reports/truefake/truefake_sample.csv`,
   sha256 `a8954c2f…8fd6`, checked by the pre-flight against the committed bytes) is read
   by one byte range, decompressed and checked against the index's CRC-32 and size.
   - Written under `truefake_sample/` at its archive path; sha256 appended per member to
     `reports/truefake/truefake_fetch_manifest.csv`.
   - A re-run skips members already verified on disk.
3. **Any refusal stops the session at once:** an HTTP error status, a non-range reply (a
   quota or sign-in page), or a range mismatch.
   - Queued members are cancelled, and in-flight ones send nothing further.
   - **No retry and no workaround**; the status file records the first refusal.
   - Only plain network failures are retried, twice.
- **CLAUDE.md §6a(c):** `gdown` fetches whole files only (here 647 GB). The approved rule
  names member-by-member range reads, so the script uses plain HTTPS range requests to
  `drive.usercontent.google.com`, **on Kaggle only**: the scoped exception the rule
  implies.
- **Privacy:** images only under `truefake_sample/`, to be published as a **PRIVATE**
  dataset. The reports hold names, sizes and hashes only.
- **Defect caught while writing:** a `with ThreadPoolExecutor` block waits for every
  queued member on exit, so the fetch would have continued after a refusal. Fixed with a
  stop flag and `shutdown(cancel_futures=True)`; tested.
- **Tests:** `test_p3_truefake.py` (5):
  - a complete fetch is byte-identical and fetches nothing outside the sample;
  - **a refusal (simulated 429) stops the run, exit 3, nothing retried**;
  - a different archive size or index is refused before any member;
  - a re-run skips verified members;
  - the runbook's stop and privacy text.

**Local dry-run: PASSED** at HEAD `408c1f5` (12 s): a local archive with TrueFake's layout,
deflated and stored members, one member outside the sample. Index verified; all 8 sample
members byte-identical.

**Full suite at `408c1f5`: 2 failures, both explained and fixed or confirmed:**
1. `test_a_preflight_runs_before_any_long_job[kaggle_run_p3_truefake]`: the test did not
   know the new long job. Its marker `code = py("p3_fetch_truefake.py"` was added (the
   runbook already runs its pre-flight first). Test-only change.
2. `test_smoke_tests_every_run_on_every_gpu`: a subprocess timeout reported as
   **−10,626 s**, a wall-clock jump of about 3 h in the middle of the run. The suite took
   4 h 45 min against the usual 8–12 min, consistent with the PC sleeping (as on
   2026-10-14). It **passes on its own**: `test_runbook_tooling.py` +
   `test_s2_queue.py`, 165 passed.

**Code zip:** `F:\thesis-2\kaggle_raw\dbp-code-408c1f5.zip`, 12,654,579 bytes, sha256
`fb49c22a6e00326a603f95745569758100e290ed7b8d4791ce052e6fc27c5370`. The later test-only
commit changes nothing the session runs.

**Estimate (ESTIMATED):**
- **Wall:** 0.5–1.5 h on CPU: about 16,000 range requests × 0.3–1.0 s over 4 connections
  (per-request latency NOT-MEASURED on Kaggle), plus 2–7 min of transfer.
- **Download:** 7.985 GiB (+ the 123 MB index).
- **Output:** 8.61 GB decompressed. Kaggle's working disk is 19.5 GB (MEASURED earlier).
- **Risk:** a Drive download quota on the one heavily read file, which means stop and
  report.

**S2-D:** measured for the mixing question. S2-B seed 0 trained on 1,926 COCO train reals
(23,112 pairs, its log). The Open Images train split adds 3,600 reals (43,200 pairs).
Pooled uniformly by real, about 65% of sampled pairs would be Open Images. The mixing rule
is asked of the user before the S2-D runbook is built.

### P3-C — training sessions 2 and 3 COMPLETE: BANK-2 and BANK-4, both orders — 2026-10-20

**Runs:** Kaggle GPU T4 × 2, `dbp-code-8171516`. Each session had the previous version's
output attached.
- **Archives:**
  - session 2: `F:\thesis-2\p3_train_reports_phase-2.zip` (11,329,413 bytes, sha256
    `744caeb1…42d3`);
  - session 3: `F:\thesis-2\p3_train_reports-phase-3.zip` (11,386,934 bytes, sha256
    `268efbe0…a7ae`);
  - both `testzip` clean; extracted to `kaggle_raw\p3_train_session{2,3}_x\`.
- **Committed:** under `reports/kaggle_20261019/p3_train_session2/` and
  `reports/kaggle_20261020/p3_train_session3/`; the four per-image score files under
  `reports/p3_heldout_scores/`.

**Checks, both sessions:**
- **Pre-flight PASSED:** 6 checks.
- **The chooser:**
  - session 2 picked `bank_2_near`, `bank_2_far`, with `bank_1` and `s2b` seed 0 complete;
  - session 3 picked `bank_4_near`, `bank_4_far`, with the four earlier runs complete;
  - the rescoring was not due in either.
- **Carry-forward byte-identical:** every carried report equals its original (BANK-1 and
  S2-B s0 in sessions 2 and 3 equal session 1's; BANK-2 in session 3 equals session 2's),
  as do the carried score files.
- **Logs:** no error, traceback or NaN. The only matches are the 11 known peft
  "401 Client Error" save-time warnings per log.

**MEASURED:**

| run | trained on (decoders) | train pairs | steps (schedule / stop) | GPU-h | train-loop s/step | val AUC at 5000 | held-out AUC (pooled) | per-image file |
|---|---|---|---|---|---|---|---|---|
| `bank_2_near` | 5 | 9,630 | 5,000 (6,000 / 5,000) | 1.579 | 1.00702 | 1.0000 | 0.91434 | 491 reals, 3,437 recons; sidecar sha256 OK |
| `bank_2_far` | 4 | 7,704 | 5,000 | 1.579 | 0.98793 | 0.9999 | 0.91798 | 491, 3,928; OK |
| `bank_4_near` | 8 | 15,408 | 5,000 | 2.032 | 1.33363 | 0.9998 | 0.92003 | 491, 1,964; OK |
| `bank_4_far` | 6 | 11,556 | 5,000 | 2.063 | 1.32099 | 0.9988 | 0.99303 | 491, 2,946; OK |

- **The LR at step 5000** is 6.712e-06 in every run (the 6,000-step schedule, D1).
- **Queue wall:** session 2 1.5833 h (device 3.1667); session 3 2.075 h (device 4.1167).
- **Smoke:** 0.075 and 0.1083 h.
- **Held-out AUC per member (descriptive; the bank-size analysis comes at the end):**
  FLUX.2 stays the hardest held-out lineage (0.598–0.721 where it is held out).

**Observation: the step got slower, on the Kaggle side.**
- **Train-loop s/step:** 0.759–0.780 (session 1), then 0.988–1.007 (session 2), then
  1.321–1.334 (session 3).
- **Same code, batch plan and data path.** The slower pace is uniform from step 1 to 5,000
  in the logs, so it is the host, not a phase of the run. The logs do not identify a cause.
- **Effect:** about 1.6 → 2.1 h of quota per session. Recorded as MEASURED.

**Queue status:** 6 of 10 complete (`bank_1`, `s2b` s0, `bank_2_*`, `bank_4_*`).
- **Remaining:** `bank_6_near` + `bank_6_far`, then `bank_1l` + `s2b` seed 1.

### P3-C — dev panel rebuilt on `muhammadadeelshah`: VERIFIED byte-identical — 2026-10-20

**Run:** Kaggle CPU, notebook `dbp-p3-dev-panel`, code `dbp-code-6782139`
(`scripts/p3_rebuild_dev_panel.py`, last changed in `6782139`).
- **Records:** `F:\thesis-2\kaggle_raw\p3_dev_panel_digest.txt` (1,665 bytes) and
  `F:\thesis-2\kaggle_raw\p3_eval_views_rebuild.txt` (2,591 bytes). Both are JSON with a `.txt` name.
- **Committed** as `.json` under `reports/kaggle_20261020/p3_dev_panel/`, bytes unchanged
  (`sha256sum`: `75b0bcd6…0ea4` and `da67d0dd…6a55`).

**MEASURED (from `p3_dev_panel_digest.json`):**
- **`ok`:** true; `problems` empty.
- **`digest_all`:** `2e1c1920…b769` over 12,150 rows. It equals `expected.digest_all`,
  the P2 panel digest committed in the script.
- **`digest_scoreable`:** `bff608c3…6116` over 11,775 rows. It equals
  `expected.digest_scoreable`.
- **Rows by view:**

| view | rows |
|---|---|
| e3_matched | 1,500 |
| e4_wildrf_matched | 1,500 |
| e5_chameleon_pipeline | 4,000 |
| e7_dda_coco | 3,500 |
| e8_evalgen | 1,250 |
| t_ctrl_diagnostic | 400 |

- **`panel_bytes`:** 2,344,945,736.
- **Every file** was re-hashed against the committed per-file sha256
  (`p2_materialise_panel.materialise_view`, unchanged).

**From `p3_eval_views_rebuild.json`:**
- **Each of the five rebuilt evaluation views** (e3, e4, e5, e7, e8) has
  `identical_to_p1: true`.
- **Rows reproduced from P2:** 1,500 / 1,500 / 4,000 / 3,500 / 1,250.
- **Every `manifest_sha256`** equals the one in the committed Panel B session record
  (`reports/kaggle_20261011/p3_eval_views_rebuild.json`, `0f099ac`). The comparison was
  run locally.
- **T-CTRL is absent here by design:** it comes from the training corpus, not from a view
  rebuild. Its 400 rows are covered by both panel digests above.

**Consequence:**
- **The dev panel P3 uses** is the `dbp-p3-dev-panel` notebook output on
  `muhammadadeelshah`, PRIVATE (WildRF and Chameleon, §6c).
- **The `adeelshah29` copy** is never attached again.
- **Next:** `dbp-p3-tune` (GPU) may run.

### P3-C — training session 4 COMPLETE: BANK-6, both orders; the slowdown is first-read cost, not the host's compute — 2026-10-21

**Run:** Kaggle GPU T4 × 2, `dbp-code-8171516`, 2026-10-20 (log timestamps), with
session 3's output attached.
- **Archive:** `F:\thesis-2\p3_train_reports-phase-4.zip`, 11,425,012 bytes, sha256
  `3b2a3370…3c75`, `testzip` clean, 352 entries, no unsafe path. Extracted to
  `F:\thesis-2\kaggle_raw\p3_train_session4_x\`.
- **Committed:** under `reports/kaggle_20261020/p3_train_session4/`; the two per-image score
  files are under `reports/p3_heldout_scores/`.

**Checks:**
- **Pre-flight PASSED:** 6 checks (2 × T4, `HF_TOKEN present: true`, 5 input hashes,
  imports, 2,488 gen_matched hashes).
- **The chooser:** picked `bank_6_near`, `bank_6_far`, with six runs complete; the
  rescoring was not due; 16 LOLO score files were carried.
- **Carry-forward byte-identical:**
  - every carried run report equals its committed original (BANK-1 and S2-B s0 equal
    session 1's, BANK-2 session 2's, BANK-4 session 3's);
  - every carried score file equals the committed one.
- **Sidecars:** both `scores_sha256` match their CSVs.
- **Held-out AUCs recomputed locally** from the per-image files equal the logged ones.
- **Logs:** no error, traceback or NaN. The only matches are the 11 known peft
  "401 Client Error" save-time warnings per training log.

**MEASURED:**

| run | trained on (decoders) | train pairs | steps (schedule / stop) | GPU-h | train-loop s/step | val AUC at 5000 | held-out AUC (pooled) | per-image file |
|---|---|---|---|---|---|---|---|---|
| `bank_6_near` | 10 | 19,260 | 5,000 (6,000 / 5,000) | 5.364 | 3.52127 | 0.9987 | 0.9966 | 491 reals, 982 recons; sidecar OK |
| `bank_6_far` | 9 | 17,334 | 5,000 | 5.485 | 3.50414 | 0.9987 | 0.99866 | 491, 1,473; OK |

- **Held out per member:**
  - `bank_6_near`: dcae 0.99326, realesrgan_x4 0.99994;
  - `bank_6_far`: movq_k22 0.99948, sdxl 0.9986, sdxl_fp16fix 0.9979.
- **The LR at step 5000** is 6.712e-06 in both runs.
- **Queue wall:** 5.4917 h (device 10.8667); smoke 0.425 h.

**The step time, per 500-step interval (minutes, from the log timestamps):**

| session | run | intervals to step 500, 1000, …, 5000 |
|---|---|---|
| LOLO 1–4 | all 8 folds | flat at 7.1–8.1 throughout |
| 1 | `bank_1`, `s2b` | flat at 6.6–6.9 (9.1 / 6.8 at the last, held-out step) |
| 2 | `bank_2_near` / `_far` | 14.6, 9.5, 8.6–8.4, … 7.4–7.6; last 14.4 / 16.1 |
| 3 | `bank_4_near` / `_far` | 24.6, 13.8, 12.2–12.0, … 8.4; last 15.2 / 17.9 |
| 4 | `bank_6_near` / `_far` | 97.6, 44.1, 31.0, 29.3, 21.6–20.9, 17.7–18.1, 18.2–18.5, 14.3–14.2, 12.5–11.9; last 35.6 / 43.5 |

**Correction to the session 2–3 entry above.** It said the slower pace "is uniform from
step 1 to 5,000 in the logs, so it is the host, not a phase of the run". **That was wrong.**
- The intervals fall steadily within every run since session 2 and approach session 1's
  rate.
- The first interval has grown each session: 14.6 → 24.6 → 97.6 min.
- Both GPUs track each other to within 1 min, so the bottleneck is shared.

**INFERRED, NOT-MEASURED:** the extra time is the first read of each image from the
`/kaggle/input` mount. Once a file has been read, it is fast.
- **The trainer and the scorer read every image serially in the main process**
  (`dbp.utils.io.load_rgb_uint8`; no loader workers). A slow first read therefore adds
  directly to the step.
- **The interval to step 5000 holds the held-out scoring**, which reads 1,473–1,964 images
  never read before. It cost 23–32 min more than the warm rate here, against about 6–8
  min in session 3.
- **The code is unchanged since `8171516`,** and session 1 (same code, `s2b` over all 12
  members) showed no such cost. The cause sits on Kaggle's side; what changed there is
  NOT-VERIFIED.

**Consequences (ESTIMATED):**
- **Session 5** (`bank_1l` + `s2b` s1, running now) may take about as long as this one.
  The 9.5 h limit and the resumable queue cover it.
- **`dbp-p3-tune` as committed is at risk.** Cell 6 scores S2-B s0 on 11,775 dev-panel
  and 6,883 val images, all read for the first time, under a **1.0 h** limit. At this
  session's first-read cost it would not finish. The notebook would then train both
  candidates (about 5.5 h+) and fail at Cell 9 without applying the rule. The same scoring
  would repeat on every re-run.
- **Tuning is held** until the user decides on an I/O fix (asked 2026-10-21).

**Queue status:** 8 of 10 complete. Remaining: `bank_1l` + `s2b` seed 1 (session 5,
running).

### P3-C — DECISIONS of 2026-10-21: S2-D real mixing (pooled by real); hardware log; tuning scoring limit — before the S2-D runbook is built

**User, 2026-10-21:**
- Sessions 2 and 3 accepted.
- Asked for a hardware log and a throughput line in the runbooks, and a clean full suite
  now that the PC no longer sleeps.

**1. S2-D mixing: option (a), pooled by real.**
- **One real pool:**
  - the COCO train reals the trainer already uses;
  - plus the **3,600 Open Images train reals**;
  - sampled **uniformly by real, exactly as the trainer does today**
    (`data.train_reals`).
- **Partners:** each real's partners are its own reconstructions by the 12 members.
- **gen_matched:** unchanged at 25% of fakes, the batch plan's 2 of 8.
- **The same rule applies to `S2-B+reals`.**
- **The pool, MEASURED:** S2-B s0's log says "23112 train pair rows over **1926** reals"
  (`reports/kaggle_20261019/p3_train_session1/s2_logs/s2_s2b_seed0_gpu1_train.log`).
  - The user's message said "~3,600" COCO train reals. The measured count is 1,926.
  - So the pool is 1,926 + 3,600 = 5,526 reals, and the Open Images share by
    construction is 3,600 / 5,526 = **65.15%**.
  - The rule, pooled by real, is unaffected.
- **No change to the trainer** (user, 2026-10-21: "the trainer must stay byte-identical";
  asked and confirmed).
  - S2-D and S2-B+reals reach it through **merged input files**: the splits, the pairs
    manifest and the reals manifest, with the Open Images train rows added.
  - These are built and verified by a new script before training.
- **The real share reported:**
  - the pool share from the trainer's own log line ("… over N reals"), as
    3,600 / N Open Images;
  - the expected sampling share follows from it.
  - The trainer does not record which reals it sampled, so a **realised** share would need
    a trainer change. The user chose not to make one (2026-10-21).
  - With about 40,000 real draws per run (5,000 steps × 8), the realised share lies within
    about ±0.5 points of the pool share (binomial, 2 SD). This is ESTIMATED, not
    measured.
- **The 400 held-out Open Images reals and their 4,800 reconstructions never enter
  training.**
  - They are absent from every merged file.
  - The builder refuses if any of them appears.
  - A test and a pre-flight check enforce it.
  - They remain reserved for the union k = 0 threshold (`P3_PLAN.md`).
- **Held-out per-image scores:** S2-D and S2-B+reals train on all 12 members. They have
  no held-out lineage, so, like S2-B s0, they write no per-image held-out file at the
  final step. The recipe is otherwise the same: the 6,000-step schedule stopped at 5,000.

**2. The S2-D runbook (user, 2026-10-21):**
- **Runs:**
  - `S2-D` seeds 0 and 1, plus `S2-B+reals` seed 0, two per session;
  - session 1 is `S2-D` s0 + `S2-B+reals`; session 2 is `S2-D` s1.
- **The chain probability p** is read from the committed `reports/p3_degradation_choice.json`.
  - `configs/p3_runs.yaml` gains `s2d_p70`, `s2d_p35` and `s2b_reals`.
  - The runbook picks the `s2d_*` entry whose p was chosen. It never picks one by hand.
- **Inputs:** every input file is verified against its committed digest before training:
  - the Open Images reals against `p3_open_images_manifest.csv`;
  - the pairs manifest against `p3_oi_pairs_manifest_digest.json`;
  - each reconstruction against the manifest's sha256.

**3. Slowdown (user, 2026-10-21):**
- **Added, in runbook cells only** (the trainer stays byte-identical):
  - a hardware log at session start: CPU model and core count, GPU names, and
    `nvidia-smi` clocks, power and temperature;
  - a one-line throughput summary at the end.
- **No other change** to how the runbooks read data.

**4. Tuning scoring limit (asked and approved 2026-10-21):**
- `env/p3_tune_expect.json` `base_score_deadline_hours` goes from **1.0 → 4.0**. Nothing
  else changes.
- **Why:** Cell 6 scores S2-B s0 on 11,775 dev-panel and 6,883 val images, all read for the
  first time. At session 4's first-read cost (see the session 4 entry) that would likely
  pass 1.0 h, fail Cell 9 after both candidates had trained, and fail again on every
  re-run.

### P3-C — S2-D runbook BUILT (trainer unchanged); hardware log and throughput line added; tuning scoring limit 4.0 h — 2026-10-21

**Full suite before any change** (`44ec269`, the PC set to never sleep): **1047 passed,
7 skipped, 0 failed** in 8 min 03 s.

**Built (decisions of 2026-10-21, entry above):**
- **`scripts/p3_s2d_inputs.py`** (new):
  - **`build`:**
    - verifies the committed Open Images reals manifest (sha256, 3,600 train / 400
      held out);
    - verifies the attached step-4a pairs manifest against
      `p3_oi_pairs_manifest_digest.json` (rows, rows per member, content digest);
    - hashes every train real and reconstruction (46,800 files, 32 threads) and never
      reads a held-out one;
    - refuses any id that collides with a COCO id;
    - writes merged copies of the pairs manifest, the split and the `t_real` reals
      manifest, COCO rows unchanged and first;
    - resolves `s2d_p70` or `s2d_p35` from the committed `p3_degradation_choice.json`
      (`chosen_p`) into the session queue;
    - builds each run's pool with the trainer's own `TrainingData` (imported, unchanged)
      and counts it by source.
  - **`check`:** the pre-flight. It refuses if a held-out Open Images id is in any merged
    file, or if a merged file differs from what `build` wrote.
- **`env/kaggle_run_p3_s2d.py` + `env/p3_s2d_expect.json`** (new):
  - **Queue:** `S2-D` s0 + `S2-B+reals` s0, then `S2-D` s1 (`p3_train_next.py`, unchanged,
    reads the resolved queue).
  - **Order:** data pre-flight, then build, then held-out check, then smoke, then
    training.
  - **Trainer inputs:** it receives `--pairs-manifest`/`--splits` of the merged files and
    `--manifests=<merged>/manifests`.
  - **Recipe:** 6,000-step schedule stopped at 5,000.
  - **No per-image held-out file:** there is no held-out lineage, as with S2-B s0.
- **`configs/p3_runs.yaml`:** gains `s2d_p70`, `s2d_p35` and `s2b_reals`.
- **Verified locally on the real manifests** (COCO inputs from
  `F:\thesis-2\kaggle_raw\dbp-p2-s2-inputs`, sha256 equal to the expect file;
  the 4a manifest `F:\thesis-2\p3_oi_pairs_manifest.txt`; file hashing stubbed, the
  images are on Kaggle; scratch choice p = 0.7; nothing committed from this run):
  - content digest `0a4e9a3c…1dc4`, the committed one;
  - merged: +3,600 reals, +43,200 pair rows, +46,800 split rows;
  - excluded: 400 reals, 4,800 pair rows;
  - **pool, per the trainer's `TrainingData`: 5,526 reals = 1,926 COCO + 3,600 Open
    Images, share 0.651466, 66,312 train pair rows** (23,112 + 43,200);
  - held-out rows found by `check`: 0 in all three files.
- **The trainer is byte-identical:** `tests/test_p3_s2d.py` pins
  `scripts/p2_train_s2.py` at its git sha256 `6b9c30c4…b7b6` (last changed `1bc18e2`).

**Runbook cells only (user, 2026-10-21):**
- `kaggle_run_p3_s2d.py`, `kaggle_run_p3_tune.py` and `kaggle_run_p3_train.py` each gain:
  - **a hardware log at session start:** CPU model and logical cores; `nvidia-smi`
    `index, name, clocks.sm, clocks.max.sm, clocks.mem, clocks.max.mem, power.draw,
    power.limit, temperature.gpu`. It is printed and saved as
    `reports/<stage>_hardware.txt`, so it lands in the reports zip;
  - **one `THROUGHPUT |` line** in the report cell: steps, train-loop s/step and GPU-h per
    run, and the queue wall.
- **No other change** to those runbooks.
- **`env/p3_tune_expect.json`:** `base_score_deadline_hours` 1.0 → 4.0 (approved). Its
  `expected` text (1.6–2.0 h) is left as it was ("no other change"). The hand-over
  states the current estimate instead.

**Tests:**
- **`tests/test_p3_s2d.py` (9):**
  - pooled by real, with COCO rows unchanged;
  - queue from the tuning result;
  - held-out files never read (deleted, and the build still passes) and never merged;
  - the pre-flight refuses a held-out row;
  - a corrupted train file is refused before anything is written;
  - a manifest off the committed digest is refused;
  - a `chosen_p` without a config entry is refused;
  - the trainer is byte-identical;
  - runbook order and the hardware/throughput lines.
- **`test_p3_degrade.py`:** the "nothing else degrades" guard now names the S2-D entries
  explicitly; `s2b_reals` stays at p = 0.

**Fixture fix (dry-run tooling only):**
- **The defect:** the real LOLO rescoring summary has been committed since `0e1945d`, so
  the training fixture's code copy reported the rescoring as done. Session 1 then skipped
  it, and the fixture checks failed (dry-run of 2026-10-21, before the fix).
- **The fix:** the fixture now removes the committed `reports/p3_heldout_scores/` from its
  code copy, the state before session 1.
- **On Kaggle:** nothing changes. Session 5 correctly sees the rescoring as done.

**Local dry-runs, `--ref :` (the index, the bytes committed next):**
- **`p3_s2d`: PASSED** (370 s, 2 sessions).
  - **Fixture:** 3 Open Images train reals + 1 held out. The held-out files **do not
    exist on disk**, so any read of them would have failed.
  - **Queue:** `s2d_p35:0, s2b_reals:0, s2d_p35:1`.
  - **Pools:** the trainer logged "over 7 reals" for every run (4 COCO + 3 Open Images),
    equal to the builder's pool.
  - **Degradation p:** 0.35 / 0.0 / 0.35.
  - **Session 2:** carried both finished runs and trained S2-D s1.
- **`p3_tune`: PASSED** (482 s); the hardware and throughput lines printed.
- **`p3_train`: PASSED** (645 s, 3 sessions, after the fixture fix). Rescoring diff 0.0 on
  all 8 folds.

### P3-C — `dbp-p3-tune` hand-over at `fcc2ab4`: dry-run PASSED, code zip — 2026-10-21

**Checks before the hand-over:**
- **Full suite** on the `fcc2ab4` tree, before the commit: **1063 passed, 7 skipped,
  0 failed** in 6 min 05 s.
  - All 7 skips are `tests/test_gpu_smoke.py`: "no CUDA device in this runtime; run on
    Kaggle T4 (PENDING-GPU)".
  - ruff: all checks passed.
- **Integrity:** CLEAN before the commit.

**Local dry-run: PASSED** at `--ref HEAD` = `fcc2ab4` (458 s, 2 sessions,
`kaggle_run_p3_tune.py`).
- **Session 1:** both candidates killed by design; the hardware log printed.
- **Session 2:** resumed, scored, and the rule applied. The fixture chose
  `s2b_deg_p35_seed0` (fixture data, no meaning).
- **The throughput line printed.**

**Code zip:** `F:\thesis-2\kaggle_raw\dbp-code-fcc2ab4.zip`, 12,906,039 bytes, sha256
`e400f71d44dedd4b6fe2026f235ef0482f23bcb1cf7d75e2cfcb6d60be27b9f2`.
- 561 entries, byte-identical to the commit; no CR in any `.sh`/`.py`.
- `base_score_deadline_hours` is 4.0.
- No `p3_degradation_choice.json` is in it, which is correct: the S2-D zip is built after
  the tuning result is committed.

**Estimate for `dbp-p3-tune` (ESTIMATED):** about 3–10 h of wall time and quota.
- **Training:** 5,000 steps at 1.3–3.5 s/step (MEASURED range, sessions 3–4) = 1.8–4.9 h,
  plus the degradation's per-batch cost (planned +25%; MEASURED by this session's smoke).
- **S2-B s0 dev scoring before training:** 18,658 images read for the first time,
  0.2–3.6 h (INFERRED from the session 4 first-read cost; limit 4.0 h).
- **Two candidate scorings after training:** 0.2–1 h.
- **Setup and smoke:** about 0.5 h.
- **Resumable:** a re-run with the version attached skips the finished S2-B scoring and
  resumes candidates from their checkpoints.

### P3-C — tuning scores only the rule's rows (user, 2026-10-21); estimate basis; printed estimate updated — 2026-10-21

**User, 2026-10-21:** S2-D runbook accepted; the 65.15% Open Images pool is reported as
MEASURED. Three questions on the tuning estimate.

**1. The basis of "up to 3.6 h" for scoring 18,658 images.**
- **The S2 scoring rate MEASURED in P2** (2026-10-05, `reports/kaggle_20261005/
  p2_s2_scores_*.json`): 0.02636–0.06408 s/image on the same dev panel. This covers
  decoding and the forward pass, read when reads were fast. It is not decoding: the
  panel averages 193 KB per image (2,344,945,736 B / 12,150, the dev-panel record).
- **The extra cost is the first read of each file from `/kaggle/input`.** This is
  INFERRED from training-log intervals and is not a direct measurement. Per image read:
  - session 1: about 0;
  - session 2: (14.6 − 6.8) min / (500 × 16) = 0.06 s;
  - session 3: 0.13 s (first 500 steps), and 0.14–0.15 s from the held-out scoring at
    step 5000: (15.2 − 9.5) min / 2,455 images and (17.9 − 9.5) min / 3,437;
  - session 4: 0.68 s (first 500 steps: (97.6 − 6.8) min / 8,000), and 0.94–0.97 s from
    the held-out scoring: (35.6 − 12.5) min / 1,473 and (43.5 − 11.9) min / 1,964.
- **"Up to 3.6 h"** was 18,658 × 0.68 s ≈ 3.5 h. At 0.97 s it would be 5.0 h, past the
  4.0 h limit.
- **A scoring stopped by its limit keeps nothing:** the watchdog exits and the list is
  redone in full on the next run.

**2. Restrict tuning to the rule's rows: done** (asked and approved, 2026-10-21).
- **The rule uses 7,208 rows per model:**
  - 6,717 dev rows: E3 reals 477; E3 GPT-Image2 500 and Nano-Banana2 500 fakes; E4 1,492;
    E5 3,748;
  - the 491 val reals.
- **11,450 more were scored and never read by the rule.**
- **`p3_tune_degradation.py` refuses to run unless every row of its `--panel-list` is
  scored.** So the rule now receives the restricted list as `--panel-list`: its input
  list changes; its code and every number it computes do not.
  - Its `--val-list` stays the full S1 list; the rule picks the 491 val reals from it.
- **New `scripts/p3_tune_lists.py`:**
  - selects the rows with the rule's own `panel_rows`, `FPR_POOLS` and `unseen_slices`
    and P2's `slice_key`, and writes them verbatim;
  - checks them against `derived_lists` in the expect file:
    - `p3_tune_panel_rows.csv` 6,717 rows, sha256 `c4614dbb…5104`;
    - `p3_tune_val_reals.csv` 491 rows, sha256 `cf6dda47…90ba`.
  - Both were produced from the real lists in `F:\thesis-2\kaggle_raw\dbp-p2-s2-inputs`, whose
    sha256 equal the expect file's.
- **Runbook cells:**
  - the lists are derived before the data pre-flight;
  - `--resolve` checks the derived lists, the rows actually read;
  - S2-B s0 and both candidates score them;
  - the rule gets the restricted dev list.
- **Equivalence, MEASURED locally on the real lists:** the rule's `evaluate()` on full vs
  restricted inputs gave identical output over 5 random score draws, and the same val-real
  ids.
  - `tests/test_p3_tune_lists.py` (4) repeats this on a synthetic panel holding every
    view. It also checks the refusal off-digest, the committed counts, and that the runbook
    derives first and hands the rule the restricted list.
- **Saving (ESTIMATED), per tuning session:** 11,450 × (first-read cost + 0.026–0.064 s)
  = 0.1 h at session-1 speed, 0.5–0.7 h at session 3's, 2.3–3.3 h at session 4's.
  - At session 4's speed, the S2-B scoring drops from 3.5–5.0 h to 1.4–1.9 h, inside the
    4.0 h limit.
- **Not produced any more:** dev scores of E7, E8 and T-ctrl for S2-B s0 and the two
  candidates. The rule never read them.
- **Fixture:** `fixture_p3_tune` adds four panel rows the rule never reads (E7, E8, T-ctrl),
  with **no image on disk**, and checks that each model scored exactly the rule's rows.

**3. The printed estimate (`expected` in `env/p3_tune_expect.json`), text only:** about
2–10.5 h of wall time and quota.
- **Scoring:** 7,208 images per model at the rates above. S2-B s0 before training takes
  0.05–1.9 h; the candidates score in parallel afterwards, 0.05–1.9 h.
- **Training:** 5,000 steps at 0.77–3.52 s/step (MEASURED, sessions 1–4) = 1.1–4.9 h, plus
  the planned +25% degradation cost.
- **Setup and smoke:** about 0.5 h.

**Local dry-run (`--ref :`): PASSED** (785 s, 2 sessions).
- 18 of 22 dev rows and 2 val reals scored per model.
- The 4 rows the rule never reads were never resolved or read.

### P3-C — `dbp-p3-tune` hand-over REPLACED: `042f6d9` supersedes `fcc2ab4` — 2026-10-21

- **Full suite** on the `042f6d9` tree: **1067 passed, 7 skipped, 0 failed** in 9 min 28 s.
  - All 7 skips are PENDING-GPU (`tests/test_gpu_smoke.py`: no CUDA device).
  - ruff: all checks passed.
  - Integrity CLEAN before the commit.
- **Local dry-run: PASSED** at `--ref HEAD` = `042f6d9` (748 s, 2 sessions).
  - Lists derived and checked; resolve: 18 and 2 rows.
  - Each model scored 18 of the 22 dev rows plus 2 val reals.
  - The rule applied; the throughput line printed.
- **Code zip:** `F:\thesis-2\kaggle_raw\dbp-code-042f6d9.zip`, 12,913,972 bytes, sha256
  `f6a8af2709985862619e4efd93b6c7105feef43c17faf770b6d5291f95988997`.
  - 563 entries, byte-identical to the commit; `COMMIT.txt` = `042f6d9`.
  - It was first written under a guessed name (`dbp-code-b63ed0b.zip`) before the commit
    hash had been read. It was renamed at once; no copy under that name remains.
- **`dbp-code-fcc2ab4.zip` is superseded** for tuning. Do not upload it.

### P3-C — training session 5 COMPLETE: BANK-1L and S2-B seed 1; the non-LOLO queue is finished (10 of 10) — 2026-10-21

**Run:** Kaggle GPU T4 × 2, `dbp-code-8171516`, 2026-10-21, with session 4's output attached.
- **The first attempt** (version 8) failed before any cell ran: `ERRORED_MOUNTING_DATASET`,
  Kaggle could not mount the attached `dbp-p3-train` output (an `Internal` error after 30
  retries). The output was 0 B, and no code of ours ran.
- **The re-run** mounted normally.
- **Archive:** `F:\thesis-2\p3_train_reports-phase-5.zip`, 11,468,146 bytes, sha256
  `6a326772…f13d`, `testzip` clean, 356 entries, no unsafe path. Extracted to
  `F:\thesis-2\kaggle_raw\p3_train_session5_x\`.
- **Committed:** `reports/kaggle_20261021/p3_train_session5/` and
  `reports/p3_heldout_scores/bank_1l_seed0_step_05000.{csv,json}`.

**Checks:**
- **Pre-flight PASSED:** 6 checks.
- **The chooser:** picked `bank_1l:0`, `s2b:1`, with eight runs complete; the rescoring was
  not due.
- **Carry-forward byte-identical:**
  - all eight carried run reports equal their committed originals (sessions 1–4);
  - every carried score file equals the committed one.
- **Sidecar:** BANK-1L's `scores_sha256` matches its CSV (4,910 rows). The held-out AUCs
  recomputed locally equal the logged ones.
- **Logs:** no error, traceback or NaN. The only matches are the 11 known peft
  "401 Client Error" warnings per training log.

**MEASURED:**

| run | trained on | train pairs | GPU-h | train-loop s/step | val AUC at 5000 | held-out AUC (pooled) |
|---|---|---|---|---|---|---|
| `bank_1l` | sd15, sd21, sd_ft_ema | 5,778 | 1.487 | 0.90628 | 0.99997 | 0.92693 |
| `s2b` seed 1 | all 12 | 23,112 | 1.628 | 1.13027 | 0.99811 | (no held-out lineage) |

- **BANK-1L held out per member:**
  - dcae 0.99696, flux 0.92253, **flux2_klein 0.60221**, movq_k22 0.99954;
  - qwen_image 0.84998, realesrgan_x4 1.0, sd35 0.97554;
  - sdxl 0.99839, sdxl_fp16fix 0.99725.
- **S2-B seeds:** s0 val AUC 0.99818 at 5000 (session 1), s1 0.99811. Both have the LR
  6.712e-06 at 5000.
- **Queue wall:** 1.6333 h (device 3.125); smoke 0.0917 h.

**Step time per 500-step interval (min):**
- **`bank_1l`:** 13.1, 9.1, 7.9, 7.5, 7.1, 7.0, 6.9, 6.8, 6.8; held-out step 17.0.
- **`s2b` s1:** 13.9, 10.9, 10.2, 9.9, 9.4, 9.2, 8.8, 8.8, 8.4, 8.2.
- **Compared with session 4** (first interval 97.6 min), session 4's slow first reads did
  not recur. The first interval is close to session 2's (14.6). The first-read cost varies
  from session to session (INFERRED; cause NOT-VERIFIED).
- **For tuning:** the range in its estimate stands.

**Queue status:** 10 of 10 non-LOLO runs complete. `dbp-p3-train` is finished; its
latest version holds every model.

**Dev-panel output downloaded to this PC.**
- **The file:** the user placed `F:\thesis-2\dbp-p3-dev-panel_output.zip` (2,478,036,227
  bytes) on the local machine.
- **What it holds:** only its listing was read. 12,690 entries: the 12,150 dev-panel images
  (E3 1,500, E4 WildRF 1,500, E5 Chameleon 4,000, E7 3,500, E8 1,250, T-ctrl 400) and a
  540-file code tree.
- **The rule:** CLAUDE.md section 6a(a) keeps dataset downloads on Kaggle. WildRF is
  research-only and Chameleon academic-only (section 6c).
- **Handling:** not extracted, hashed or used. The panel was verified on Kaggle (entry of
  2026-10-20). It lies outside REPO_ROOT, so integrity is unaffected.
- **Asked:** the user is asked to delete it.

### P3-C — S2-D: every file the runs read is staged once to local disk before training (user, 2026-10-21) — 2026-10-22

**User, 2026-10-21:**
- **The request:** warm every file the S2-D runs read with a parallel read pass, combined
  with the Open Images hash check so that each file is read once.
- **Logging:** the warm-up time, files/s and the training s/step go into the THROUGHPUT
  line.
- **If RAM cannot hold the page cache:** say so and copy to `/tmp` instead, if it fits,
  reporting the sizes before choosing.
- The trainer stays byte-identical; no other change.

**Sizes, before choosing:**
- **Open Images:** 3,600 train reals and 43,200 train reconstructions, about 10.45 GB.
  - Taken pro rata from the MEASURED totals: crops 872,531,042 B for 4,000; pairs
    10,737,317,606 B for 48,000.
- **COCO:** the train and val pairs of all 12 members plus their reals, 31,421 files,
  about 6.9–9.4 GB.
  - ESTIMATED at 0.22–0.30 MB per file: no COCO byte count was ever recorded.
- **gen_matched:** already on local disk.
- **Total:** about 17.5–20 GB.
- **Kaggle RAM:** 33.7 GB; `/tmp` free: 1,100 GB. Both MEASURED in
  `reports/kaggle_20260920/env_report.json`.
- **Two training processes share that RAM;** their resident size is NOT-MEASURED.

**Choice: a copy to local disk, not a page-cache warm-up.**
- 17.5–20 GB could not be shown to stay cached beside two trainers.
- The kernel would evict the cache silently, and the first-read cost would return
  unseen.
- `/tmp` holds the copy many times over.

**Built (`scripts/p3_s2d_inputs.py` `build`; the trainer unchanged):**
- **Which files:** every file the queued runs read is listed with the trainer's own
  `TrainingData`: train and val pairs, those pairs' reals, and gen_matched.
- **One parallel pass (32 threads)** reads each file **once** from the corpus root (links to
  `/kaggle/input`) and writes it to `--local-root` (`/tmp/dbp_s2d_corpus`).
  - The 46,800 Open Images train files are hash-checked **in that same read**.
  - The held-out files are never read.
- **Before copying,** the pass measures (stat) the total size, records MemTotal,
  MemAvailable and Cached plus the free space, and refuses if size + 2 GB exceeds the free
  space.
- **Recorded in `reports/p3_s2d_inputs.json` `staged`:** files and bytes by source,
  seconds, files/s, MB/s.
- **gen_matched is counted but not copied:** this session's matcher wrote it to local disk,
  and its manifest holds absolute paths the trainer reads whatever its root.
- **The merged files** are prepared in `.pending` and moved into place only when staging
  passed, so a failed check leaves no merged files and no queue.
- **`check`** (pre-flight) also refuses any held-out Open Images file in the local copy.

**Runbook cells (`env/kaggle_run_p3_s2d.py`):**
- the build gets `--local-root`;
- the queue's `--root` is the local copy (smoke and training);
- `check` gets `--local-root`;
- **THROUGHPUT** now begins `staging <files> files <GB> GB in <s> s (<files/s>, <MB/s>)`,
  followed by each run's train-loop s/step and GPU-h.

**Tests (`tests/test_p3_s2d.py`, 12):**
- **The local copy:** holds every file the runs read, byte-identical, counted by source.
- **Held-out:** no held-out file in the local copy (`check` counts it).
- **Refusal:** a copy that does not fit is refused before anything is written.
- **Wiring:** the runbook trains from the local copy.
- **gen_matched:** absolute paths are counted, not copied.
- **The earlier tests** are adapted: the fixture's COCO files now exist on disk.

**The local dry-run fixture `p3_s2d`:**
- The trainer's command line in each training log must show the local root.
- Staging must hash-verify all 9 Open Images train files.

**Local dry-run (`--ref :`): PASSED** (301 s, 2 sessions).
- Staging: 29 files (COCO 18, Open Images 9 hash-verified, gen_matched 2 already local).
- Every run trained from `dbp_s2d_corpus`.
- The THROUGHPUT line printed the staging rate and each run's s/step.

**Full suite: 1070 passed, 7 skipped, 0 failed** (5 min 22 s).
- All 7 skips are PENDING-GPU (`tests/test_gpu_smoke.py`).
- ruff: all checks passed.

**Hand-over waits for the tuning result.** The S2-D code zip is built from the commit that
adds `reports/p3_degradation_choice.json`, and dry-run again at that commit.

**Estimate for the staging step:** 17.5–20 GB at the session's read rate.
- About 5–20 min if reads behave as in session 5.
- Up to about 28–40 min at session 4's first-read cost: 78,221 files × 0.68–0.97 s
  / 32 threads, if the cost is per-request latency.
- ESTIMATED, NOT-MEASURED.

### P3-C — degradation tuning COMPLETE: the committed rule chose **p = 0.35**; S2-B+deg = `s2b_deg_p35` seed 0 — 2026-10-22

**Run:** Kaggle GPU T4 × 2, `dbp-code-042f6d9`, 2026-10-21 (log timestamps), notebook
`dbp-p3-tune`, first version.
- **Inputs:** the `dbp-p3-dev-panel` output and the session-5 `dbp-p3-train` output.
- **Archive:** `F:\thesis-2\p3_tune_reports.zip`, 11,841,903 bytes, sha256 `59ba03be…8d8b`,
  `testzip` clean, 422 entries, no unsafe path. Extracted to
  `F:\thesis-2\kaggle_raw\p3_tune_x\`.
- **Committed:** `reports/kaggle_20261021/p3_tune/`, and the result as
  **`reports/p3_degradation_choice.json`**, the file the S2-D runbook reads.

**Checks:**
- **Pre-flight PASSED:** 8 checks, including that every row of both derived lists resolves.
- **Derived lists:** `p3_tune_panel_rows.csv` 6,717 rows and `p3_tune_val_reals.csv` 491,
  both equal to the committed sha256.
- **Each of the three models** scored exactly those rows (`limit` null).
- **S2-B s0's report** is byte-identical to session 1's.
- **Carried score files** are identical to the committed ones.
- **Logs:** only the 11 known peft warnings per training log.
- **The rule re-applied locally** to the recorded per-model numbers
  (`p3_tune_degradation.choose`) gives the same choice. Each macro equals the mean of its
  parts.

**MEASURED, dev data (the rule's rows):**

| model | macro achieved FPR (E4, E5) at the COCO k = 0 threshold | E4 FPR | E5 FPR | macro AUC (4 unseen slices) | GPT-Image2 | Nano-Banana2 | E4 | E5 |
|---|---|---|---|---|---|---|---|---|
| `s2b` s0 | 0.6204 | 0.6851 | 0.5558 | 0.7709 | 0.6456 | 0.8176 | 0.9281 | 0.6921 |
| `s2b_deg_p70` s0 | 0.5542 | 0.6326 | 0.4759 | 0.7738 | 0.6481 | 0.7874 | 0.9260 | 0.7338 |
| `s2b_deg_p35` s0 | 0.5595 | 0.6447 | 0.4742 | 0.7745 | 0.6500 | 0.8024 | 0.9301 | 0.7157 |

- **Guard floor:** 0.7609 (S2-B s0 macro AUC − 0.01). **Both candidates pass.**
- **|Δ macro FPR|:** 0.0053, within the 1-point tie margin. **The committed rule chooses the
  milder p: `s2b_deg_p35_seed0`, p = 0.35.**
- **This is the pre-registered `S2-B+deg`.** S2-D uses `s2d_p35`. The real committed result
  resolves the S2-D queue to `s2d_p35:0, s2b_reals:0, s2d_p35:1` (checked locally with
  `p3_s2d_inputs.resolve_queue`).
- **Descriptive only:**
  - Degradation lowered the dev macro FPR by about 6 points at either p.
  - At p = 0.35: E5 AUC rose 2.4 points, and Nano-Banana2 fell 1.5.
  - The FPRs stay very high (0.47–0.69) at a COCO-only threshold. That is the problem H3
    and the union threshold address; it is not a tuning outcome.

**Training (MEASURED):**
- **Both candidates:** OK at step 5000 of the 6,000-step schedule, LR 6.712e-06.
  - val AUC: p70 0.99302, p35 0.99614.
  - GPU-h: 3.994 each.
- **Chain share:** 0.6975 (p70) and 0.3478 (p35). The four operations each take about
  1/4, and re-share about 1/2 of social.
- **Chain cost:** 16.4 and 8.0 ms per pair.
- **Train-loop s/step:** 2.8305 / 2.83348.
  - **Batch build: 2,216 / 2,280 ms of each step.** The step is dominated by reading
    images, not by the chain: 8 pairs × 8–16 ms ≈ 0.1 s.
- **Per 500-step interval (min):** 49.4, 32.5, 27.5, 24.5, 21.1, 19.1, 18.6, 18.0, 14.9, 14.1.
  - Again the falling first-read profile.
  - The two GPUs are within 0.1 min of each other.
- **Wall:** queue 4.0667 h; smoke 0.1333 h; S2-B s0 dev scoring 369 s.

**The first direct measurement of the first-read cost.**
- **S2-B s0, scoring files read for the first time:**
  - the 491 COCO val reals (dataset `report`): **0.32671 s/image**;
  - the 6,717 dev-panel rows (notebook output `dbp-p3-dev-panel`): **0.03107 s/image**.
- **The candidates, scoring the same files later in the session:** 0.0186–0.0201 s and
  0.0286–0.0287 s per image.
- **So:** about **0.31 s per first read** of a COCO-corpus file in this session, and
  almost none for the dev-panel notebook output.
- **Supports S2-D's staging.** Every file read once, in parallel, to local disk
  (`805ee7c`).

**Hardware (Cell 2, first log):**
- **CPU:** Intel Xeon @ 2.00 GHz, **4 logical cores**.
- **GPUs:** 2 × Tesla T4 at idle (300 / 1590 MHz SM, 405 / 5001 MHz memory, 10.45 / 9.56
  W of 70 W, 41 / 40 °C).
- **Not known:** whether earlier sessions ran on hosts with more cores (no hardware log
  before `fcc2ab4`).

**Where the models are:**
- S2-B+deg (`s2b_deg_p35_seed0`) and the p70 candidate are in the `dbp-p3-tune` notebook
  output.
- That output is attached at the final Panel B scoring.

### P3-C — S2-D hand-over at `4dfe54f`: dry-run PASSED, code zip — 2026-10-22

- **Local dry-run: PASSED** at `--ref HEAD` = `4dfe54f` (360 s, 2 sessions,
  `kaggle_run_p3_s2d.py`).
  - Build and check PASSED; staging hash-verified all 9 Open Images files.
  - Session 1 trained `s2d_p35:0` + `s2b_reals:0`; session 2 carried both and trained
    `s2d_p35:1`.
  - Every run trained from the local copy; THROUGHPUT printed.
- **Code zip:** `F:\thesis-2\kaggle_raw\dbp-code-4dfe54f.zip`, 13,114,423 bytes, sha256
  `6840d0ac18ac12566b412e45e3e68f59670711e2cafdb3e6c26428fa58041d4d`.
  - 598 entries, byte-identical to the commit; `COMMIT.txt` = `4dfe54f`.
  - It holds `reports/p3_degradation_choice.json` (`s2b_deg_p35_seed0`, 0.35) and the
    staging code.
- **Full suite** on the S2-D code (`805ee7c`): 1070 passed, 7 PENDING-GPU skips.
  - `4dfe54f` adds only reports and the log.
- **Estimate (ESTIMATED), per S2-D session** (session 2 trains one run):
  - **Staging:** 5–40 min.
  - **Training:** 5,000 steps at about 1.0–1.7 s/step once reads are local = 1.4–2.4 h.
    - The basis is the MEASURED warm intervals: S2-B s1 8.2 min / 500; the tuning
      candidates 14.1 min / 500 with degradation.
  - **Setup and smoke:** about 0.3–0.6 h.
  - **Total:** about 2–3.5 h, up to about 5 h if local reads are slower than assumed.

### P3-C — the final scoring BUILT (Panel B once per frozen model; H5); k-shot methods committed; P2 references prepared — 2026-10-22

**User, 2026-10-22:**
- Tuning accepted (p = 0.35); S2-D session 1 running.
- **Build, test and dry-run the final steps.** Nothing on Panel B runs until both S2-D
  seeds are done.
  - The final scoring runbook: every frozen model **once** on Panel B (+ luma), H5, and
    TrueFake if fetched.
  - Where the P2 references live (upload them from the local copy if only on
    `adeelshah29`).
  - A pre-flight that refuses unless every section-5 value is committed and the Panel B
    digest matches.
  - The TrueFake fetch status; a GPU estimate.
- **Decisions, asked and answered on 2026-10-22:**
  - H5 on `S2-D` **seed 0**, P2 `full_bank` and P2 P-A;
  - **test rows only** (8,655);
  - the perturbation definitions as proposed;
  - **two sessions** (clean, then robust).

**The 24 frozen models**, each with its source and the hash recorded for it. The
`snapshot_sha256` values come from the committed sidecars `reports/p3_heldout_scores/*`.
The LoRA and head values come from `reports/kaggle_20261021/p3_tune/p2_s2_scores_*.json`
and `reports/kaggle_20261005/p2_s2_scores_*.json`.

| model | source on Kaggle | step | recorded hash (committed) |
|---|---|---|---|
| `lolo_f1..f8_seed0` | dataset `dbp-p3-lolo-output` | N*(-f): 5000, except f6 4500 | snapshot digests f1 `f8a3c5bf…`, f2 `eaddfae7…`, f3 `cad1f6fc…`, f4 `ffcf62d0…`, f5 `8a410f10…`, f6 `117e2a98…`, f7 `8a27ee16…`, f8 `077bae61…` |
| `bank_1`, `bank_2/4/6_{near,far}`, `bank_1l` (seed 0) | notebook `dbp-p3-train` (session 5) | 5000 | `e4e8210a…`, `8358babe…`/`1443953b…`, `fcce08ac…`/`6831d377…`, `8ce750b4…`/`24448870…`, `d0b9dd76…` |
| `s2b_seed0` | `dbp-p3-train` | 5000 | LoRA `6b8c6e68…`, head `4d41d09a…` |
| `s2b_seed1` | `dbp-p3-train` | 5000 | none: first recorded by the pre-flight |
| `s2b_deg_p35_seed0` (S2-B+deg) | notebook `dbp-p3-tune` | 5000 | LoRA `7658a77b…`, head `80fe4a48…` |
| `s2b_reals_seed0`, `s2d_p35_seed0`, `s2d_p35_seed1` | notebook `dbp-p3-s2d` | 5000 | none: first recorded by the pre-flight |
| P2 `full_bank_seed0`, `p_a_sd1_seed0` | new dataset `dbp-p2-reference-models` | P2's selected | LoRA `4cd74c92…`/`f411fba0…`, head `ed47b140…`/`4fc35c3c…` |

**The P2 references live only on `adeelshah29`** (P2 Session D, the `phase-2` notebook).
- **Local copy:** `F:\thesis-2\kaggle_raw\P2_Training-results_x\`.
- **Re-hashed this session (`sha256sum`):** all four files equal the P2 scoring summaries'
  `lora_sha256` / `head_sha256`.
- **Prepared for upload:** `F:\thesis-2\kaggle_raw\dbp-p2-reference-models\` (4.6 MB).
  - It holds `<id>/head.pt`, `<id>/lora/adapter_config.json` and
    `<id>/lora/adapter_model.safetensors` for both models, and nothing else.
  - To be uploaded by the user as a PRIVATE dataset on `muhammadadeelshah`. These are our
    own weights, not licensed data.

**Built:**
- **`src/dbp/eval/kshot.py`** (+ `tests/test_p3_kshot.py`, 8): **the section-6.3 k-shot
  methods, committed before Panel B is scored** (section 5).
  - (Q) the binomial j: tests reproduce the frozen k = 50 → j 1 (P 0.0203; j = 2 gives
    0.1025), k = 100 → j 4 (0.0526), and k = 10 / 25 → j 1, bound not met (0.4586 /
    0.1424).
  - (S) the logit shift.
  - The draws: seeded per (pool, k), one draw at the whole pool.
- **`src/dbp/eval/perturb.py`** (+ `tests/test_p3_perturb.py`, 17): the 12 H5 conditions +
  clean. They are deterministic and size-preserving, with JPEG at 4:2:0.
  - **`P3_PLAN.md` §5 note of 2026-10-22** records the definitions, the rows, the models
    and the two sessions.
- **`scripts/p3_final_preflight.py`:** refuses to plan unless every condition holds.
  - **Section 5:** the degradation result, `N*`/`N*(-f)`/`N_max`, the Open Images split,
    and the k-shot j values. Each file's sha256 equals the expect file's (git bytes); the
    `s2b_deg`/`s2d` models must be the chosen p.
  - **Panel B:** the committed manifests' sha256 and the four frozen digests. Recomputed
    locally from the committed manifests: all `267d9fde…`, test `af544cc1…`, calibration
    `fd5fb95e…`, luma `41445030…`.
  - **All 24 models:** weights present, the training report finished at the right step,
    and every recorded hash matched; the others are recorded.
  - **Staging:** every image (Panel B, luma, the 491 + 400 threshold reals) is copied once
    to local disk, sha256 checked in the same read.
  - **Once only:** complete stages are carried forward; an interrupted one is listed in
    `incomplete_previous`.
- **`scripts/p3_final_score.py`:** one worker per T4 under `p2_run_shards.py` (fail-fast).
  - It re-hashes the weights and loads them as every S2 scorer does.
  - **clean:** Panel B + luma + threshold reals.
  - **robust:** the Panel B test rows × 12 conditions.
  - It refuses a model whose summary exists ("Panel B is scored once per frozen model").
  - The scores go to `reports/p3_final_scores/`, the summaries to `reports/p3_final/`.
- **`env/kaggle_run_p3_final.py` + `env/p3_final_expect.json`:**
  - inputs found by contents;
  - a hardware log;
  - pre-flight 1 (environment, then the plan);
  - pre-flight 2: a 2-row smoke of every model due;
  - scoring;
  - a report that cannot fail (`p2_report.py --stage final`, new);
  - the verdict.
- **`tests/test_p3_final.py`** (8):
  - section 5 passes, and refuses a wrong p, a wrong N*(-f) or a wrong k-shot rule;
  - Panel B refuses a changed manifest or digest;
  - a model refuses a hash mismatch, an unfinished report, or a missing source;
  - carry and interruption;
  - every hash in the expect file equals its committed record;
  - the runbook's order.

**The local dry-run fixture `p3_final`:**
- **Models:** one trained S2-B snapshot, copied into all 24 models in the five sources.
- **Panel B:** a fixture Panel B (13 rows + 3 luma) with its own digests.
- **Threshold reals:** fixture threshold reals; fixture section-5 records.
- **Two sessions, and a check** that a second scoring of a complete model is refused.

**Defects caught by the dry-run, before any hand-over:**
1. **The head ran outside `torch.no_grad()`** in the new worker. The 2-row smoke failed
   (`Can't call numpy() on Tensor that requires grad`). The forward now follows
   `p2_s2_infer.py`.
2. **The fixture also mounted the base fixture's P2 dev panel,** which holds a second
   `e7_dda_coco_luma`. The runbook refused the ambiguity, as it should. The fixture now
   removes it; the final session attaches no dev panel.

**Local dry-run (`--ref :`): PASSED** (491 s).
- **Session 1:** pre-flight PASSED (19 files staged and hash-verified); 24 models scored
  clean.
- **Session 2:** all 24 carried forward; robust for `s2d_p35_seed0`, `full_bank_seed0` and
  `p_a_sd1_seed0` × 12 conditions × 10 test rows.
- **Once only:** a re-score was refused (exit 1).

**Full suite: 1110 passed, 7 skipped, 0 failed** (10 min 58 s).
- All 7 skips are PENDING-GPU.
- ruff: all checks passed.

**TrueFake fetch:**
- **Status:** built (`408c1f5`), never run.
- **Local dry-run: PASSED** at `4dfe54f` (13 s); the runbook is unchanged since `408c1f5`.
- **TrueFake scoring** is built after the fetch: its inputs and the pre-registered
  disjointness check (sha256 + pHash ≤ 6 against our training reals) depend on what the
  fetch delivers.

**GPU estimate (ESTIMATED), quota = wall time (I-042):**
- **Session 1, clean:** about 1.3–2.5 h.
  - 24 models × 10,996 images at the MEASURED warm S2 rate 0.019–0.031 s/image =
    1.4–2.3 device-h, two T4s in parallel.
  - Plus staging (about 2 GB) and 24 model loads.
- **Session 2, robust:** about 1.5–3.5 h.
  - 3 × 12 × 8,655 = 311,580 images at 0.02–0.05 s/image. The CPU cost of the
    perturbations on 4 cores is NOT-MEASURED.
  - Two of the three models share one T4.
- **Later, TrueFake:** 4 models × 16,000 images at 0.02–0.064 s/image = 0.36–1.14
  device-h, about 0.5–1.2 h with staging.
- **Total:** about 3.3–7.2 h.

### P3-C — final scoring and TrueFake fetch: hand-over at `e2561f9` — 2026-10-22

**A third defect caught by the dry-run at HEAD `860521c`, before any hand-over:**
- **The defect:** the once-only guard also refused the **smoke**. An earlier smoke's
  summary in the same scratch root blocked a second one.
  - The dry-run's scratch outlives a run; on Kaggle `/tmp` starts empty.
- **The fix (`e2561f9`):** only a real scoring (no `--limit`) is refused
  (`refuses_rescore`), with a test. `tests/test_p3_final.py` now has 9 tests, all
  passing.

**Local dry-runs at `--ref HEAD` = `e2561f9`:**
- **`p3_final`: PASSED** (468 s).
  - **Session 1:** 24 models clean.
  - **Session 2:** 24 carried forward; robust for 3 models × 12 conditions.
  - **A re-score** was refused (exit 1).
- **`p3_truefake`: PASSED** (9 s); 8 members byte-identical.

**Code zip:** `F:\thesis-2\kaggle_raw\dbp-code-e2561f9.zip`, 13,148,386 bytes, sha256
`14078afb6a63d43a1c7dd15a7bc5d7df6430fa665d98b56455eda93b54bf1efa`.
- 607 entries, byte-identical to the commit; `COMMIT.txt` = `e2561f9`.
- **It serves the TrueFake fetch now and the final scoring later.**
- `dbp-code-860521c.zip` is superseded.

**The final scoring does not run yet.** Its pre-flight refuses until S2-D seed 1 exists,
by design. Nothing on Panel B runs until both S2-D seeds are done (user).

### P3-C — TrueFake sample FETCHED: 16,000 / 16,000 members, no refusal — 2026-10-23

**Run:** Kaggle CPU, `dbp-code-e2561f9`, 2026-10-22 21:05–22:18 (log timestamps). The
source is the user's pasted log; the status file and the fetch manifest are not yet
copied back.
- **Pre-flight PASSED:** 2 checks; `truefake_sample.csv` has 16,000 rows and the committed
  sha256.
- **Index verified:** 600,876 entries; central directory sha256 `869a102f…dc0b`; size
  694,732,361,838 B. Both equal the committed values.
- **Members:** 16,000 of 16,000 fetched by range, each CRC-checked. `outcome: COMPLETE`,
  `refusal: null`.
- **Measured:**
  - `bytes_compressed_fetched` 8,574,203,526;
  - `seconds` 4,166.8 (69.4 min), at a steady 500 members per 2.1–2.3 min;
  - the estimate was 0.5–1.5 h.

**Next:**
- The user publishes `truefake_sample/` as a PRIVATE dataset.
- `reports/truefake/` (status and the per-member fetch manifest: names, sizes, hashes; no
  images) is copied back and committed.
- TrueFake scoring is then built, with the pre-registered disjointness check, after the
  final clean stage.

### P3-C — S2-D session 1 COMPLETE: `s2d_p35` seed 0 and `s2b_reals` seed 0; staging removed the first-read cost — 2026-10-24

**Run:** Kaggle GPU T4 × 2, `dbp-code-4dfe54f`, 2026-10-23 05:44–07:05 (log timestamps),
account `muhammadadeelshah`.
- **Earlier attempts** (2026-10-22/23) failed with `ERRORED_MOUNTING_DATASET` on the input
  `notebooks/adeelshah29/dbp-p3-reconstruct`. Per the user, one attempt hung after about
  an hour with no GPU time charged; that log was not retrieved. This attempt mounted
  normally.
- **Inputs:** the Open Images reconstructions were read from `adeelshah29`'s
  `dbp-p3-reconstruct` notebook output, shared to `muhammadadeelshah` (user, 2026-10-23).
- **The user also stated (2026-10-23)** that one of the two accounts was obtained from
  another person. Which one is an **open question**.
  - CLAUDE.md §13 keeps all thesis work on the user's own account, because Kaggle allows
    one account per person.
  - GPU planning stays on one account.
- **Archive:** `F:\thesis-2\p3_s2d_reports.zip`, 11,922,555 bytes, sha256 `7fd4535b…4f12`,
  `testzip` clean, 457 entries, no unsafe path. Extracted to
  `F:\thesis-2\kaggle_raw\p3_s2d_session1_x\`.
- **Committed:** `reports/kaggle_20261023/p3_s2d_session1/`.

**Checks:**
- **Pre-flight PASSED:** 6 checks.
- **Build PASSED:**
  - the tuning result sha256 `6cda0151…`, chosen `s2b_deg_p35_seed0`, p 0.35;
  - queue `s2d_p35:0, s2b_reals:0, s2d_p35:1`.
- **Held-out check PASSED:** 0 held-out rows in all three merged files and **0 held-out
  files in the local copy**.
- **Both trainers logged** "66312 train pair rows over **5526** reals". This equals the
  builder's pool: **1,926 COCO + 3,600 Open Images, Open Images share 0.651466**. Each
  trainer's command line shows the local root.
- **Logs:** only the 11 known peft warnings per training log.

**Staging, MEASURED:**
- **What was staged:** 80,709 files, 18,985,165,518 B (17.68 GB):
  - Open Images 46,800 / 10,460,105,487 B;
  - COCO 31,421 / 8,320,413,881 B;
  - gen_matched 2,488 / 204,646,150 B, already local and not copied.
- **Speed:** a 12.6 s stat pass, then **248.0 s** to read, check and copy (325.4 files/s,
  73.0 MB/s).
- **Integrity:** **46,800 hash-verified, 0 mismatched, 0 missing.**
- **Host:** RAM MemTotal 31.35 GB; Cached 5.77 → 26.66 GB after staging. Free on `/tmp`:
  1,048.72 → 1,031.07 GB.
- **The earlier size estimate:** 17.5–20 GB; COCO measured 8.32 GB against the estimated
  6.9–9.4 GB.

**Training (MEASURED):**

| run | train-loop s/step | GPU-h | val AUC at 5000 | batch build |
|---|---|---|---|---|
| `s2d_p35` s0 | 0.80041 | 1.165 | 0.99453 | 268 ms |
| `s2b_reals` s0 | 0.81432 | 1.188 | 0.99826 | 201 ms |

- **Both runs:** OK at step 5000 of the 6,000-step schedule, LR 6.712e-06. No held-out
  lineage, so no per-image file, as for S2-B.
- **Degradation (`s2d_p35`):**
  - chain share 0.3478 over 40,000 pairs; 7.0 ms per pair;
  - the operation counts are identical to the tuning candidate `s2b_deg_p35`, as the
    separate stream `[0, 417]` intends.
- **Per 500-step interval:**
  - `s2d_p35` 7.0, 7.0, 7.0, 7.0, 6.9, 7.0 … 7.0 min;
  - `s2b_reals` 7.0 … 7.2 min.
- **Flat from step 1:** the first-read profile of sessions 2–4 and tuning (first interval
  14.6–97.6 min) is gone.
  - Tuning measured 2,216–2,280 ms of batch build per step; this session 201–268 ms.
  - The staging (user request of 2026-10-21) did what it was for.
- **Queue wall:** 1.1917 h (device 2.3667); smoke 0.0333 h.
- **Hardware:** Xeon @ 2.00 GHz, 4 logical cores; 2 × T4 at idle at start.

**Queue:** 2 of 3 complete. Remaining: `s2d_p35` seed 1 (session 2, this notebook, session
1's output attached).

### P3-C — Kaggle accounts: `muhammadadeelshah` is the user's own; `adeelshah29` was obtained from another person — 2026-10-24

**User, 2026-10-23/24:**
- The account the user created is **`muhammadadeelshah`**. `adeelshah29` was obtained from
  another person.
- Earlier records calling `adeelshah29` "also the user's own account" (CLAUDE.md §13, this
  log) were therefore wrong. The CLAUDE.md wording awaits the user's approval.

**What lives where (from this log):**
- **On `adeelshah29`:**
  - P2's sessions: the `phase-2` notebook, the P2 S2 runs, the original dev panel;
  - the step-4a reconstruction output `dbp-p3-reconstruct`, read by S2-D through a share.
- **On `muhammadadeelshah`:** every other P3 run and dataset: Panel B, Open Images, LOLO,
  `dbp-p3-train`, `dbp-p3-tune`, the rebuilt dev panel, `dbp-p3-s2d`, the TrueFake sample.

**Consequences:**
- **GPU work stays on `muhammadadeelshah` only** (§13: one account per person).
- **The final scoring attaches nothing from `adeelshah29`:**
  - the P2 reference weights come from `dbp-p2-reference-models` on
    `muhammadadeelshah`, hash-equal to the P2 records;
  - the Open Images inputs it reads (400 held-out reals) are on `dbp-p3-open-images`.
- **The reconstructions S2-D reads from `adeelshah29`** are checked file by file against the
  committed manifest digest (46,800 verified in session 1). The result does not depend on
  which account served the bytes.

### P3-C — S2-D session 2 COMPLETE: `s2d_p35` seed 1; every P3 model is trained — 2026-10-27

**Run:** Kaggle GPU T4 × 2, `dbp-code-4dfe54f`, 2026-10-24 10:45–11:58 (log timestamps),
`muhammadadeelshah`, with session 1's output attached.
- **Archive:** `F:\thesis-2\p3_s2d_reports_phase-2.zip`, 11,917,503 bytes, sha256
  `6e403e5c…22ba`, `testzip` clean, 455 entries, no unsafe path.
- **Committed:** `reports/kaggle_20261024/p3_s2d_session2/`.

**Checks:**
- **Pre-flight, build and held-out check PASSED:** 0 held-out rows in every merged file and
  in the local copy.
- **The merged files' sha256 and both pools** are identical to session 1's.
- **Staging:** 80,709 files, 17.68 GB in **107.7 s** (749.3 files/s, 168.1 MB/s); 46,800
  hash-verified, 0 mismatched.
- **The chooser:** `s2d_p35:0` and `s2b_reals:0` complete; this session trained
  `s2d_p35:1`.
  - The carried reports are byte-identical to session 1's.
- **Logs:** only the 11 known peft warnings.

**MEASURED, `s2d_p35` seed 1:**
- **Training:** OK at step 5000 of the 6,000-step schedule, LR 6.712e-06.
  - train-loop **0.83421 s/step**; GPU-h 1.212; **val AUC 0.99423** (seed 0: 0.99453);
  - batch build 255 ms.
- **Degradation:** chain share 0.3458 over 40,000 pairs; 7.02 ms per pair.
  - The stream is seeded per seed, so the counts differ from seed 0's.
- **Pool:** "over 5526 reals", as the builder said.
- **Intervals:** flat at 7.2–7.3 min per 500 steps.
- **Wall:** queue 1.2167 h.

**All P3 training is complete:** LOLO 8, non-LOLO 10, the tuning candidates, and S2-D 3.

**Next, before the final-scoring hand-over (user, 2026-10-23):** record the hashes of
`s2b_seed1`, `s2b_reals_seed0`, `s2d_p35_seed0` and `s2d_p35_seed1` from their weight files
in `env/p3_final_expect.json`, so the pre-flight checks all 24 models.

### P3-C — the last 4 model hashes recorded; final-scoring hand-over at `0af33cf` — 2026-10-27

**User, 2026-10-27:**
- 8ee02fe approved.
- 12 weight files downloaded to `F:\thesis-2\kaggle_raw\final_hashes\<run>\`.
- CLAUDE.md §13 corrected (`ba022d3`, the user's wording conditions).

**The weight files:** placed flat in each run's folder by the user, then moved by Claude into
`lora\` at the user's instruction.
- **The heads arrived as `head.zip`:** the browser's name for a PyTorch `.pt`, which is a
  zip archive.
  - Each was checked to be a valid PyTorch file loaded with `weights_only=True` (`fc.weight`
    1 × 1536, `fc.bias` 1, the P2 heads' structure), then renamed `head.pt`. sha256 is
    unchanged by the rename.
- **Sizes:** no "(1)" duplicates; every adapter 2,371,648 B; every head 8,109 B.
- **Hashes (`sha256sum`, this session):**

| model | LoRA `adapter_model.safetensors` | `head.pt` |
|---|---|---|
| `s2b_seed1` | `10441dbd…7fdb` | `c66ebe49…6cdd` |
| `s2b_reals_seed0` | `4018105f…7079` | `29ca8244…596d` |
| `s2d_p35_seed0` | `d7148267…1689` | `638447e6…7b27` |
| `s2d_p35_seed1` | `ad7aaa41…2c05` | `937491ef…0639` |

- **Committed** to `env/p3_final_expect.json` (`0af33cf`). The pre-flight now **checks all
  24 models, records none**.
  - They are recorded as LoRA + head sha256, as for S2-B s0 and S2-B+deg. The snapshot
    digest also covers peft's `lora/README.md`, which was not downloaded.
  - `tests/test_p3_final.py`: every model carries a record, and the 8 LoRA hashes are
    distinct.
  - The `adapter_config.json` files differ in sha256 across runs, but not in meaning (peft
    writes the target-module set in process order). They are not checked.

**Checks:**
- **Local dry-run: PASSED** at `--ref HEAD` = `0af33cf` (266 s).
  - **Session 1:** 24 models clean.
  - **Session 2:** 24 carried; robust for 3 models × 12 conditions.
  - **A re-score** was refused.
- **Full suite: 1111 passed, 7 skipped** (PENDING-GPU), 0 failed; ruff clean.

**Code zip:** `F:\thesis-2\kaggle_raw\dbp-code-0af33cf.zip`, 13,186,825 bytes, sha256
`349efb521cca7fe41e6e6bcd78d28a42b6176819a4852cee0041e1a6f729782d`.
- 635 entries, byte-identical to the commit; `COMMIT.txt` = `0af33cf`.
- It supersedes `dbp-code-e2561f9` for the final scoring.

### P3-C — FINAL SCORING session 1 (clean) COMPLETE: all 24 frozen models scored once on Panel B — 2026-10-28

**Run:** Kaggle GPU T4 × 2, `dbp-code-0af33cf`, notebook `dbp-p3-final`, account
`muhammadadeelshah`, 2026-10-27 09:12–09:50 (worker log timestamps).
- **Inputs (Cell 1, as printed):** `dbp-p3-panel-b`, `report`, `dbp-p3-open-images`,
  `dbp-p3-lolo-output` (datasets); `dbp-p3-train`, `dbp-p3-tune`, `dbp-p3-s2d` (notebooks);
  `dbp-p2-reference-models` (dataset). **All are on `muhammadadeelshah`; nothing came from
  `adeelshah29`.**
- **The reference-model dataset:** its first upload carried an extra top-level folder.
  - The CPU check (Cells 1–2) refused it: "expected exactly one P2 reference models,
    found []".
  - The user re-uploaded the two model folders at the top level. The CPU check then found
    all 9 inputs.
- **Archive:** `F:\thesis-2\p3_final_reports.zip`, 16,518,820 bytes, sha256 `89151cab…acb9`,
  `testzip` clean, 632 entries, no unsafe path.

**Pre-flight (`p3_final_plan.json`): PASSED, `problems` empty.**
- **Section 5:**
  - `s2b_deg_p35_seed0`, p 0.35;
  - N* 5000, N_max 6000;
  - N*(-f) 5000 for every fold except f6 at 4500;
  - Open Images 3,600 / 400;
  - k-shot j: 10 → 1 (not met), 25 → 1 (not met), 50 → 1, 100 → 4.
- **Panel B:** 9,405 rows + 700 luma. Digests **`267d9fde…`, test `af544cc1…`, calibration
  `fd5fb95e…`, luma `41445030…`, equal to the frozen ones.**
- **Models:** **all 24 equal to their committed records** (none recorded for the first
  time).
  - The 4 hashes committed from the downloaded files on 2026-10-27 matched on Kaggle.
- **Staging:** 10,996 images (Panel B + luma + 491 + 400 threshold reals), 1.818 GB, read,
  checked and copied in 37.4 s. **10,996 hash-verified, 0 mismatched.**
- **Once only:** stage `clean`, 24 due; nothing carried; nothing interrupted.

**Scoring (MEASURED):**
- **Smoke:** COMPLETE.
- **Queue:** COMPLETE, wall **0.6833 h** (device 1.3667 h), 12 models per T4, both workers
  exit 0. No error or traceback in either worker log.
- **Per model:** 195.8–204.8 s for 10,996 images, about 0.018 s/image.
- **Checked locally, every model:**
  - the summary's LoRA and head sha256 equal the plan's;
  - `limit` null; the Panel B digest recorded;
  - each of the 3 score files' sha256 equals the summary's;
  - its image ids equal its list's, in order (9,405 / 700 / 891);
  - every score is finite.
- **Estimate vs measured:** estimated 1.3–2.5 h; measured 0.68 h of queue.

**Committed:**
- `reports/p3_final_scores/<model>/clean/{panel_b,luma,threshold_reals}.csv` (72 files,
  6.8 MB) and `reports/p3_final/<model>_clean.json` (24).
- The plan, statuses, hardware, worker logs and the three scoring lists under
  `reports/kaggle_20261027/p3_final_session1/`.
- **`.gitattributes`** gains `-text` for the score files and the lists, as for
  `p3_heldout_scores`, so the committed bytes keep the sha256 recorded on Kaggle.

**Panel B has now been scored once per frozen model (pre-registration section 2).** No
section-5 value can change from here.

**Next:**
1. Final scoring session 2 (robust, H5): the same notebook with this version attached.
2. TrueFake scoring: after the user's `reports/truefake/` files.
3. The pre-registered analyses on CPU (H1-b, H2, H3, H3-k, H4, H5).

### P3-C — the Panel B analysis BUILT and tested on synthetic scores only (no real score read) — 2026-10-28

**User, 2026-10-28:**
- **Build the analysis now:** H1-b, H2, H3 with the union threshold, H3-k, H4 and H5.
- **Rules:**
  - exactly the frozen pre-registration, with anything extra in a separate "exploratory"
    section;
  - develop and test **only on synthetic or fixture scores**, never computing a metric on
    the real Panel B scores;
  - commit, then run once on the committed scores, reporting every verdict as it comes
    out;
  - H5 runs when the robust scores arrive.
- Session 2 (robust) is running.

**`scripts/p3_analyze.py`:**
- **Section 4:** per slice AUC/AP; TPR at the k = 0 union threshold; slice-calibrated
  TPR@5%FPR; per pool achieved FPR at the union and the COCO-only thresholds; E8 TPR
  only (fakes-only).
- **Thresholds:** k = 0 = `calibrate_on_reals` at 5% on the 891 threshold reals (491
  COCO validation + 400 held-out Open Images); COCO-only on the 491.
- **H1-b:**
  - folds 1–3 on their E7 subsets, each ≥ 0.85, 97.5% CIs;
  - FLUX.1 standard beside fold 3 as an upper bound (§6d);
  - the §7 claim needs H1-a and ≥ 2 of 3 folds.
- **H3:** E4 and E5 test reals at the union threshold, both S2-D seeds.
  - Criteria: macro ≤ 15%; each pool ≤ 25%; a paired relative reduction vs P2 `full_bank`
    ≥ 50% with its 97.5% CI excluding 0.
  - Descriptive: ≤ 10%, and the three-step table (E3 included).
- **H3-k:**
  - `dbp.eval.kshot` (Q) and (S) on each view's calibration pool, at the frozen k values
    and 1,000 draws;
  - single-draw cells with a bootstrap CI over the test reals;
  - mean FPR, share ≤ 7.5%, TPR and ECE (15 bins);
  - at k = 50, ≥ 80% on E4 and E5.
- **H2:** S2-D − `full_bank` macro AUC over the 4 unseen slices; 95% lower bound > −2 points,
  both seeds; also without E3, with both verdicts stated.
- **H4:** BANK-8 (= S2-B s0) − BANK-1, ≥ +2 points with the 95% CI excluding 0.
  - The near and far curves with CIs, and each run's held-out-lineage validation AUC from
    its committed sidecar.
  - BANK-1L descriptive.
- **H5:** S2-D s0 under `jpeg50` and `resize050` within 5 points of clean (test rows).
  The 12-condition curves for S2-D s0, `full_bank` and P-A.
- **Bootstrap (section 4):**
  - 2,000 resamples, seed 2026, a fresh generator per analysis;
  - per resample each real pool drawn once (shared by its slices and by every model),
    then each fake slice;
  - paired comparisons; thresholds fixed;
  - percentile 95%, or 97.5% for H1-b and H3.
- **`INTERPRETATIONS`:** where the frozen text leaves a detail unstated, the reading is a
  named entry in the code and in the output. It is fixed **before any real number** and
  awaits the user's confirmation before the single real run.
- **Exploratory:** empty at this commit.

**`tests/test_p3_analysis.py` (15), synthetic scores only:**
- The frozen constants, and the slot names against the committed Panel B *list* (rows, no
  scores).
- Score files must match their list.
- Calibration and threshold reals never enter the test pools.
- A paired bootstrap of identical models has zero width and is deterministic.
- The PASS and FAIL paths of H1-b, H3, H2, H4, H3-k and H5; the (S) shift; ECE; and the
  description.
- ruff: all checks passed.

**No metric has been computed on the real Panel B scores.**

### P3-C — the analysis interpretations CONFIRMED by the user, before the run — 2026-10-28

Asked after `c88a806`, before any real Panel B metric existed. The user chose each
recommended reading, all as already coded:
1. **"value with its CI excluding 0"** (H3 reduction ≥ 50%, H4 gain ≥ +2 points): the point
   estimate must reach the value, and the CI's lower bound must be > 0.
2. **H3-k:** a method passes only if it passes on **both** S2-D seeds.
3. **H3-k (S):** the shift's "validation reals" are the **891** threshold reals.
4. **H5:** |macro AUC(condition) − macro AUC(clean)| ≤ 0.05.

**Also coded and stated:**
- H1-b, H3 and H5 thresholds are judged on point estimates, with CIs reported;
- "judged on E4 and E5" means each pool reaches 80%;
- H3-k ECE is over the view's test reals and fakes.

### P3-C — PANEL B RESULTS (clean stage): H1-b PASS (3/3), H3 FAIL, H3-k PASS (Q and S), H2 FAIL, H4 PASS; H5 pending — 2026-10-28

**The single run.**
- **Code and inputs:** `scripts/p3_analyze.py` at `41fee6d`, on the committed clean-stage
  scores (`reports/p3_final_scores/`, `4332110`). The inputs were unchanged against HEAD.
- **Run:** 2026-10-28 00:22–00:26 (3 min 41 s).
- **Output:** `reports/p3_results/p3_analysis_clean.json`, 138,960 bytes, sha256
  `bdd090c6…0120`.
- **Settings:** bootstrap 2,000 resamples, seed 2026; the four confirmed interpretations
  recorded in the output.
- **Every verdict below is as computed. Nothing was re-run or adjusted.**
- **Cross-check:** independently recomputed afterwards (scikit-learn AUC, plain NumPy
  threshold): the union thresholds, the E4/E5 FPRs, the 4-slice macro AUCs and fold 3's
  luma AUC equal the analysis's.

**H1-b (primary; 97.5% CIs; each subset ≥ 0.85): PASS on 3 of 3 folds.**

| fold | subset | AUC [97.5% CI] |
|---|---|---|
| 1 | sd-vae-ft-ema | 0.9985 [0.9963, 0.9998] |
| 1 | sd-vae-ft-mse | 0.9984 [0.9962, 0.9998] |
| 1 | stable-diffusion-2-1 | 0.9979 [0.9955, 0.9995] |
| 2 | sdxl-vae | 0.9977 [0.9948, 0.9995] |
| 3 | stable-diffusion-3.5-large | 0.9859 [0.9778, 0.9927] |
| 3 | FLUX.1 **luma** | 0.8994 [0.8705, 0.9253] |
| 3 (upper bound, §6d) | FLUX.1 standard | 0.9741 |

- With H1-a PASS (2026-10-17), the section-7 claim **"S2 on a decoder bank generalises to
  unseen decoder lineages" is supported** (H1-a + H1-b on ≥ 2 of 3 folds).

**H3 (primary; k = 0 union threshold; E4, E5 test reals 392 / 2,000): FAIL.**

| | S2-D s0 | S2-D s1 | P2 `full_bank` |
|---|---|---|---|
| union threshold | 0.3929 | 0.3663 | 2.4845 |
| FPR E4 | 0.1582 | 0.1633 | 0.1582 |
| FPR E5 | 0.1205 | 0.1185 | 0.1370 |
| macro FPR | **0.1393** | **0.1409** | 0.1476 |
| relative reduction vs `full_bank` [97.5% CI] | 0.0559 [−0.0972, 0.1990] | 0.0454 [−0.1013, 0.1775] | — |

- **Criteria, each seed:**
  - macro ≤ 15%: **met** (0.139, 0.141);
  - no pool above 25%: **met**;
  - reduction ≥ 50%: **not met**;
  - reduction CI excluding 0: **not met**.
- **H3 fails on the reduction criterion.**
- **Descriptive:** macro ≤ 10% not reached.
- **Three-step table (FPR):**

| pool | `full_bank` COCO-only threshold | `full_bank` union threshold | S2-D union (s0 / s1) |
|---|---|---|---|
| E3 (descriptive) | 0.3719 | 0.0661 | 0.1322 / 0.0579 |
| E4 | 0.6276 | 0.1582 | 0.1582 / 0.1633 |
| E5 | 0.4855 | 0.1370 | 0.1205 / 0.1185 |
| macro E4/E5 | 0.5565 | 0.1476 | 0.1393 / 0.1409 |

- **Reading:** the threshold effect (union threshold, 55.7% → 14.8%) does almost all the
  work; the model effect (S2-D over `full_bank` at the same threshold) is small.

**H3-k (secondary; k = 50; ≥ 80% of draws ≤ 7.5% on E4 and E5, both seeds): (Q) PASS,
(S) PASS.**

| seed | pool | k = 0 FPR / TPR | (Q) k = 50: share ≤ 7.5% / mean FPR / mean TPR | (S) k = 50: share / mean FPR / mean TPR |
|---|---|---|---|---|
| s0 | E4 | 0.158 / 0.869 | 0.987 / 0.006 / 0.392 | 1.000 / 0.017 / 0.617 |
| s0 | E5 | 0.121 / 0.275 | 0.963 / 0.032 / 0.100 | 1.000 / 0.024 / 0.079 |
| s1 | E4 | 0.163 / 0.867 | 0.993 / 0.010 / 0.417 | 1.000 / 0.020 / 0.616 |
| s1 | E5 | 0.119 / 0.344 | 0.968 / 0.031 / 0.113 | 1.000 / 0.023 / 0.096 |

- The FPR is controlled, at a large TPR cost on E5 (k = 0 TPR 0.28–0.34 → 0.08–0.11).
- E3 and E7 are descriptive, in the JSON.

**H2 (secondary; S2-D − `full_bank` macro AUC; non-inferior if the 95% lower bound > −2
points, both seeds): FAIL.**
- **4 unseen slices:**
  - s0: −0.0082 [−0.0271, 0.0111];
  - s1: −0.0032 [−0.0218, 0.0160].
  - Neither lower bound is above −0.02.
- **Without E3:**
  - s0: −0.0141 [−0.0214, −0.0072], fail;
  - s1: −0.0006 [−0.0088, 0.0071], non-inferior.
  - FAIL, since both seeds are required.
- **The two verdicts do not differ.**
- **Point estimates** are within 1.5 points; the CIs reach just past −2.

**H4 (secondary; BANK-8 − BANK-1, ≥ +2 points with the 95% CI excluding 0): PASS.**
- **Gain:** +0.1364 [0.1161, 0.1588].
- **Curve (macro AUC over the 4 unseen slices [95% CI]; held-out-lineage validation AUC):**
  - **near:** k1 (1 dec) 0.6268 [0.5943, 0.6583], val 0.927; k2 (5) 0.7180, 0.914; k4 (8)
    0.7332, 0.920; k6 (10) 0.7655, 0.997; k8 (12) 0.7632.
  - **far:** k2 (4) 0.6914, 0.918; k4 (6) 0.7213, 0.993; k6 (9) 0.7555, 0.999; k8 0.7632.
  - **BANK-1L (descriptive):** 0.6974, val 0.927.

**H5: NOT-RUN.** The robust-stage scores (final scoring session 2) are not in yet.

**Section-7 claims, from these verdicts:**

| claim | needs | status |
|---|---|---|
| "generalises to unseen decoder lineages" | H1-a + H1-b ≥ 2/3 | **supported** |
| "deployable without target data" | H3 + H2 | **not supported** (H3 FAIL, H2 FAIL) |
| "a few target-domain reals suffice" | H3-k at k = 50 | **supported** (Q and S), with the TPR cost stated |
| "the bank beats one decoder on lineages neither saw" | H1-c | supported (2026-10-17) |
| "more decoders help" | H4 | **supported** |

- **The pre-registered failure path applies** (§7: H3 fails but H3-k passes): the
  deployment recommendation becomes **"S2-D plus k-shot calibration"**. This is stated as
  the result, not as a rescue.

### P3-C — FINAL SCORING session 2 (robust, H5) COMPLETE: 3 models × 12 conditions — 2026-10-28

**Run:** Kaggle GPU T4 × 2, `dbp-code-0af33cf`, notebook `dbp-p3-final` with session 1's
output attached, 2026-10-27 19:13–21:40 (worker log timestamps), `muhammadadeelshah`.
- **Archive:** `F:\thesis-2\p3_final_reports_phase-2.zip`, 21,153,272 bytes, sha256
  `e39f35fd…6276`, `testzip` clean, 674 entries, no unsafe path.
- **Committed:**
  - `reports/p3_final_scores/<model>/robust/<condition>.csv` (36);
  - `reports/p3_final/<model>_robust.json` (3);
  - `reports/kaggle_20261027/p3_final_session2/`.

**Checks:**
- **Pre-flight PASSED** again: Panel B digest `267d9fde…`; 10,996 images re-verified in
  56.0 s, 0 mismatched; all 24 model hashes matched.
- **Stage `robust`:** due `s2d_p35_seed0`, `full_bank_seed0`, `p_a_sd1_seed0`.
  - The clean stage was carried for all 24.
  - **All 72 carried clean score files and the 24 clean summaries are byte-identical to the
    committed ones.**
  - Nothing was interrupted. The scoring lists are byte-identical to session 1's.
- **Each robust summary:**
  - its LoRA and head sha256 equal the plan's; `limit` null;
  - its 12 conditions are the plan's;
  - each file's sha256 equals the summary's;
  - its ids are the 8,655 Panel B test rows, in order; every score is finite.
- **Pillow:** 13.0.0 (recorded, as the perturbations are deterministic per Pillow build).
- **Logs:** no error or traceback.

**MEASURED:**
- **Queue:** COMPLETE, wall **2.525 h** (device 5.0501). The estimate was 1.5–3.5 h.
- **Split:** GPU 0 `s2d_p35_seed0` + `p_a_sd1_seed0`; GPU 1 `full_bank_seed0`.
- **Per condition:** 0.023–0.110 s/image (the image work on 4 CPU cores dominates).
- **Per model:** 5,322 / 5,214 / 3,741 s.

### P3-C — H5 RESULT: FAIL (JPEG q50 −13.3 points; resize 0.5 −4.8 points) — 2026-10-28

**The single H5 run.**
- **Code and inputs:** `scripts/p3_analyze.py --robust` at `4f21607`. The analysis code is
  unchanged since `41fee6d` (`git diff --stat` empty). Inputs: the committed clean and
  robust scores.
- **Run:** 2026-10-28 10:38–11:00 (21 min 53 s).
- **Output:** `reports/p3_results/p3_analysis_full.json`, 148,880 bytes, sha256
  `81b8c506…7383`.
- **The clean-stage hypotheses and the descriptive tables** in this output are
  **identical** to `p3_analysis_clean.json`: deterministic, recomputed, not re-decided.
- **Cross-check:** the two decisive macro AUCs recomputed independently (scikit-learn)
  match: 0.6219 and 0.7074.

**H5 (S2-D seed 0, macro AUC over the 4 unseen slices, Panel B test rows; within 5 points of
clean): FAIL.**

| condition | macro AUC | − clean (0.7553) | within 5 points |
|---|---|---|---|
| JPEG q50 | 0.6219 | **−0.1334** | **no** |
| resize-and-restore 0.5 | 0.7074 | −0.0480 | yes |

**The 12-condition curves (macro AUC; descriptive):**

| condition | S2-D s0 | P2 `full_bank` | P2 P-A |
|---|---|---|---|
| clean | 0.7553 | 0.7636 | 0.6951 |
| JPEG 95 / 75 / 50 / 30 | 0.7458 / 0.7109 / 0.6219 / 0.5914 | 0.7501 / 0.7205 / 0.6443 / 0.6273 | 0.6845 / 0.6856 / 0.6239 / 0.6231 |
| WebP 90 / 70 / 50 | 0.7252 / 0.6670 / 0.6396 | 0.7434 / 0.6936 / 0.6727 | 0.6814 / 0.6488 / 0.6334 |
| resize 0.75 / 0.5 | 0.7302 / 0.7074 | 0.7580 / 0.7224 | 0.6929 / 0.6767 |
| blur σ 0.5 / 1 / 2 | 0.7391 / 0.7249 / 0.6693 | 0.7461 / 0.7523 / 0.6907 | 0.6838 / 0.6920 / 0.6415 |

- **Reading:** S2-D is not more robust than P2 `full_bank`. Under every lossy condition it
  is at or slightly below `full_bank`, despite S2-D's degradation training.
- **The steepest losses** are strong JPEG and WebP compression.
- **The confirmatory part of the P3 analysis is now complete:** H1-a PASS, H1-b PASS,
  H1-c PASS, H3 FAIL, H3-k PASS (Q, S), H2 FAIL, H4 PASS, H5 FAIL.

### P3-C — TrueFake fetch reports verified and committed; Twitter left 499 SD1.5/SD2 fakes unchanged — 2026-10-28

**Files** (downloaded by the user from the fetch notebook's output to `F:\thesis-2\`):
- `truefake_fetch_status.json` (428 B): outcome COMPLETE, refusal null, 16,000 members,
  `bytes_compressed_fetched` 8,574,203,526, 4,166.8 s;
- `truefake_fetch_manifest.csv` (2,879,036 B).
- **Committed** under `reports/truefake/`.

**Checked locally against the frozen sample** (`reports/truefake/truefake_sample.csv`,
sha256 `a8954c2f…` in git):
- **16,000 rows and 16,000 distinct members**, the sample's set exactly.
- **0 mismatches** in `id`, `share`, `stratum` and decompressed size.
- `crc_ok` is True for every member; every sha256 is well formed.
- The decompressed total is 8,611,966,656 B, and the sample's compressed total equals the
  status's 8,574,203,526.

**Finding (MEASURED):**
- **15,501 distinct sha256** over 16,000 files.
- **All 499 duplicate pairs are a Twitter file byte-identical to its own pre-share PNG**:
  StableDiffusion1.5 and StableDiffusion2 fakes, 62–63 per prompt category. That is 499 of
  those two generators' 500 Twitter files.
- For them the Twitter − pre-share Δ is **zero by construction**.
- Recorded in `docs/data_cards/truefake.md`; it is stated wherever that Δ appears.

**Still to do:**
- The user publishes `truefake_sample/` as the private dataset `dbp-truefake-sample`.
- TrueFake scoring is built next, with the pre-registered disjointness check (sha256 +
  pHash ≤ 6 of the sampled reals against our training reals).

### P3-C — TrueFake scoring BUILT (disjointness incl. COCO originals; 4 models once; descriptive analysis); hand-over at `29e66f5` — 2026-10-28

**User decisions (2026-10-28):** "Hash the COCO originals"; "Yes, build it". The rule is the
frozen TrueFake section of `docs/P3_PLAN.md` (2026-10-20); nothing was added to it.

**Built (commit `29e66f5`):**
- `scripts/p3_truefake_prepare.py` (CPU pre-flight and plan):
  - the sample (`a8954c2f…`) and the fetch manifest (`f8dd9ada…`, `sha256sum` this
    session) must be the committed files and list the same 16,000 members, all `crc_ok`;
  - the 4 models (`s2b_seed0`, `s2b_deg_p35_seed0`, `s2d_p35_seed0`, `full_bank_seed0`)
    are checked by `p3_final_preflight.check_model` against `env/p3_final_expect.json`;
  - every member is staged once to local disk with its sha256 checked in the same read;
    its size is read from the header, and a member under 320 px is marked `too_small`;
  - **disjointness:** all 8,000 real versions against T-real (the 2,417 not excluded) and
    Open Images train (3,600), by sha256 (crops and originals) and pHash <= 6 bits (crops,
    Open Images originals, and the **COCO originals hashed in-session** from
    `coco-2017-dataset`/`train2017`, each first checked against its committed
    `source_sha256`; any missing or mismatching original is fatal). A matching id is
    dropped in all four versions, no backfill, counted per pool;
  - once only: complete models are carried forward from an attached previous version.
- `scripts/p3_truefake_score.py`: the deployed protocol (the run's centre crop, head
  logit, fp16 autocast on the backbone only, as the final scorer); a re-score is refused.
- `scripts/p3_truefake_analyze.py`: AUC per generator slice × share, FFHQ and FORLAB
  separately; Δ = platform − pre-share by paired bootstrap (2,000, seed 2026, 95%, one id
  draw per group shared by every share and model); diffusion and StyleGAN macros; FORLAB
  FPR at each model's k = 0 union threshold (descriptive); Twitter-identical counts; the
  plan's caveats are written into the output.
- `env/kaggle_run_p3_truefake_score.py`, `env/p3_truefake_score_expect.json`;
  `p2_report.py --stage truefake`; dry-run fixture `p3_truefake_score`;
  `tests/test_p3_truefake_score.py` (11 tests, synthetic scores only).
- **Privacy:** `.gitignore` (append) keeps `thesis_dbp/reports/p3_truefake_scores/` out of
  git; the returned per-image scores stay in `F:\thesis-2\kaggle_raw\`. `.gitattributes` gains
  `reports/kaggle_*/p3_truefake*/** -text` for the returned list and summaries.

**Two readings stated in the code (not new rules):**
- an id with any version too small for the 320 crop is excluded from every version, so
  Δ stays paired (counted; the plan lists them);
- the pHash references for COCO T-real are the crops' pHash plus the originals' pHash.

**Checks:**
- **ruff:** all checks passed.
- **pytest:** 1144 passed, 7 skipped (`test_gpu_smoke.py`: no CUDA device, PENDING-GPU),
  0 failed, in 550.9 s.
  - An earlier attempt run under a 590 s shell `timeout` printed one `F` at about 82% and
    was killed before its summary, so that test was never identified. The untimed re-run
    passed in full.
- **Local dry-run at `--ref HEAD` (`29e66f5`): PASSED** in 132 s. Planted cases:
  - a FORLAB real equal to a COCO original was dropped by sha256;
  - an FFHQ real equal to an Open Images crop was dropped by pHash;
  - one 300 px fake was marked too small;
  - 4 models scored 71 rows each;
  - a re-score was refused;
  - the analysis ran on the returned files.
- **Analysis cost (synthetic, full sizes):** 0.386 s per resample, so about 13 min for 2,000.
- **Integrity:** CLEAN before the commit.

**Code zip:** `F:\thesis-2\kaggle_raw\dbp-code-29e66f5.zip`, 23,298,236 B, sha256
`c275802a3d316734a8be8d87efff718a5a881be99806e6faf602df9d250e9933`.

**Estimate (ESTIMATED):**
- About 0.8–1.5 h wall on T4 × 2.
- **Measured rate:** 0.019 s/image on Panel B (`reports/p3_final/*_clean.json`). This is a
  lower bound, because TrueFake files are larger.
- **Plan cap:** 0.064 s/image, which is 1.14 device-h.
- **Quota:** 1 h per session hour.

### P3-C — thesis results tables and figures (`9626eca`); P4 baseline plan proposed (no runs) — 2026-10-28

**User request (2026-10-28):** draft the thesis tables and figures from the committed results;
add the descriptive TPR tables if missing; then propose a plan for the P4 baselines.

**TPR tables:** no earlier request for them reached this session after `2f87670`. The
analysis already stored the TPRs (descriptive section), so they are included now
(`t2`, `t3`, `t4`).

**Built (`scripts/p3_thesis_figures.py`; outputs byte-identical on re-run):**
- **Inputs:** `reports/p3_h1a_ci.json`, `reports/p3_h1c.json` and
  `reports/p3_results/p3_analysis_full.json`.
- **Tables** (`reports/p3_results/tables/`, Markdown + CSV). Every row has a `status`
  column: decision (pre-registered) or descriptive.
  - t1: verdicts with CIs, every hypothesis H1-a to H5;
  - t2: the H3 three-step table with TPR beside every FPR;
  - t3: H3-k FPR and TPR by k;
  - t4: per-slice AUC, AP, TPR at k = 0 and slice-calibrated TPR@5%FPR, for the main
    models and for all 24;
  - t5: the H4 curve;
  - t6: the H5 curves.
- **Figures** (`reports/p3_results/figures/`, PNG + SVG, 91–245 KB each, under the 1 MB
  rule):
  - H3-k FPR and TPR by k;
  - H4: the Panel B curve beside the held-out-lineage validation curve, on separate
    charts;
  - H5: 3 models × 12 conditions, with the two decision points ringed.
- **Palette:** the first three categorical slots of the dataviz reference palette.
  `validate_palette.js` could not run (no `node` on this machine), so its documented
  all-pairs validation of those three slots is what is relied on.
- **One computation beyond the stored JSON:** the descriptive TPRs at P2 `full_bank`'s
  COCO-only threshold (three-step table), computed from the committed scores.
  - The same loader first reproduced every stored union FPR and TPR exactly (tolerance
    1e-12).
  - Values: E3 62.4%, E4 98.3%, E5 73.4%, macro E4/E5 85.8%.
- **Labelling:** H2's "without E3" rows are labelled a pre-registered companion test,
  reported beside the decision (section 6.4).
- **Tests:** `tests/test_p3_thesis_figures.py`, 4 passed; `test_repo_figures.py` passed;
  ruff clean. Integrity CLEAN before the commit.

**P4 baselines: metadata read this session (Hugging Face Hub only, the allowed host):**

| baseline | checkpoint | revision | file | licence (card) |
|---|---|---|---|---|
| DDA (NeurIPS 2025) | `Junwei-Xi/Dual-Data-Alignment` (README: "the official checkpoint") | `4390d9023899196b437480bb6a441915ef5d816c` | `DDA_ckpt.pth`, 1,255,621,296 B | apache-2.0; training set `Junwei-Xi/DDA-Training-Set` (`6319d54c…`) apache-2.0 |
| CO-SPY (CVPR 2025) | `ruojiruoli/Co-Spy-Pretrained-Weights`: README holds only the licence; **official status NOT-VERIFIED** | `890ab2ee8dcf0254ff6ef34bcdc2f4906344007f` | `sd-v1_4.zip` 316,332,881 B; `progan.zip` 5,397,681 B | mit (uploader's tag) |
| AEROBLADE (CVPR 2024) | training-free; our AEROBLADE-style score (min LPIPS-vgg over 11 bank VAEs) exists since P2; a faithful variant would add `kandinsky-community/kandinsky-2-1` (`04b36665…`, apache-2.0) | — | — | code licence NOT-VERIFIED |
| FIRE (CVPR 2025) | reproduced on Kaggle in P0 from the official checkpoints; host and licence NOT-VERIFIED this session | — | — | NOT-VERIFIED |

- **Proposed:** a frozen P4 baseline specification, then once-only scoring on the
  committed Panel B lists with the k = 0 union rule.
- **No baseline run, download or new file beyond the two READMEs** (read into the
  scratchpad).
- **Awaiting the user's decisions.**

### P3-C — TrueFake input is the fetch notebook's output; Open Images coverage confirmed; the one-off test failure traced; hand-over at `c9b8c0b` — 2026-10-29

**User, 2026-10-29:**
- Approved `29e66f5` and `ec1fcfb`, both readings, and the three extra items (the
  `.gitignore` line, the `.gitattributes` line, `p2_report --stage truefake`).
- Kaggle would not publish `dbp-truefake-sample` as a dataset (size). The fetch
  notebook's output (`truefake_sample/` plus its reports) is attached directly instead.

**Open Images training reals: already covered (no change).**
- `p3_truefake_prepare.references()` reads the committed manifest
  `reports/kaggle_20261017/p3_open_images/p3_open_images_manifest.csv` (sha256
  `0623a6b0…5961`) and keeps the rows with `split == "train"`: **3,600 of 4,000**.
- **sha256** of the crops (`sha256`) and the originals (`source_sha256`).
- **pHash** of the crops (`phash`) and the originals (`source_phash`, computed on the
  `cvdf_s3_rescaled_max1024px` copy).
- All four columns are well formed in 3,600 / 3,600 rows (checked this session).
- No new images are read for them.

**The notebook output as input:**
- The runbook already found the sample by content: its pattern
  `*/*/*/*/PreSocial/Real/FFHQ` matches `notebooks/<owner>/<slug>/truefake_sample`.
- Every member is sha256-checked against the committed fetch manifest in the
  pre-flight, whatever the source.
- The user's copies of `truefake_fetch_status.json` and `truefake_fetch_manifest.csv` in
  `F:\thesis-2\` are byte-identical to the committed files (`8a2c3fe2…`, `f8dd9ada…`).
- **Changed (`c9b8c0b`):**
  - the runbook text names the fetch notebook's output;
  - the plan records the sample root it staged from;
  - the dry-run fixture puts the sample in a fetch-notebook output, beside that
    notebook's own code tree and reports;
  - a regression test checks that the sample patterns match a notebook output, a
    dataset, or a dataset holding the folder, and nothing else.
- My first version of that test gave two layouts one shared input tree and failed. That
  was the test's setup, fixed before the commit.

**The one-off failure of 2026-10-28 (an `F` under a 590 s shell `timeout`):**
- **Identified:**
  - the `F` was the 71st character after 13 full progress lines, so test #1,007 of the
    1,151 collected then;
  - that is `tests/test_s2_queue.py::test_bad_device_lists_are_refused[0,0]`, #1,011
    today, after the 4 figure tests sorted ahead of it.
- **Re-run in the same order** (`-x`, the same `timeout 590`, `-v --durations=0`):
  1148 passed, 7 skipped, 0 failed, in 472 s; that test passed.
- **Cause: NOT-DETERMINED.** The likely mechanism:
  - the test asserts that a child `p2_run_queue.py` exits with code 1;
  - GNU `timeout` signals the whole process group, and a child killed first would fail
    that assertion;
  - this fits the timing: in an undisturbed run the test executes at about 442 s of
    463 s of test time, so the slower first run reached it at about the kill;
  - this fits the output: `-x` would otherwise have printed the failure report, and none
    appeared.
- **Not reproduced:** 21 kills of the test itself and 13 mid-child kills of a test with
  a longer-lived child produced no `F`.
- **No test was changed.**

**Checks:**
- **pytest at `c9b8c0b`:** 1149 passed, 7 skipped (`test_gpu_smoke.py`, PENDING-GPU),
  0 failed, in 485.6 s, without a timeout.
- **ruff:** clean.
- **Local dry-run at `--ref HEAD` (`c9b8c0b`): PASSED** in 127 s. The sample was found in
  the fetch notebook's output; 80 / 80 members hash-checked; the planted matches were
  dropped (FFHQ 1, FORLAB 1); 1 member too small; 4 models × 71 rows; a re-score was
  refused.
- **Integrity:** CLEAN before the commit.

**Code zip:** `F:\thesis-2\kaggle_raw\dbp-code-c9b8c0b.zip`, 23,943,853 B, sha256
`35dd7231f30b09b01a4aff81f76377a3d80b02611d6f1d5ef1d30cc6d8ea8859`.
- It replaces `dbp-code-29e66f5.zip` for this run.

**Snapshot table refreshed:** the "Last prompt" and "Git" rows.

**Items 2–4 of the user's earlier message** (tables, figures, the TPR tables, the P4
plan) were completed on 2026-10-28 (`9626eca`, `1219af5`).

### P3-C / P4 — H3-k ECE table; citations verified; P4 baseline specification (freeze candidate) — 2026-10-29

**User decisions (2026-10-29):**
- The P4 baselines are descriptive only, with paired bootstrap 95% CIs (2,000, seed 2026).
- Each baseline's published preprocessing is the primary row; crop-320 is a sensitivity
  row for the cheap baselines only.
- AEROBLADE is reproduced faithfully (Kandinsky 2.1 AE approved under §9b), beside our
  "AEROBLADE-style (ours)" row.
- DDA is also scored under the two H5 conditions.
- A read-only GitHub/arXiv metadata look-up is approved: README, LICENSE, paper
  preprocessing, official weight links; no cloning, no running, no weight downloads; every
  URL logged.
- The TrueFake scoring run (`c9b8c0b`) is running on Kaggle.

**(a) H3-k ECE: stored, now tabulated.**
- `reports/p3_results/tables/t7_h3k_ece.{md,csv}`, from `p3_analysis_full.json`.
- Coverage: E4 and E5, both seeds, k = 0 and (Q)/(S) at each k; every row descriptive.
- (Q) moves only the threshold, so its ECE equals k = 0's by construction
  (interpretation `h3k_ece`).
- The other tables and figures are byte-unchanged.
- Test added (`test_h3k_ece_table_restates_the_stored_values`).

**(b) Citations, verified through the arXiv API and the HF dataset card
(`docs/CITATIONS.md`):**
- **arXiv 2605.14486:**
  - v1 (2026-05-14) names the method **Separate Expert Fusion (SEF)**; authors Yiheng Li,
    Yang Yang, Zichang Tan, Gao Li, Zhen Lei, Wenhao Wang.
  - v2 (2026-08-23) renames it **ACEF**; authors Yiheng Li, Yang Yang, Wenhao Wang,
    Zichang Tan, Zecheng Lin, Li Gao, Zhen Lei.
  - Same title in both: "Reduce the Artifacts Bias for More Generalizable AI-Generated
    Image Detection".
- **arXiv 2601.20461v2:** "Exploiting the Final Component of Generator Architectures for
  AI-Generated Image Detection"; Yanzhu Liu, Xiao Liu, Yuexuan Wang, Mondal Soumik.
- **E3 has a paper:** arXiv 2608.01258v1, "A Benchmark Dataset for MLLM-Generated Image
  Detection: GPT Image2 & Nano Banana2"; Zirui Zhang, Yinbo Yu, Donghai Guan, Chunwei
  Tian, Daoqiang Zhang, Qi Zhu. The README's BibTeX was copied verbatim.
- **E3 restricts use:** the README (sha256 `1c7e4f0b…`) limits it "solely for research on
  AI-generated content detection … and related academic purposes", narrowing the
  `apache-2.0` tag. CLAUDE.md §6c still lists E3 as comparatively permissive; raised with
  the user, CLAUDE.md not edited.
- **Also verified:** DDA, CO-SPY, AEROBLADE and FIRE (titles, authors, venues).

**(c) `docs/P4_BASELINES_SPEC.md`: a freeze candidate, not frozen.**
- **Contents:**
  - checkpoints pinned by revision with the Hub's LFS sha256 (DDA `b27a31d3…`, CO-SPY
    `sd-v1_4.zip` `9626f640…`, Kandinsky 2.1 MoVQ safetensors `66731921…`);
  - code licences: DDA Apache-2.0, CO-SPY MIT, FIRE MIT, AEROBLADE **none**, so it will be
    our own implementation from the paper;
  - the native preprocessing as published, the frozen list of reported numbers, the
    caveats, the GPU plan, and 29 URLs with sha256.
- **Findings that shape the spec:**
  - DDA: DINOv2 ViT-L/14 with LoRA rank 8, centre crop 336, trained on the MSCOCO
    *training* set. E7 (MSCOCO val) is its authors' benchmark.
  - 1,281 of the 9,405 Panel B rows are under 336 px on the short side.
  - CO-SPY: SO400M CLIP plus an SD-1.5 VAE round trip, resize 224. The VAE round trip
    puts it outside the user's "cheap" rule, so no sensitivity or H5 rows.
  - The official CO-SPY `download.sh` fetches exactly the `ruojiruoli` files.
  - FIRE: ResNet-50, short side resized to 256; weights on Google Drive only, with no
    published hash.
  - AEROBLADE: SD1, SD2 and Kandinsky 2.1 AEs, LPIPS-VGG16 second layer, minimum over the
    AEs, whole image.
  - Panel B reaches 64 Mpx, so a size cap is needed.
- **Open items before the freeze (section 7):**
  - DDA below 336 px;
  - the FIRE and CO-SPY crop and aspect details;
  - DDA crop-320 vs the 14-px patch;
  - the AEROBLADE cap (768 proposed, AEROBLADE's own data limit);
  - the SD1/SD2 AE choice and the VQ path in `reconstruct.py`;
  - the FIRE checkpoint and Google Drive approval;
  - the CO-SPY checkpoint.
- **GPU plan (ESTIMATED):** 11.5–18.1 device-h, 6–9 h wall on T4 × 2, 2–3 sessions.
- **Nothing was downloaded or run** beyond metadata and the text files named in the
  spec's section 9.

### P3-C — a second TrueFake fetch run was REFUSED by Google Drive (HTTP 503); stopped as the rule requires — 2026-10-29

**What happened** (from the Kaggle output pasted by the user; not re-measured here):
- A new version of the **fetch** notebook ran on 2026-10-29: index verified (600,876
  entries), 16,000 members to fetch, 0 on disk.
- At 07:11:10, after 7,955 of 16,000 members (3,483,493,684 compressed bytes, 2,911.5 s),
  Google Drive answered **HTTP 503** ("Error 503 (Server Error)").
- `p3_fetch_truefake.py` stopped with exit 3, and Cell 6 failed the notebook with "GOOGLE
  DRIVE REFUSED … Do not re-run or work around it". **That is the rule working as
  written; no code change.**

**Effect:**
- **None on the sample.** The successful fetch of 2026-10-23 stands: its committed status
  (`reports/truefake/truefake_fetch_status.json`) reads COMPLETE, 16,000 / 16,000,
  8,574,203,526 B, 4,166.8 s.
- **Its output is a separate notebook version.** It is the one to attach to the TrueFake
  scoring run (`c9b8c0b`), not this refused version, which holds only 7,955 members.
- **The pre-flight guards it:** a partial sample fails staging ("N file(s) missing") in
  Cell 3, before any model is loaded or scored.
- **Rule:** the fetch is not re-run. Whether Drive is ever contacted again is the user's
  decision.
- **The refused run's status file is not committed:** the committed one describes the
  sample in use.

### P3-C — TrueFake scoring Cell 1 failed on Kaggle (two outputs hold `s2b_seed0`); fixed at `2dcec9a` — 2026-10-31

**Kaggle failure** (pasted by the user):
- Cell 1 stopped at 63 s with `AssertionError: expected exactly one output holding
  s2b_seed0, found [.../dbp-p3-train/Implementation/thesis_dbp,
  .../dbp-p3-tune/Implementation/thesis_dbp]`.
- Nothing was staged or scored, and no GPU work ran.

**Cause (my runbook mistake, §12 rule 6):**
- `c9b8c0b` found the train output by `s2b_seed0`, and the tune output also carries
  `s2b_seed0`.
- The final-scoring runbook, proven on Kaggle with these inputs, uses `s2b_seed1` (train)
  and `s2d_p35_seed1` (s2d). I had changed both markers.
- The dry-run fixture did not model the duplicate, so the dry-run passed.

**Fix (`2dcec9a`):**
- **Runbook:** the final runbook's markers.
- **Fixture:** the tune output also carries `s2b_seed0`, as on Kaggle.
  - The old runbook now fails the dry-run with Kaggle's exact error (checked with the
    `c9b8c0b` runbook, exit 1).
  - The fixed one passes.
- **Regression test:** `test_model_sources_use_the_final_runbooks_proven_markers`.
- **Corrected test:** the older `test_runbook_is_thin_and_finds_every_input_by_content`
  asserted the faulty markers.

**Checks:**
- **pytest:** 1151 passed, 7 skipped (`test_gpu_smoke.py`, PENDING-GPU), 0 failed, in
  655.5 s.
- **ruff:** clean.
- **Local dry-run: PASSED**, at the index (142 s) and at `--ref HEAD` (`2dcec9a`, 242 s).
- **Integrity:** CLEAN before the commit.

**Code zip:** `F:\thesis-2\kaggle_raw\dbp-code-2dcec9a.zip`, 23,959,848 B, sha256
`0856a27179957b47df09306c71a8681593b371218c040769c3cdd14ea88468e7`.
- It replaces `dbp-code-c9b8c0b.zip`.
- The inputs to attach are unchanged.

### P3-C — TrueFake holds a truncated JPEG; the pre-flight now decodes every member and excludes such ids (a new reading, for user approval); hand-over at `218da57` — 2026-10-31

**Kaggle, Cell 3 (pasted by the user; code `2dcec9a`):**
- **Pre-flight up to the failure:** environment pre-flight PASSED (2 × T4, HF_TOKEN
  present, Pillow 13.0.0).
- **Staging:** **16,000 / 16,000 members hash-verified** against the committed fetch
  manifest, 0 mismatched (8.02 GB in 99 s, 161.6 files/s).
- **COCO originals:** 2,417 hashed, 0 missing, 0 sha256 mismatches.
- **The failure:** the pHash pass over the reals then raised `OSError: image file is
  truncated (35 bytes not processed)`.
- **Nothing was scored.** Which member failed is not in the output.

**Cause:**
- **A defect in TrueFake as distributed, not a transfer error.** Every member equals the
  archive byte for byte: CRC-32 at the fetch, sha256 at staging.
- **Not caught locally:** the dry-run fixture had no undecodable file. The frozen rule
  does not cover the case.

**Fix (`218da57`), a new reading of the frozen rule, alongside the approved
"too small" one:**
- **Full decode:** the pre-flight decodes all 16,000 members, not only the reals.
- **Recorded, not repaired:** every undecodable member is listed in the plan by name with
  Pillow's error.
- **Excluded from every version:** an id with an undecodable version is excluded from all
  four versions (pairing), and the analysis counts it under `excluded["unreadable"]`.
- **Never repaired:** `ImageFile.LOAD_TRUNCATED_IMAGES` would fill the missing data with
  grey, which invents pixels in a forensic test; rejected.
- **One shared rule:** `scorable()` now decides which rows are scored, in the pre-flight,
  the scorer and the analysis.
- **Fixture:** a real's and a fake's file are truncated, hashed as they are (so staging
  passes, as on Kaggle). Both are caught and excluded.
- **Tests:** +2 (`test_a_truncated_file_is_recorded_not_repaired`,
  `test_the_scorer_and_the_analysis_score_the_same_rows`), and the exclusion test now
  covers the unreadable case.

**Checks:**
- **pytest:** 1153 passed, 7 skipped (`test_gpu_smoke.py`, PENDING-GPU), 0 failed, in
  409.7 s.
- **ruff:** clean.
- **Local dry-run: PASSED**, at the index (108 s) and at `--ref HEAD` (`218da57`, 90 s).
  88 members, 2 undecodable caught, 4 models × 77 rows.
- **Integrity:** CLEAN before the commit.

**Code zip:** `F:\thesis-2\kaggle_raw\dbp-code-218da57.zip`, 23,961,724 B, sha256
`187a27b8f8d46c0755de0533567bfa8d5f3ef239f67af8a56657edb61af732a6`.
- It replaces `dbp-code-2dcec9a.zip`.
- The inputs are unchanged.
- **Cell 3 takes longer:** add a few minutes for decoding the 16,000 members (ESTIMATED).

### P3-C — TrueFake SCORED and ANALYSED (secondary, descriptive): 4 models × 15,999 files; 3,999 ids kept — 2026-10-31

**Run:** Kaggle GPU T4 × 2, `dbp-code-218da57`, `muhammadadeelshah`, 2026-10-31 (worker
logs 11:19–11:34).
- **Archive:** `F:\thesis-2\p3_truefake_reports.zip`, 24,008,960 B, sha256
  `f74f7550e1a95c4ac7d7fea621eb22cd723125ef3480995028e003762d6fea0d`. `testzip` clean,
  739 entries, no unsafe path, no image inside.
- **Extracted to:** `F:\thesis-2\kaggle_raw\p3_truefake_x\`.
- **The rest of the returned tree:** 587 files byte-identical to the committed ones
  (line endings normalised), 0 differ, 18 new (all TrueFake).

**Checks (this session, `scratchpad/verify_tf.py`):**
- **Status:** COMPLETE, both jobs return code 0; smoke COMPLETE first. Pre-flight env
  PASSED (2 × T4, `HF_TOKEN present: true`, Pillow 13.0.0).
- **Staging:** 16,000 / 16,000 sha256-verified, 0 missing, 0 mismatched.
- **Disjointness:** 7,999 real versions checked against 12,034 sha256 and 12,034 pHash
  references (T-real 2,417 incl. the COCO originals hashed in-session, 0 missing, 0
  mismatched; Open Images train 3,600). **0 ids dropped** in either pool; the smallest
  Hamming distance found is 10 (threshold 6).
- **Too small:** 0 (smallest side in the sample is 450 px).
- **Undecodable: 1 member**, `TrueFake/PreSocial/Real/FORLAB/09258.jpg`, `OSError: image
  file is truncated (35 bytes not processed)`. Its 3 shared versions decode (720×960,
  600×800, 900×1200) and were scored (15,999 rows per model); the analysis excludes the
  id from every version, so those 3 scores are not used.
- **List:** sha256 `6c3002e1…a1df` equals the plan's; the sample (`a8954c2f…`) and fetch
  manifest (`f8dd9ada…`) in the zip equal the committed files.
- **Each of the 4 summaries:** LoRA and head sha256 equal the plan's and the records in
  `env/p3_final_expect.json`; `limit` null; crop 320, centre; backbone revision
  `5931719e…` (as pinned in `src/dbp/backbones/loader.py`); `dry_run_random_init` false;
  the score file's sha256 equals the summary's; its ids are the scorable list rows in
  order; every score finite.
- **Per-image score sha256** (kept in kaggle_raw only, git-ignored): `s2b_seed0`
  `5cbb5f05…dd1f`, `s2b_deg_p35_seed0` `4b2481e8…5d4b`, `s2d_p35_seed0` `ddaf6fb6…b5fe`,
  `full_bank_seed0` `3d74c927…89f8`.

**MEASURED:** wall 0.25 h (device 0.50 h); 0.0266–0.0274 s/image (425.9–437.6 s per
model). The plan's cap was 1.14 device-h.

**Analysis:** `scripts/p3_truefake_analyze.py` at `8ea9e56`, run once, 2026-10-31
18:46–19:09 (23 min). Output `reports/p3_results/p3_truefake_analysis.json`, 88,064 B,
sha256 `d3eda6454df26be071d20e436f3f963d38d0358bd125c3491c5d5846c990f1de`.
- **Kept ids:** FFHQ 1,000, FORLAB 999, each generator 250. Excluded: unreadable FORLAB 1.
- **Twitter byte-identical to pre-share:** SD1.5 249, SD2 250, all others 0.

**Results (descriptive; point [95% paired bootstrap CI]; no hypothesis, no verdict):**

| Model | FFHQ diffusion AUC pre-share | Facebook Δ | Telegram Δ | Twitter Δ |
|---|---|---|---|---|
| `s2b_seed0` | 0.992 [0.989, 0.994] | −0.161 [−0.175, −0.149] | −0.047 [−0.055, −0.040] | −0.016 [−0.020, −0.013] |
| `s2b_deg_p35_seed0` | 0.993 [0.990, 0.995] | −0.134 [−0.146, −0.123] | −0.040 [−0.048, −0.033] | −0.012 [−0.015, −0.009] |
| `s2d_p35_seed0` | 0.997 [0.996, 0.998] | −0.187 [−0.200, −0.174] | −0.043 [−0.050, −0.036] | −0.012 [−0.015, −0.009] |
| `full_bank_seed0` | 0.989 [0.986, 0.992] | −0.191 [−0.204, −0.177] | −0.045 [−0.052, −0.038] | −0.020 [−0.024, −0.016] |

| Model | FFHQ StyleGAN AUC pre-share (content-matched) | Facebook Δ | Telegram Δ | Twitter Δ |
|---|---|---|---|---|
| `s2b_seed0` | 0.685 [0.661, 0.709] | −0.135 [−0.160, −0.110] | −0.021 [−0.042, 0.000] | −0.062 [−0.080, −0.046] |
| `s2b_deg_p35_seed0` | 0.780 [0.760, 0.800] | −0.186 [−0.209, −0.164] | −0.072 [−0.089, −0.054] | −0.100 [−0.116, −0.085] |
| `s2d_p35_seed0` | 0.900 [0.886, 0.913] | −0.294 [−0.319, −0.269] | −0.122 [−0.140, −0.104] | −0.119 [−0.135, −0.104] |
| `full_bank_seed0` | 0.649 [0.624, 0.673] | −0.100 [−0.126, −0.074] | 0.015 [−0.006, 0.037] | −0.032 [−0.050, −0.015] |

- **FORLAB FPR at each model's k = 0 union threshold** (descriptive only, never a claim;
  target 5% on the 891 threshold reals), pre-share / Facebook / Telegram / Twitter:
  `s2b_seed0` 0.190 / 0.287 / 0.086 / 0.097; `s2b_deg_p35_seed0` 0.281 / 0.350 / 0.117 /
  0.119; `s2d_p35_seed0` 0.158 / 0.405 / 0.087 / 0.092; `full_bank_seed0` 0.171 / 0.313 /
  0.110 / 0.105.
- **FORLAB vs StyleGAN** pre-share AUC is 0.434–0.608 (faces vs camera scenes: the
  content caveat applies).
- **Facebook is the hardest platform** for every model; FLUX.1 and SD3 drop most there
  (FFHQ, pre-share → Facebook: FLUX.1 0.964–0.988 → 0.503–0.619; SD3 0.998–1.000 →
  0.639–0.733).

**Observation on the written caveat (not changed):** the analysis' Twitter caveat says the
SD1.5 / SD2 Twitter Δ "is zero by construction". Only the fake side is unchanged there; the
reals' Twitter files all differ from pre-share (0 identical), so Δ is near zero, not zero
(e.g. `s2b_seed0` FFHQ SD1.5 −0.0005 [−0.0021, 0.0004]). Raised with the user; the output
is left as written.

**Committed:** `reports/kaggle_20261031/p3_truefake/` (plan, list, statuses, hardware,
pre-flight, 4 worker logs, `summaries/<model>.json`) and
`reports/p3_results/p3_truefake_analysis.json`. No per-image score, no image.

**Pending user approval:** the "undecodable → id excluded from every version" reading
(`218da57`); the run used it, and it affects 1 FORLAB id.

### P4 — spec decisions applied; DDA, FIRE and CO-SPY evaluation code read; four new open items; NOT frozen — 2026-10-31

**User decisions (2026-10-31):**
- Approved `32ca8b9`.
- Spec §7 items 1–7 decided as proposed: DDA's own handling below 336, crop-308
  sensitivity, AEROBLADE cap 768 (crop, never drop), `sd15` + `sd21` + KD2.1 MoVQ with a VQ
  path in `reconstruct.py`, FIRE ImageNet-ADM by `gdown` on Kaggle, CO-SPY `sd-v1_4`.
- CLAUDE.md §6c: E3 corrected to research only.
- The "AEROBLADE-style (ours)" Panel B row is dropped.

**Read-only code reads** (13 URLs, each with its time, status, size and sha256, in spec §9;
the scratchpad fetcher `tools/meta_fetch.py`, hosts api.github.com and
raw.githubusercontent.com only):
- **Trees** of the 3 repos at their pins.
- **DDA:** `Inference/inference.py` (one file, as approved): `CenterCrop(336)`, CLIP
  normalisation, sigmoid output.
- **FIRE:** `eval.py`, `eval_df.sh`, `eval_fi.sh`, `dataset.py`. `resize` defaults to
  False and neither script sets it, so it is `ToTensor` only.
- **CO-SPY:** `evaluate.py` and `detectors/sd-v1_4/{fusion,artifact,semantic,base}.py`.
  The outer transform is Resize(384) + CenterCrop(384); the artifact branch does
  Resize(224) and its own VAE round trip; the semantic branch is open_clip SigLIP
  SO400M-384.
- **HF API:** `timm/ViT-SO400M-14-SigLIP-384` @ `ac16108d…`. CompVis SD-v1.4's VAE
  safetensors has sha256 `a2b5134f…194c`, equal to our pinned `sd15` VAE's (same API
  call on `stable-diffusion-v1-5` @ `451f4fe1…`).
- **Local check:** torchvision 0.26 `CenterCrop(336)` zero-pads a 300 × 500 image (18 zero
  columns on each side).

**Changed (commit below):**
- `docs/P4_BASELINES_SPEC.md`: sections 1, 3, 4, 6, 7, 8 and 9.
- `CLAUDE.md` §6c: the E3 line.
- `docs/CITATIONS.md`: one sentence (§6c now corrected).

**Not frozen**, because the reads raised four items for the user (spec §7.8–7.11):
- CO-SPY's in-forward VAE round trip vs CLAUDE.md §7;
- a new dependency, `open_clip_torch`, plus the pinned SigLIP weights;
- FIRE: the code does not resize, but the paper resizes to 256;
- DDA's backbone source (two more files to read).

**Revised GPU plan (ESTIMATED):** 3.7–10.5 device-h (was 11.5–18.1), about 2–5.5 h wall
on T4 × 2.

### P3-C — TrueFake exclusion rule approved; Twitter caveat corrected and the analysis re-run (numbers identical); thesis table t8 — 2026-10-31

**User decisions (2026-10-31):** `7700d0a` approved.
- **Exclusion rule approved:** an id with any unreadable version is removed from all four
  versions. It is label-blind, keeps the pairing, and affects 1 FORLAB id
  (`TrueFake/PreSocial/Real/FORLAB/09258.jpg`).
- Recorded here and in `docs/data_cards/truefake.md`, new section 3a.

**Twitter caveat corrected:**
- **Changed:**
  - `scripts/p3_truefake_analyze.py`: the caveat string and the docstring now say
    "near zero, not exactly zero, because the reals were re-encoded";
  - the data card's section 3 bullet says the same.
- **Re-run once:** 2026-10-31 19:40–19:51, with the committed list and summaries
  (`reports/kaggle_20261031/p3_truefake/`) and the kaggle_raw scores. Output sha256
  `59f316227fbfbd3d68f88aa839995013fcad0d7309cc3362ec1c3341dd6f7eb0` (was `d3eda645…`).
- **Compared field by field with `7700d0a`:** all **1,730 numeric fields are identical**.
  The one difference is `caveats[5]`, the corrected text.

**Thesis table t8 (descriptive):**
- **Code:** `scripts/p3_thesis_figures.py` reads `p3_truefake_analysis.json`.
- **`t8a_truefake_auc`** (80 rows): 4 models × 2 real pools × (8 generators + 2 macros).
  For each: pre-share AUC; Facebook, Telegram and Twitter AUC; Δ vs pre-share; paired
  bootstrap 95% CIs.
- **`t8b_truefake_forlab_fpr`** (4 rows): FORLAB FPR at each model's k = 0 union
  threshold, per platform. Point values only, since the analysis stores no CI for them.
- Every row is labelled descriptive; the caveats are written into the table note.
- **Unchanged:** the other tables and figures regenerated byte-identical.
- **Tests:** +1, `test_truefake_tables_restate_the_stored_values`.

**Checks:**
- **pytest:** `test_p3_truefake_score.py`, `test_p3_thesis_figures.py` and
  `test_repo_figures.py`: 28 passed.
- **ruff:** clean.
- **Integrity:** CLEAN before the commit.

**P4:** the user's message repeated the decisions already applied in `80a8939`. The
freeze still waits on spec items 7.8–7.11.

### P4 — spec decisions 7.8–7.11 applied; FIRE and DDA model code read; three new open items (7.12–7.14); NOT frozen — 2026-11-01

**User decisions (2026-11-01):** `80a8939` approved.
- **CO-SPY:** a stated exception to CLAUDE.md §7 for its in-forward VAE round trip, run as
  written, fp32, no tiling, on our pinned `sd15` file. `open_clip_torch` approved (§9b),
  with the dry-run rule; SigLIP pinned and loaded from a local path.
- **FIRE:** the code's `--resize` option. The `network_utils.py` read was approved.
- **DDA:** the read of the two model files was approved. The 2-image equivalence check is
  approved only in an isolated CPU notebook (no Secrets, no private data, 2 COCO val2017
  images, pass within 1e-3, nothing reused).
- **GPU plan:** 3.7–10.5 device-h approved.

**Reads** (3 URLs, logged in spec §9):
- **DDA** `Inference/models/dinov2_models_lora.py` and `dinov2_models.py`:
  - the backbone comes from `torch.hub.load('facebookresearch/dinov2', ..., pretrained=True)`,
    which is unpinned and runs downloaded code;
  - LoRA sits on `attn.qkv`, `attn.proj`, `mlp.fc1`, `mlp.fc2`;
  - the score is `fc(x_norm_clstoken)`.
- **FIRE** `utils/network_utils.py`: building `FIRE_model` fetches the
  `runwayml/stable-diffusion-v1-5` pipeline in fp16 and torchvision's ResNet-50
  IMAGENET1K_V2 weights. Its forward runs two VAE round trips through `retrieve_latents`,
  which samples the latent by default; an ESPCN-predicted FFT mid-band filter; and a 6-channel
  instance-norm ResNet-50.
- **HF API:** `runwayml/stable-diffusion-v1-5` resolves to `451f4fe1…`, our `sd15` pin.
- **Local:** diffusers 0.40.0 `retrieve_latents` defaults to `sample_mode="sample"`.

**Not frozen.** Three new items for the user (spec §7.12–7.14):
- DDA's `lora.py` (scaling and key names), and `timm` as the architecture source;
- FIRE's fp16 and sampled-latent round trips: proposed as FIRE wrote them, seeded per
  image;
- FIRE's non-multiple-of-8 sizes: proposed centre crop to a multiple of 8.

**GPU plan revised (ESTIMATED):** FIRE 0.1–0.3 → 0.5–1.2 device-h (two VAE round trips
found), total 4.1–11.4 device-h, which is above the approved 3.7–10.5 and is flagged.

### P4 — spec items 7.12–7.14 decided; `lora.py` and DINOv2 hub code read; spec FROZEN (`p4-baselines-frozen`) — 2026-11-01

**User decisions (2026-11-01):** `d27e2aa` approved.
- **7.12, DDA:**
  - read `Inference/models/lora.py` and implement its LoRA exactly;
  - architecture: `timm` `vit_large_patch14_dinov2`, `pretrained=False`;
  - load `DDA_ckpt.pth` with `weights_only=True` after the sha256 check, and stop on any
    missing or unexpected key;
  - if the isolated 2-image check fails, fix the cause; the 1e-3 tolerance is never
    loosened.
- **7.13, FIRE:** a stated §7 exception. The two round trips run as published (fp16,
  `latent_dist.sample()`, the filtered copy), seeded per image from a hash of its id. VAE
  from the checkpoint if it carries one, else our `sd15` file. No pipeline or torchvision
  download.
- **7.14, FIRE sizes:** centre-crop the long side down to a multiple of 8 (at most 7 px),
  and record the count.
- **GPU plan:** the revised total of 4.1–11.4 device-h is approved.

**Reads** (same fetcher, 6 URLs; spec section 9):
- **DDA `lora.py`:**
  - output = `original(x) + (x Aᵀ) Bᵀ · alpha / rank` (scale 1/8 at rank 8, alpha 1);
  - targets are matched by substring;
  - keys are `original_layer.*`, `lora.lora_A` and `lora.lora_B`.
- **`facebookresearch/dinov2` @ `7764ea0f…`** (`main` per the GitHub API): `hubconf.py`,
  `dinov2/hub/backbones.py`, `dinov2/models/vision_transformer.py`, `LICENSE`. These were
  read so the positional-embedding interpolation matches exactly: bicubic, fp32,
  `scale_factor = (w0 + 0.1) / 37`, no antialias. There is also an unused `mask_token`.
- **Disclosure:** the DINOv2 read was not separately named in the approval. It was made
  because item 7.12 asks for DINOv2's interpolation to be matched exactly, and that cannot
  be done without reading it.

**Local:** `pip install --dry-run` of `open_clip_torch==3.3.0` resolves to `timm` 1.0.30,
`ftfy` 6.3.1 and `wcwidth` 0.9.4. torch, torchvision and numpy are untouched.

**Spec frozen:** header, section 1 (DDA, FIRE), section 3 (FIRE row: crop to a multiple of
8, the §7 exception, the seed rule), section 4 (pins), section 7 (12–14 decided), section 8
(approved), section 9 (reads). Tag `p4-baselines-frozen` on this commit; no baseline code
or score exists.

### P4 — baseline code built (DDA, CO-SPY, FIRE, AEROBLADE), three runbooks dry-run; NOT-RUN on Kaggle — 2026-11-02

**Built after the freeze** (tag `p4-baselines-frozen`, spec unchanged; its sha256 at the tag,
from git's bytes, `6576eade…2a54`, is checked by the pre-flight):
- **`src/dbp/baselines/`:**
  - `dda.py`: DDA's LoRA from `lora.py`, on `timm` `vit_large_patch14_dinov2`, with
    DINOv2's own interpolation; `mask_token` registered; strict load.
  - `cospy.py`: CO-SPY re-implemented (MIT) with upstream module names; SigLIP from the
    local file through `open_clip`; a VAE key in `artifact_weights.pth` stops the run.
  - `fire.py`: FIRE re-implemented (MIT); fp16 sampled round trips with the seed rule;
    long side cropped to a multiple of 8.
  - `aeroblade.py`: our implementation; 768 cap; LPIPS-VGG layer 2; minus the minimum.
  - `common.py`: sha256, then `weights_only=True`; exact key sets; the FIRE seed.
- **`configs/p4_aeroblade.yaml`:** the Kandinsky 2.1 MoVQ, in the bank's schema (not a bank
  member).
- **`reconstruct.py` is unchanged:** its `vq` path (`reconstruct_vq`, the Kandinsky 2.2
  MoVQ's) already serves the 2.1 MoVQ. Its config, read 2026-11-01 at the pinned revision,
  is the same class (VQModel, spatial norm, vq_embed_dim 4). Tests were added for that
  config shape.
- **`requirements/base.txt`:** `open_clip_torch==3.3.0` and `timm==1.0.30` (approved
  2026-11-01). The local venv gained exactly these two plus `ftfy` 6.3.1 and `wcwidth`
  0.9.4; torch 2.11.0+cpu, torchvision 0.26.0+cpu and numpy 2.5.2 are unchanged (printed
  before and after).
- **Scripts:**
  - `p4_weights.py` (step 1, CPU): fetch, hash, record FIRE's sha256, extract the CO-SPY
    `.pth` files, inspect every checkpoint;
  - `p4_dda_equivalence.py` (isolated step 2);
  - `p4_preflight.py` and `p4_score.py` (step 3, GPU);
  - `build_code_zip.py --bundle p4-dda-equiv`: a code-only subset;
  - `p2_report.py --stage p4`.
- **Runbooks:** `env/kaggle_run_p4_weights.py`, `env/kaggle_run_p4_dda_equivalence.py`,
  `env/kaggle_run_p4_baselines.py`, each with its expect file.
- **Isolated check images:** `000000039477.jpg` and `000000185292.jpg`, the first two
  val2017 names in `reports/p2_panel_sets/p2_panel_e7_dda_coco.csv`. They are read from
  the public Kaggle dataset `awsaf49/coco-2017-dataset`, so no new host is needed.
- **FIRE file id:** `1bTVrfyUNcjOKFiP1HlvyE9xNqWLhJrGq`, from FIRE's README @ `ff1a316d`,
  line 122, read 2026-10-29.

**Checks:**
- **pytest (full suite):** 1,210 passed, 7 skipped (all 7 in tests/test_gpu_smoke.py: no CUDA device, PENDING-GPU), 0 failed, 457 s.
- **ruff:** clean.
- **Local dry-runs, against the index of this commit:** all three PASSED.
  - `p4_weights`: 144 s.
  - `p4_dda_equivalence`: 45 s. Uses a stand-in archive; DDA's own code is never run
    locally.
  - `p4_baselines`: 1,373 s, 2 sessions. Session 2 carried all 7 rows and refused a
    re-score.

**Dry-run fixes:**
- **FIRE on CPU:** the first scoring dry-run stopped at its 1 h smoke deadline, because an
  fp16 SD VAE on a CPU is impractically slow.
  - **Seam added:** in `DBP_DRYRUN=1` on a CPU only, FIRE's VAE runs in fp32.
  - **Coverage:** Kaggle runs fp16, and its GPU smoke exercises it. The unit tests cover
    fp16 with a tiny VAE.
- **A held directory:** the next attempt failed because the agent's own shell held the
  dry-run's log directory open; the shell moved out and the run passed.

**Deviations, stated:**
- **AEROBLADE in the dry-run** uses the cached `sd15` and Kandinsky **2.2** MoVQ (the same
  `vq` path). `sd21` and the 2.1 MoVQ are first loaded by Kaggle's pre-flight smoke.
- **CO-SPY `artifact_weights.pth` may hold the full VAE** (INFERRED: upstream's base
  `save_weights` saves the whole state dict). If it does, the frozen spec stops the run.
  The weights session reports whether those tensors equal the pinned `sd15` VAE, so the
  user can decide.
- **The isolated notebook's code bundle** is a Kaggle dataset the user uploads. It is
  private by Kaggle's default but holds code only (no data, no reports). Flagged for the
  user's confirmation.

**NOT-RUN:** every Kaggle step. FIRE's and the three CO-SPY `.pth` sha256 values are null
in `env/p4_baselines_expect.json` until the weights session returns them. The pre-flight
refuses to plan while any is null.

### P4 — steps 1 and 2 returned; CO-SPY VAE rule; hashes committed; repository re-rooted; step 3 handed over — 2026-11-05

**User decisions (2026-11-05):**
- **Approved:** `c37ece3` (the freeze) and `bffdfca` (the code), and the four stated
  deviations.
- **The step-2 bundle meets the isolation rule.** That rule: no licensed images, no model
  weights of ours, no reports and no Secrets in a session that runs third-party code.
- **Hosts for DDA's own code:** it may fetch from github.com and dl.fbaipublicfiles.com
  in the isolated step-2 session only. Record the URLs and sha256 of everything it
  downloads. These hosts are never used in a scoring session.
- **CO-SPY's `artifact_weights.pth` VAE:**
  - equal to the pinned `sd15` VAE (exactly or after a dtype cast): continue with sd15;
  - different: use the checkpoint's own VAE, record each tensor's max abs difference, and
    continue;
  - this does not stop the run. It replaces the stop rule in spec section 1, as a
    post-freeze user decision; the spec file itself is unchanged, and its sha256 is still
    checked.

**Step 1, weights (Kaggle CPU, code `bffdfca`):**
- **Files:** all four present, with the recorded size and sha256 (DDA, CO-SPY zip, SigLIP).
- **FIRE recorded:** sha256 `fabd7a700b8fbcc826407d96cc1afdd86cc2722a72e6300ba558cd0419efe689`,
  261,923,836 B.
- **CO-SPY `.pth` files:** semantic `f391482e…2dd4` (6,186 B); artifact `3ec2b5aa…4f76`
  (340,566,472 B); fusion `7af0160e…f6fb` (1,578 B).
- **DDA:** 537 keys, strict load OK.
- **CO-SPY artifact file:** 394 keys, 248 of them the VAE, **equal to pinned sd15 (max abs
  diff 0.0)**.
- **FIRE:** 318 keys, the full VAE, **equal to sd15 cast to fp16 (0.0)**.
- **The only problem** was the CO-SPY stop that code `bffdfca` raised, now waived.
- **Committed:** `reports/kaggle_20261105/p4_weights/p4_weights.json`.

**Step 2, isolated DDA check (Kaggle CPU, no Secrets, code-only bundle, public COCO):**
**PASS**.
- **Result:** DDA's own code and `dbp.baselines.dda` give identical logits: −3.6115453 and
  −2.9998522, abs diff 0.0, against a tolerance of 1e-3.
- **Committed:** `reports/kaggle_20261105/p4_dda_equivalence.json`.
- **Downloads: NOT-VERIFIED.** That run's downloads were not recorded: code `bffdfca` did
  not log them.
- **Transcription:** both JSON files are transcribed from the pasted output; their equality
  with the Kaggle files is NOT-VERIFIED (`reports/kaggle_20261105/README.md`).

**Changes:**
- **CO-SPY VAE rule:**
  - `dbp.baselines.common.compare_state` (exact or dtype-cast equality, with per-tensor
    max abs difference);
  - `cospy.choose_vae` / `load_cospy` apply the 2026-11-05 rule and return a VAE record,
    which goes into the row summary;
  - `p4_weights.py` records the comparison and no longer stops;
  - `p4_preflight.py` accepts a weights report whose only problems are the waived CO-SPY
    notice.
- **Hashes committed:** `env/p4_baselines_expect.json` holds FIRE's and the three CO-SPY
  `.pth` sha256 values (no null remains).
- **Host guard:** `p4_score.py` refuses `torch.hub.load`, and lets `torch.hub` download
  from download.pytorch.org only.
- **Download log:** `p4_dda_equivalence.py` now records the URL and sha256 of every
  download (the DDA archive, and `torch.hub`'s files).
- **Re-rooting fixes:**
  - **`.gitignore` correction:** the user's re-root made `thesis_dbp/` the repository
    root. The new `thesis_dbp/.gitignore` written on 2026-11-02 had an unanchored `data/`,
    which also excluded `src/dbp/data/` (7 files) from the first commit `db2aca9`, which
    is already pushed. Appended `!src/dbp/data/`; the package is committed here.
  - **`build_code_zip.py`:** works with the repository rooted at `thesis_dbp/` or at its
    parent. Zip entries still start with `thesis_dbp/`.
  - **`tests/test_repo_figures.py`:** lists from `thesis_dbp/`. In the new layout it would
    otherwise have skipped silently. It now also fails if fewer than 100 files are listed.
- **Comparison with the old history:** `db2aca9` holds the same bytes as `bffdfca` for
  every file, except the 7 missing `src/dbp/data/` files and the new `.gitignore`.
  Compared file by file against `F:\thesis-2\kaggle_raw\dbp-code-bffdfca.zip`.

**Checks:**
- **pytest:** 1,218 passed, 7 skipped (all in tests/test_gpu_smoke.py: no CUDA device, PENDING-GPU), 0 failed, 435 s.
- **ruff:** clean.
- **Integrity:** CLEAN.
- **Dry-runs:** at HEAD after this commit; see the step-3 hand-over.

### P4 — steps 1-2 originals verified; step 2 provenance recorded; pre-push audit (no push) — 2026-11-05

**User decisions (2026-11-05):** `9ba3dde` approved.
- **Step 2 is not re-run.**
- **No push yet:** a pre-push audit comes first, and the user confirms the repository is
  private.
- **Step 3** (GPU scoring, code `9ba3dde`) was started by the user.

**Step 2 provenance (recorded as the user asked):** step 2 ran on code `bffdfca`, which did
not log the URLs or sha256 of what DDA's code downloaded (the DDA archive from github.com,
DINOv2's code and weights through `torch.hub`). Those values are **NOT-VERIFIED** and will
not be filled in. **Its result was an exact match:** both logits identical, abs diff 0.0,
tolerance 1e-3.

**Verification against the Kaggle originals** (the user placed them in `F:\thesis-2\kaggle_raw\`):
- **`p4_weights.json`:** extracted from `p4_weights_reports.zip` (sha256
  `d60c871fe2052928642a13ba0716b962af59cc4025137d4a7413474c28162c36`, 739 entries) into
  `F:\thesis-2\kaggle_raw\p4_weights_reports_x\`. It is **byte-identical** to the committed
  `reports/kaggle_20261105/p4_weights/p4_weights.json`: sha256
  `42cf31bad09424a4b8c4181d03c7ebc3ff2dc55fb1b505973b31e5277f7f32b5`, 3,360 B, 66 fields,
  0 differing.
- **`p4_dda_equivalence.json`** (`F:\thesis-2\kaggle_raw\p4_dda_equivalence.json`): **byte-identical** to the
  committed `reports/kaggle_20261105/p4_dda_equivalence.json`: sha256
  `8c413410cdb1ec8dc6ed95d128296ff84d608ee42dfde1fdfa29bb1dd71a9673`, 1,191 B, 30 fields,
  0 differing.
- **Git blobs:** the copies in git (`git cat-file -p HEAD:...`) have the same two sha256
  values.
- **README:** the NOT-VERIFIED note in `reports/kaggle_20261105/README.md` is now
  resolved, as recorded here.

**Pre-push audit** (read-only; every blob reachable from `master`: 2 commits, 868 paths,
875 blobs; nothing pushed):
- **(a) Images:**
  - by extension, only 12 paths: 6 PNG + 6 SVG charts under `reports/**/figures/`, each
    under 1 MB (CLAUDE.md section 2b);
  - by magic bytes, only those 6 PNGs;
  - **no dataset image** under any extension, and no non-CSV file over 1 MB.
- **(b) Secrets:** **0 hits** for Hugging Face, GitHub, AWS, Slack and OpenAI-style tokens,
  private keys and Kaggle API keys. 0 risky file names (`.env`, `kaggle.json`, `*.pem`,
  `*token*`, `*secret*`, `credentials*`).
- **(c) Per-image files that would go to GitHub:** 150 CSVs (any with an image-id or path
  column), 50,513,800 B, 788,963 rows.
  - **Scores:** the P3 final scores (24 models), the held-out scores, and P2 session E's
    real scores.
  - **Manifests and lists:** Panel B, the TrueFake member manifests (no scores), the P2
    panel sets, Open Images with its attribution, the COCO reals.
  - **The TrueFake per-image scores are not among them** (git-ignored, kept in
    `kaggle_raw`).

### Repository — pre-push audit tool; first push of the P4 work to the private GitHub repository — 2026-11-05

**User decisions (2026-11-05):**
- `2730bb5` verified.
- The GitHub repository `Muhammad-Adeel-Shah/thesis` is **PRIVATE** (the user checked and
  changed its visibility).
- Add the audit as `scripts/prepush_audit.py` with a test, commit and run it, then push
  `master` to `origin`. The push also restores `src/dbp/data/` on GitHub.

**Added:**
- **`scripts/prepush_audit.py`** (read-only) checks every blob in every commit reachable
  from `--ref`:
  - **(a) Images:** any image or video extension, or an image signature in the bytes, must
    be an SVG/PDF/PNG chart under `reports/**/figures/` of at most 1 MB.
  - **(b) Secrets:** token and private-key patterns, reported redacted, and risky file
    names.
  - **(c) Per-image files:** CSVs with an image-id or path column, summarised by
    directory.
  - **Exit codes:** 0 when clean, 1 on any (a) or (b) violation, 2 when it cannot run.
- **`tests/test_prepush_audit.py`:** 17 cases on throwaway repositories:
  - a dataset JPEG, a PNG outside `figures/`, a chart over 1 MB, a renamed JPEG, a video;
  - an image deleted in a later commit;
  - four token kinds and four risky names;
  - the per-image summary, and the exit codes;
  - the fake tokens are assembled at run time.

**Correction to the chat report of 2026-11-05 (not to this log):** the `p3_final_scores`
group was given as 16.2 MB / 452,428 rows. The audit gives 15.22 MB / 575,484 rows in 108
files. The totals in the previous entry are correct.

**Audit of `master` = `2730bb5` before this commit:** CLEAN.
- **(a):** 12 image files, all allowed charts.
- **(b):** 0 content hits, 0 risky names.
- **(c):** 150 CSVs, 50,513,800 B, 788,963 rows.

### P4 — scoring session 1: 6 of 7 rows complete; AEROBLADE stopped at the deadline; measured rate; option A approved in principle — 2026-11-09

**Session** (Kaggle T4 x2, code `9ba3dde`, 2026-11-05 08:51–19:27):
- **Pre-flight:** passed. The spec sha256, the three lists, all six weight files and the
  accepted weights report were checked; 10,996 files were staged and hash-verified in 26 s.
- **Smoke:** all 7 rows passed.
- **Scoring:** **exited 2 (DEADLINE at 10.5 h)**.
  - GPU 0 finished its 6 rows in about 3.7 h, then idled.
  - GPU 1 (AEROBLADE) was terminated at 10.5 h with nothing saved, because the worker
    writes a list only when it completes, and Panel B (9,405) had not.
- **Cost:** 10.6 h session wall, which is what Kaggle bills; 21.0 device-h by the runner's
  count (2 x 10.5 h). The plan said 2–6 h wall / 4.1–11.4 device-h. **The AEROBLADE
  estimate was wrong:** see below.
- **Files:** in `F:\thesis-2\kaggle_raw\`:
  - `p4_reports.zip`: sha256 `07bf1e39aeae8d79d88c725cfcc90a848770dd0e2ea1d1b3b1728e052caf2595`,
    779 entries, extracted to `F:\thesis-2\kaggle_raw\p4_reports_x\`;
  - `p4_smoke_job1_gpu1.log` (`328565a1…2d`) and `p4_score_job1_gpu1.log` (`68c7dd12…59`),
    each byte-identical to the zip's copy.

**The 6 complete rows, verified and committed**
- **Verified:**
  - each CSV's sha256 equals its summary's;
  - the image ids are in the committed lists' order;
  - every score is finite;
  - the flag counts equal the summaries';
  - `limit` is null; the device is cuda.
- **Committed:** the summaries in `reports/p4_baselines/`, the scores in
  `reports/p4_baseline_scores/` (`-text` in `.gitattributes`), and the plan, status,
  hardware and the four worker logs in `reports/kaggle_20261105/p4_baselines_session1/`.

| row | lists (rows) | flag | s/img (MEASURED) | seconds |
|---|---|---|---|---|
| `dda_native` | 9,405 / 700 / 891 | padded 1,281 / 31 / 0 | 0.146–0.148 | 1,624 |
| `dda_crop308` | 9,405 / 700 / 891 | padded 0 / 0 / 0 | 0.115–0.117 | 1,284 |
| `dda_jpeg50` | 8,655 test | padded 1,050 | 0.156 | 1,349 |
| `dda_resize050` | 8,655 test | padded 1,050 | 0.164 | 1,422 |
| `cospy_native` | 9,405 / 700 / 891 | upscaled 1,942 / 120 / 0 | 0.434–0.439 | 4,824 |
| `fire_native` | 9,405 / 700 / 891 | cropped 4,582 / 618 / 0 | 0.187–0.260 | 2,738 |

- **Cross-check:** DDA padded 1,281 Panel B rows, the count of rows with a short side
  under 336 that the spec recorded on 2026-10-29.
- **CO-SPY:** used the pinned sd15 VAE, equal to the file's.
- **FIRE:** used the checkpoint's VAE in fp16, equal to sd15 cast to fp16.
- **Pillow:** 13.0.0 on Kaggle, where DDA's README pins 12.3.0; a stated caveat, spec
  section 4.

**AEROBLADE, MEASURED** (smoke, 2 images per list, one T4, fp32, 3 AEs + LPIPS):

| list | first 2 images, capped to <= 768 and a multiple of 8 | s/img | us per capped pixel |
|---|---|---|---|
| Panel B | 768 x 768, 768 x 768 | 10.44 | 17.7 |
| luma | 640 x 520, 640 x 424 | 5.25 | 17.4 |
| threshold reals | 384 x 384, 384 x 384 | 2.79 | 18.9 |

- **The rate is linear in pixels:** 17.4–18.9 us per capped pixel across a 4x range of
  image area.
- **Bottleneck:** the work is GPU compute of the three fp32 round trips plus LPIPS, not
  image loading. The same images load and score at 0.12 s/img in the DDA rows, so loading
  is under about 2% of AEROBLADE's per-image time.
- **Not separable:** the split between the three AEs and LPIPS is NOT-MEASURED; the
  worker logs only whole lists.
- **Pixel totals:** computed from the committed manifests (`p3_panel_b_selection.csv`,
  `p1_reals_manifest_t_real.csv`, the Open Images manifest), after the 768 cap and the
  multiple-of-8 crop:
  - Panel B 3.405 Gpx (2,888 images cut by the cap);
  - luma 0.192 Gpx;
  - threshold reals 0.131 Gpx;
  - **total 3.728 Gpx** (the spec's "about 3.7 Gpx").
- **Single T4:** 3.728 Gpx x 17.4–18.9 us = **18.0–19.6 GPU-h**.
- **Consistent with the session:** Panel B alone is 16.5–17.9 h, so it could not finish
  in 10.5 h.
- **Why the spec's 2.3–7.5 device-h was wrong:** it scaled P2's 320-px measurements
  (sd15 0.18 s, Kandinsky 2.2 MoVQ 0.38 s, about 7.2 us/px for three AEs) and left out
  LPIPS. The measured rate is 2.4–2.6x that; the cause is NOT-MEASURED.

**User decisions (2026-11-09):** option A is approved in principle (execution only:
split AEROBLADE across both GPUs, load images in the background, checkpoint). Option B
(fp16) is rejected. Option C (drop or subset) only if option A proves unaffordable.
1. **Exception to the "interrupted means re-scored from scratch" rule:** AEROBLADE may
   resume from saved partial scores, because the method is deterministic.
   - Checkpoints are append-only, hashed and never overwritten.
   - On every resume, 20 already-saved rows are re-scored, and the run stops if any
     differs beyond fp32 rounding.
2. **Fail fast, for every row from now on:** the runbook measures s/img in the smoke and
   refuses to start, or re-plans, if the projected time exceeds the session deadline.
3. **Quota:** the user has 30 GPU h left this week (it resets on Saturday).

**No option-A code yet;** the projection is reported to the user first.

### P4 — option A built (AEROBLADE across both T4s, append-only checkpoints, resume check; fail-fast for every row) — 2026-11-09

**User decisions (2026-11-09):** `c0b641f` approved.
- **Option A:**
  - split AEROBLADE across both T4s by pixel count;
  - checkpoint every 200 images (append-only, hashed, never overwritten);
  - on every resume, re-score 20 saved rows and stop if any differs beyond fp32 rounding;
  - merge into the normal `aeroblade` row and verify against the committed lists.
- **No background loading:** loading is under 2% of the time.
- **Fail fast, every row:** time each row in the smoke (excluding the first, warm-up image),
  and refuse to start if the projection exceeds the remaining deadline.
- **Push:** `master` is pushed after the pre-push audit passes (the repository is private).

**Exception, recorded as the user asked:** the earlier rule "an interrupted model is scored
again from scratch" does not apply to AEROBLADE's shards.
- **Resuming** from saved partial scores is allowed, because the method is deterministic.
- **Checkpoints** are append-only, hashed and never overwritten.
- **Every resume** first re-scores 20 saved rows (spread over what is saved), and stops if
  any value differs beyond fp32 rounding.
- **The tolerance:** a relative difference of at most 1e-6 (about 8 fp32 ulps), with an
  absolute floor of 1e-9; values are stored at full float precision.

**Built:**
- **`src/dbp/baselines/shards.py`:**
  - `capped_pixels`; `assign` (longest-processing-time on capped pixels, deterministic);
  - `ShardStore`, the append-only store: part files plus `ledger.jsonl`. Each part is
    checked against its ledger line on load; a planted, changed or missing part stops
    the run; a part name is never written twice;
  - `resume_sample`, `within_fp32`, and `merge` (every (list, id) exactly once, in the
    committed order).
- **`scripts/p4_preflight.py`:**
  - deals a sharded row's images by capped pixels, read from the staged, hash-checked
    copies, and writes `reports/p4_shards/<row>_shard<k>.csv`;
  - carries earlier versions' checkpoints forward without overwriting (a differing file
    stops the run);
  - plans units `aeroblade#0`/`#1`, and the stage `merge` when every shard is done.
- **`scripts/p4_score.py`:**
  - **Shard units:** score in parts of 200. A resume runs the re-score check first, and
    `resumes.jsonl` records every check. Each store's `meta.json` pins the manifest
    sha256 and the column set. A finished shard writes `shard<k>.done.json`.
  - **The smoke:** scores one untimed warm-up image per model, then times every smoke
    image, recording `timing` (images, seconds, capped pixels) and `load_seconds`. It
    also rehearses a resume check on its own store.
  - **Dry-run seam:** `DBP_P4_STOP_AFTER_PARTS` (honoured only with `DBP_DRYRUN=1`)
    rehearses a deadline.
- **`scripts/p4_failfast.py`:**
  - a row's time is load + s/image x its images; a shard's is load + s/pixel x its pixels
    left + the resume re-score;
  - a job is the sum of its units;
  - the time available is min(deadline, 12 h session − elapsed − 0.25 h);
  - exit 1 refuses the run, and `reports/p4_projection.json` gives the deadline to score
    with.
- **`scripts/p4_merge.py`:**
  - verifies every store and ledger, then writes the normal `aeroblade` CSVs exactly as an
    unsharded row's;
  - its summary records each shard, its ledger hash and every resume check;
  - exit 2 means a shard is not done.
- **`env/kaggle_run_p4_baselines.py`:**
  - records the session start;
  - the smoke and the fail-fast check come before any long job;
  - a merge cell follows the scoring;
  - the verdict checks every row due.
- **`env/p4_baselines_expect.json`:** `shards` (n 2, every 200, re-score 20); jobs
  `0:…,aeroblade#0` / `1:aeroblade#1`; session limit 12 h, margin 0.25 h.
- **Not built:** background image loading (user decision).

**Tests:**
- **`tests/test_p4_shards.py`** (14): the deal; the store (append, reload at full
  precision, planted part, changed part, no overwrite); the resume sample; the tolerance;
  the merge; the projection (s/image, s/pixel, the resume cost, the refusal, a late
  session, a missing smoke).
- **Full suite:** 1,247 passed, 7 skipped (GPU-only), 0 failed.
- **ruff:** clean.

**Local dry-run** (`p4_baselines`, against the index): PASSED in 2,834 s.
- **Session 1:** all 7 rows; each shard stopped after one part (a rehearsed deadline); the
  runbook exited nonzero, as expected.
- **Session 2:** session 1 attached; shard 1 resumed from 2 saved rows (re-score check: 2
  rows, ok); the fail-fast projection fit; merged 4 + 6 images into `aeroblade`.
- **Refusals:** a second merge and a re-score of a finished shard were refused.
- **The dry-run at HEAD follows the commit.**

### P4 — scoring session 2: AEROBLADE complete (option A); all 7 rows of the frozen spec scored — 2026-11-10

**Session** (Kaggle T4 x2, code `34a399a`, 2026-11-09). `F:\thesis-2\kaggle_raw\p4_reports_2.zip`: sha256
`79e55a8848b516f597e73b9cc7401065b6a1cd4bda4838cf5709106f7dcec4ae`, 862 entries, extracted
to `F:\thesis-2\kaggle_raw\p4_reports_2_x\`.
- **Plan:**
  - the 6 committed rows were complete (from the code), carried 0;
  - due were `aeroblade#0` on GPU 0 and `aeroblade#1` on GPU 1;
  - 3.728 Gpx dealt 1,863,819,392 / 1,863,824,448 px, 5,498 images each.
- **Fail-fast projection** (from the smoke after one warm-up image): 17.17 / 18.55 us per
  capped pixel, so 8.90 h / 9.61 h, against 10.5 h available after 0.08 h elapsed.
  **Started.**
- **Scoring:** **COMPLETE.** 10.09 h wall, 20.18 device-h by the runner's count; both jobs
  exited 0.
  - Shard 0: 9.35 h, 28 parts, 18.05 us/px.
  - Shard 1: 10.08 h, 28 parts, 19.47 us/px.
  - **The projection was 5% low:** the smoke's 6 images ran slightly faster per pixel than
    the full shards. Shard 1 finished 0.42 h before the deadline.
  - No resume was needed, so no `resumes.jsonl`.
- **Merge:** 9,405 / 700 / 891 rows.

**Verified** (locally, against the extracted output):
- **Each shard:**
  - the manifest sha256 equals the plan's;
  - every part matches its ledger line (`ShardStore.load`);
  - 5,498 rows in manifest order;
  - the ledger sha256 equals the done record's.
- **The merged CSVs:**
  - the image ids are in the committed lists' order;
  - every value equals its checkpoint value (rounded to the CSV's 6 decimals);
  - every score is minus the minimum of its three LPIPS layer-2 distances (within 1.5e-6);
  - all finite; the sha256 and row counts equal the summary's.
- **Cross-check:** `cut_by_cap` 2,888 / 0 / 0. Panel B's 2,888 equals the count of
  images with a side over 768, computed independently from `p3_panel_b_selection.csv`.
- **The 6 earlier rows** in this output are byte-identical to the committed ones.
- **AEs:** `sd15` @ `451f4fe1…`, `sd21` @ `bb215482…`, `movq_k21` @ `04b36665…`.
- **Versions:** torch 2.11.0+cu130, timm 1.0.30, diffusers 0.42.0, Pillow 13.0.0.

**Committed:**
- `reports/p4_baselines/aeroblade.json`;
- `reports/p4_baseline_scores/aeroblade/` (3 CSVs; the shard ledgers, meta and done records);
- `reports/kaggle_20261109/p4_baselines_session2/` (plan, projection, status, smoke status,
  hardware, 4 worker logs, the 2 shard manifests).
- **Not committed:** the 56 checkpoint part files duplicate the merged scores and stay in
  `F:\thesis-2\kaggle_raw\p4_reports_2_x\`. The committed ledgers carry their sha256 values.

**P4 scoring is complete:** all 7 rows of the frozen spec are scored once. Next is the
section 5 analysis (descriptive).

