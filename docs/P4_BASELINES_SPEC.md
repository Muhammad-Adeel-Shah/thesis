# P4 baselines on Panel B — specification (FROZEN, tag `p4-baselines-frozen`)

**Status:** written 2026-10-29, **before any baseline score exists**. Nothing here has
been run or downloaded; every value below was read by a command on the day stated
(section 9). The user's decisions of 2026-10-31 and 2026-11-01 are applied (section 7);
no item is open. **Frozen 2026-11-01** by the commit tagged `p4-baselines-frozen`,
before any baseline code or score existed. After the freeze, nothing in sections 1–6
changes.

**What this is (user decisions of 2026-10-29):**
- **Descriptive only.** No hypothesis, no decision rule, no PASS/FAIL. Every comparison
  is a point estimate with a paired bootstrap 95% CI.
- **Each baseline's own published preprocessing is the primary row.**
- **Our 320 centre crop is a sensitivity row for the cheap baselines only**, i.e. those
  without a VAE round trip: DDA (CO-SPY has one, section 1).
- **AEROBLADE is reproduced faithfully** (the paper's autoencoders, LPIPS setting and
  minimum over AEs; the Kandinsky 2.1 AE is approved under §9b).
- **"AEROBLADE-style (ours)" is not scored on Panel B** (user, 2026-10-31: saves about
  7.5 device-h). Its P2-panel result stays in the thesis as it is.
- **DDA is also scored, descriptively, under the two H5 decision conditions** (JPEG q50,
  resize-and-restore 0.5); CO-SPY is not, because it is not cheap (VAE round trip).

---

## 1. The baselines (read 2026-10-29)

| baseline | paper | code (licence) | weights | trained on |
|---|---|---|---|---|
| **DDA** | arXiv 2505.14359v6, NeurIPS 2025 | `roy-ch/Dual-Data-Alignment` @ `08d6b414a33826403b9096cfdb7758d5eae12dbb` (Apache-2.0) | `Junwei-Xi/Dual-Data-Alignment` @ `4390d9023899196b437480bb6a441915ef5d816c`, `DDA_ckpt.pth`, 1,255,621,296 B, sha256 `b27a31d39374803ddeff02bfabb2be76e190b04300490cddfafb24f683f37e3e` (README: "the official checkpoint"; card apache-2.0). Backbone `dinov2_vitl14`, LoRA rank 8, alpha 1, on `attn.qkv`, `attn.proj`, `mlp.fc1`, `mlp.fc2`; the score is `fc` on DINOv2's `x_norm_clstoken` (`Inference/models/dinov2_models*.py`). Upstream builds the backbone with **`torch.hub.load('facebookresearch/dinov2', name, pretrained=True)`**, which runs code downloaded from GitHub at an unpinned branch. So **we build the architecture from a pinned source that runs no downloaded code and load `DDA_ckpt.pth` into it** (user, 2026-11-01): `weights_only=True`, sha256 checked, **any missing or unexpected key stops the run**. The source is `timm`'s `vit_large_patch14_dinov2` with DINOv2's own positional-embedding interpolation, and the LoRA is DDA's `lora.py` (item 7.12) | MSCOCO **training set** reals + DDA-aligned SD-2.1-VAE reconstructions (`Junwei-Xi/DDA-Training-Set` README @ `6319d54c…`; paper App. A) |
| **CO-SPY** | arXiv 2503.18286v1, CVPR 2025 | `Megum1/CO-SPY` @ `4be0ba51fc6c96a06d5d3b12e36439d5d1507e16` (MIT) | `ruojiruoli/Co-Spy-Pretrained-Weights` @ `890ab2ee8dcf0254ff6ef34bcdc2f4906344007f`, `sd-v1_4.zip`, 316,332,881 B, sha256 `9626f6408b8f05a707ebaa3db98873e14ccd20a1224f4da0a48784ac5163d36e` (user decision 2026-10-31). **Official:** the repository's `pretrained/download.sh` fetches exactly this file. Plus two frozen backbones the code loads unpinned: **SigLIP** `ViT-SO400M-14-SigLIP-384`, `webli` via `open_clip` (`detectors/sd-v1_4/semantic.py`), on the Hub as `timm/ViT-SO400M-14-SigLIP-384` @ `ac16108d567c4389e6cd2b11c9b8585f7474435b`, `open_clip_model.safetensors`, 3,511,918,424 B, sha256 `97faeafa0576c1ba4c92bae739ca06ef5f5d0c2db92f86a6086653d82a262043`, apache-2.0. **Loaded from a local path only**; `open_clip_torch` is added at a pinned version (user, 2026-11-01, §9b; section 4); and the **VAE** of `CompVis/stable-diffusion-v1-4` (`detectors/sd-v1_4/artifact.py`), whose `vae/diffusion_pytorch_model.safetensors` (sha256 `a2b5134f4dbc140d9c11f11cba3233099e00af40f262f136c691fb7d38d2194c`) is **byte-identical to our pinned `sd15` member's** at `451f4fe1…`, so our pinned file is used | DRCT-2M / SD-v1.4: SD-v1.4 fakes from MSCOCO-2017 captions + MSCOCO reals (paper section 4) |
| **AEROBLADE** | arXiv 2401.17879v2, CVPR 2024 | `jonasricker/aeroblade` @ `50fc2b22c929fdc481757adc4bc81bf98d36f9fc` — **no licence** (GitHub API: none), so **our own implementation from the paper**; their code is not used, and the thesis says so | training-free; AEs (user decision 2026-10-31): our pinned **`sd15`** (SD1) and **`sd21`** (SD2) members, and **Kandinsky 2.1 MoVQ** (`kandinsky-community/kandinsky-2-1` @ `04b366659663bf60deacb96fd3e38f1cb39a3be4`, `movq/diffusion_pytorch_model.safetensors`, 271,380,362 B, sha256 `667319216c14764042b3e7cae7c4d1a309a409a674d413144e9ac65b572c6ea8`, apache-2.0) | — |
| **FIRE** | arXiv 2412.07140v3, CVPR 2025 | `Chuchad/FIRE` @ `ff1a316d668dc7fb7493ba70448769cf68d6029b` (MIT) | **"Imagenet with adm"** (user decision 2026-10-31), **Google Drive only**, no published hash. Fetched with `gdown` in the Kaggle session (section 4); its sha256 is recorded at the first download and committed before any baseline is scored. **Building `FIRE_model` upstream fetches** (`utils/network_utils.py`): (a) the whole `StableDiffusionPipeline` of `runwayml/stable-diffusion-v1-5` in **fp16**, unpinned. On 2026-11-01 that id resolves to `451f4fe16113bff5a5d2269ed5ad43b0592e9a14`, **the revision of our `sd15` pin**. (b) torchvision's ResNet-50 `IMAGENET1K_V2` weights (`download.pytorch.org`), loaded `strict=False`. Our adapter fetches neither: the ResNet is built with `weights=None`, and the VAE as in item 7.13. The upstream `eval.py` loads the whole `FIRE_model` state dict strictly, so the checkpoint should cover the ResNet, the ESPCN mask network and the VAE (INFERRED). **We stop if it does not cover every ResNet and ESPCN key**, and we report which VAE keys it holds | ADM fakes vs ImageNet reals (paper section 4) |

**Model-loading rules (CLAUDE.md §5):** `DDA_ckpt.pth` and the `.pth` files inside the
CO-SPY zip and the FIRE download are loaded with `torch.load(..., weights_only=True)`
only, after their sha256 is checked. **If `weights_only=True` refuses a file, stop and
ask; nothing is unpickled.** The Kandinsky MoVQ, the SigLIP model and the SD VAE are
loaded from `.safetensors`. The upstream code calls `torch.load` with no
`weights_only` (`Inference/inference.py`, `detectors/sd-v1_4/*.py`, `eval.py`); our
adapters do not.
- **CO-SPY's artifact weights** are loaded `strict=False` upstream. Our adapter records
  which keys the file holds, and it stops if those keys cover the VAE, because the pinned
  VAE would then not be the one in use.

## 2. Inputs (unchanged from the P3 final scoring)

- Panel B test and calibration rows (9,405, digest `267d9fde…`) and the luma companion
  (700): the committed lists `reports/kaggle_20261027/p3_final_session1/p3_final_lists/`.
- The 891 union threshold reals (491 COCO validation + 400 held-out Open Images).
- Every file is staged once with its sha256 checked (`p3_s2d_inputs.stage`), as in the
  final scoring.

## 3. Rows and preprocessing

Each row is one (baseline, preprocessing, condition) and is scored once.

| row | preprocessing (source) | condition | images |
|---|---|---|---|
| DDA native | DDA's `test_transform` (`Inference/inference.py`): torchvision `CenterCrop(336)` → `ToTensor` → `Normalize` with the CLIP mean (0.48145466, 0.4578275, 0.40821073) and std (0.26862954, 0.26130258, 0.27577711). No resize. **An image under 336 px on a side is zero-padded to 336** by `CenterCrop`. That is DDA's behaviour, adopted (user, 2026-10-31); a local check with torchvision 0.26 padded a 300 × 500 image with 18 zero columns on each side. Every row is scored, and the number of padded rows is recorded. Score: the logit (upstream applies a sigmoid, which is monotone and gives the same AUC, AP and threshold rank) | clean | 10,996 |
| DDA crop-308 (sensitivity) | as DDA native, but `CenterCrop(308)` (22 × 14, the largest multiple of the 14-px patch ≤ 320; user, 2026-10-31). DDA's inference file shows only 336, so nothing in the code says 320 is handled natively | clean | 10,996 |
| DDA native, H5 conditions | as DDA native, after `dbp.eval.perturb` (`jpeg50`, `resize050`) on the whole image | jpeg50, resize050 | 2 × 8,655 test rows |
| CO-SPY native | CO-SPY's `test_transform` (`detectors/sd-v1_4/fusion.py`): `Resize(384)` on the **short side**, keeping the aspect (torchvision default bilinear; images under 384 are upscaled) → `CenterCrop(384)` → `ToTensor`, in [0, 1]. Inside `forward`: the **semantic** branch normalises with mean = std = 0.5 and runs SigLIP; the **artifact** branch applies `Resize(224, antialias=False)` to the 384 crop and identity normalisation. It then does an SD VAE round trip of the [0, 1] tensor (`latent_dist.mean`, no scaling factor) and forms the residual `(x − x_recon) / 7 × 100`, which feeds a ResNet stem and its first two layers (`artifact.py`). A linear layer fuses the two logits. The paper's "224" is this inner resize. **Stated exception to CLAUDE.md §7** (user, 2026-11-01): this round trip runs **exactly as CO-SPY wrote it**, with its [0, 1] input, its 224 resize and the float residual, in fp32, with no tiling, loading our pinned `sd15` VAE file (byte-identical to the one CO-SPY loads). **Reason:** it is part of the baseline's learned forward pass, not a reconstruction of ours. Replacing it with `reconstruct.py` (a [-1, 1] centre crop, clamped uint8) would score a different detector from the published one. `reconstruct.py` stays the only path for every reconstruction of ours, AEROBLADE's included. Score: the fused logit | clean | 10,996 |
| AEROBLADE faithful | a centre crop of **at most 768 × 768**, no resize, then to a multiple of 8 (`reconstruct.py`). **Crop, never drop:** every row is scored, and the number of rows cut by the 768 cap is recorded (user, 2026-10-31; 768² is AEROBLADE's own data limit, README `--max_image_area 589824`). Each AE's round trip uses `reconstruct.py`, which gains a VQ path for the MoVQ, with tests. LPIPS-VGG16 **second layer only**, spatially averaged; score = − min over {`sd15`, `sd21`, KD2.1 MoVQ} (paper section 3; README "second LPIPS layer") | clean | 10,996 |
| FIRE native | the code's own `--resize` option (user, 2026-11-01; matches the paper's "resized to 256×256 along the shortest side"): `Resize(256, antialias=True)` on the PIL image's **short side**, aspect kept, `ToTensor`, in [0, 1], no normalisation (`dataset.py`), **one image per batch**. **Then the long side is centre-cropped down to the nearest multiple of 8, at most 7 px** (item 7.14; FIRE's own code fails on such sizes); the number of cropped rows is recorded. `FIRE_model()` with its defaults (`mode="frq"`, instance norm, radii 40/120; `eval.py`). Inside `forward` (`utils/network_utils.py`):<br>1. An FFT mid-band filter whose mask is predicted by a small ESPCN network. Each filtered image is divided by (max − min) over the whole batch, which is per image at batch 1.<br>2. **Two SD VAE round trips**, of the image and then of its filtered version, through `retrieve_latents`, which **samples** the latent (`latent_dist.sample`), in **fp16**, on the [0, 1] input, no scaling factor, no tiling.<br>3. The absolute residuals `abs(recon(x) − x)` and `abs(recon(filtered) − x)`, concatenated (6 channels), go to a ResNet-50 with instance norm.<br>**Stated exception to CLAUDE.md §7** (user, 2026-11-01; item 7.13): these two round trips run **exactly as FIRE published them** (fp16, sampled latent, filtered copy). **Reason:** like CO-SPY's, they are part of the baseline's learned forward pass; its ResNet was trained on residuals of fp16 sampled round trips, so `reconstruct.py` (fp32, `.mode()`) would score a different detector. **Seed rule:** each image's two draws come from one CPU `torch.Generator` seeded with the first 8 bytes of sha256(image id, UTF-8) as a big-endian integer masked to 63 bits; the image's draw comes first, then the filtered copy's (FIRE's order). Every score is reproducible. **VAE weights** from the FIRE checkpoint if it carries the full `vae.*` key set, otherwise our pinned `sd15` file; no pipeline and no torchvision download. Score: the logit `o` (upstream applies a sigmoid) | clean | 10,996 |

## 4. Scoring protocol (as the P3 final scoring)

- A pre-flight checks the inputs, the weight hashes (section 1) and the imports, then runs
  a 2-image smoke test per row, before any long job.
- Workers are fail-fast, one per T4. **Once only:** a row's summary marks it done; a
  re-score is refused, and complete rows are carried into a later session.
- **Score direction:** larger means more likely fake, for every row (AEROBLADE: the
  negated distance).
- **Threshold:** each row's own k = 0 union threshold (`calibrate_on_reals` at 5% FPR on
  its 891 threshold-real scores).
- **Environment:** Kaggle's installed Pillow and libjpeg are recorded, not changed
  (§3 least change). The DDA README warns that results depend on the JPEG decoder and
  pins Pillow 12.3.0 with libjpeg-turbo 3.1.4.1; the difference is a stated caveat.
- **`open_clip_torch`** (user, 2026-11-01, §9b): added to `requirements/` at a pinned
  version. Before it is installed on Kaggle, `pip install --dry-run` runs, and the torch,
  torchvision and numpy versions are printed before and after. **If any of the three would
  change, stop and report.** It also brings `timm`, which is pinned with it:
  `open_clip_torch==3.3.0` and `timm==1.0.30` (section 9). `env/kaggle_bootstrap.sh`
  already runs this dry-run and before/after check on every install.
- **FIRE's checkpoint** (user, 2026-10-31, under the WildRF rule): `gdown` at a pinned
  version, installed with `--no-deps`, in the Kaggle session only, never in
  `requirements/`. torch, torchvision and numpy versions are printed before and after the
  install. The file's sha256 is recorded at the first download and **committed before any
  baseline is scored**; every later session checks it. **If Google Drive refuses, stop and
  report; no mirror.**
- **Undecodable files** are not repaired: `ImageFile.LOAD_TRUNCATED_IMAGES` (set in
  FIRE's `dataset.py`) and FIRE's skip-to-the-next-image `except` are not adopted.
  Panel B was fully decoded when it was built.
- Per-image scores are written under `reports/p4_baseline_scores/` and committed like the
  P3 final scores (Panel B holds no private images; TrueFake is not scored here).

## 5. What is reported (frozen list; all descriptive)

For **every row** of section 3:
1. **Per slice** (every Panel B test slice, E7 luma included; E8 fakes-only): AUC and AP
   with 95% CIs; TPR at the row's union threshold.
2. **Per real pool** (E3, E4, E5, E7, E7 luma): achieved FPR at the row's union
   threshold, with a 95% CI.
3. **Macro AUC over the 4 unseen slices** (the P3 definition), with a 95% CI;
   **macro FPR over E4 and E5**.
4. **Paired differences** against S2-D seed 0, S2-D seed 1 and P2 `full_bank`, on their
   committed clean scores: the macro AUC over the 4 unseen slices, each slice's AUC, and
   the macro FPR over E4/E5, each with a 95% CI.
5. **DDA under the H5 conditions:** macro AUC under `jpeg50` and `resize050` with CIs, its
   difference from DDA clean, and the paired difference against S2-D seed 0 under the same
   condition (committed robust scores).

**Bootstrap:** 2,000 resamples, seed 2026, one RNG per analysis. In each resample the real
pools are drawn once and shared by every row and model; fakes are drawn within their
slice. Percentile 95% intervals. This is the P3 analysis's `bootstrap`, unchanged.
**Labels:** E7 FLUX.1 standard is an upper bound (§6d) for every row.

## 6. Caveats stated wherever a baseline number appears

- **E7 is DDA's home benchmark:** DDA-COCO was built by DDA's authors, from MSCOCO
  validation images reconstructed by VAEs. DDA trained on the MSCOCO *training* set, so
  E7's reals are not in its training data.
- **CO-SPY's training reals are MSCOCO** (DRCT-2M). Their split is **NOT-VERIFIED**; an
  overlap with E7's val2017 reals is not ruled out.
- **FIRE was trained on ADM fakes** (a pixel-space model, no VAE), and collapsed zero-shot
  in our P0 reproduction; its row measures that baseline as published.
- **AEROBLADE is our implementation from the paper**, because its repository has no
  licence. Its whole-image setting is capped at a 768 × 768 centre crop (the count of cut
  rows is reported). Its SD1/SD2 AEs are our pinned `sd15`/`sd21` members.
- **DDA zero-pads images under 336 px** (its own `CenterCrop`); the count is reported.
- **CO-SPY upscales images under 384 px** (its own `Resize(384)`).
- **Deployment (CLAUDE.md §6c):** the baseline numbers carry Panel B's licence limits;
  E3 is research only (its README; CLAUDE.md §6c corrected 2026-10-31).

## 7. Decisions of 2026-10-31, and items still open

**Resolved (user, 2026-10-31), with what the code showed:**
1. **DDA below 336 px:** DDA's `Inference/inference.py` evaluates with
   `CenterCrop(336)`, which zero-pads; adopted (section 3). Of the 9,405 Panel B rows,
   1,281 have a short side under 336 (E5 1,180, E7 94, E3 7; counted 2026-10-29).
2. **FIRE and CO-SPY transforms:** read (section 3). CO-SPY: resize the short side to 384,
   centre crop 384, with an inner resize to 224 in the artifact branch. FIRE: see 7.10.
3. **DDA sensitivity crop:** 308 (22 × 14). The inference file shows only 336.
4. **AEROBLADE size:** a centre crop of at most 768 × 768; crop, never drop; the number of
   cut rows is recorded.
5. **AEROBLADE AEs:** `sd15`, `sd21` and Kandinsky 2.1 MoVQ. `reconstruct.py` gains a VQ
   path with tests. Our own implementation from the paper (no licence upstream).
6. **FIRE weights:** ImageNet-ADM, by `gdown` on Kaggle (section 4).
7. **CO-SPY weights:** `sd-v1_4.zip`.

**Change (user, 2026-10-31):** the "AEROBLADE-style (ours, 11 VAEs)" row is dropped from
Panel B scoring.

**Decided 2026-11-01 (user), on the items the 2026-10-31 reads raised:**

8. **CO-SPY's VAE round trip:** the stated exception to CLAUDE.md §7 (section 3, CO-SPY
   row, with its reason).
9. **`open_clip_torch`:** approved under §9b. SigLIP is pinned at `ac16108d…` and loaded
   from a local path only; dry-run rule in section 4. CO-SPY is not dropped.
10. **FIRE:** the code's `--resize` option (section 3). The read of
    `utils/network_utils.py` shows that building `FIRE_model` fetches the SD-1.5 pipeline
    (fp16) and the ImageNet ResNet-50 weights (section 1).
11. **DDA:** the two model files were read. They use `torch.hub`, so the architecture is
    built from a pinned source and `DDA_ckpt.pth` is loaded into it (section 1).
    - **Equivalence check, approved only in isolation:** a separate CPU-only Kaggle
      notebook, with no Secrets and no private dataset attached.
    - **Inputs:** 2 COCO val2017 images. **Outputs:** the two logits from each
      implementation.
    - **Pass:** the logits agree within 1e-3. Nothing from that session is reused.

**Decided 2026-11-01 (user), on the items the model-code reads raised:**

12. **DDA: LoRA formulation, and the architecture source.**
    - **LoRA, as `Inference/models/lora.py` defines it** (read 2026-11-01, approved):
      - `LoRALinear(original_layer, rank, alpha)` wraps a target `nn.Linear`; its output
        is `original_layer(x) + (x Aᵀ) Bᵀ · (alpha / rank)`, with `lora_A` of shape
        rank × in and `lora_B` of shape out × rank. With rank 8 and alpha 1 the scale is
        **1/8**.
      - **Targets** are matched by substring of the module name (`attn.qkv`,
        `attn.proj`, `mlp.fc1`, `mlp.fc2`), so every block's four linears are wrapped.
      - **Parameter names:** `<module>.original_layer.weight`,
        `<module>.original_layer.bias`, `<module>.lora.lora_A`, `<module>.lora.lora_B`.
        The backbone sits at `base_model.model`, the score head at `base_model.fc`, and
        DDA's inference loads `checkpoint['model']` strictly.
      - Our adapter implements exactly this; DDA's code is not imported.
    - **Architecture:** `timm`'s `vit_large_patch14_dinov2` with `pretrained=False`
      (`timm` pinned in `requirements/` with `open_clip_torch`). DINOv2's hub code was
      read to match it (`facebookresearch/dinov2` @ `7764ea0f…`, the `main` branch that
      `torch.hub` fetched on 2026-11-01; section 9). It shows:
      - `dinov2_vitl14` is `vit_large` with image size 518, patch 14, LayerScale,
        an MLP FFN, no block chunks and no register tokens;
      - it carries an unused `mask_token` (1 × 1024);
      - the positional embedding (1 + 37 × 37 tokens) is interpolated bicubically in
        fp32 with `scale_factor = ((w0 + 0.1) / 37, (h0 + 0.1) / 37)` and no antialias
        (`interpolate_offset=0.1`, `interpolate_antialias=False`).
    - **So our model:**
      - computes the tokens with DINOv2's interpolation exactly as above, not with
        `timm`'s own resampling;
      - registers a `mask_token` parameter, so the key sets can match;
      - scores `fc` on the final-norm class token, as DDA does.
    - **Loading:** sha256 checked, `weights_only=True`, `checkpoint['model']`, strict.
      **Any missing or unexpected key stops the run.**
    - **If the isolated 2-image check (item 11) fails,** the cause is fixed, for example
      by matching DINOv2's interpolation more exactly. **The 1e-3 tolerance is never
      loosened.**
13. **FIRE's in-forward VAE round trips:** the stated exception to CLAUDE.md §7, run as
    published (fp16, `latent_dist.sample()`, the FFT-filtered copy), with the seed rule
    (section 3, FIRE row, with its reason). VAE from the FIRE checkpoint if it carries
    one, else our pinned `sd15` file. No pipeline and no torchvision download.
14. **FIRE image sizes:** after the 256 short-side resize, the long side is centre-cropped
    down to the nearest multiple of 8 (at most 7 px), and the count is recorded
    (section 3).

**GPU plan:** the revised total of 4.1–11.4 device-h is approved (user, 2026-11-01).

## 8. GPU plan (ESTIMATED; T4 × 2; nothing measured for any baseline yet; revised 2026-11-01, approved)

| row | basis | device-h |
|---|---|---|
| DDA native + crop-308 + 2 H5 conditions (39,302 passes) | DINOv2 ViT-L/14 at 336: 0.06–0.12 s/image ESTIMATED, against our ViT-B at 320, 0.019 s (MEASURED, `reports/p3_final/*_clean.json`) | 0.7–1.3 |
| CO-SPY native (10,996) | **revised:** SigLIP SO400M at **384** (729 tokens) plus the SD VAE round trip at 224 (P2 MEASURED `sd15` 0.18 s at 320, about 0.09 s area-scaled to 224) plus a ResNet stem: 0.2–0.45 s/image | 0.6–1.4 |
| FIRE native (10,996) | **revised 2026-11-01:** two SD VAE round trips in fp16 at a short side of 256, plus an FFT filter, ESPCN and ResNet-50 (`FIRE_model` read). Per round trip, P2 MEASURED `sd15` 0.18 s at 320² fp32, area-scaled to about 256 × 341 and with fp16 taken as no slower: 0.15–0.4 s/image | 0.5–1.2 |
| AEROBLADE faithful, cap 768 (about 3.7 Gpx) | per AE at crop 320 (MEASURED P2: SD 0.18 s, Kandinsky-2.2 MoVQ 0.38 s as a proxy for 2.1); lower bound per-call cost, upper bound area-scaled (I-048) | 2.3–7.5 |
| ~~AEROBLADE-style (ours)~~ | dropped (user, 2026-10-31) | 0 |
| **total** | | **4.1–11.4** |

- **Wall time:** about 2–6 h on two T4s, including about 0.3 h per session for the
  pre-flight, staging and model loads.
  - **One session** if the low end holds; two if AEROBLADE runs at the area-scaled rate.
  - AEROBLADE is the long pole: put it alone on one GPU, and DDA, CO-SPY and FIRE on the
    other.
- **Resumable:** complete rows are carried forward between sessions.
- **Quota:** 1 hour per session hour, about 2–6 h, on `muhammadadeelshah` only (§13).
- **Revision of 2026-11-01:** FIRE turned out to hold two VAE round trips, so its row rose
  from 0.1–0.3 to 0.5–1.2 device-h, and the total from the approved 3.7–10.5 to 4.1–11.4.
  The wall time stays within the same 2–6 h. The user approved the revised total on
  2026-11-01.

## 9. Provenance (sources read on 2026-10-29, 2026-10-31 and 2026-11-01)

Hugging Face Hub API (metadata; files read in full where named):
- `Junwei-Xi/Dual-Data-Alignment`: model info with file metadata; `README.md` @ `4390d902…`.
- `Junwei-Xi/DDA-Training-Set`: dataset info; file list; `README.md` @ `6319d54c…`
  (2,289 B, sha256 `11720cc77862afd3d9090c1369522ce3632d13703d68bc6bef17b53057508fbc`).
- `ruojiruoli/Co-Spy-Pretrained-Weights`: model info with file metadata; `README.md`
  @ `890ab2ee…`.
- `kandinsky-community/kandinsky-2-1`: model info with file metadata.
- `zr-zhang/MLLM-Generated-Image-Detection-Dataset`: dataset info; `README.md` @
  `1498eead…` and @ `651a325f…` (byte-identical, sha256 `1c7e4f0b…2ef6f`).

GitHub and arXiv (status 200 for every URL; sha256 of the bytes received):

| time (UTC) | status | bytes | sha256 | URL |
|---|---|---|---|---|
| 2026-10-29T06:20:58Z | 200 | 8,354 | `be2249c861b41bd2e6d142b75c8a59fd6cd0d35b2f6acd6b1d1fa97022dfa9af` | https://export.arxiv.org/api/query?id_list=2605.14486,2601.20461,2505.14359&max_results=10 |
| 2026-10-29T06:21:02Z | 200 | 2,976 | `f401ea2d239bc090807deee38b3d272e0db68e338c39dd2083ff84f6e7f3782b` | https://export.arxiv.org/api/query?search_query=ti:%22CO-SPY%22&max_results=5 |
| 2026-10-29T06:21:06Z | 200 | 2,961 | `fb332fc95dc8666769f786e302a70efc49e672527835a9ced7ca77f55508c66a` | https://export.arxiv.org/api/query?search_query=ti:AEROBLADE&max_results=5 |
| 2026-10-29T06:21:09Z | 200 | 2,929 | `e69f1624929325f6b0ce6887fbe0336d67100798ad817da134567c7c595b373b` | https://export.arxiv.org/api/query?search_query=ti:%22Frequency-Guided%20Reconstruction%20Error%22&max_results=5 |
| 2026-10-29T06:21:45Z | 200 | 3,545 | `61b9191a4dedbaa26bc3b8cfd18676a0434f6bd9b8a80bb6bad49153511dc70e` | https://export.arxiv.org/api/query?id_list=2605.14486v1 |
| 2026-10-29T06:22:28Z | 200 | 3,537 | `36508dd8799cf3fd8ecf302aca510acf5700a40e460663d51be4f88314c3c02e` | https://export.arxiv.org/api/query?id_list=2608.01258 |
| 2026-10-29T06:22:51Z | 200 | 5,644 | `90bc12b93c3b9e710ffc1d45c12e0c7f06259c1a75884c9ec4f85dbceae5e14e` | https://api.github.com/repos/roy-ch/Dual-Data-Alignment |
| 2026-10-29T06:22:55Z | 200 | 5,067 | `ec532bd0021c7d3289d3ce126446ac2f2a095ff830bf20ffbe14e0f7f0e24b5d` | https://api.github.com/repos/Megum1/Co-Spy |
| 2026-10-29T06:22:58Z | 200 | 5,546 | `216b3de0846708c9f4233db1750f082453ce981903e4c9e2470ec4a0a98c7335` | https://api.github.com/repos/jonasricker/aeroblade |
| 2026-10-29T06:23:02Z | 200 | 5,427 | `b23518aeda83901895cf07cd2f55d9e144720351f20eb8d1ba4aa41951abf882` | https://api.github.com/repos/liyih/SEF_AIGC_detection |
| 2026-10-29T06:23:06Z | 200 | 55 | `4af480b8ee5b87b369a76c49bd22c9a783908272ebffbe97898f8ab0f0772a5f` | https://api.github.com/search/repositories?q=%22Frequency-Guided+Reconstruction+Error%22 |
| 2026-10-29T06:23:09Z | 200 | 55 | `4af480b8ee5b87b369a76c49bd22c9a783908272ebffbe97898f8ab0f0772a5f` | https://api.github.com/search/repositories?q=FIRE+diffusion+detection+frequency+reconstruction |
| 2026-10-29T06:23:26Z | 200 | 6,734 | `e8b977b0b9f0dc60f74d283a97b6e0092e8c189e592b7b7a86bbc905daacd78c` | https://api.github.com/repos/roy-ch/Dual-Data-Alignment/commits/main |
| 2026-10-29T06:23:29Z | 200 | 4,968 | `9ab2751b5dba097b502b0e6aedfc83929295958b43dcc7d585ed84eac1114378` | https://api.github.com/repos/Megum1/CO-SPY/commits/main |
| 2026-10-29T06:23:33Z | 200 | 5,722 | `002f232d202ab571a5469fb10174842460b308c08ee4b3e5fcae3ae3a5d4408f` | https://api.github.com/repos/jonasricker/aeroblade/commits/main |
| 2026-10-29T06:23:48Z | 200 | 12,090 | `1d7db3f692ccd4c5d2bb2d32971bb01f5acf3b8409052cc8d69b4c020bf6aa09` | https://raw.githubusercontent.com/roy-ch/Dual-Data-Alignment/08d6b414a33826403b9096cfdb7758d5eae12dbb/README.md |
| 2026-10-29T06:23:52Z | 200 | 11,357 | `c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4` | https://raw.githubusercontent.com/roy-ch/Dual-Data-Alignment/08d6b414a33826403b9096cfdb7758d5eae12dbb/LICENSE |
| 2026-10-29T06:23:56Z | 200 | 15,596 | `ad0dd0e67ea624f32f090cf13c0adcab31590b79a2b9f6fc4293f03cae5f61ce` | https://raw.githubusercontent.com/Megum1/CO-SPY/4be0ba51fc6c96a06d5d3b12e36439d5d1507e16/README.md |
| 2026-10-29T06:23:59Z | 200 | 1,069 | `39522a3c720bdecf4b6afcda92278f9455be1a81f21acd228a8ac7b4afbe1488` | https://raw.githubusercontent.com/Megum1/CO-SPY/4be0ba51fc6c96a06d5d3b12e36439d5d1507e16/LICENSE |
| 2026-10-29T06:24:03Z | 200 | 5,834 | `3cc84fb8a44b83890160f336e90486c3c726126e0a28475ce2676f878cb89a23` | https://raw.githubusercontent.com/jonasricker/aeroblade/50fc2b22c929fdc481757adc4bc81bf98d36f9fc/README.md |
| 2026-10-29T06:24:37Z | 200 | 810 | `fe1f0b39c32d3a558253078e7600155fcf69c4cdb721bfa773d2664be16b1ed1` | https://raw.githubusercontent.com/Megum1/CO-SPY/4be0ba51fc6c96a06d5d3b12e36439d5d1507e16/pretrained/download.sh |
| 2026-10-29T06:24:53Z | 200 | 495,042 | `fb077ad9ea095db50cd1ac52582541b1a73f891f600b4fce6dcc6946eaaa0d55` | https://arxiv.org/html/2401.17879v2 |
| 2026-10-29T06:24:58Z | 200 | 519,132 | `f7d17dee237a40bc4e7fbdfefb8acc40f8ec618080961edd0d5ff1704a3f3835` | https://arxiv.org/html/2505.14359v6 |
| 2026-10-29T06:25:02Z | 200 | 1,085,185 | `6621516981997c204c2a26dea1eae2817d35004a3c291774b8f39aa2ddda3307` | https://arxiv.org/html/2503.18286v1 |
| 2026-10-29T06:25:11Z | 200 | 261,928 | `fdd05d2da389ac4f4df58be6a95b8cfab99e09ae9c594424b0207973d426e570` | https://arxiv.org/html/2412.07140v3 |
| 2026-10-29T06:25:46Z | 200 | 4,863 | `39a1831568fef4e1f17d1acc5fa82985c19244612c7046c7434c5bc3bfa3d458` | https://api.github.com/repos/Chuchad/FIRE |
| 2026-10-29T06:25:49Z | 200 | 4,268 | `79beab2b7e8afac02edbb4325e0ab2f929fa9caef1c4a7a5e95e872db0b3f457` | https://api.github.com/repos/Chuchad/FIRE/commits/main |
| 2026-10-29T06:26:03Z | 200 | 5,246 | `72e2796f11c7403786c3d72c34c4d99d9272f3373c59dfb771625347ea7ec76d` | https://raw.githubusercontent.com/Chuchad/FIRE/ff1a316d668dc7fb7493ba70448769cf68d6029b/README.md |
| 2026-10-29T06:26:06Z | 200 | 1,067 | `fa03e7a7f25052df0cd510158af143e4e4e654857bbdf23cdc4f2abf698d1d09` | https://raw.githubusercontent.com/Chuchad/FIRE/ff1a316d668dc7fb7493ba70448769cf68d6029b/LICENSE |
| 2026-10-31T14:29:29Z | 200 | 10,163 | `47b03191b58390e1b391ba4ff3e9ecbf1adfcac5e4b708498293c1aac4f312f4` | https://api.github.com/repos/roy-ch/Dual-Data-Alignment/git/trees/08d6b414a33826403b9096cfdb7758d5eae12dbb?recursive=1 |
| 2026-10-31T14:29:37Z | 200 | 11,230 | `aae45daa74dbca599b5fbfdad6a3831eed59b99776906705cb1cfe9180f4fa28` | https://api.github.com/repos/Megum1/CO-SPY/git/trees/4be0ba51fc6c96a06d5d3b12e36439d5d1507e16?recursive=1 |
| 2026-10-31T14:29:44Z | 200 | 4,864 | `cdc5e4ed9991cc68abf6396bf110e4903503f7485c5e6232976e0db14b20badf` | https://api.github.com/repos/Chuchad/FIRE/git/trees/ff1a316d668dc7fb7493ba70448769cf68d6029b?recursive=1 |
| 2026-10-31T14:29:58Z | 200 | 1,551 | `2ce5bb15a95a10b3c06a31b5b561b749e1e5acec735fee1b7596be629dbcbeba` | https://raw.githubusercontent.com/roy-ch/Dual-Data-Alignment/08d6b414a33826403b9096cfdb7758d5eae12dbb/Inference/inference.py |
| 2026-10-31T14:30:04Z | 200 | 3,980 | `6b678bb26cf7b0c9691fe69f06e02fcf574f0dd8d1cb236106a4469140fea2c6` | https://raw.githubusercontent.com/Chuchad/FIRE/ff1a316d668dc7fb7493ba70448769cf68d6029b/eval.py |
| 2026-10-31T14:30:08Z | 200 | 8,511 | `2679a09051ba4199d17ab2778af837a20c7354e3eac6f3d251115518f4b607a6` | https://raw.githubusercontent.com/Megum1/CO-SPY/4be0ba51fc6c96a06d5d3b12e36439d5d1507e16/evaluate.py |
| 2026-10-31T14:30:34Z | 200 | 10,622 | `b98f52e675a9654ab5a3575d8b7cd76d323fec87dd8b335210d21bc05621fa78` | https://raw.githubusercontent.com/Chuchad/FIRE/ff1a316d668dc7fb7493ba70448769cf68d6029b/dataset.py |
| 2026-10-31T14:30:39Z | 200 | 3,854 | `30bf2587fb8e7b7f64dd5b03b40e5b748d791d5d1345865532954baa5d4e21c5` | https://raw.githubusercontent.com/Megum1/CO-SPY/4be0ba51fc6c96a06d5d3b12e36439d5d1507e16/detectors/sd-v1_4/fusion.py |
| 2026-10-31T14:31:03Z | 200 | 261 | `0c7f3feb2391343c23f7ab9fe6dea2435b3962da533b4414a89b1f7515e0472e` | https://raw.githubusercontent.com/Chuchad/FIRE/ff1a316d668dc7fb7493ba70448769cf68d6029b/eval_df.sh |
| 2026-10-31T14:31:07Z | 200 | 236 | `b35eadaf0b8521b658b623349543de24edf59d97d3fd4e76076e80212123f852` | https://raw.githubusercontent.com/Chuchad/FIRE/ff1a316d668dc7fb7493ba70448769cf68d6029b/eval_fi.sh |
| 2026-10-31T14:31:10Z | 200 | 6,284 | `e3971d1c6731f1ab1fdcc5c1871b082177c6e13f26021506741e5a635f619bf9` | https://raw.githubusercontent.com/Megum1/CO-SPY/4be0ba51fc6c96a06d5d3b12e36439d5d1507e16/detectors/sd-v1_4/artifact.py |
| 2026-10-31T14:31:50Z | 200 | 3,522 | `c1a52f7c4e813c644f4349bdcbaef139bb5d9a6e6ec8bef2e17c5c3eb0e401dd` | https://raw.githubusercontent.com/Megum1/CO-SPY/4be0ba51fc6c96a06d5d3b12e36439d5d1507e16/detectors/sd-v1_4/semantic.py |
| 2026-10-31T14:31:53Z | 200 | 1,112 | `ef7a6f9e307c00a2580ac2646f358d849e68071f19de622b65454065b42d204d` | https://raw.githubusercontent.com/Megum1/CO-SPY/4be0ba51fc6c96a06d5d3b12e36439d5d1507e16/detectors/sd-v1_4/base.py |

2026-10-31, Hugging Face Hub API (`HfApi.model_info(..., files_metadata=True)`, token
present, not printed):
- `timm/ViT-SO400M-14-SigLIP-384`: sha `ac16108d567c4389e6cd2b11c9b8585f7474435b`, not
  gated, apache-2.0; `open_clip_model.safetensors` 3,511,918,424 B, sha256
  `97faeafa0576c1ba4c92bae739ca06ef5f5d0c2db92f86a6086653d82a262043`.
- `CompVis/stable-diffusion-v1-4`: sha `133a221b8aa7292a167afc5127cb63fb5005638b`, not
  gated, creativeml-openrail-m; `vae/diffusion_pytorch_model.safetensors` 334,643,276 B,
  sha256 `a2b5134f4dbc140d9c11f11cba3233099e00af40f262f136c691fb7d38d2194c`.
- `stable-diffusion-v1-5/stable-diffusion-v1-5` @ `451f4fe16113bff5a5d2269ed5ad43b0592e9a14`
  (our `sd15` pin): `vae/diffusion_pytorch_model.safetensors` 334,643,276 B, the same
  sha256.

2026-10-31, local check (torchvision 0.26.0): `CenterCrop(336)` on a 300 × 500 image
returns 3 × 336 × 336, with the 18 leftmost and 18 rightmost columns all zero.

2026-11-01, read-only (same fetcher; approved by the user 2026-11-01):

| time (UTC) | status | bytes | sha256 | URL |
|---|---|---|---|---|
| 2026-11-01T14:23:51Z | 200 | 2,591 | `c974400dfab533d773fa013c33f0fd24f5a3478c122936a5f5b8e327264f559d` | https://raw.githubusercontent.com/roy-ch/Dual-Data-Alignment/08d6b414a33826403b9096cfdb7758d5eae12dbb/Inference/models/dinov2_models_lora.py |
| 2026-11-01T14:23:55Z | 200 | 1,062 | `68e6453533c88f4423396572fd7ef7c587f5c33a94583ffb63b5b0d913f46c4b` | https://raw.githubusercontent.com/roy-ch/Dual-Data-Alignment/08d6b414a33826403b9096cfdb7758d5eae12dbb/Inference/models/dinov2_models.py |
| 2026-11-01T14:23:58Z | 200 | 9,861 | `117073c16b63c9afde204691fb4ac82a9b7d1bd08a2419f2e23d23ff9834e461` | https://raw.githubusercontent.com/Chuchad/FIRE/ff1a316d668dc7fb7493ba70448769cf68d6029b/utils/network_utils.py |

2026-11-01, Hugging Face Hub API: `runwayml/stable-diffusion-v1-5` resolves to sha
`451f4fe16113bff5a5d2269ed5ad43b0592e9a14`, our `sd15` pin. Local, diffusers 0.40.0:
`retrieve_latents(encoder_output, generator=None, sample_mode="sample")` returns
`latent_dist.sample(generator)` by default (`inspect.getsource`). PyPI (`pip index
versions`): `open_clip_torch` latest 3.3.0.

2026-11-01, second read (same fetcher). `lora.py` was approved by the user; DINOv2's hub
code was read to match its interpolation, as the user's item 7.12 asks. The DINOv2 commit
is `main` as the GitHub API returned it (`7764ea0f912e53c92e82eb78a2a1631e92725fc8`,
committed 2026-06-03T15:28:49Z); its licence file is Apache-2.0.

| time (UTC) | status | bytes | sha256 | URL |
|---|---|---|---|---|
| 2026-11-01T16:52:35Z | 200 | 3,355 | `610b3c436a4e3fa47315f7f98c3e4e5887e352d173687fc2b7b7a64d8fc13f10` | https://raw.githubusercontent.com/roy-ch/Dual-Data-Alignment/08d6b414a33826403b9096cfdb7758d5eae12dbb/Inference/models/lora.py |
| 2026-11-01T16:54:09Z | 200 | 15,483 | `8d514590738b315c8e13a194e5e3527039f8ed13a81ae7a7263d5db97460aa95` | https://api.github.com/repos/facebookresearch/dinov2/commits/main |
| 2026-11-01T16:54:21Z | 200 | 1,126 | `c1f5090e78ff940b72c076d2bf9c0310d1707c946b3d10e2d6f2b0bdf56a6f64` | https://raw.githubusercontent.com/facebookresearch/dinov2/7764ea0f912e53c92e82eb78a2a1631e92725fc8/hubconf.py |
| 2026-11-01T16:54:25Z | 200 | 5,586 | `871fca671b12a9ff02e810654baf509e97ccf461bf8196ce5ddeefff2fd87d3e` | https://raw.githubusercontent.com/facebookresearch/dinov2/7764ea0f912e53c92e82eb78a2a1631e92725fc8/dinov2/hub/backbones.py |
| 2026-11-01T16:54:29Z | 200 | 16,438 | `7799a260f2d7d0fe197331d08502fb8c542f9b7424723650f6a39b64fa2639ea` | https://raw.githubusercontent.com/facebookresearch/dinov2/7764ea0f912e53c92e82eb78a2a1631e92725fc8/dinov2/models/vision_transformer.py |
| 2026-11-01T16:54:34Z | 200 | 11,359 | `600cc67cc4cb2f5ea317dcfc687ad1c74dc4bec8782bbe9db0afd83513b935b7` | https://raw.githubusercontent.com/facebookresearch/dinov2/7764ea0f912e53c92e82eb78a2a1631e92725fc8/LICENSE |

2026-11-01, local `pip install --dry-run --report open_clip_torch==3.3.0` in the project
venv: it would install `open_clip_torch` 3.3.0 (requires `timm>=1.0.17`), `timm` 1.0.30,
`ftfy` 6.3.1 and `wcwidth` 0.9.4, and nothing else. torch, torchvision and numpy are
untouched. These two versions are pinned in `requirements/` (section 4).

