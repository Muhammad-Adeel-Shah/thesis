# P5 — Video module: plan (DRAFT, not frozen; no training)

**Status:** DRAFT, written 2026-11-20 on branch `p5-video` (worktree
`F:/thesis-2/worktrees/p5-video/thesis_dbp`, from `master` at `54180d3`). Nothing here is
frozen. The pre-registration (`docs/P5_PREREGISTRATION.md`, not yet written) is drawn from
this plan after the pilot (section 8) and frozen on **2026-11-22**, before any P5 model
trains and after the test panel is frozen.

> **Decisions of 2026-11-21 (the user's review of `0f0d5a4`; they override the text below
> wherever the two differ; the pre-registration consolidates them):**
>
> - **A1:**
>   - **Sources:** Pexels (training), AIGVDBench (the main test set) and the AEGIS hard
>     test (clean, descriptive only).
>   - **Rejected:** Mixkit, GenVideo and VBench.
>   - **Licence:** Pexels is treated as the Pexels License (research, private, no
>     redistribution); the Hub's apache tag is recorded beside it.
>   - **Citation:** AIGVDBench is cited as Ma et al., "Your One-Stop Solution for
>     AI-Generated Video Detection", CVPR 2026, arXiv:2601.11035.
>   - **Calibration pool:** a YouTube calibration pool for V4 k-shot is reserved from the
>     AIGVDBench reals, disjoint from the test reals.
> - **A2:**
>   - **Where:** D3 and DeCoF code and weights are fetched only in an isolated Kaggle session
>     (no Secrets, no private data), with pinned commits and recorded hashes.
>   - **If DeCoF's weights are unavailable:** it is dropped, and the baselines become D3 +
>     image S2-D per frame + DDA per frame.
> - **A3:**
>   - **Dependencies:** `av==19.0.2` only. On Kaggle, the preinstalled pyarrow is used.
>   - **Encoders:** libx264/libx265 are verified and the FFmpeg build is recorded.
>   - **Re-encodes:** every test-panel re-encode runs on Kaggle.
> - **A4 / compute:**
>   - **Kaggle:** the pilot; the panel and its shortcut gate; the baseline fetch; all
>     scoring.
>   - **AWS g5.2xlarge (us-east-2):** Pexels, the reconstruction and training, under a
>     **$80** budget.
>   - **RunPod:** LOLO overflow.
>   - **The uni PC:** 5-hour resumable jobs only.
>   - **Test data never leaves Kaggle.**
> - **A5:**
>   - **Clips:** **25 consecutive native frames**, crop 320, no resize, no temporal
>     resampling. The pre-declared minimum-size rule and exclusions are in
>     `src/dbp/data/video.py`.
>   - **Pairs:** a real and its reconstruction share the decode/crop path and are stored
>     lossless.
>   - **Degradation:** p = 0.35, plus H.264/H.265 re-encodes (a CRF range), applied
>     identically to both clips of a pair.
>   - **Timing:** the training-data reconstruction may start before the freeze.
> - **A7:**
>   - **Thresholds:** P3's.
>   - **Primary:** V1 (LOLO) and V2 (bank vs single, on unseen AIGVDBench generators), with
>     97.5% CIs.
>   - **Secondary:** V3–V5. **V6 baselines:** descriptive.
>   - **V5:** H.264 CRF 28 and resize 0.5, within 5 points.
>   - **The shortcut gate:** before the panel freeze, a metadata-only AUC (fps, resolution,
>     duration, codec/bitrate). If any is > 0.70, the reals and fakes are
>     stratified/rebalanced, then frozen.
>   - **The primary view is format-matched:** reals and fakes re-encoded identically. Raw is
>     secondary.
> - **Pilot:** bank membership may change **only for technical failure**. The Qwen-Image vs
>   Wan2.1 weight comparison is reported; if they are identical, Wan results are flagged
>   "seen decoder".
> - **Schedule:**
>   - **Sat 21:** P5-B and the Kaggle pilot.
>   - **Sun 22:** AWS reconstruction; the panel fetch and gate.
>   - **Mon 23:** pre-registration and panel frozen and pushed (if missed: LOLO 5 → 3).
>   - **Tue 24–Wed 25:** training.
>   - **Thu 26:** scoring. **Fri 27:** baselines. **Sat 28:** analysis, once.
>   - **All of P5 by Sat 29.**
>   - **Cut order** (every control kept): (1) LOLO 5 → 3; (2) robustness H.264 only;
>     (3) baselines D3 + image S2-D per frame; (4) k-shot → k = 0 only.
> - **Technical exclusion (2026-11-21, not a pilot outcome):** **Allegro**. diffusers
>   implements its encode/decode only with tiling, which the protocol forbids.
> - **Limitation (user, review of `4bed7c0`):** in-the-wild HunyuanVideo videos are often
>   chunk-decoded. diffusers' default frame-wise path temporal-tiles anything over 16
>   frames, with blending. Our training pairs decode the full 25-frame clip in one pass, so
>   the Hunyuan reconstructions do not carry chunk seams that real Hunyuan outputs may
>   carry.
> - **One diffusers version** (user, 2026-11-21) for the pilot, the AWS reconstruction and
>   all Kaggle scoring: **0.42.0**, the pilot's.
>   - It is pinned in `configs/video_bank.yaml` and `requirements/base.txt`, and asserted by
>     the loader.
>   - From 0.40.0 to 0.42.0, no run-time default of any bank class differs. The only source
>     change in their modules is Wan's resample check (assert → ValueError) and type hints.
> - **AWS rules (user, 2026-11-21):**
>   - **Budget:** setup + Pexels 5, reconstruction ≤ 45, training 15, features/heads 3,
>     EBS 5, reserve 27; hard stop 75.
>   - **Data:** downloaded once to one 200 GB gp3 data volume with
>     DeleteOnTermination=false.
>   - **Shutdown:** a trap on exit and on error, a 20-minute idle watchdog, and a hard cap of
>     estimate × 1.5; shutdown = stop.
>   - **Instances:** one at a time.
>   - **Backups:** Google Drive, small files only.
>   - **Launches:** a launch only after the user's OK, behind `p5_aws_check.py prelaunch`.
>   - The runbook is `env/cloud_p5_runbook.md`.
> - **AWS access (user, 2026-11-22):**
>   - **No AWS Agent Toolkit:** no MCP server, no `aws login`, no rules file, no uv. The
>     AWS CLI v2 is installed by the user from the official `AWSCLIV2.msi`.
>   - **Two IAM users** (policies in `env/aws_iam/`):
>     - `dbp-agent`: ReadOnlyAccess, plus Stop and Terminate in us-east-2. Claude Code runs
>       with these keys only.
>     - `dbp-launch`: launch, start, attach and teardown of the `dbp-p5*` resources;
>       g5.2xlarge and us-east-2 only; everything else denied. The user types these keys,
>       only for a launch, a start or the teardown.
>   - **Keys:** environment variables of one PowerShell session only, through
>     `env/aws_p5_env.ps1`. No file, no `~/.aws`.
> - **HunyuanVideo (user, 2026-11-22; pre-declared):** it stays in the bank for the A10G
>   speed test. It is dropped only if it fails technically on the A10G, or if it breaks the
>   reconstruction's $45 share.
> - **The first AWS session is approved** (user, 2026-11-22): setup + Pexels + the A10G
>   smoke, about 2 h, about $2.40, once the IAM users exist; it doubles as the AWS test
>   launch. The full reconstruction waits for the user's approval of the projection measured
>   there.
> - **Review of bf6837b (user, 2026-11-22):**
>   - The dbp-launch policy is approved as written. The user creates it as the
>     customer-managed `dbp-launch-policy`.
>   - `dbp-agent` stays as the user made it: ReadOnlyAccess + `dbp-agent-stop`.
>   - SSH is open only to the user's current IP as a /32. `prelaunch` refuses 0.0.0.0/0, ::/0,
>     wider ranges, other ports and more than one address.
>   - The root volume is gp3, set explicitly, at the AMI's size; `prelaunch` checks it.
>   - The guard's rate is $1.217/h: the instance at $1.212, plus $0.005 for the public IPv4
>     address.
>   - rclone is approved, on the AWS instance only (CLAUDE.md §14).
>   - Regression digests under 0.42.0: identical to 0.40.0.
>   - Credits are USD 120; the hard stop stays at 75.
> - **Review of 95b3a25 + 4e10e27 (user, 2026-11-22): accepted.**
>   - **dbp-launch** also allows `ec2:RevokeSecurityGroupIngress`, so the old /32 can be
>     removed when the public IP changes. The user creates the policy from that version.
>   - **The pilot is accepted.**
>     - Qwen-Image vs Wan2.1: **different decoders, a shared encoder.** This is recorded as
>       a caveat. There is **no "seen decoder" flag.**
>     - The cogvx determinism fix and "0.40 = 0.42" for the digests are recorded.
>   - **The video training recipe**, fixed now, from the pilot, before the freeze (section 3.4
>     and the `S2V-F` row in section 4). **It was fixed from training-domain pilot data only**
>     (200 Pexels clips and their reconstructions), **never from test data.**
>   - **The reconstruction's cost:**
>     - The 200 threshold-held-out reals are **not reconstructed**: they are only ever used
>       as reals. The bank builds 2,000 clips (1,800 train + 200 validation).
>     - **fp16** for every VAE whose pilot fp16 check passed. **fp32** for mochi (its check
>       failed) and for cogvx (determinism).
>     - The reconstruction is re-projected for 2,000 clips after the A10G speed test.
>   - **HunyuanVideo is decided after the speed test,** by the user, against the
>     reconstruction's $45 share. If it does not fit, the numbers go to the user, who may
>     re-allocate from the reserve. The agent neither drops nor keeps it alone.
> - **User, 2026-11-23:**
>   - **HunyuanVideo's fp16 check** runs in the A10G smoke, with the same rule as the others:
>     the PSNR threshold at every clip, and a repeat run that matches. That it ran on an A10G
>     is recorded (`gpu_name` in the smoke record).
>   - **F1 without HunyuanVideo:** wan21, ltx090, cogvx (section 5).
>   - **The training recipe's codec, CRF and share** are in section 3.4.
>   - **EBS encryption by default is off.** The volumes hold public Pexels data and model
>     outputs only; test data never leaves Kaggle. `dbp-launch` has no KMS permission, so an
>     encrypted volume is deleted right after it is created.

**Method:** the image method on video, at reduced scale, without reduced rigour. The
rules are the image module's: a frozen pre-registration pushed before training; a test
panel frozen and scored once per frozen model; leak and dedup checks; two seeds for the
main models; the integrity check; the pre-push audit; licensed data stays private;
`HF_TOKEN` only from the environment or Kaggle Secrets.

**Provenance.** Every revision, size, hash, licence tag and config value below was
produced on 2026-11-20 by `HfApi.model_info(..., files_metadata=True)`,
`HfApi.dataset_info(..., files_metadata=True)`, `HfApi.list_repo_tree` and
`hf_hub_download` of `config.json`, README, index and licence files at the pinned SHA.
That was metadata only: no weights, no videos. The audit scripts ran from the session
scratchpad and are not committed (approval item A6). Values marked **NOT-VERIFIED** were
not produced by a command this session. **PENDING-GPU** means it can only be measured on
a GPU. **ESTIMATE** gives its basis.

---

## 0. What needs the user's decision before anything runs

| # | item | why the approved rules do not cover it |
|---|---|---|
| A1 | **The source list** (section 1): Pexels subset (training reals), AIGVDBench and AEGIS (test panel), VBench (optional, gated) | §6a(b): a new dataset needs approval |
| A2 | **Hosts:** none new for data, because every proposed source is on the Hugging Face Hub. For the D3 and DeCoF *code*, `raw.githubusercontent.com` / `api.github.com` are allowed only for P1 data (§6a), so reading them needs approval, or the baselines are re-implemented from the papers | §6a |
| A3 | **Two new dependencies:** `av==19.0.2` (PyAV: decode, H.264/H.265 encode) and `pyarrow==26.0.1` (the panel and Pexels manifests are parquet). `pip install --dry-run` (2026-11-20) resolves exactly these two and leaves torch, torchvision and numpy untouched. No video decoder exists in the venv today: `av`, `decord`, `cv2`, `imageio_ffmpeg` and `torchcodec` are all absent, and there is no `ffmpeg` on PATH | §9b, §3 |
| A4 | **A new runtime:** the university PC (1× RTX 4000 Ada, 20 GB). §6a(a) says dataset downloads happen on Kaggle only. Proposed amendment: **training reals only** may be downloaded to the uni PC; **test-panel data never leaves Kaggle**. That puts training and test on physically separate machines, which is itself a leak barrier. AWS is optional (section 6) | §6a(a), §13 |
| A5 | **Extending the reconstruction protocol (CLAUDE.md §7) to video:** new kinds in `reconstruct.py`, rules in section 2.3 | §7 names image kinds only |
| A6 | **New files** (section 9) | §9b |
| A7 | **The rule values** in section 5, proposed from P3's values for comparability; P3's were set by the user | as in P3 (2026-10-08) |
| A8 | **Licence questions** in section 1.3 (Pexels upstream terms; AIGVDBench citation; AEGIS "no modification") | §6c |

**Information needed (blocking the compute plan):**
- the uni PC's OS, NVIDIA driver/CUDA, free disk, whether it can run 24/7 and reach the
  HF Hub;
- the remaining Kaggle GPU quota (read with the editor on CPU, §13);
- whether to accept the VBench gate (section 1.2);
- AWS yes or no.

---

## 1. Data sources

### 1.1 Training reals (one stock-footage source; test reals come from YouTube)

| source | pin | licence (as stated) | what we take | download |
|---|---|---|---|---|
| **`zengxianyu/open-sora-pexels-subset`** (Pexels, repackaged from `LanguageBind/Open-Sora-Plan-v1.1.0` as WebDataset) | `c595493c834ec2c47ceee5d18ecca002aa646540` | Hub tag `apache-2.0` | **2,400 distinct videos:** 1,800 train + 200 validation + 200 threshold-held-out (k = 0) + 200 pilot/dev. One clip per video | 124 tars, mean 1.08 GB. The README states 9,750 videos, so 78.6 per shard. 2,400 videos → **31 shards ≈ 33.6 GB** (arithmetic from measured tar sizes and the README count) |
| *alternative / second source:* `FastVideo/Mixkit-Src` | `b5fff04254af8bbf716b1c166e85b8fe4000bb0e` | **no licence tag, no README** | — | 8,360 files, 25.4 GB. Its index `video2caption.json` lists 6,145 clips; **all are 1080p (1920×1080 or 1080×1920) at 30 fps**, median 6 s, mean 2.88 MB. 2,000 clips ≈ 5.8 GB |
| *rejected:* `LanguageBind/Open-Sora-Plan-v1.0.0` | `70d23b672d2c84d32f32eee70c1ccd3a3623ba74` | tag `mit`; README: mixkit/pexels/pixabay "crawled … under the CC0 license" | — | 1.33 TB in split archives; no subset can be taken |

**Recommendation:** Pexels only. Mixkit has no stated licence on the Hub. Its uniform 30 fps
would also make frame rate a label shortcut wherever the test fakes run at other rates
(section 3.4).

### 1.2 Test panel (unseen generators; reals from a different platform than training)

| view | pin | licence (as stated) | content (measured) | role |
|---|---|---|---|---|
| **AIGVDBench** (`AIGVDBench/AIGVDBench`) | `bf48acaa4920990af3dd2a511ea88e63354df305` | Hub tag `cc-by-4.0`; the README is 30 bytes (no card, no citation) | 32 zips, one per generator: **11 closed-source** (Gen2, Gen3, Jimeng, Luma, Opensora, Sora, causvid_24fps, kling, pika, vidu, wan; 94.5 GB), **20 open-source** T2V/I2V/V2V (189.6 GB, of which `T2V/Wan2.1.zip` is 86.05 GB), `Real/Real.zip` 94.24 GB; 378.32 GB in total. `Split/test.jsonl`: 3,000 rows of `Video_id` + prompt; the ids look like YouTube ids (`59U-7GPCjNQ_0_0to892.mp4`) | **primary** |
| **AEGIS hard test** (`Clarifiedfish/AEGIS`) | `d2ad8bd38def2945d5fb7d988f2fda0099953296` | no Hub tag; README "Custom Academic Use License": academic use only, no redistribution, "not allowed to perform any form of broadcasting, **modification**…" | `hard_test_set_video.tar` 2.84 GB; `mapping_test_data.csv`: 436 videos, 218 fake (**kling 111, sora 107**) and 218 real (**youtube 109, dvf 109**) | **secondary**, clean only: no robustness views, because of the no-modification clause |
| *optional:* **VBench sampled videos** (`Vchitect/VBench_sampled_video`) | `8c89bb0218a12323d04821d11ec03c780d0dfb9c` | tag `mit`; gate: "You agree to not use the data to conduct experiments that cause harm to human subjects" (asks name, organisation, e-mail) | 71 generator folders, fakes only. It includes Sora, Veo 3, Kling, Kling-1.6, Jimeng, Luma, MiniMax-Video-01, Vidu Q1, Pika-1, Gen-2/3, Step-Video-T2V, MAGI-T2V, Open-Sora 2.0, LTX-2, Mochi-1, Wan2.1 and CogVideoX. One file per video, so it can be sampled | fakes-only view, TPR at k = 0 (like E8 in P3). **BLOCKED-ACCESS: our token gets 403** until the user accepts the gate |

**How the panel is fetched (Kaggle only):** whole zips are too large (Real.zip alone is
94 GB). The fetch reads each zip's central directory and then only the chosen members, by
HTTP range requests through `huggingface_hub.HfFileSystem` (already a dependency).
- The member counts per zip, and so the sampled download size, are **PENDING**: they come
  from the fetch session's first cell.
- The upper bound is the full-zip sizes above.

**Rejected:** GenVideo/DeMamba, AIGVDet, GenVidBench, Ivy-Fake and DVF.
- GenVideo/DeMamba's official host is not on the allowlist; the Hub copies
  (`sshaiton/GenVideo-50k-shard`, 84 GB) are third-party mirrors.
- `xmulz/AIGVDet` is Moonvalley-only.
- GenVidBench has no licence tag and is packed as RAR.
- Ivy-Fake and DVF ship multi-volume 7z/zip archives of 355–363 GB that cannot be
  subsampled.

### 1.3 Licence and citation items (CLAUDE.md §6c applies to all of it)

- **AIGVDBench:** the only licence evidence is the Hub tag. **The citation text is
  missing**: arXiv is not on the allowlist, so the user supplies it for `docs/CITATIONS.md`.
  No AIGVDBench number is reported before that entry is filled.
- **AEGIS:** academic only. It must also cite TIP-I2V (CC BY-NC 4.0) and Vript ("academic
  use only, with prior permission"). The no-modification clause rules out re-encoding, so
  it gets no V5 views.
- **Pexels subset:** the `apache-2.0` tag is the repackager's. Pexels' own terms were not
  fetched (pexels.com is not an allowed host): **NOT-VERIFIED**.
- **The reconstructions are model outputs.** The HunyuanVideo licence forbids use outside
  its Territory and "to repurpose or distribute output…". They stay private, like every
  derived artefact.

---

## 2. The video decoder bank

### 2.1 Members (5 video lineages + the image lineage per frame)

All revisions are the full commit SHAs returned by `model_info`. Weight bytes and sha256
are the LFS records of `vae/diffusion_pytorch_model.safetensors` (fp32).

| key | repo @ revision | class | z ch | compression t×h×w | licence | VAE bytes | VAE sha256 | the same file (byte-identical) also ships in |
|---|---|---|---|---|---|---|---|---|
| `wan21` | `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` @ `0fad780a534b6463e45facd96134c9f345acfa5b` | AutoencoderKLWan | 16 | 4×8×8 ¹ | apache-2.0 | 507,591,892 | `d6e524b3fffede1787a74e81b30976dce5400c4439ba64222168e607ed19e793` | Wan2.1-T2V-14B, **Wan2.2-T2V-A14B**, SkyReels-A2, SkyReels-V2 (8 Diffusers repos), SkyReels-V3-R2V |
| `hunyuan` | `hunyuanvideo-community/HunyuanVideo` @ `e8c2aaa66fe3742a32c11a6766aecbf07c56e773` | AutoencoderKLHunyuanVideo | 16 | 4×8×8 (config) | **tencent-hunyuan-community** ² | 985,943,868 | `7c68a6295f9034a88225fbafb1f3258291a08d57a1fdb938233fa57b1b8f4883` | HunyuanVideo-I2V |
| `cogvx` | `zai-org/CogVideoX-2b` @ `1137dacfc2c9c012bed6a0793f4ecf2ca8e7ba01` | AutoencoderKLCogVideoX | 16 | 4×8×8 ¹ | apache-2.0 (the 2b repo; 5b is `other`) | 862,388,596 | `a410e48d988c8224cef392b68db0654485cfd41f345f4a3a81d3e6b765bb995e` | CogVideoX-5b, CogVideoX1.5-5B |
| `ltx090` | `Lightricks/LTX-Video` @ `8984fa25007f376c1a299016d0957a37a2f797bb` | AutoencoderKLLTXVideo | 128 | 8×32×32 ³ | `other`: LTX Video 0.9 RAIL-M ("Permitted Purpose") | 1,676,798,532 | `265ca87cb5dff5e37f924286e957324e282fe7710a952a7dafc0df43883e2010` | — (LTX-0.9.5 is a different VAE, `7eb65b16…`) |
| `mochi` | `genmo/mochi-1-preview` @ `cd1d0fe89aa8382fcd9ae760bc3d7afe6183579f` | AutoencoderKLMochi | 12 | 6×8×8 ³ | apache-2.0 | 1,839,047,460 | `0d74f617ddcaa1b4a53884581d8d21b6836761f6175b20e3df177d4a12ed77df` | — |
| `sd15_pf` | `stable-diffusion-v1-5/stable-diffusion-v1-5` @ `451f4fe16113bff5a5d2269ed5ad43b0592e9a14` (= image-bank `sd15`) | AutoencoderKL, per frame | 4 | 1×8×8 | creativeml-openrail-m | 334,643,276 | `a2b5134f4dbc140d9c11f11cba3233099e00af40f262f136c691fb7d38d2194c` | image lineage, used per frame by AnimateDiff/VideoCrafter-style generators ⁴ |

¹ From the config: Wan's `dim_mult` has 4 levels (3 spatial halvings) and
`temperal_downsample [False, True, True]`. CogVideoX states `temporal_compression_ratio 4`;
its spatial 8× is inferred from 4 blocks. Checked by output shape at the smoke step.
² The community repo carries no licence. The weights are Tencent's, and `tencent/HunyuanVideo`
@ `6204ad6aea1a77ff5aba337c88278bb9500eb37d` licenses them as tencent-hunyuan-community:
"DOES NOT APPLY IN THE EUROPEAN UNION, UNITED KINGDOM AND SOUTH KOREA".
³ Not stated in the pinned 0.9.0 / Mochi configs; taken from the architecture (LTX:
`patch_size 4` × 3 downsamples; 0.9.5's config states 8 / 32). **NOT-VERIFIED** until the
smoke step's shape check.
⁴ The per-generator VAE identity is a tensor-level check (section 3.2), not a file hash: the
Latte, ModelScope and I2VGen files differ in bytes from `sd15`.

**Reserve lineages (never trained on; unseen-VAE reconstruction tests, V1-c):**
- `allegro`: `rhymes-ai/Allegro` @ `c1b9207bb5cb79e2aa08f3d139c17d26c0de55b6`,
  apache-2.0, 4 ch, `temporal_compression_ratio 4`, 699,904,972 B, `47871a69…`; the same file
  ships in all 4 Allegro repos.
- `wan22_5b`: `Wan-AI/Wan2.2-TI2V-5B-Diffusers` @
  `b8fff7315c768468a5333511427288870b2e9635`, apache-2.0, 48 ch,
  `scale_factor_spatial 16` / `scale_factor_temporal 4`, 2,818,777,808 B, `62cd18f1…`. This is
  the VAE of Wan2.2-5B, a different lineage from `wan21`.

**Excluded:**
- **Cosmos:** `nvidia/Cosmos-1.0-Diffusion-7B-Text2World` and Cosmos-Predict2 are gated, and
  our token gets 403 on `config.json`, so BLOCKED-ACCESS. The `Cosmos-*-Tokenizer` repos
  ship TorchScript, which would mean executing downloaded code (§6).
- **LTX-2.5:** gated, 403.
- **HunyuanVideo-1.5:** its VAE is 5.04 GB and its licence is `other`; kept as a reserve
  only if time allows.
- **The SVD temporal decoder:** licence `stable-video-diffusion-community`.

### 2.2 A lineage overlap with the image model (measured in the pilot)

- **What the configs show:** Qwen-Image's VAE has the same architecture as `wan21`.
  - Both have `z_dim 16`, `base_dim 96`, `dim_mult [1,2,4,4]` and
    `temperal_downsample [False, True, True]`.
  - The files differ: 253,806,966 B against 507,591,892 B, consistent with half precision.
- **Why it matters:** the image `S2-D` was trained with `qwen_image` in its bank.
  - If the two VAEs share tensors, image S2-D has already seen a Wan-family decoder.
  - The pilot and V6 must be read with that in mind.
- **The check:** the pilot compares the encoder and decoder state-dict digests in fp32, the
  P0 D-016 method.

### 2.3 The video reconstruction protocol (extends `reconstruct.py`; approval A5)

- **One source of truth:** new kinds `vae_video_kl` (Wan, HunyuanVideo, CogVideoX, LTX,
  Mochi) and `vae_kl_per_frame` are added to `src/dbp/decoders/reconstruct.py`. Nothing
  is reimplemented elsewhere.
- **The input** is `(1, 3, T, H, W)` from `uint8` frames, `x = u8/127.5 − 1`, fp32.
  - `z = vae.encode(x).latent_dist.mode()`, then `vae.decode(z).sample`.
  - No `scaling_factor` / `shift_factor`.
  - Output clamp → round → `uint8` → **PNG frames**.
- **Frame counts are native per member.** Each member gets the smallest valid length ≥ 17:
  17 (1 + 4k / 1 + 8k) for wan21, hunyuan, cogvx and ltx090; 19 (1 + 6k) for mochi
  (NOT-VERIFIED until the smoke).
  - The decode is asserted to return exactly T frames, as `reconstruct_vae_kl_5d` asserts
    today.
  - **Every model uses output frames 1–16, reals and reconstructions alike.** Frame 0 of
    a causal VAE is encoded alone, image-like, so including it would put a
    position-dependent artefact into one class.
- **No spatial tiling, no temporal tiling or frame-wise blending.** `enable_tiling()` and
  `use_framewise_encoding/decoding` with overlap blending stay off: their seams are
  artificial traces.
  - A model's **native causal cache** (Wan's `feat_cache`, CogVideoX's conv cache) is the
    architecture, not tiling, and is allowed.
  - Slicing over the batch is allowed.
- **Precision:** fp32 unless `fp16_build_allowed` (≥ 55 dB at every measured clip, no
  NaN/Inf) passes in the smoke.
- **Determinism:** each member re-runs twice; `max_abs_delta = 0` is required, as at P0.

### 2.4 Memory on 1× RTX 4000 Ada (20 GB)

Weights are the measured fp32 file sizes. The activation column is an **analytic lower
bound**: one full-resolution tensor `T × 320 × 320 × C × 4 B` at the member's widest
full-resolution stage, with all frames resident. **Peak VRAM is PENDING-GPU**: the pilot's
smoke step measures it with `torch.cuda.max_memory_allocated` on all 8 members.

| key | weights fp32 | largest single activation | expected to fit |
|---|---|---|---|
| wan21 | 507.6 MB | 668.5 MB (T 17, C 96; less in practice, the causal cache holds 4 frames) | T4 and Ada |
| hunyuan | 985.9 MB | 891.3 MB (T 17, C 128) | PENDING-GPU; Ada first |
| cogvx | 862.4 MB | 891.3 MB (T 17, C 128) | PENDING-GPU |
| ltx090 | 1,676.8 MB | 55.7 MB (after the 4× patchify) | T4 and Ada |
| mochi | 1,839.0 MB | 996.1 MB (T 19, C 128) | PENDING-GPU; Ada first |
| sd15_pf | 334.6 MB | 891.3 MB if all 17 frames are batched; 52 MB per frame | T4 and Ada (P0: 1,159 MB peak at 512², T4) |
| allegro (reserve) | 699.9 MB | 891.3 MB | PENDING-GPU |
| wan22_5b (reserve) | 2,818.8 MB | 278.5 MB (patch 2, C 160) | PENDING-GPU |

**Rule:** a member whose measured peak exceeds 14 GB runs on the Ada only. No member is
ever run tiled to make it fit.

---

## 3. Clips, pairs, panel and leak checks

### 3.1 Clips

- **Decode:** PyAV, native fps, no frame-rate conversion.
- **Window:** 17 consecutive frames (19 for mochi) from a seeded start. The seed is a hash
  of the video id, as in §6b.
- **Crop:** a centre 320 × 320, the same window in every frame. **No resize.**
- **Eligible:** short side ≥ 320, at least 19 decodable frames, and decoded without error.
- **One clip per source video.** Splits are grouped by source video id.
- **Storage:** frames stored as PNG.
  - 2,000 reals × (1 + 6 members) × 16 frames × 320² × 3 B = **68.8 GB raw**. PNG size is
    PENDING; the pilot measures bytes per clip.
  - This is why the training pixels stay on the machine that builds them (section 6).

### 3.2 Panel strata (fixed in the pre-registration after the pilot's identity checks)

| stratum | rule | AIGVDBench examples |
|---|---|---|
| **in-bank VAE** | the generator's VAE is hash- or tensor-identical to a bank member | Wan2.1; HunyuanVideo; Cogvideox1.5 (CogVideoX VAE is byte-identical across CogVideoX repos); AccVideo (ships a Hunyuan VAE `.pt`, tensor check); LTX (version unknown, so tensor check or stratum "unknown"); AnimateDiff / VideoCrafter / SEINE (SD family, tensor check) |
| **out-of-bank VAE** | the generator's VAE is public and matches no member | EasyAnimate (own VAEs, 4 distinct files, none in the bank); Pyramid-Flow (own `causal_video_vae`); SVD (SD encoder + temporal decoder); IPOC and RepVideo (VAE NOT-VERIFIED) |
| **closed / unknown** | no public weights | Sora, Kling, Gen2, Gen3, Luma, Pika, Vidu, Jimeng, wan (closed), causvid_24fps, Opensora (listed as closed by the benchmark) |

**"Unseen" for V2–V6** means out-of-bank and closed slices. In-bank slices are reported
separately, as E7 was for the images.

### 3.3 Draw, dedup and leak checks (all before the panel is frozen)

- **Size:** AIGVDBench, at most **150 fakes per generator zip** and **1,500 test reals**
  (1,200 test + 300 calibration), drawn from the `Split/test.jsonl` ids where the zip
  layout allows. AEGIS: all 436.
- **Exact duplicates:** sha256 of every file.
- **Near duplicates:** pHash (`dbp.data.phash`, ≤ 6 bits) on keyframes 1, 8 and 16.
- **Source ids:** YouTube id for AIGVDBench, Pexels id for training.
- **Checked against:**
  - every training, validation, threshold and pilot clip;
  - **across the three panel views.** Sora and Kling showcase videos may sit in AEGIS,
    AIGVDBench and VBench at once; a match keeps the copy in the first-listed view only,
    with counts logged.
- **I2V conditioning leak:** pHash of each panel fake's frame 1 against every training-real
  frame.
- **Matches** are removed from the draw and counted per view.
- **Freeze:** the panel digest is committed before any P5 model trains.

**The AIGVDBench draw and pairing rules.** These come from the user's principles of
2026-11-23 and are implemented in `scripts/p5_fetch_panel.py` with
`env/p5_panel_expect.json`. They are fixed before any panel video is fetched or scored, and
they go into the pre-registration unchanged.
- **The layout,** measured from the zip directories (`reports/p5_panel/
  aigvdbench_listing_summary.json`):
  - the 20 open-source zips hold exactly Real.zip's 20,000 names, so every open-source fake
    has a same-named real;
  - the 11 closed-source zips share 2,000 numeric names, none of them in Real.
- **The universe:** the names in both `Split/test.jsonl` and Real.zip, which is the
  benchmark's own test split.
- **Grouping:** by source video, the part of `<source>_<k>_<a>to<b>.mp4` before `_<k>`.
  This is stricter than by name: no scene is shared across pools.
- **The pools.** Groups are ordered by `sha256("2026:p5-panel:" + source)`. Whole groups
  fill, in order:
  - the k-shot **calibration** pool, 300 names;
  - the **closed-source real** pool, 300 names;
  - the **open-source** pool, 900 names.
  A name (its real and every fake made from it) is in exactly one pool, and only whole
  groups are taken, so a pool may hold slightly more than its number.
- **Open-source slices** (one per zip): the open pool is ranked by
  `sha256("2026:p5-panel:" + zip + ":" + name)`.
  - The slice keeps the first 150 names whose real AND that zip's fake are both eligible.
  - These are **content-matched pairs**, reported as the content-matched view.
- **Closed-source slices** (one per zip):
  - One shared order, `sha256("2026:p5-panel:closed:" + name)`, so the closed generators
    are compared on the same prompts.
  - Each slice keeps its first 150 eligible fakes, scored against the closed-source real
    pool (names used nowhere else).
- **Eligibility** is the clip protocol of `src/dbp/data/video.py`:
  - 8-bit, not rotated, short side ≥ 320;
  - a seeded window of **17 frames** (scoring uses frames 1–16), centre crop 320, never
    resized;
  - not a still.
  - A slice stops after 50 consecutive ineligible fakes.
  - Every exclusion is counted per zip.
- **Duplicates:**
  - exact by sha256;
  - near, when every keyframe pHash (frames 1, 8, 16 of the cropped window) is within 6
    bits.
  - A real and its own fake (which a V2V fake may be close to) are counted separately as
    `own_pair`, not as duplicates.
  - AIGVDBench is listed before AEGIS, so a cross-view match keeps the AIGVDBench copy.
- **The metadata gate:** a direction-free single-feature AUC per slice (its fakes against
  its reals) for fps, width, height, duration, bitrate, reported frames, codec, profile and
  pix_fmt. Any value > 0.70 is flagged.
  - **Rebalancing is decided by the user from the flags, before the freeze.** Nothing is
    rebalanced or dropped automatically.
- **Still to do before the freeze:**
  - the training-side pHash cross-check (from the AWS fetch);
  - the I2V conditioning-leak check;
  - the AEGIS label join;
  - the gate decisions.

### 3.4 Compression and format must not predict the label (§6b / §6d, extended to video)

- **The table:** for every view, `label × (codec, profile, pix_fmt, fps, resolution class,
  bits-per-pixel decile, duration)`.
- **The lift:** for each parameter, the best single-parameter rule and its **lift** over
  the majority class, per generator stratum.
- **Shortcuts:** any lift > 5 points is recorded as a shortcut and stated wherever that
  slice is reported.
- **Frame rate is the expected one.** Fakes often run at 8, 16 or 24 fps.
  - In training, reconstructions inherit their real's frames and fps, so fps cannot
    separate the training classes.
  - At test it can. V3 is therefore reported per fps stratum as well, and V5 includes an
    fps change.
- **Training-side compression shortcut:**
  - **The problem:** a real's frames are H.264-decoded; its reconstruction is clean PNG.
  - **The rule:** each reconstruction is re-encoded with libx264 at the CRF whose
    bits-per-pixel matches its real source clip, seeded, then decoded.
  - **The degradation chain (fixed 2026-11-22, from the pilot; user):**
    - **The pilot's evidence:** the image S2-D, per frame, on 200 Pexels clips and their
      wan21 and ltx090 reconstructions. Clip AUC is 0.882–0.910 on the raw view and
      0.6425–0.7543 after one libx264 CRF 23 re-encode of both clips
      (`reports/kaggle_20261121/p5_pilot/p5_pilot_summary.json`, both seeds). A single
      ordinary re-encode removes most of what the image model sees, so the video model must
      be trained through re-encodes.
    - **The rule:**
      - An H.264 or H.265 re-encode, at a random CRF in **the range in the open item
        below**, is applied to **p = 0.80** of the training pairs.
      - It is applied **identically to both clips of a pair**: the same codec, CRF, preset
        and fps, drawn once per pair and seeded.
      - The other degradations of S2-D's chain keep **p = 0.35**.
      - The bpp-matched re-encode of each reconstruction (above) stays: it removes the
        format shortcut. The p = 0.80 re-encode comes on top of it, for both classes.
    - **Provenance:** fixed from training-domain pilot data only (Pexels reals and their
      reconstructions), never from test data. No test clip has been fetched.
    - **Codec, CRF and share (user, 2026-11-23):**
      - codec: H.264 with p = 0.7, H.265 with p = 0.3;
      - CRF: an integer, uniform over 18–35;
      - re-encode share: exactly p = 0.80, identical for both clips of a pair.
    - **The thesis states** that V5 (H.264 CRF 23/28/35, H.265 CRF 28/35) tests robustness
      inside the trained range, not outside it.

---

## 4. Models

| id | what | seeds |
|---|---|---|
| `S2V-F` | **frame model:** the S2 recipe per frame. DINOv3 ViT-B/16 + LoRA r = 8, batch 16, LR 1e-4, wd 0.01, warm-up 300, crop 320, degradation p = 0.35 + a pair-identical H.264/H.265 re-encode at p = 0.80 (section 3.4), **`N*` = 5,000 inherited from P3** (no new selection) | 0, 1 |
| `S2V-T` | **temporal model:** `S2V-F`'s frozen backbone. Per-frame features `f_t` and deltas `f_t − f_{t−1}`, t = 1..16, go to a small temporal head (2-layer transformer, width 256) trained on cached features. Two-stage, so V3 compares the same backbone with and without time | 0, 1 |
| `S2V-1` | single VAE: `wan21` only, the same recipe and head (V2) | 0, 1 |
| `LOLO-v` | v ∈ {wan21, hunyuan, cogvx, ltx090, mochi} held out; `sd15_pf` is always in | 0 |
| frozen comparators | image `S2-D` ViT-B s0/s1 (P3), per frame; DDA per frame (P4); D3 (training-free); DeCoF | — |

**DeCoF's weights** are `AIGVDBench/Weights/DeCoF.tar` (56,815,450 B, on the Hub).
- **Loading:** `torch.load(weights_only=True)` with its sha256 recorded; BLOCKED if that fails.
- **If it was trained on AIGVDBench's train split** (NOT-VERIFIED), AIGVDBench is
  in-domain for it, and it is reported as an in-domain reference, not as a cross-generator
  baseline.

---

## 5. Hypotheses (to be pre-registered; values proposed, the user sets them — A7)

- **Metrics:** P3's: clip-level AUC/AP per slice against the view's whole real class.
- **Macro averages** are taken over unseen slices.
- **Confidence intervals:** paired bootstrap, 2,000 resamples, seed 2026. The two
  primaries use 97.5% CIs.
- **Threshold at k = 0:** fixed at 5% FPR on the 200 threshold-held-out Pexels clips.

| id | question | proposed success rule (P3 analogue) |
|---|---|---|
| **V1 (PRIMARY)** | LOLO over the 5 video lineages | **V1-a:** `G_V` = mean held-out-lineage AUC (validation reals vs held-out member's reconstructions, `S2V-T`) ≥ 0.90, 97.5% lower bound ≥ 0.85, no fold < 0.75 (H1-a). **V1-b:** per fold, panel generators whose VAE is the held-out lineage, AUC ≥ 0.85 where the slice has ≥ 100 fakes (H1-b). **V1-c:** `allegro` and `wan22_5b` reconstructions, descriptive |
| **V2 (PRIMARY)** | bank vs single VAE on unseen generators | `S2V-T` − `S2V-1` macro AUC over unseen slices ≥ +2 points, 97.5% CI excluding 0, both seeds (H1-c / H4) |
| V3 | temporal vs frame-only | `S2V-T` − `S2V-F` (mean frame logit) ≥ +2 points, 95% CI excluding 0, both seeds; also reported per fps stratum |
| V4 | FPR at k = 0, and k-shot (k = 50) | k = 0: macro FPR over the panel real pools ≤ 15%, none > 25% (H3). k = 50: (Q) and (S) as in P3 H3-k, FPR ≤ 7.5% in ≥ 80% of 1,000 draws, **judged on AIGVDBench** (calibration pool 300); AEGIS descriptive |
| V5 | robustness | H.264 CRF 28 and resize-and-restore 0.5 each within 5 AUC points of clean (H5). H.264 CRF 23 / 35, H.265 CRF 28 / 35 and fps change (every 2nd frame) descriptive. AIGVDBench only |
| V6 | baselines | `S2V-T` − image `S2-D` per frame ≥ +2 points, 95% CI excluding 0, both seeds (the video bank adds value). DDA per frame, D3 and DeCoF: paired, descriptive |

**Pre-declared fallback order (user's), with triggers:**
1. **F1: LOLO 5 → 3 folds** (wan21, hunyuan, ltx090 — the three most different) if the
   training data is not complete by **Mon 23 Nov, end of day**.
   - **If HunyuanVideo leaves the bank** (user, 2026-11-23, decided before any result): the
     three folds are **wan21, ltx090, cogvx**.
   - If it stays: F1 exactly as written.
2. **F2: drop k-shot** if scoring has not started by **Wed 25 Nov, end of day**.
3. **F3: robustness H.264 only** (CRF 28, with 23 and 35 descriptive) if clean scoring is not
   done by **Thu 26 Nov, 12:00**.
4. **F4: baselines D3 + S2-D per frame** if F3 triggers, or if DeCoF or DDA fails its
   pre-flight.

Each trigger is logged with its time; nothing else may change.

---

## 6. Compute plan

| resource | role | why |
|---|---|---|
| **Uni PC, 1× RTX 4000 Ada 20 GB** | training reals (download, clips, 6-member reconstruction), all training (9 frame models, temporal heads, LOLO) | the heavy pixels (≈ 69 GB raw) stay where they are made; hunyuan and mochi need the larger card if they exceed 14 GB on a T4 |
| **Kaggle 2× T4** (`muhammadadeelshah` only, §13) | pilot; panel fetch, dedup, separability, robustness views; **every scoring session**; baselines | test data never leaves Kaggle; the models (LoRA + head, MBs) are uploaded as a private dataset with hashes |
| **AWS G instance (optional)** | overflow for the data build only | a single-GPU instance; **region outside the EU/UK** (HunyuanVideo Territory clause); same private-data rules; needs approval (A4). Prices NOT-VERIFIED |

**GPU-hours, ESTIMATE:**

| job | ESTIMATE | basis |
|---|---|---|
| pilot + bank smoke (8 members) | ≤ 3 T4-h | 200 × 3 clips × 17 frames ≈ 10,200 frame-reconstructions + ≈ 51,000 ViT-B frame forwards (2 seeds, raw + matched views) |
| training data build | **PENDING-GPU** | 2,400 clips × 6 members; s/clip measured in the pilot |
| 9 frame models | ≈ 11 T4-h (fewer on the Ada) | P3 `s2d_p35` seed 1: 1.212 GPU-h for 5,000 steps at batch 16, crop 320, on T4 (log, 2026-10-27) |
| feature caches + temporal heads | PENDING-GPU | 9 models × 16,800 clips × 16 frames |
| panel scoring, clean + robust | PENDING-GPU | depends on the drawn panel size |

The firm budget replaces these estimates in the pre-registration, from the pilot's
measurements.

---

## 7. Day-by-day schedule

| date | uni PC / Kaggle | repository work | gate |
|---|---|---|---|
| **Fri 20 Nov** (today) | — | this plan; on approval: video IO, `reconstruct.py` video kinds, `configs/video_bank.yaml`, the Pexels fetch and the pilot, with tests and a runbook **dry-run** (§12) | **A1–A8 approved** |
| **Sat 21** | Kaggle: Pexels fetch (31 shards) → **pilot + 8-member smoke**; panel fetch session 1 (central directories, then members; AEGIS tar) | panel builder: eligibility, dedup, separability tables; tensor-level lineage checks | pilot results logged (EXPLORATORY) |
| **Sun 22** | uni PC: environment + data build starts | **panel frozen** (digest committed); **pre-registration frozen, tagged `p5-prereg-frozen`, pre-push audit, pushed** | **freeze** |
| **Mon 23** | uni PC: data build done; `S2V-F` s0/s1, `S2V-1` s0/s1 | training-set leak and dedup report; splits | F1 trigger (end of day) |
| **Tue 24** | uni PC: LOLO ×5 (or ×3); temporal heads | baselines (D3, DeCoF, DDA-pf, S2-D-pf) built, dry-run | — |
| **Wed 25** | Kaggle: robustness views; **scoring session 1 (clean)** | analysis built and tested on synthetic scores only | F2 trigger (end of day) |
| **Thu 26** | Kaggle: **scoring session 2** (robust + baselines) | analysis interpretations confirmed by the user before it runs | F3 / F4 trigger (12:00) |
| **Fri 27** | — | **analysis run once**; thesis tables and figures; log | **scored** |

**The critical path is the pilot on Saturday.** It needs the approvals tonight, so that the
fetch and pilot code can be written, tested and dry-run before Saturday's Kaggle session.

---

## 8. The 1-day pilot: does image S2-D per frame already catch video-VAE reconstructions?

**It is EXPLORATORY, and its pass rules are fixed now, before it runs.**
- **It may set only operational parameters:**
  - bank membership (a member failing PSNR, NaN, determinism or memory is dropped);
  - fp16 per member;
  - frames per member;
  - storage;
  - the schedule.
- **It cannot change any V-rule.**

- **Data:** 200 Pexels clips (the pilot/dev pool, never trained on or tested on), seeded,
  17 frames, crop 320.
- **VAEs: 3, chosen for spread:**
  - `wan21`: the architecture of `qwen_image`, which S2-D trained on, so the expected-easy
    control;
  - `hunyuan`: 16 ch, unrelated;
  - `ltx090`: 128 ch, 32× spatial, far.
- **Models:** the frozen image `S2-D` ViT-B s0 and s1 (P3), per frame. The clip score is the
  mean of the frame-1–16 logits. The threshold is S2-D's committed k = 0 union threshold.
- **Views:** **raw** (reconstruction PNG vs decoded real frames) and **H.264-matched**
  (section 3.4).
- **Measured per VAE:**
  - PSNR;
  - frame and clip AUC (real vs reconstruction, paired);
  - TPR at the threshold, and the 200 reals' FPR;
  - s/clip;
  - peak VRAM;
  - fp16 parity;
  - re-run determinism.
- **Also measured:**
  - Qwen-Image vs `wan21` state-dict digests;
  - PNG bytes per clip;
  - the 8-member smoke (all members, 3 clips each).
- **Reading, pre-declared:** a VAE is **"already caught"** if, on the H.264-matched view:
  - clip AUC ≥ 0.95;
  - TPR ≥ 80% at the threshold;
  - real FPR ≤ 15%.
- **What it changes:** whatever the outcome, the training plan stands. V6 is the test of
  whether the video bank adds value on unseen generators.
- **Where:** Kaggle. `dbp-p3-s2d` holds the S2-D models; the session also needs the Pexels
  shards. About 3 T4-h (ESTIMATE, section 6).

---

## 9. Files to be added (approval A6) — all inside `thesis_dbp/`

- **Config:** `configs/video_bank.yaml`, the members and reserve lineages of section 2.1
  with these pins.
- **Code:**
  - `src/dbp/data/video.py`: PyAV decode, window, crop, PNG frames, H.264 matched
    re-encode;
  - new kinds in `src/dbp/decoders/reconstruct.py` and `loaders.py`.
- **Scripts:**
  - `scripts/p5_hub_audit.py`: today's audit, committed with its JSON output under
    `reports/p5_plan/`;
  - `scripts/p5_fetch_reals.py`, `scripts/p5_fetch_panel.py`;
  - `scripts/p5_pilot.py`, `scripts/p5_preflight.py`.
- **Runbooks:** `env/kaggle_run_p5_pilot.py`, `env/kaggle_run_p5_panel.py`.
- **Tests:** `tests/test_p5_video.py` (frame counts, crop, no resize, determinism on a tiny
  VAE, PNG round-trip), plus the dry-run fixtures.
- **Dependencies:** `av==19.0.2` and `pyarrow==26.0.1` in `requirements/base.txt`, exact
  pins. `env/kaggle_bootstrap.sh` keeps refusing any change to torch, torchvision or numpy.
- **Docs:** `docs/P5_PREREGISTRATION.md` (Sunday), and a P5 section in
  `docs/CITATIONS.md`.

## 10. Risks

- **Seven days for the full design.** At least F1 or F3 is likely to trigger. They are
  pre-declared, so triggering them is not a deviation.
- **The data build has no measured rate yet.** The schedule assumes the 6-member
  reconstruction of 2,400 clips fits Sunday to Monday on the Ada.
- **The panel's per-zip member counts and layout are unknown** until the central-directory
  read. If a zip's members cannot be matched to `Split/test.jsonl`, the draw is seeded
  within the zip instead, and that is recorded.
- **The Qwen↔Wan relation** may make image S2-D look strong on Wan-family slices. The
  pilot measures it first.
