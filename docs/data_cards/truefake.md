# Data card — TrueFake (secondary, descriptive social-media test)

| field | value |
|---|---|
| role | **secondary, descriptive** social-media test, scored **once, at the end, with Panel B** (user decision 2026-10-20). Never training, never development, no decision rule |
| source | Dell'Anna, Montibeller, Boato et al., IJCNN 2025 ("TrueFake"); repository `MMLab-unitn/TrueFake-IJCNN25` at commit `806f848ad04883916a4e229c853e0e9ae482465b` |
| what we use | a stratified sample of **4,000 ids × 4 versions** (pre-share, Facebook, Telegram, Twitter) = 16,000 files, **7.985 GiB**: `reports/truefake/truefake_sample.csv` (sha256 `a8954c2f63e81d95173cbec99e3f7c1ee2dd3d49527fac5ff7075de5e1ac8fd6`) |
| fetched | **not yet** -- a Kaggle CPU session, member by member from `TrueFake.zip`, only after the user approves its hand-over |
| privacy | **the images stay private**: never in the thesis, a report, a figure or a commit; aggregate numbers only |

Every value below comes from a page or file read on 2026-10-19/20 (sha256 given), or from
the archive index read the same days. Nothing is from memory (CLAUDE.md §9a).

---

## 1. Licences and terms (read from the sources)

| component | terms | source read |
|---|---|---|
| **TrueFake** (the repository that publishes the dataset links) | **CC BY 4.0**: the repository's `LICENSE` is the full "Attribution 4.0 International" legal code | `LICENSE` at commit `806f848a…`, sha256 `9ba9550ad48438d0836ddab3da480b3b69ffa0aac7b7878b5a0039e7ab429411` |
| TrueFake paper (arXiv 2504.20658) | CC BY 4.0 -- **this is the paper's licence**, not a dataset licence; the paper states no dataset licence | `arxiv.org/abs/2504.20658` |
| **FFHQ** (70,000 pre-share reals; 10,000 in the shared subset) | the individual images: Flickr, under CC BY 2.0, CC BY-NC 2.0, Public Domain Mark 1.0, CC0 1.0 or U.S. Government Works -- "All of these licenses allow free use, redistribution, and adaptation for **non-commercial purposes**"; the dataset itself (metadata, script, documentation): **CC BY-NC-SA 4.0** (NVIDIA), with citation required | `NVlabs/ffhq-dataset` at commit `4826aa6ea77aa7f1a7802b938ed7c40afb985cda`: `LICENSE.txt` sha256 `620105e38526e788e422e21599ecbad75e78216cb594ea7680b2c2431a2433d9`, `README.md` sha256 `c12efdc2759654840c9b2f8fe1cd9a6716be218b831c27056e13e3ec5cd3d663` |
| **FORLAB** (30,719 pre-share reals; 10,000 shared) | **NOT-VERIFIED.** The source paper (Iuliani, Fontani, Piva, IEEE Access 2021; arXiv 2009.04878) describes "over 33.000 Flickr images belonging to 45 smartphone and 25 DSLR camera models"; neither it nor the TrueFake paper states distribution terms. The individual images are Flickr photographs under their authors' terms | `arxiv.org/abs/2009.04878` |

**Consequence (CLAUDE.md §6c):** FFHQ is non-commercial; FORLAB's terms are unknown. Any
TrueFake number is an **academic, descriptive** result and supports no deployment claim.

## 2. What the archive holds (index read 2026-10-19/20)

`TrueFake.zip`: 694,732,361,838 bytes, ZIP64, 600,876 entries (central directory 123,346,958
bytes, sha256 `869a102f65504dc25df759de4a51ae6931cecac7b255b0b5ad270dad2544dc0b`). Layout
`TrueFake/<PreSocial|Facebook|Telegram|Twitter>/<Fake|Real>/<source>/[<subfolder>/]<file>`.

| share | fakes | reals |
|---|---|---|
| PreSocial | 8 generators × 40,000 (FLUX.1, SD1.5, SD2, SD3, SDXL, StyleGAN, StyleGAN2, StyleGAN3) | FFHQ 70,000; FORLAB 30,719 |
| Facebook, Telegram, Twitter (each) | 8 × 5,000 | FFHQ 10,000; FORLAB 10,000 |

- **One-to-one:** every one of the 60,000 shared files has a pre-share twin with the same
  path stem, and the **same 60,000 ids** are on all three platforms. Shared files are
  JPEG; the pre-share twins are 50,000 PNG + 10,000 JPEG.
- Platform processing (paper): Facebook downscales to 720 px wide, QF 61–92; X to
  1200 px wide, QF 87 (no change at ≤ 768 × 768); Telegram height ≤ 800, QF 85.

## 3. Relation to our bank and data

- **Generators:** every diffusion generator decodes through a lineage **in our bank**
  (SD1.5 and SD2.1 → `kl_4ch_sd1`; SDXL → `kl_4ch_sdxl`; SD3 medium → `kl_16ch`, VAE
  identity with our SD3.5 member NOT-VERIFIED; FLUX.1 dev → `kl_16ch`). **StyleGAN 1–3
  have no decoder in the bank: an unseen generator family.**
- **Reals: both pools are Flickr** -- FFHQ (faces) and FORLAB (camera originals obtained
  from Flickr) -- the platform of our training reals (COCO, Open Images). Under the
  pre-registration's platform principle (§2) **neither carries an FPR claim**; both are
  reported descriptively, separately.
- **Content:** StyleGAN fakes are faces (FFHQ-trained), so **StyleGAN vs FFHQ is the
  content-matched comparison**; any comparison against FORLAB also reflects content.
- **Twitter did not re-encode most SD1.5 / SD2 fakes (MEASURED 2026-10-28):**
  - In the fetched sample (`reports/truefake/truefake_fetch_manifest.csv`), **499 of the 500
    Twitter files of StableDiffusion1.5 and StableDiffusion2** are byte-identical to their
    pre-share PNG. Their `.jpg` name notwithstanding, the sha256 is equal.
  - 15,501 distinct sha256 over the 16,000 files; no other duplicates.
  - **For those two generators the Twitter − pre-share difference is near zero, not
    exactly zero**: the fakes are unchanged, but the reals were re-encoded (0 of the 1,999
    kept real ids has a byte-identical Twitter file). It is reported with that caveat
    (wording corrected 2026-10-31; MEASURED, e.g. `s2b_seed0` FFHQ SD1.5 Δ −0.0005
    [−0.0021, 0.0004]).

## 3a. Scoring exclusions (rule approved by the user 2026-10-31)

- **Rule:** an id is removed from **all four versions** if any version is dropped by the
  disjointness check, is under the 320 px crop, or **cannot be decoded**. The rule is
  label-blind and keeps Δ paired. Undecodable files are recorded, never repaired
  (`LOAD_TRUNCATED_IMAGES` is not used).
- **Result of the 2026-10-31 run** (`reports/kaggle_20261031/p3_truefake/p3_truefake_plan.json`):
  - **1 undecodable member**, `TrueFake/PreSocial/Real/FORLAB/09258.jpg`:
    `OSError: image file is truncated (35 bytes not processed)`. It equals the archive
    byte for byte (CRC-32 at the fetch, sha256 at staging), so it is a defect in TrueFake
    as distributed.
  - Its id is excluded from all four versions. Its three shared versions decode and were
    scored, but are not used.
  - 0 ids dropped by disjointness; 0 too small (the smallest side is 450 px).
  - **Kept:** FFHQ 1,000, FORLAB 999, each of the 8 generators 250.

## 4. Citation

`docs/CITATIONS.md`, "TrueFake", with FFHQ and FORLAB.
