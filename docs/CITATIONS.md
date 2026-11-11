# Citations required by the datasets

Some sets in this corpus require a citation as a condition of use. A set **may not be
reported on until its entry here is filled in** (CLAUDE.md §6c).

Each block below is reproduced **verbatim** from the source's own README, read by command
on the date given. Nothing here is reconstructed from memory.

---

## E7 — DDA-COCO

Read 2026-09-20 from `Junwei-Xi/DDA-COCO` (`README.md`, revision
`8c9330a3b374bcac46a8045a0e3c09ebcf7868fb`). Licence `apache-2.0`.

```bibtex
@inproceedings{chen2025dual,
  title={Dual Data Alignment Makes {AI}-Generated Image Detector Easier Generalizable},
  author={Ruoxin Chen and Junwei Xi and Zhiyuan Yan and Ke-Yue Zhang and Shuang Wu and Jingyi Xie and Xu Chen and Lei Xu and Isabel Guan and Taiping Yao and Shouhong Ding},
  booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
  year={2025},
  url={https://openreview.net/forum?id=C39ShJwtD5}
}
```

Paper: NeurIPS 2025, `https://arxiv.org/abs/2505.14359`.

---

## E8 — EvalGEN

Read 2026-09-20 from `Junwei-Xi/EvalGEN` (`README.md`, revision
`86c719a9bfc72b7e5d91bfbb04432774feb4e3b7`). Licence `apache-2.0`.

The same work publishes both benchmarks, so the citation is identical to E7's:

```bibtex
@inproceedings{chen2025dual,
  title={Dual Data Alignment Makes {AI}-Generated Image Detector Easier Generalizable},
  author={Ruoxin Chen and Junwei Xi and Zhiyuan Yan and Ke-Yue Zhang and Shuang Wu and Jingyi Xie and Xu Chen and Lei Xu and Isabel Guan and Taiping Yao and Shouhong Ding},
  booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
  year={2025},
  url={https://openreview.net/forum?id=C39ShJwtD5}
}
```

EvalGEN's prompts come from the **GenEval** benchmark (553 aligned prompts, per its
README). If the thesis uses EvalGEN, GenEval should be cited too; its reference has
**NOT-VERIFIED** status here because no command in this session read it.

---

## E6 — MNW (Microsoft-Northwestern-WITNESS)

Read 2026-09-21 from `https://raw.githubusercontent.com/microsoft/MNW/main/README.md`
(repository `microsoft/MNW`, default branch `main`, `pushed_at` 2026-09-19; mirror
`nsail-lab/MNW`). Fetched by direct URL, not via search. The repository carries **no
licence file** (`license: null` from the GitHub API); the terms are stated in the README.

Terms, verbatim from the README:

> **This dataset is intended for evaluation purposes and cannot be used for training or
> for commercial purposes.**

Citation, verbatim from the README:

```bibtex
@ARTICLE{11479406,
author={Roca, Thomas and Postiglione, Marco and Gao, Chongyang and Gortner, Isabel and Wojciak, Zuzanna and Wang, Pengce and Alimardani, Mahsa and Anlen, Shirin and White, Kevin and Ferres, Juan Lavista and Kraus, Sarit and Gregory, Sam and Subrahmanian, V. S. and Murugesan, San},
journal={ IEEE Intelligent Systems },
title={{ The Microsoft-Northwestern-WITNESS Benchmark for Deepfake Detection }},
year={2026},
volume={41},
number={02},
ISSN={1941-1294},
pages={15-23},
doi={10.1109/MIS.2026.3668398},
url = {https://doi.ieeecomputersociety.org/10.1109/MIS.2026.3668398},
publisher={IEEE Computer Society},
address={Los Alamitos, CA, USA},
month=mar}
```

(The README's block also carries `abstract` and `keywords` fields; they are omitted here
as bibliography noise. Everything else is reproduced exactly.)

The volume, number, pages and DOI given in the P1-R2 review all check out against the
README: IEEE Intelligent Systems 41(02):15-23, 2026, doi 10.1109/MIS.2026.3668398.

Two things the README adds that matter for the data card:

- MNW is **multi-modal** — "more than 50,000 artifacts (images, videos, and audio
  files)". Only the image portion is in P1 scope.
- It "includes adversarial examples created with state-of-the-art attacks", so a poor
  score on part of it may be an adversarial-robustness result rather than a
  generalisation result, and must be reported that way.

**Still needed before ingestion:** the README contains no download URL. The access route
is to be supplied separately. If it carries a token, that token goes in **Kaggle Secrets
only** — never in the repository, the logs or a report (CLAUDE.md §4).

---

## E4 — WildRF

Licence `NOASSERTION` (research use only). The README read on 2026-09-20 from
`barcavia/RealTime-DeepfakeDetection-in-the-RealWorld` states no citation requirement in
the section describing the dataset, and no BibTeX block was present in the portion read.
The paper is *Real-Time Deepfake Detection in the Real-World*, project page
`https://vision.huji.ac.il/ladeda/`. A full reference is **NOT-VERIFIED** and should be
taken from the paper itself before the thesis cites it.

---

## E3 — MLLM-Generated Image Detection Dataset

`zr-zhang/MLLM-Generated-Image-Detection-Dataset`, revision
`1498eead24292a9a2d134476b2c559193b68b9de`, licence tag `apache-2.0`.

**Read 2026-10-29** (`README.md` at that revision, 1,692 B, sha256
`1c7e4f0b9e6f9720a8e8cba9d848ec2b78f1ee363bcd79c18bf3253a22f2ef6f`; byte-identical at
the current revision `651a325f0ece2744c21d6c68280d2aaac227177b`). The 2026-09-20 note
("no citation requirement found in the repository metadata") had read the metadata only,
not the README, which **does** ask for a citation and **restricts use**.

Citation, verbatim from the README:

```bibtex
@article{zhang2026benchmark,
  title   = {A Benchmark Dataset for MLLM-Generated Image Detection: GPT Image2 & Nano Banana2},
  author  = {Zhang, Zirui and Yu, Yinbo and Guan, Donghai and Tian, Chunwei and Zhang, Daoqiang and Zhu, Qi},
  journal = {arXiv preprint arXiv:2608.01258},
  year    = {2026}
}
```

Checked against arXiv (API, 2026-10-29): `2608.01258v1`, submitted 2026-08-02, same title,
authors Zirui Zhang; Yinbo Yu; Donghai Guan; Chunwei Tian; Daoqiang Zhang; Qi Zhu. Code:
`github.com/xbrainnet/SAP-DSP` (from the README; not read).

**Use restriction, verbatim from the README ("Licensing and Restrictions"):** *"This
dataset is made available solely for research on AI-generated content detection, image
authenticity analysis, digital forensics, generative model evaluation, and related
academic purposes. It must not be used to develop or optimize technologies for fraud,
forgery, identity impersonation, fabrication of false evidence, or circumvention of content
detection systems."* This narrows the `apache-2.0` tag; CLAUDE.md section 6c was corrected to
match on 2026-10-31 (user decision; raised 2026-10-29).

---

## E5 — Chameleon (AIDE)

Read 2026-09-21 from `https://raw.githubusercontent.com/shilinyan99/AIDE/main/README.md`
(repository `shilinyan99/AIDE`, **MIT** licence on the *code*, ICLR 2025).

Dataset terms, verbatim from the README:

> Chameleon is only used for academic research. Commercial use in any form is prohibited.

Citation, verbatim from the repository README:

```bibtex
@article{yan2024sanity,
  title={A Sanity Check for AI-generated Image Detection},
  author={Yan, Shilin and Li, Ouxiang and Cai, Jiayin and Hao, Yanbin and Jiang, Xiaolong and Hu, Yao and Xie, Weidi},
  journal={arXiv preprint arXiv:2406.19435},
  year={2024}
}
```

**Note:** the repository gives the **arXiv preprint** form, while the work was published
at **ICLR 2025**. The proceedings entry is therefore different from the block above and is
**NOT-VERIFIED** — no command in this session read the ICLR proceedings. Cite the
proceedings version in the thesis, and replace this block once it has been verified.

The official release is a **URL list** obtained by emailing the first author
(`tattoo.ysl@gmail.com`). The images used here came from a third-party community mirror;
see `docs/data_cards/e5_chameleon.md` §7.

---

## Open Images V7 (T-OI training reals, P3)

Read 2026-10-17 from `https://storage.googleapis.com/openimages/web/factsfigures_v7.html`
(sha256 `03f731edf0a6524ae0403253fb3d805b51a42bbaf45a436bfc4491a650fa4fb0`). The page
says, verbatim: "If you use the Open Images dataset in your work (also V5-V7), please cite
this article."

```bibtex
@article{OpenImages,
  author = {Alina Kuznetsova and Hassan Rom and Neil Alldrin and Jasper Uijlings and Ivan Krasin and Jordi Pont-Tuset and Shahab Kamali and Stefan Popov and Matteo Malloci and Alexander Kolesnikov and Tom Duerig and Vittorio Ferrari},
  title = {The Open Images Dataset V4: Unified image classification, object detection, and visual relationship detection at scale},
  year = {2020},
  journal = {IJCV}
}
```

The same page asks for further citations only if the segmentations, the Localized
Narratives or the point-labels are used; none of them is used here. Images are CC BY 2.0 per image:
per-image attribution in `docs/p3_open_images_attribution.csv`. The image files are the
CVDF copies (`cvdfoundation/open-images-dataset`, README at commit
`16c6bae89b22dddc7194618bbb0e24852a911d94`); see `docs/data_cards/t_oi_open_images.md`.

---

## TrueFake (secondary, descriptive social-media test, P3)

Read 2026-10-20 from `MMLab-unitn/TrueFake-IJCNN25` (`README.md` at commit
`806f848ad04883916a4e229c853e0e9ae482465b`, sha256
`6529ff73755a1f7a7ad2f96c42e1788c31b083b9c95f947e05571cbfc8065124`). Repository licence CC BY
4.0 (`LICENSE`). Reproduced exactly as the README gives it.

```bibtex
@INPROCEEDINGS{11228911,
  author={Dell’Anna, Stefano and Montibeller, Andrea and Boato, Giulia},
  booktitle={2025 International Joint Conference on Neural Networks (IJCNN)}, 
  title={TrueFake: A Real World Case Dataset of Last Generation Fake Images also Shared on Social Networks}, 
  year={2025},
  volume={},
  number={},
  pages={1-8},
  keywords={Training;Image coding;Social networking (online);Forensics;Large language models;Neural networks;Transform coding;Detectors;Transformers;Mirrors;Multimedia forensics;AI-generated image detection;Social Networks;Deepfakes},
  doi={10.1109/IJCNN64981.2025.11228911}}
```

**Components** (each read 2026-10-20; see `docs/data_cards/truefake.md`):
- **FFHQ** -- `NVlabs/ffhq-dataset` `LICENSE.txt` (commit `4826aa6e…`) asks for credit "by
  citing our paper": "A Style-Based Generator Architecture for Generative Adversarial
  Networks", Tero Karras, Samuli Laine, Timo Aila (NVIDIA), http://stylegan.xyz/paper.
  Full BibTeX **NOT-VERIFIED** here.
- **FORLAB** -- the TrueFake paper's reference [1]: M. Iuliani, M. Fontani, A. Piva, "A leak
  in PRNU based source identification -- questioning fingerprint uniqueness", IEEE Access,
  vol. 9, pp. 52455-52463, 2021 (arXiv 2009.04878). DOI **NOT-VERIFIED** (seen only in a
  search result).

---

## Training corpora

- **COCO 2017** — CC BY-SA 4.0 (annotations). Cite the COCO paper per its own terms.
  Reference **NOT-VERIFIED** here.
- **DIV2K** — academic research only; the NTIRE challenge papers are the expected
  citation. Reference **NOT-VERIFIED** here.

Both are marked NOT-VERIFIED rather than filled in, for the same reason as MNW: no
command in this session produced them.

---

## Methods cited or compared (verified 2026-10-29)

Author lists and titles read from the arXiv API (`export.arxiv.org/api/query`) on
2026-10-29; repositories and licences from the GitHub API and the files at the commits
given. Every URL read is listed in `docs/P4_BASELINES_SPEC.md`, section 9.

- **SEF** -- arXiv `2605.14486`. **The method was renamed between versions:**
  - v1 (submitted 2026-05-14): "Reduce the Artifacts Bias for More Generalizable
    AI-Generated Image Detection"; Yiheng Li, Yang Yang, Zichang Tan, Gao Li, Zhen Lei,
    Wenhao Wang; the abstract names the method **Separate Expert Fusion (SEF)**; code
    `github.com/liyih/SEF_AIGC_detection` (MIT per the GitHub API; not read further).
  - v2 (updated 2026-08-23): same title; Yiheng Li, Yang Yang, Wenhao Wang, Zichang Tan,
    Zecheng Lin, Li Gao, Zhen Lei; the abstract names it **Artifact-Complementary Expert
    Fusion (ACEF)**. arXiv comment: "preprint".
  - The thesis cites "SEF (2026)"; cite v1 for that name, or v2 with the new name.
- **Final component** -- arXiv `2601.20461v2` (submitted 2026-01-28, updated 2026-03-10):
  "Exploiting the Final Component of Generator Architectures for AI-Generated Image
  Detection"; Yanzhu Liu, Xiao Liu, Yuexuan Wang, Mondal Soumik. No venue or code link in
  the arXiv record.
- **DDA** -- arXiv `2505.14359v6`: "Dual Data Alignment Makes AI-Generated Image Detector
  Easier Generalizable"; Ruoxin Chen, Junwei Xi, Zhiyuan Yan, Ke-Yue Zhang, Shuang Wu,
  Jingyi Xie, Xu Chen, Lei Xu, Isabel Guan, Taiping Yao, Shouhong Ding; NeurIPS 2025
  (Spotlight). BibTeX in the repository README (`roy-ch/Dual-Data-Alignment` at
  `08d6b414…`) matches.
- **CO-SPY** -- arXiv `2503.18286v1`: "CO-SPY: Combining Semantic and Pixel Features to
  Detect Synthetic Images by AI"; Siyuan Cheng, Lingjuan Lyu, Zhenting Wang, Xiangyu
  Zhang, Vikash Sehwag; journal ref. CVPR 2025.
- **AEROBLADE** -- arXiv `2401.17879v2`: "AEROBLADE: Training-Free Detection of Latent
  Diffusion Images Using Autoencoder Reconstruction Error"; Jonas Ricker, Denis
  Lukovnikov, Asja Fischer; CVPR 2024.
- **FIRE** -- arXiv `2412.07140v3`: "FIRE: Robust Detection of Diffusion-Generated Images
  via Frequency-Guided Reconstruction Error"; Beilin Chu, Xuan Xu, Xin Wang, Yufei Zhang,
  Weike You, Linna Zhou; CVPR 2025, DOI 10.1109/CVPR52734.2025.01197 (both from the arXiv
  record).

