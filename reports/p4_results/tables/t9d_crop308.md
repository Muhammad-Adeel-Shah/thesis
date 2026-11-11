# Table 9d (iii): DDA crop sensitivity, CenterCrop 336 vs 308

Descriptive only: P4 has no hypothesis and no decision rule (spec header). AEROBLADE is our implementation from the paper (its repository has no licence). E7 FLUX.1 standard is an upper bound (CLAUDE.md section 6d). Kaggle's Pillow 13.0.0 is not the 12.3.0 DDA's README pins (spec section 4).

| row | crop | GPT-Image-2 (E3) | Nano-Banana-2 (E3) | E4 WildRF | E5 Chameleon | macro AUC (4 unseen) | macro FPR E4/E5 | status |
|---|---|---|---|---|---|---|---|---|
| DDA | CenterCrop(336), DDA's own | 0.476 [0.412, 0.543] | 0.838 [0.784, 0.887] | 0.950 [0.935, 0.962] | 0.873 [0.863, 0.883] | 0.784 [0.758, 0.810] | 12.8% [10.8%, 14.6%] | descriptive |
| DDA crop-308 (sensitivity) | CenterCrop(308), sensitivity | 0.475 [0.408, 0.542] | 0.835 [0.784, 0.885] | 0.943 [0.928, 0.956] | 0.880 [0.870, 0.890] | 0.784 [0.757, 0.809] | 12.1% [10.2%, 14.0%] | descriptive |
