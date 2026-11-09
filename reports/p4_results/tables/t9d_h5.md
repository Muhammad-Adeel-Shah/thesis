# Table 9d (iv): DDA under the two H5 conditions beside S2-D seed 0

Descriptive only: P4 has no hypothesis and no decision rule (spec header). AEROBLADE is our implementation from the paper (its repository has no licence). E7 FLUX.1 standard is an upper bound (CLAUDE.md section 6d). Kaggle's Pillow 13.0.0 is not the 12.3.0 DDA's README pins (spec section 4). Panel B test rows. S2-D seed 0's macro AUC under each condition reproduces the committed P3 H5 curve; its change from clean is P3's H5 decision value.

| condition | DDA macro AUC | DDA minus DDA clean, points | S2-D s0 macro AUC | S2-D s0 minus its clean, points (P3 H5) | S2-D s0 minus DDA, points | status |
|---|---|---|---|---|---|---|
| JPEG q50 | 0.704 [0.673, 0.734] | -8.0 [-9.6, -6.4] | 0.622 [0.591, 0.651] | -13.3 | -8.2 [-12.2, -4.6] | descriptive |
| resize-and-restore 0.5 | 0.738 [0.709, 0.767] | -4.7 [-6.0, -3.3] | 0.707 [0.677, 0.738] | -4.8 | -3.0 [-6.7, +0.6] | descriptive |
