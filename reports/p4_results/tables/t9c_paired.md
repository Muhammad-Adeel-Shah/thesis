# Table 9c: paired differences, ours minus each baseline (AUC and FPR points, 95% CI)

Descriptive only: P4 has no hypothesis and no decision rule (spec header). AEROBLADE is our implementation from the paper (its repository has no licence). E7 FLUX.1 standard is an upper bound (CLAUDE.md section 6d). Kaggle's Pillow 13.0.0 is not the 12.3.0 DDA's README pins (spec section 4). A positive macro-AUC difference favours ours; a negative macro-FPR difference means ours has the lower FPR. 'CI excludes 0' is stated, not judged.

| ours | minus baseline | delta macro AUC, points | AUC CI excludes 0 | delta macro FPR E4/E5, points | FPR CI excludes 0 | status |
|---|---|---|---|---|---|---|
| S2-D s0 | DDA | -2.9 [-6.0, +0.1] | no | +1.2 [-1.1, +3.6] | no | descriptive |
| S2-D s0 | DDA crop-308 (sensitivity) | -2.8 [-5.8, +0.1] | no | +1.9 [-0.6, +4.4] | no | descriptive |
| S2-D s0 | CO-SPY | -1.4 [-5.2, +2.1] | no | -13.6 [-16.6, -10.4] | yes | descriptive |
| S2-D s0 | FIRE | +20.6 [+16.9, +24.3] | yes | +10.6 [+8.6, +12.8] | yes | descriptive |
| S2-D s0 | AEROBLADE (ours, from the paper) | +12.4 [+8.1, +16.9] | yes | -1.9 [-4.3, +0.7] | no | descriptive |
| S2-D s1 | DDA | -2.4 [-5.4, +0.4] | no | +1.3 [-1.1, +3.8] | no | descriptive |
| S2-D s1 | DDA crop-308 (sensitivity) | -2.3 [-5.3, +0.4] | no | +2.0 [-0.3, +4.5] | no | descriptive |
| S2-D s1 | CO-SPY | -0.9 [-4.5, +2.4] | no | -13.4 [-16.4, -10.3] | yes | descriptive |
| S2-D s1 | FIRE | +21.1 [+17.3, +24.8] | yes | +10.8 [+8.8, +13.0] | yes | descriptive |
| S2-D s1 | AEROBLADE (ours, from the paper) | +13.0 [+8.9, +17.1] | yes | -1.7 [-4.3, +0.9] | no | descriptive |
| P2 full_bank | DDA | -2.1 [-5.0, +0.8] | no | +2.0 [-0.3, +4.4] | no | descriptive |
| P2 full_bank | DDA crop-308 (sensitivity) | -2.0 [-4.9, +0.9] | no | +2.7 [+0.2, +5.3] | yes | descriptive |
| P2 full_bank | CO-SPY | -0.6 [-4.1, +2.9] | no | -12.8 [-15.7, -9.7] | yes | descriptive |
| P2 full_bank | FIRE | +21.5 [+17.4, +25.5] | yes | +11.5 [+9.4, +13.7] | yes | descriptive |
| P2 full_bank | AEROBLADE (ours, from the paper) | +13.3 [+9.0, +17.6] | yes | -1.0 [-3.6, +1.6] | no | descriptive |
