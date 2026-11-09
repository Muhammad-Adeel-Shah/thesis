# Table 9a: AUC on the 4 unseen slices and their macro, 95% CI

Descriptive only: P4 has no hypothesis and no decision rule (spec header). AEROBLADE is our implementation from the paper (its repository has no licence). E7 FLUX.1 standard is an upper bound (CLAUDE.md section 6d). Kaggle's Pillow 13.0.0 is not the 12.3.0 DDA's README pins (spec section 4).

| model | group | GPT-Image-2 (E3) | Nano-Banana-2 (E3) | E4 WildRF | E5 Chameleon | macro AUC (4 unseen) | status |
|---|---|---|---|---|---|---|---|
| DDA | baseline | 0.476 [0.412, 0.543] | 0.838 [0.784, 0.887] | 0.950 [0.935, 0.962] | 0.873 [0.863, 0.883] | 0.784 [0.758, 0.810] | descriptive |
| CO-SPY | baseline | 0.774 [0.717, 0.828] | 0.744 [0.676, 0.808] | 0.821 [0.792, 0.848] | 0.739 [0.723, 0.753] | 0.770 [0.740, 0.799] | descriptive |
| FIRE | baseline | 0.506 [0.443, 0.568] | 0.526 [0.450, 0.602] | 0.554 [0.514, 0.594] | 0.610 [0.593, 0.627] | 0.549 [0.519, 0.579] | descriptive |
| AEROBLADE (ours, from the paper) | baseline | 0.586 [0.516, 0.654] | 0.635 [0.562, 0.705] | 0.686 [0.651, 0.721] | 0.616 [0.598, 0.635] | 0.631 [0.597, 0.665] | descriptive |
| S2-D s0 | ours | 0.604 [0.538, 0.666] | 0.829 [0.776, 0.879] | 0.935 [0.919, 0.950] | 0.652 [0.637, 0.670] | 0.755 [0.729, 0.781] | descriptive |
| S2-D s1 | ours | 0.601 [0.539, 0.666] | 0.825 [0.770, 0.877] | 0.925 [0.907, 0.941] | 0.690 [0.674, 0.707] | 0.760 [0.734, 0.786] | descriptive |
| P2 full_bank | ours | 0.632 [0.566, 0.696] | 0.807 [0.750, 0.861] | 0.944 [0.927, 0.957] | 0.672 [0.656, 0.689] | 0.764 [0.737, 0.790] | descriptive |
| S2-B s0 | ours | 0.622 [0.555, 0.687] | 0.801 [0.743, 0.854] | 0.946 [0.929, 0.959] | 0.685 [0.669, 0.701] | 0.763 [0.735, 0.790] | descriptive |
