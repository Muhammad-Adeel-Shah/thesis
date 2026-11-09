# Table 9d (ii): E8 TPR at the k = 0 union threshold (fakes only)

Descriptive only: P4 has no hypothesis and no decision rule (spec header). AEROBLADE is our implementation from the paper (its repository has no licence). E7 FLUX.1 standard is an upper bound (CLAUDE.md section 6d). Kaggle's Pillow 13.0.0 is not the 12.3.0 DDA's README pins (spec section 4).

| model | group | Flux | GoT | Infinity | NOVA | OmniGen | status |
|---|---|---|---|---|---|---|---|
| DDA | baseline | 91.2% | 100.0% | 98.8% | 99.2% | 100.0% | descriptive |
| DDA crop-308 (sensitivity) | baseline | 90.8% | 100.0% | 98.4% | 98.8% | 100.0% | descriptive |
| CO-SPY | baseline | 89.6% | 99.2% | 93.6% | 96.4% | 100.0% | descriptive |
| FIRE | baseline | 8.4% | 9.6% | 14.0% | 8.4% | 5.2% | descriptive |
| AEROBLADE (ours, from the paper) | baseline | 57.6% | 93.2% | 24.0% | 53.6% | 72.8% | descriptive |
| S2-D s0 | ours | 82.4% | 98.0% | 100.0% | 95.6% | 97.2% | descriptive |
| S2-D s1 | ours | 80.4% | 97.2% | 99.6% | 94.0% | 96.8% | descriptive |
| P2 full_bank | ours | 66.8% | 92.8% | 98.0% | 90.0% | 92.4% | descriptive |
| S2-B s0 | ours | 70.4% | 93.6% | 98.0% | 90.4% | 92.8% | descriptive |
