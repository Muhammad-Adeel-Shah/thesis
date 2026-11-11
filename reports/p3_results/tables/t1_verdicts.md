# P3 hypothesis verdicts

Every row is read from the committed results; `status` says whether it is a pre-registered decision or descriptive.

| hypothesis | role | status | statistic | estimate | level | criterion | verdict | source |
|---|---|---|---|---|---|---|---|---|
| H1-a | primary | decision (pre-registered) | G, mean held-out-lineage AUC over 8 LOLO folds | 0.9662 [0.9612, 0.9709] | 0.975 | G >= 0.90, CI lower >= 0.85, no fold < 0.75 | PASS | reports/p3_h1a_ci.json |
| H1-b fold 1 | primary | decision (pre-registered) | AUC, e7_dda_coco/sd-vae-ft-ema | 0.9985 [0.9963, 0.9998] | 0.975 | each subset >= 0.85 (point) | PASS | p3_analysis_full.json |
| H1-b fold 1 | primary | decision (pre-registered) | AUC, e7_dda_coco/sd-vae-ft-mse | 0.9984 [0.9962, 0.9998] | 0.975 | each subset >= 0.85 (point) | PASS | p3_analysis_full.json |
| H1-b fold 1 | primary | decision (pre-registered) | AUC, e7_dda_coco/stable-diffusion-2-1 | 0.9979 [0.9955, 0.9995] | 0.975 | each subset >= 0.85 (point) | PASS | p3_analysis_full.json |
| H1-b fold 2 | primary | decision (pre-registered) | AUC, e7_dda_coco/sdxl-vae | 0.9977 [0.9948, 0.9995] | 0.975 | each subset >= 0.85 (point) | PASS | p3_analysis_full.json |
| H1-b fold 3 | primary | decision (pre-registered) | AUC, e7_dda_coco/stable-diffusion-3.5-large | 0.9859 [0.9778, 0.9927] | 0.975 | each subset >= 0.85 (point) | PASS | p3_analysis_full.json |
| H1-b fold 3 | primary | decision (pre-registered) | AUC, e7_dda_coco_luma/FLUX.1 | 0.8994 [0.8705, 0.9253] | 0.975 | each subset >= 0.85 (point) | PASS | p3_analysis_full.json |
| H1-b fold 3 | primary | descriptive | AUC, e7_dda_coco/FLUX.1 (upper bound, section 6d shortcut) | 0.9741 [0.9618, 0.9843] | 0.975 | not judged | - | p3_analysis_full.json |
| H1-b | primary | decision (pre-registered) | folds passing | 3 / 3 | - | >= 2 of 3 folds (claim, with H1-a) | PASS | p3_analysis_full.json |
| H1-c | secondary | decision (pre-registered) | mean (LOLO-f - BANK-1) AUC, folds 2-8 | 0.0533 [0.0466, 0.0600] | 0.95 | >= +0.02 and CI excludes 0 | PASS | reports/p3_h1c.json |
| H3 | primary | decision (pre-registered) | S2-D s0: macro FPR E4/E5 at the k = 0 union threshold (E4 15.8%, E5 12.0%) | 13.9% | - | <= 15%; no pool > 25% | met | p3_analysis_full.json |
| H3 | primary | decision (pre-registered) | S2-D s0: relative FPR reduction vs P2 full_bank (14.8%) | 0.0559 [-0.0972, 0.1990] | 0.975 | >= 0.50 and CI excludes 0 | not met | p3_analysis_full.json |
| H3 | primary | decision (pre-registered) | S2-D s1: macro FPR E4/E5 at the k = 0 union threshold (E4 16.3%, E5 11.8%) | 14.1% | - | <= 15%; no pool > 25% | met | p3_analysis_full.json |
| H3 | primary | decision (pre-registered) | S2-D s1: relative FPR reduction vs P2 full_bank (14.8%) | 0.0454 [-0.1013, 0.1775] | 0.975 | >= 0.50 and CI excludes 0 | not met | p3_analysis_full.json |
| H3 | primary | decision (pre-registered) | all criteria, both seeds | - | - | all of the above | FAIL | p3_analysis_full.json |
| H3-k (Q) | secondary | decision (pre-registered) | S2-D s0: share of k = 50 draws with FPR <= 7.5% | E4 0.987, E5 0.963 | - | >= 0.80 on E4 and on E5, both seeds | met | p3_analysis_full.json |
| H3-k (S) | secondary | decision (pre-registered) | S2-D s0: share of k = 50 draws with FPR <= 7.5% | E4 1.000, E5 1.000 | - | >= 0.80 on E4 and on E5, both seeds | met | p3_analysis_full.json |
| H3-k (Q) | secondary | decision (pre-registered) | S2-D s1: share of k = 50 draws with FPR <= 7.5% | E4 0.993, E5 0.968 | - | >= 0.80 on E4 and on E5, both seeds | met | p3_analysis_full.json |
| H3-k (S) | secondary | decision (pre-registered) | S2-D s1: share of k = 50 draws with FPR <= 7.5% | E4 1.000, E5 1.000 | - | >= 0.80 on E4 and on E5, both seeds | met | p3_analysis_full.json |
| H3-k (Q) | secondary | decision (pre-registered) | both seeds | - | - | both seeds | PASS | p3_analysis_full.json |
| H3-k (S) | secondary | decision (pre-registered) | both seeds | - | - | both seeds | PASS | p3_analysis_full.json |
| H2 | secondary | decision (pre-registered) | S2-D s0 - P2 full_bank macro AUC, 4 unseen slices | -0.0082 [-0.0271, 0.0111] | 0.95 | CI lower > -0.02 (non-inferior), both seeds | not shown | p3_analysis_full.json |
| H2 | secondary | decision (pre-registered) | S2-D s1 - P2 full_bank macro AUC, 4 unseen slices | -0.0032 [-0.0218, 0.0160] | 0.95 | CI lower > -0.02 (non-inferior), both seeds | not shown | p3_analysis_full.json |
| H2 | secondary | decision (pre-registered) | both seeds, 4 unseen slices | - | - | both seeds | FAIL | p3_analysis_full.json |
| H2 | secondary | pre-registered companion test, reported beside the decision (section 6.4) | S2-D s0 - P2 full_bank macro AUC, without E3 (E4, E5) | -0.0141 [-0.0214, -0.0072] | 0.95 | CI lower > -0.02 (non-inferior), both seeds | not shown | p3_analysis_full.json |
| H2 | secondary | pre-registered companion test, reported beside the decision (section 6.4) | S2-D s1 - P2 full_bank macro AUC, without E3 (E4, E5) | -0.0006 [-0.0088, 0.0071] | 0.95 | CI lower > -0.02 (non-inferior), both seeds | non-inferior | p3_analysis_full.json |
| H2 | secondary | pre-registered companion test, reported beside the decision (section 6.4) | both seeds, without E3 (E4, E5) | - | - | both seeds | FAIL | p3_analysis_full.json |
| H4 | secondary | decision (pre-registered) | BANK-8 - BANK-1 macro AUC, 4 unseen slices | 0.1364 [0.1161, 0.1588] | 0.95 | >= +0.02 and CI excludes 0 | PASS | p3_analysis_full.json |
| H5 | secondary | decision (pre-registered) | S2-D s0 macro AUC, jpeg50 (clean 0.7553; difference -0.1334) | 0.6219 [0.5910, 0.6510] | 0.95 | \|condition - clean\| <= 0.05 (point) | outside | p3_analysis_full.json |
| H5 | secondary | decision (pre-registered) | S2-D s0 macro AUC, resize050 (clean 0.7553; difference -0.0480) | 0.7074 [0.6772, 0.7378] | 0.95 | \|condition - clean\| <= 0.05 (point) | within | p3_analysis_full.json |
| H5 | secondary | decision (pre-registered) | both conditions | - | - | both within 5 points | FAIL | p3_analysis_full.json |
