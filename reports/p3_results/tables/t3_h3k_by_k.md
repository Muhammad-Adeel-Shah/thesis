# H3-k: FPR and TPR by k

Decision rows: k = 50, E4 and E5. Single-draw cells give one FPR with its bootstrap CI over the test reals.

| seed | pool | method | k | mean_fpr | mean_tpr | share_fpr_le_7_5 | draws | status |
|---|---|---|---|---|---|---|---|---|
| S2-D s0 | E3 | k = 0 | 0 | 13.2% | 28.4% | - | - | descriptive |
| S2-D s0 | E3 | Q | 10 | 8.9% | 23.5% | 0.487 | 1000 | descriptive |
| S2-D s0 | E3 | Q | 25 | 2.0% | 12.8% | 0.884 | 1000 | descriptive |
| S2-D s0 | E3 | Q | 50 | 0.0% | 8.9% | single draw: FPR 0.0% [0.0%, 0.0%] | 1 | descriptive |
| S2-D s0 | E3 | S | 10 | 1.8% | 13.1% | 0.989 | 1000 | descriptive |
| S2-D s0 | E3 | S | 25 | 1.5% | 12.4% | 1.000 | 1000 | descriptive |
| S2-D s0 | E3 | S | 50 | 1.7% | 11.6% | single draw: FPR 1.7% [0.0%, 4.1%] | 1 | descriptive |
| S2-D s0 | E4 | k = 0 | 0 | 15.8% | 86.9% | - | - | descriptive |
| S2-D s0 | E4 | Q | 10 | 9.0% | 68.9% | 0.469 | 1000 | descriptive |
| S2-D s0 | E4 | Q | 50 | 0.6% | 39.2% | 0.987 | 1000 | decision (pre-registered) |
| S2-D s0 | E4 | Q | 100 | 1.0% | 56.1% | single draw: FPR 1.0% [0.3%, 2.3%] | 1 | descriptive |
| S2-D s0 | E4 | S | 10 | 1.9% | 62.3% | 0.997 | 1000 | descriptive |
| S2-D s0 | E4 | S | 50 | 1.7% | 61.7% | 1.000 | 1000 | decision (pre-registered) |
| S2-D s0 | E4 | S | 100 | 1.8% | 61.3% | single draw: FPR 1.8% [0.5%, 3.3%] | 1 | descriptive |
| S2-D s0 | E5 | k = 0 | 0 | 12.0% | 27.5% | - | - | descriptive |
| S2-D s0 | E5 | Q | 10 | 10.1% | 23.2% | 0.476 | 1000 | descriptive |
| S2-D s0 | E5 | Q | 50 | 3.2% | 9.9% | 0.963 | 1000 | decision (pre-registered) |
| S2-D s0 | E5 | Q | 100 | 5.8% | 16.3% | 0.876 | 1000 | descriptive |
| S2-D s0 | E5 | S | 10 | 2.4% | 8.2% | 0.999 | 1000 | descriptive |
| S2-D s0 | E5 | S | 50 | 2.4% | 7.9% | 1.000 | 1000 | decision (pre-registered) |
| S2-D s0 | E5 | S | 100 | 2.4% | 7.9% | 1.000 | 1000 | descriptive |
| S2-D s0 | E7 | k = 0 | 0 | 0.7% | 93.1% | - | - | descriptive |
| S2-D s0 | E7 | Q | 10 | 7.6% | 97.7% | 0.618 | 1000 | descriptive |
| S2-D s0 | E7 | Q | 50 | 1.8% | 96.1% | 0.999 | 1000 | descriptive |
| S2-D s0 | E7 | Q | 100 | 2.0% | 96.7% | single draw: FPR 2.0% [0.7%, 3.7%] | 1 | descriptive |
| S2-D s0 | E7 | S | 10 | 1.5% | 95.2% | 0.998 | 1000 | descriptive |
| S2-D s0 | E7 | S | 50 | 1.4% | 95.6% | 1.000 | 1000 | descriptive |
| S2-D s0 | E7 | S | 100 | 1.3% | 95.7% | single draw: FPR 1.3% [0.3%, 2.7%] | 1 | descriptive |
| S2-D s1 | E3 | k = 0 | 0 | 5.8% | 21.4% | - | - | descriptive |
| S2-D s1 | E3 | Q | 10 | 8.3% | 23.5% | 0.495 | 1000 | descriptive |
| S2-D s1 | E3 | Q | 25 | 2.1% | 12.8% | 0.889 | 1000 | descriptive |
| S2-D s1 | E3 | Q | 50 | 0.0% | 7.6% | single draw: FPR 0.0% [0.0%, 0.0%] | 1 | descriptive |
| S2-D s1 | E3 | S | 10 | 0.5% | 11.6% | 1.000 | 1000 | descriptive |
| S2-D s1 | E3 | S | 25 | 0.5% | 11.7% | 1.000 | 1000 | descriptive |
| S2-D s1 | E3 | S | 50 | 0.8% | 11.9% | single draw: FPR 0.8% [0.0%, 2.5%] | 1 | descriptive |
| S2-D s1 | E4 | k = 0 | 0 | 16.3% | 86.7% | - | - | descriptive |
| S2-D s1 | E4 | Q | 10 | 8.0% | 67.2% | 0.555 | 1000 | descriptive |
| S2-D s1 | E4 | Q | 50 | 1.0% | 41.7% | 0.993 | 1000 | decision (pre-registered) |
| S2-D s1 | E4 | Q | 100 | 1.0% | 56.3% | single draw: FPR 1.0% [0.3%, 2.3%] | 1 | descriptive |
| S2-D s1 | E4 | S | 10 | 2.3% | 61.8% | 0.990 | 1000 | descriptive |
| S2-D s1 | E4 | S | 50 | 2.0% | 61.6% | 1.000 | 1000 | decision (pre-registered) |
| S2-D s1 | E4 | S | 100 | 2.0% | 61.5% | single draw: FPR 2.0% [0.8%, 3.6%] | 1 | descriptive |
| S2-D s1 | E5 | k = 0 | 0 | 11.8% | 34.4% | - | - | descriptive |
| S2-D s1 | E5 | Q | 10 | 10.1% | 27.9% | 0.503 | 1000 | descriptive |
| S2-D s1 | E5 | Q | 50 | 3.1% | 11.3% | 0.968 | 1000 | decision (pre-registered) |
| S2-D s1 | E5 | Q | 100 | 5.4% | 18.1% | 0.902 | 1000 | descriptive |
| S2-D s1 | E5 | S | 10 | 2.5% | 9.9% | 0.997 | 1000 | descriptive |
| S2-D s1 | E5 | S | 50 | 2.3% | 9.6% | 1.000 | 1000 | decision (pre-registered) |
| S2-D s1 | E5 | S | 100 | 2.3% | 9.5% | 1.000 | 1000 | descriptive |
| S2-D s1 | E7 | k = 0 | 0 | 0.7% | 92.8% | - | - | descriptive |
| S2-D s1 | E7 | Q | 10 | 7.0% | 97.4% | 0.691 | 1000 | descriptive |
| S2-D s1 | E7 | Q | 50 | 1.7% | 95.2% | 1.000 | 1000 | descriptive |
| S2-D s1 | E7 | Q | 100 | 2.7% | 96.0% | single draw: FPR 2.7% [1.0%, 4.7%] | 1 | descriptive |
| S2-D s1 | E7 | S | 10 | 1.8% | 94.7% | 1.000 | 1000 | descriptive |
| S2-D s1 | E7 | S | 50 | 1.5% | 94.9% | 1.000 | 1000 | descriptive |
| S2-D s1 | E7 | S | 100 | 1.3% | 94.8% | single draw: FPR 1.3% [0.3%, 2.7%] | 1 | descriptive |
