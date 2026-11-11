# H3-k: ECE by k (E4, E5)

Expected calibration error, 15 equal-width bins, of sigmoid(score) over the view's test reals and test fakes (mean over draws; one draw where the pool is used whole). Pre-registered as reported, never judged. (Q) moves only the threshold, so its ECE equals k = 0's by construction (interpretation h3k_ece, confirmed 2026-10-28).

| seed | pool | method | k | ece | draws | status |
|---|---|---|---|---|---|---|
| S2-D s0 | E4 | k = 0 | 0 | 0.0735 | - | descriptive |
| S2-D s0 | E4 | Q | 10 | 0.0735 | 1000 | descriptive |
| S2-D s0 | E4 | Q | 50 | 0.0735 | 1000 | descriptive |
| S2-D s0 | E4 | Q | 100 | 0.0735 | 1 | descriptive |
| S2-D s0 | E4 | S | 10 | 0.1469 | 1000 | descriptive |
| S2-D s0 | E4 | S | 50 | 0.1494 | 1000 | descriptive |
| S2-D s0 | E4 | S | 100 | 0.1503 | 1 | descriptive |
| S2-D s0 | E5 | k = 0 | 0 | 0.1764 | - | descriptive |
| S2-D s0 | E5 | Q | 10 | 0.1764 | 1000 | descriptive |
| S2-D s0 | E5 | Q | 50 | 0.1764 | 1000 | descriptive |
| S2-D s0 | E5 | Q | 100 | 0.1764 | 1000 | descriptive |
| S2-D s0 | E5 | S | 10 | 0.3583 | 1000 | descriptive |
| S2-D s0 | E5 | S | 50 | 0.3592 | 1000 | descriptive |
| S2-D s0 | E5 | S | 100 | 0.3594 | 1000 | descriptive |
| S2-D s1 | E4 | k = 0 | 0 | 0.0662 | - | descriptive |
| S2-D s1 | E4 | Q | 10 | 0.0662 | 1000 | descriptive |
| S2-D s1 | E4 | Q | 50 | 0.0662 | 1000 | descriptive |
| S2-D s1 | E4 | Q | 100 | 0.0662 | 1 | descriptive |
| S2-D s1 | E4 | S | 10 | 0.1528 | 1000 | descriptive |
| S2-D s1 | E4 | S | 50 | 0.1572 | 1000 | descriptive |
| S2-D s1 | E4 | S | 100 | 0.1604 | 1 | descriptive |
| S2-D s1 | E5 | k = 0 | 0 | 0.1625 | - | descriptive |
| S2-D s1 | E5 | Q | 10 | 0.1625 | 1000 | descriptive |
| S2-D s1 | E5 | Q | 50 | 0.1625 | 1000 | descriptive |
| S2-D s1 | E5 | Q | 100 | 0.1625 | 1000 | descriptive |
| S2-D s1 | E5 | S | 10 | 0.3506 | 1000 | descriptive |
| S2-D s1 | E5 | S | 50 | 0.3521 | 1000 | descriptive |
| S2-D s1 | E5 | S | 100 | 0.3526 | 1000 | descriptive |
