# Large initial conditions of the CTRV EKF (IASC revision, Reviewer 1 #7)

`run.py` starts the unmodified `CTRVMotionModel` (default Q, R, P0, two-detection
bootstrap) from its zero prior `[cx, cy, 0, 0, 0]` on synthetic constant-turn
tracks and records, after k visible detections, the absolute heading error, the
relative speed error and whether P stays finite, symmetric and positive definite.

Grid: true speed 1, 2, 5, 10, 15, 20, 25 px/frame; 24 true headings over the full
circle (initial heading error up to +-pi); turn rate -0.08, 0, +0.08 rad/frame;
centre noise 0 px (1 run) and 2 px (10 seeds). 5,544 runs. Full numbers in
`summary.json` (`by_k`, k = 2..10).

| noise | speed (px/frame) | heading err. median / p95 after 4 det. (rad) | after 10 det. (rad) |
|---|---|---|---|
| 0 | 1 | 0.190 / 3.088 (bootstrap needs > 1 px) | 0.039 / 3.132 |
| 0 | 2-25 | p95 <= 0.201, max <= 0.206 | p95 <= 0.062, max <= 0.063 |
| 2 px | 1-2 | 0.44-0.91 / 2.63-2.81 | 0.59-1.43 / 2.81-2.89 |
| 2 px | 5 | 0.416 / 1.405 | 0.172 / 0.677 |
| 2 px | 10-25 | 0.19-0.30 / 0.49-0.69 | 0.04-0.09 / 0.12-0.23 |

P stayed finite, symmetric and positive definite in all 5,544 runs.

Reading: a large initial speed or heading error is absorbed within a few
detections. The hard case is a slow motorcycle whose displacement per frame is
comparable to the detection noise (Proposition 1). This replaces an earlier
manuscript sentence ("below 0.08 rad within four visible frames") that had no
run behind it and is not what the filter does.
