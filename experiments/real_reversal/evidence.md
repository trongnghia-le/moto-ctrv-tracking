# CTRV's turn-rate boundary on real HCMC trajectories (IASC revision, Reviewer 2 #5)

## Input

The motorcycle tracks that pre-labelled the *dense* segment (frames 10,700-11,299
of the NKKN stream), exported from CVAT as MOT 1.1: 27,513 boxes, 335 tracks, 600
frames. The file is derived from third-party video and is not redistributed.
- The boxes are C2's filtered EKF states written by the tracker (the `tlwh` of a
  CTRV track), not raw detections.
- Only stretches that C2 tracked without a break can be used (17 or more
  contiguous frames), so tracks that C2 actually lost are excluded.

## Method (`run.py`)

On every contiguous motorcycle stretch, three filters start 12 frames before a
candidate last-visible frame t0 and are updated through t0:
1. C2's CTRV EKF;
2. the same EKF with omega pinned at 0 (same Q, R, P0 and bootstrap), which
   isolates the turn-rate state;
3. ByteTrack's linear KF.

Each filter then coasts 5 hidden frames. Error is the distance from the coasted
position to the real centre at t0+5, in box diagonals. Only windows with a mean
speed of at least 2 px/frame count.

Each window is classified from the centres, using the turn rate from a
3-frame-baseline heading (mean over t0-7..t0-4 against mean over t0-3..t0):
- **reversal:** opposite signs, both above TURN;
- **same-sign turn:** same sign, both above TURN;
- **straight:** both below 0.01 rad/frame.

Windows of one track overlap, so shares are given as [point, 2.5%, 97.5%] from a
bootstrap over tracks (2,000 replicates).

The omega=0 arm and the bootstrap were added after an internal review of the
first version, which compared CTRV only with the linear KF. That comparison
cannot separate the turn-rate state from the other differences between the
filters: the omega=0 EKF also beats the linear KF on 96% of straight windows.

## Primary setting (TURN 0.03, `summary.json`)

| kind | windows / tracks | CTRV worse than omega=0 | median error CTRV / omega=0 / linear |
|---|---|---|---|
| reversal | 668 / 148 | 68% [64, 72] | 0.135 / 0.120 / 0.132 |
| same-sign turn | 230 / 88 | 54% [44, 65] | 0.146 / 0.178 / 0.165 |
| straight | 1,632 / 201 | 63% [59, 67] | 0.153 / 0.154 / 0.184 |

## Full grid (`sensitivity.json`, minimum speed 2 px/frame)

Each cell: share of windows where CTRV is worse than omega=0, then windows/tracks.

| TURN | reversal | same-sign turn |
|---|---|---|
| 0.03 | 68% [64, 72], 668/148 | 54% [44, 65], 230/88 |
| 0.05 | 69% [62, 75], 268/86 | 58% [40, 81], 60/27 |
| 0.08 | 73% [64, 82], 103/49 | 69% [46, 89], 13/9 |

With a minimum speed of 4 px/frame, reversal gives 64%, 62% and 68%, with lower
interval bounds of 56%, 47% and 54%.

## Persistence (`persistence.json`)

The mean turn rate over the hidden frames kept the sign of the pre-gap turn rate
in 36-43% of windows (8% for the 13 sharpest same-sign turns).

## Reading

After an apparent turn-rate reversal, the turn-rate state measurably worsens the
5-frame coast; all intervals at minimum speed 2 lie above 50%. On straight motion
the effect is negligible in size. After same-sign turns it is inconclusive.

## Figure

`figure.py` draws Fig. 2c-d of the IASC manuscript. Panel (c) is the window with
the median CTRV-minus-(omega=0) error among the 103 reversal windows at TURN
0.08.

## Caveats

- The boxes are filtered states, not raw detections.
- Stretches are selected by continuous tracking.
- The occlusion is simulated, not observed.
- The pre-label tracks may contain identity errors.
