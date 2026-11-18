# CTRV's turn-rate boundary on real HCMC trajectories (IASC revision, Reviewer 2 #5)

**Input.** The motorcycle tracks that pre-labelled the *dense* segment (frames
10,700-11,299 of the NKKN stream): C2's output before human correction,
exported from CVAT as MOT 1.1 (27,513 boxes, 335 tracks, 600 frames). The file
is derived from third-party video and is not redistributed. The final reviewed
ground truth is not used; only real box centres are.

**Method (`run.py`, fixed before the first run).** On every contiguous
motorcycle run, each filter (C2's CTRV EKF and ByteTrack's linear KF) is
started 12 frames before a candidate last-visible frame t0 and updated through
t0. It then coasts 5 hidden frames (the typical gap length on dense). Error =
distance from the coasted position to the real centre at t0+5, in box
diagonals. Only windows with mean speed >= 2 px/frame count. Each window is
classified from observed centres: turn rate from the 3-frame-baseline heading,
mean over t0-7..t0-4 versus t0-3..t0.
- Reversal: opposite signs, both above TURN.
- Same-sign turn: same sign, both above TURN.
- Straight: both below 0.01 rad/frame.

**Primary setting (TURN 0.03, `summary.json`).**

| kind | windows | CTRV worse | median err CTRV / linear |
|---|---|---|---|
| straight | 1,632 | 6.1% | 0.153 / 0.184 |
| reversal | 668 | 48.4% | 0.135 / 0.132 |
| same-sign turn | 230 | 56.5% | 0.146 / 0.165 |

**Full grid (`sensitivity.py` → `sensitivity.json`).** Share of windows where
CTRV is worse; n = windows. Speed >= 2 / >= 4 px/frame.

| TURN | reversal | same-sign turn |
|---|---|---|
| 0.03 | 48% (668) / 46% (235) | 57% (230) / 62% (87) |
| 0.05 | 61% (268) / 62% (102) | 63% (60) / 69% (32) |
| 0.08 | 77% (103) / 77% (47) | 85% (13) / 86% (7) |

**Persistence (`persistence.py` → `persistence.json`).** The mean turn rate
over the hidden frames kept the sign of the pre-gap turn rate in 36-43% of
windows (8% for the 13 sharpest same-sign turns). The median ratio of hidden to
pre-gap turn rate was negative (-0.09 to -0.30).

**Reading.** CTRV coasts better than the linear filter on straight motion. Once
the apparent turn rate before a gap is large, it coasts worse, whether or not
the turn rate reversed: on real footage, apparent turn rates (short swerves or
box jitter) rarely persist across 5 frames. The synthetic sign-reversal case is
one instance of this broader failure.

**Figure.** `figure.py` → Fig. 2c-d of the IASC manuscript. Panel (c) is the
window with the median CTRV-minus-linear error among the 103 reversal windows
at TURN 0.08 (track 306, last visible pre-label frame 560).

**Caveats.**
- Pre-label tracks may contain identity errors.
- The occlusion is simulated, not observed.
- Turn rates are estimated from detector boxes.
