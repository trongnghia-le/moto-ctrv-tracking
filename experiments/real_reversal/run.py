"""
experiments/real_reversal/run.py
────────────────────────────────
CTRV's turn-rate sign-reversal boundary on real Ho Chi Minh City motorcycle
trajectories (IASC revision, Reviewer 2 #5).

Input: the dense-segment pre-label tracks exported from CVAT (MOT 1.1 gt.txt,
600 frames, frames 10,700-11,299 of the NKKN stream). The file is derived from
third-party video and is not redistributed; pass its path on the command line.

Real trajectories, simulated occlusion: on contiguous motorcycle track runs,
each filter (the CTRV EKF of C2 and ByteTrack's linear Kalman filter) is
started WARM frames before a candidate last-visible frame t0 and updated
through t0, then coasts HIDE frames; the error is the distance between its
prediction and the real box centre at t0 + HIDE, in box diagonals.

Each t0 is classified from the observed centres before the gap (fixed before
the run, not tuned):
  heading_t = atan2 of c_t - c_{t-3};  omega_t = wrap(heading_t - heading_{t-1})
  w_before = mean omega over t0-7..t0-4,  w_after = mean omega over t0-3..t0
  reversal : opposite signs, both |w| >= TURN
  steady   : same sign, both |w| >= TURN
  straight : both |w| <  STRAIGHT
Only windows whose mean speed over t0-WARM..t0 is >= MIN_SPEED px/frame count
(slower displacement is comparable to box jitter).

Run from the repository root:
  python experiments/real_reversal/run.py <path/to/gt.txt>
"""
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from code.models.motorcycle_motion_model import CTRVMotionModel, wrap_angle  # noqa: E402
from code.tracker.vendor.kalman_filter import KalmanFilter  # noqa: E402

MOTORCYCLE = 2          # class id in the CVAT export (labels.txt: car, motorcycle, ...)
WARM, HIDE = 12, 5      # filter warm-up frames before t0; hidden frames (dense gaps were 5)
TURN, STRAIGHT = 0.03, 0.01   # rad/frame
MIN_SPEED = 2.0         # px/frame
OUT = pathlib.Path(__file__).with_name("summary.json")


def runs(gt):
    """Contiguous per-frame runs of each motorcycle track: list of (tid, frames, boxes xywh)."""
    out = []
    for tid in np.unique(gt[:, 1]):
        r = gt[(gt[:, 1] == tid) & (gt[:, 7] == MOTORCYCLE)]
        if len(r) == 0:
            continue
        r = r[np.argsort(r[:, 0])]
        cut = np.where(np.diff(r[:, 0]) != 1)[0] + 1
        for seg in np.split(r, cut):
            out.append((int(tid), seg[:, 0].astype(int), seg[:, 2:6]))
    return out


def centres(b):
    return np.c_[b[:, 0] + b[:, 2] / 2, b[:, 1] + b[:, 3] / 2]


def xyxy(b):
    return np.array([b[0], b[1], b[0] + b[2], b[1] + b[3]])


def ctrv_predict(boxes, t0, hide):
    m = CTRVMotionModel()
    m.init(xyxy(boxes[t0 - WARM]))
    for k in range(t0 - WARM + 1, t0 + 1):
        m.predict()
        m.update(xyxy(boxes[k]))
    path = []
    for _ in range(hide):
        p = m.predict(lost=True)
        path.append([(p[0] + p[2]) / 2, (p[1] + p[3]) / 2])
    return np.array(path)


def linear_predict(boxes, t0, hide):
    kf = KalmanFilter()

    def xyah(b):
        return np.array([b[0] + b[2] / 2, b[1] + b[3] / 2, b[2] / b[3], b[3]])
    mean, cov = kf.initiate(xyah(boxes[t0 - WARM]))
    for k in range(t0 - WARM + 1, t0 + 1):
        mean, cov = kf.predict(mean, cov)
        mean, cov = kf.update(mean, cov, xyah(boxes[k]))
    path = []
    for _ in range(hide):
        mean[7] = 0.0            # ByteTrack zeroes the height velocity of lost tracks
        mean, cov = kf.predict(mean, cov)
        path.append(mean[:2].copy())
    return np.array(path)


def classify(c, t0):
    head = np.arctan2(c[3:, 1] - c[:-3, 1], c[3:, 0] - c[:-3, 0])   # head[i] is heading at frame i+3
    om = np.array([wrap_angle(a - b) for a, b in zip(head[1:], head[:-1])])  # om[j] at frame j+4
    idx = lambda f: f - 4                                            # frame -> index in om
    if idx(t0 - 7) < 0:
        return None
    wb = om[idx(t0 - 7):idx(t0 - 4) + 1].mean()
    wa = om[idx(t0 - 3):idx(t0) + 1].mean()
    if abs(wb) >= TURN and abs(wa) >= TURN:
        return ("reversal" if np.sign(wb) != np.sign(wa) else "steady"), wb, wa
    if abs(wb) < STRAIGHT and abs(wa) < STRAIGHT:
        return "straight", wb, wa
    return None


def main(gt_path):
    gt = np.loadtxt(gt_path, delimiter=",")
    events = []
    for tid, frames, boxes in runs(gt):
        c = centres(boxes)
        for t0 in range(max(WARM, 7), len(frames) - HIDE):
            speed = np.linalg.norm(np.diff(c[t0 - WARM:t0 + 1], axis=0), axis=1).mean()
            if speed < MIN_SPEED:
                continue
            cls = classify(c, t0)
            if cls is None:
                continue
            truth = c[t0 + HIDE]
            diag = float(np.hypot(*boxes[t0 + HIDE][2:]))
            e_ctrv = float(np.linalg.norm(ctrv_predict(boxes, t0, HIDE)[-1] - truth) / diag)
            e_lin = float(np.linalg.norm(linear_predict(boxes, t0, HIDE)[-1] - truth) / diag)
            events.append(dict(tid=tid, frame=int(frames[t0]), kind=cls[0], w_before=round(float(cls[1]), 4),
                               w_after=round(float(cls[2]), 4), speed=round(float(speed), 2),
                               err_ctrv=round(e_ctrv, 4), err_linear=round(e_lin, 4)))
    summary = {"params": dict(WARM=WARM, HIDE=HIDE, TURN=TURN, STRAIGHT=STRAIGHT, MIN_SPEED=MIN_SPEED),
               "motorcycle_runs": sum(1 for r in runs(gt)), "by_kind": {}}
    for kind in ("reversal", "steady", "straight"):
        ev = [e for e in events if e["kind"] == kind]
        if not ev:
            continue
        ec, el = np.array([e["err_ctrv"] for e in ev]), np.array([e["err_linear"] for e in ev])
        # one event per track (the first) as a dependence check
        first = {}
        for e in ev:
            first.setdefault(e["tid"], e)
        fc = np.array([e["err_ctrv"] for e in first.values()])
        fl = np.array([e["err_linear"] for e in first.values()])
        summary["by_kind"][kind] = dict(
            windows=len(ev), tracks=len(first),
            median_err_ctrv=round(float(np.median(ec)), 3), median_err_linear=round(float(np.median(el)), 3),
            share_ctrv_worse=round(float(np.mean(ec > el)), 3),
            per_track_median_ctrv=round(float(np.median(fc)), 3), per_track_median_linear=round(float(np.median(fl)), 3),
            per_track_share_ctrv_worse=round(float(np.mean(fc > fl)), 3))
    summary["events"] = events
    OUT.write_text(json.dumps(summary, indent=1))
    for k, v in summary["by_kind"].items():
        print(k, v)


if __name__ == "__main__":
    main(sys.argv[1])
