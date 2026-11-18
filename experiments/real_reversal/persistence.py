"""Does the turn rate seen before a gap persist through it on real trajectories?

For every window that run.py classifies as 'steady' or 'reversal', compare the
mean turn rate over the last visible frames (w_after, t0-3..t0) with the mean
turn rate over the hidden frames (t0+1..t0+HIDE), from the real centres.
Run from the repository root:  python experiments/real_reversal/persistence.py <gt.txt>
"""
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import run as R  # noqa: E402

OUT = pathlib.Path(__file__).with_name("persistence.json")


def omega_series(c):
    head = np.arctan2(c[3:, 1] - c[:-3, 1], c[3:, 0] - c[:-3, 0])
    return np.array([R.wrap_angle(a - b) for a, b in zip(head[1:], head[:-1])])  # index j <-> frame j+4


def main(gt_path):
    gt = np.loadtxt(gt_path, delimiter=",")
    tracks = R.runs(gt)
    res = []
    for turn in (0.03, 0.05, 0.08):
        R.TURN = turn
        for kind in ("steady", "reversal"):
            same, ratio = [], []
            for tid, frames, boxes in tracks:
                c = R.centres(boxes)
                om = omega_series(c)
                for t0 in range(max(R.WARM, 7), len(frames) - R.HIDE):
                    speed = np.linalg.norm(np.diff(c[t0 - R.WARM:t0 + 1], axis=0), axis=1).mean()
                    if speed < R.MIN_SPEED:
                        continue
                    cls = R.classify(c, t0)
                    if cls is None or cls[0] != kind:
                        continue
                    w_gap = om[t0 + 1 - 4:t0 + R.HIDE - 4 + 1].mean()
                    same.append(np.sign(w_gap) == np.sign(cls[2]))
                    ratio.append(w_gap / cls[2])
            cell = dict(turn=turn, kind=kind, windows=len(same),
                        share_same_sign_in_gap=round(float(np.mean(same)), 3) if same else None,
                        median_gap_over_before=round(float(np.median(ratio)), 3) if ratio else None)
            res.append(cell)
            print(cell)
    OUT.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
