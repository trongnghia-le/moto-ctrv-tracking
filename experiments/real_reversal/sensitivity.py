"""Sensitivity of experiments/real_reversal to the turn threshold and minimum speed.

The primary setting (TURN 0.03, MIN_SPEED 2) is in run.py; this grid is reported
in full, whichever way it points.
Run from the repository root:  python experiments/real_reversal/sensitivity.py <gt.txt>
"""
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import run as R  # noqa: E402

OUT = pathlib.Path(__file__).with_name("sensitivity.json")


def main(gt_path):
    gt = np.loadtxt(gt_path, delimiter=",")
    tracks = R.runs(gt)
    grid = []
    for turn in (0.03, 0.05, 0.08):
        for min_speed in (2.0, 4.0):
            R.TURN, R.MIN_SPEED = turn, min_speed
            rows = {"reversal": [], "steady": []}
            for tid, frames, boxes in tracks:
                c = R.centres(boxes)
                for t0 in range(max(R.WARM, 7), len(frames) - R.HIDE):
                    speed = np.linalg.norm(np.diff(c[t0 - R.WARM:t0 + 1], axis=0), axis=1).mean()
                    if speed < min_speed:
                        continue
                    cls = R.classify(c, t0)
                    if cls is None or cls[0] not in rows:
                        continue
                    truth = c[t0 + R.HIDE]
                    diag = float(np.hypot(*boxes[t0 + R.HIDE][2:]))
                    ec = np.linalg.norm(R.ctrv_predict(boxes, t0, R.HIDE)[-1] - truth) / diag
                    el = np.linalg.norm(R.linear_predict(boxes, t0, R.HIDE)[-1] - truth) / diag
                    rows[cls[0]].append((tid, ec, el))
            cell = dict(turn=turn, min_speed=min_speed)
            for kind, r in rows.items():
                if not r:
                    cell[kind] = None
                    continue
                a = np.array([[x[1], x[2]] for x in r])
                cell[kind] = dict(windows=len(r), tracks=len({x[0] for x in r}),
                                  median_ctrv=round(float(np.median(a[:, 0])), 3),
                                  median_linear=round(float(np.median(a[:, 1])), 3),
                                  share_ctrv_worse=round(float(np.mean(a[:, 0] > a[:, 1])), 3))
            grid.append(cell)
            print(cell)
    OUT.write_text(json.dumps(grid, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
