"""Sensitivity of experiments/real_reversal to the turn threshold and minimum speed.

The primary setting (TURN 0.03, MIN_SPEED 2) is in run.py; this grid is reported
in full, whichever way it points. Shares are [point, 2.5%, 97.5%] with a
bootstrap over tracks.
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
    grid = []
    for turn in (0.03, 0.05, 0.08):
        for min_speed in (2.0, 4.0):
            R.TURN, R.MIN_SPEED = turn, min_speed
            cell = dict(turn=turn, min_speed=min_speed, **R.stats(R.evaluate(gt), kinds=("reversal", "steady")))
            grid.append(cell)
            print(json.dumps(cell))
    OUT.write_text(json.dumps(grid, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
