"""
experiments/initial_conditions/run.py
─────────────────────────────────────
Large-initial-condition test of the CTRV EKF (IASC revision, Reviewer 1 #7).

A new track starts from the zero prior [cx, cy, 0, 0, 0], so the true speed
and heading of the motorcycle are an initial-condition error of up to
v0 px/frame and +-pi rad. This replays synthetic tracks over a grid of true
speeds, headings and turn rates, with and without measurement noise, through
the unmodified CTRVMotionModel (default Q/R, two-detection bootstrap) and
records, after k visible detections, the absolute heading error, the
relative speed error and whether P stays finite, symmetric and positive
definite.

Run from the repository root:  python experiments/initial_conditions/run.py
"""
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from code.models.motorcycle_motion_model import CTRVMotionModel, wrap_angle  # noqa: E402
from code.utils.synthetic import centers_to_bboxes, constant_turn  # noqa: E402

SPEEDS = [1.0, 2.0, 5.0, 10.0, 15.0, 20.0, 25.0]   # px/frame
HEADINGS = np.linspace(-np.pi, np.pi, 24, endpoint=False)
OMEGAS = [-0.08, 0.0, 0.08]                        # rad/frame
NOISES = [0.0, 2.0]                                # px, centre jitter
SEEDS = range(10)                                  # noisy runs only
K_MAX = 10                                         # visible detections
WH = np.array([30.0, 40.0])
OUT = pathlib.Path(__file__).with_name("results.json")


def run_one(v0, th0, om, noise, seed):
    centers = constant_turn(v=v0, omega=om, n=K_MAX, noise=noise, seed=seed, heading0=th0)
    boxes = centers_to_bboxes(centers, WH)
    m = CTRVMotionModel()
    m.init(boxes[0])
    head, speed, ok = [], [], True
    for k in range(1, K_MAX):
        m.predict()
        m.update(boxes[k])
        # true direction of motion leaving frame k (generator turns after each step)
        true_th = th0 + k * om
        head.append(abs(wrap_angle(m.x[3] - true_th)))
        speed.append(abs(m.x[2] - v0) / v0)
        P = m.P
        ok &= bool(np.all(np.isfinite(P)) and np.allclose(P, P.T, atol=1e-6)
                   and np.linalg.eigvalsh((P + P.T) / 2).min() > 0)
    return head, speed, ok   # index i = after i+2 visible detections


def summarize(rows):
    h = np.array([r[0] for r in rows])   # (runs, K_MAX-1)
    s = np.array([r[1] for r in rows])
    out = {}
    for i in range(h.shape[1]):
        k = i + 2
        out[k] = dict(
            heading_err_rad=dict(median=round(float(np.median(h[:, i])), 4),
                                 p95=round(float(np.percentile(h[:, i], 95)), 4),
                                 max=round(float(h[:, i].max()), 4)),
            speed_rel_err=dict(median=round(float(np.median(s[:, i])), 4),
                               p95=round(float(np.percentile(s[:, i], 95)), 4)))
    return out


def main():
    results = {"grid": dict(speeds_px_per_frame=SPEEDS, headings=len(HEADINGS),
                            omegas_rad_per_frame=OMEGAS, noise_px=NOISES,
                            seeds_noisy=len(SEEDS), k_max=K_MAX,
                            filter="CTRVMotionModel defaults (Q/R/P0 as in code), bootstrap on")}
    for noise in NOISES:
        seeds = [0] if noise == 0 else list(SEEDS)
        for v0 in SPEEDS:
            rows = [run_one(v0, th, om, noise, sd) for th in HEADINGS for om in OMEGAS for sd in seeds]
            results[f"noise{noise:g}_v{v0:g}"] = dict(
                runs=len(rows), p_always_pd=all(r[2] for r in rows), by_k=summarize(rows))
    OUT.write_text(json.dumps(results, indent=1))
    for key, r in results.items():
        if key == "grid":
            continue
        k4 = r["by_k"][4]
        print(f"{key:14s} runs={r['runs']:4d} pd={r['p_always_pd']} "
              f"k4 heading med/p95/max={k4['heading_err_rad']['median']}/"
              f"{k4['heading_err_rad']['p95']}/{k4['heading_err_rad']['max']} "
              f"speed med/p95={k4['speed_rel_err']['median']}/{k4['speed_rel_err']['p95']}")


if __name__ == "__main__":
    main()
