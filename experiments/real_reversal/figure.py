"""Fig. 2c-d: CTRV's turn-rate boundary on real HCMC motorcycle trajectories.

(c) A typical clear reversal: among windows classified as reversal at TURN 0.08,
    the one whose CTRV-minus-(omega=0) error is the median (chosen by rule, not by eye).
(d) Share of windows in which CTRV coasts worse than the same EKF with omega fixed
    at 0, with 95% intervals from a bootstrap over tracks, against the turn
    threshold (sensitivity.json, MIN_SPEED 2); straight motion as reference.

Same style as experiments/paper_final/figures.py: serif 8 pt, TrueType, Okabe-Ito.
Run from the repository root:
  python experiments/real_reversal/figure.py <gt.txt> <out.pdf>
"""
import json
import pathlib
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import run as R  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
MM = 1 / 25.4
plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'mathtext.fontset': 'stix', 'font.size': 8, 'axes.titlesize': 8, 'axes.labelsize': 8,
    'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 7, 'axes.linewidth': 0.6,
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02,
})
CTRV, LIN, REAPPEAR = '#0072B2', '#D55E00', '#009E73'


def pick_example(gt):
    R.TURN, R.MIN_SPEED = 0.08, 2.0
    cands = []
    for tid, frames, boxes in R.runs(gt):
        c = R.centres(boxes)
        for t0 in range(max(R.WARM, 7), len(frames) - R.HIDE):
            speed = np.linalg.norm(np.diff(c[t0 - R.WARM:t0 + 1], axis=0), axis=1).mean()
            if speed < R.MIN_SPEED:
                continue
            cls = R.classify(c, t0)
            if cls is None or cls[0] != 'reversal':
                continue
            truth, diag = c[t0 + R.HIDE], float(np.hypot(*boxes[t0 + R.HIDE][2:]))
            pc, pl = R.ctrv_predict(boxes, t0, R.HIDE), R.linear_predict(boxes, t0, R.HIDE)
            p0 = R.ctrv_predict(boxes, t0, R.HIDE, freeze_omega=True)
            d = (np.linalg.norm(pc[-1] - truth) - np.linalg.norm(p0[-1] - truth)) / diag
            cands.append((d, tid, frames, boxes, c, t0, pc, pl, p0))
    cands.sort(key=lambda x: x[0])
    return cands[len(cands) // 2], len(cands)


def main(gt_path, out):
    gt = np.loadtxt(gt_path, delimiter=',')
    (d, tid, frames, boxes, c, t0, pc, pl, p0), n = pick_example(gt)
    sens = json.loads((HERE / 'sensitivity.json').read_text())
    summ = json.loads((HERE / 'summary.json').read_text())

    fig, axes = plt.subplots(1, 2, figsize=(148 * MM, 52 * MM), gridspec_kw=dict(width_ratios=[1.05, 1]))
    ax = axes[0]
    a, b = t0 - R.WARM, t0 + R.HIDE
    ax.plot(c[a:t0 + 1, 0], c[a:t0 + 1, 1], color='0.35', lw=1.0, marker='o', ms=2.6, label='Observed (visible)')
    ax.plot(c[t0:b + 1, 0], c[t0:b + 1, 1], color='0.6', lw=0.9, ls=':', marker='o', ms=2.6,
            markerfacecolor='white', label='Hidden (true)')
    ax.plot(np.r_[c[t0:t0 + 1, 0], pl[:, 0]], np.r_[c[t0:t0 + 1, 1], pl[:, 1]], color=LIN, ls='--', marker='^',
            ms=3.0, lw=1.0, label='Linear KF coast')
    ax.plot(np.r_[c[t0:t0 + 1, 0], pc[:, 0]], np.r_[c[t0:t0 + 1, 1], pc[:, 1]], color=CTRV, marker='s',
            ms=3.0, lw=1.1, label='CTRV-EKF coast')
    ax.plot(np.r_[c[t0:t0 + 1, 0], p0[:, 0]], np.r_[c[t0:t0 + 1, 1], p0[:, 1]], color='0.15', ls='-.',
            marker='x', ms=3.0, lw=0.9, label=r'Same EKF, $\omega=0$')
    ax.scatter([c[b, 0]], [c[b, 1]], s=40, marker='*', color=REAPPEAR, zorder=5, label='Reappearance')
    ax.invert_yaxis()
    ax.set_aspect('equal', adjustable='datalim')
    ax.set_xlabel('x (px)')
    ax.set_ylabel('y (px)')
    ax.set_title(f'(c) Real track, {R.HIDE}-frame gap after a turn-rate reversal', fontsize=7.5)
    ax.legend(frameon=False, fontsize=6.0, loc='best', handlelength=1.6, labelspacing=0.2)

    ax = axes[1]
    turns = sorted({g['turn'] for g in sens})
    for kind, style, label in (('reversal', dict(color='black', marker='o', ls='-'), 'After a reversal'),
                               ('steady', dict(color='0.45', marker='s', ls='--'), 'Same-sign turn')):
        cells = [next(g for g in sens if g['turn'] == t and g['min_speed'] == 2.0)[kind] for t in turns]
        sh = np.array([cl['ctrv_worse_than_omega0'] for cl in cells]) * 100
        dx = -0.0012 if kind == 'reversal' else 0.0012
        xs = np.array(turns) + dx
        ax.errorbar(xs, sh[:, 0], yerr=[sh[:, 0] - sh[:, 1], sh[:, 2] - sh[:, 0]], ms=3.5, lw=1.0,
                    capsize=2.0, elinewidth=0.7, label=label, **style)
        for x, y, cl in zip(xs, sh[:, 0], cells):
            ax.annotate(f"{cl['windows']}/{cl['tracks']}", (x, y), textcoords='offset points',
                        xytext=(-6, 4) if kind == 'reversal' else (6, 2), ha='right' if kind == 'reversal' else 'left', fontsize=5.5, color='0.25')
    straight = summ['by_kind']['straight']['ctrv_worse_than_omega0'][0] * 100
    ax.axhline(straight, color='0.6', lw=0.8, ls=':', label='Straight motion')
    ax.axhline(50, color='0.85', lw=0.6, zorder=0)
    ax.set_xticks(turns)
    ax.set_xlim(0.022, 0.092)
    ax.set_ylim(20, 100)
    ax.set_xlabel('Turn-rate threshold (rad/frame)')
    ax.set_ylabel(r'CTRV worse than $\omega=0$ (%)')
    ax.set_title(r'(d) Effect of the turn-rate state ($n$ windows/tracks)', fontsize=7.5)
    ax.legend(frameon=False, fontsize=6.0, loc='lower right', handlelength=1.8, labelspacing=0.2)
    for s in ('top', 'right'):
        for a_ in axes:
            a_.spines[s].set_visible(False)
    fig.tight_layout(w_pad=1.5)
    fig.savefig(out)
    print(f'example: track {tid}, last visible frame {int(frames[t0])}, CTRV-minus-omega0 {d:+.3f} '
          f'diagonals (median of {n} reversal windows at TURN 0.08)')


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
