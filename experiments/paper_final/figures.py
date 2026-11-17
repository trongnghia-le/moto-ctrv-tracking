"""Figures for the JRTIP manuscript, drawn only from committed result files.

    python experiments/paper_final/figures.py  ->  paper/jrtip/figures/*.pdf

Springer svjour3 sizes: one column 84 mm, two columns 174 mm. Times-like serif at
8 pt to match the body, vector PDF with TrueType fonts embedded, Okabe-Ito colours
plus distinct markers so that the figures also read in greyscale.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'paper/jrtip/figures'
SIZES = (448, 640, 960, 1280)
SEGMENTS = ('dense', 'mixed', 'freeflow')
SEG_LABEL = {'dense': 'Dense', 'mixed': 'Mixed', 'freeflow': 'Free-flow'}
SEG_STYLE = {'dense': ('#0072B2', 'o'), 'mixed': ('#D55E00', 's'), 'freeflow': ('#009E73', '^')}
ARM_STYLE = {'baseline': ('ByteTrack', '#000000', 'o', '-'),
             'ctrv_gate_nsa': ('C2', '#0072B2', 's', '-'),
             'ocsort_matched': ('OC-SORT (0.5)', '#D55E00', '^', '--')}
MM = 1 / 25.4
plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'mathtext.fontset': 'stix', 'font.size': 8, 'axes.titlesize': 8, 'axes.labelsize': 8,
    'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 7, 'axes.linewidth': 0.6,
    'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'lines.linewidth': 1.0, 'lines.markersize': 4,
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02,
    'axes.spines.top': False, 'axes.spines.right': False})


def load(path):
    return json.loads((ROOT / path).read_text(encoding='utf-8'))


def bracket(size):
    return load(f'experiments/seed_bias/bracket_best85_roi_final_{size}.json')['segments']


def corrected(b, seg, pair):
    """Seed-corrected IDF1 gap: (mid, low, high) over the audit-share interval."""
    mid, a, c = b[seg]['comparisons'][pair]['gap']['idf1']['corrected']
    return mid, min(a, c), max(a, c)


def size_axis(ax):
    ax.set_xscale('log')
    ax.set_xticks(SIZES)
    ax.set_xticklabels([str(s) for s in SIZES])
    ax.minorticks_off()
    ax.set_xlabel('Input size (px)')


def fig_sizes():
    """Dense ID switches, and the seed-corrected IDF1 gaps C2-ByteTrack and OC-SORT-C2."""
    fig, axes = plt.subplots(1, 3, figsize=(174 * MM, 48 * MM))
    ax = axes[0]
    for arm, (label, colour, marker, style) in ARM_STYLE.items():
        idsw = [load(f'experiments/paper_final/best85_roi_final/{s}/metrics.json')['segments']['dense']['metrics'][arm]
                ['motorcycle']['num_switches'] for s in SIZES]
        ax.plot(SIZES, idsw, color=colour, marker=marker, linestyle=style, label=label)
    size_axis(ax)
    ax.set_ylabel('Identity switches')
    ax.set_title('(a) Dense, motorcycle identity switches', loc='left')
    ax.set_ylim(0, 165)
    ax.legend(frameon=False, loc='lower right', bbox_to_anchor=(1.02, 0.0), fontsize=6.5,
              handlelength=1.6, labelspacing=0.25, borderaxespad=0.2)
    for ax, pair, title in ((axes[1], 'ctrv_gate_nsa - baseline', '(b) C2 $-$ ByteTrack, IDF1'),
                            (axes[2], 'ocsort_matched - ctrv_gate_nsa', '(c) OC-SORT (0.5) $-$ C2, IDF1')):
        for k, seg in enumerate(SEGMENTS):
            colour, marker = SEG_STYLE[seg]
            g = [corrected(bracket(s), seg, pair) for s in SIZES]
            x = [s * (1.06 ** (k - 1)) for s in SIZES]
            ax.errorbar(x, [m for m, _, _ in g], yerr=[[m - lo for m, lo, _ in g], [hi - m for m, _, hi in g]],
                        color=colour, marker=marker, markersize=3.8, markeredgecolor='white', markeredgewidth=0.4,
                        linestyle='-', linewidth=0.9, capsize=2.0, capthick=0.7, elinewidth=0.7,
                        label=SEG_LABEL[seg], zorder=2 + k)
        ax.axhline(0, color='0.5', linewidth=0.6, zorder=0)
        size_axis(ax)
        ax.set_xlim(390, 1460)
        ax.set_ylabel('Seed-corrected $\\Delta$IDF1')
        ax.set_title(title, loc='left')
    axes[1].legend(frameon=False, loc='upper left')
    fig.tight_layout(w_pad=1.2)
    fig.savefig(OUT / 'fig_input_size.pdf')


def fig_seed_bias(size=448):
    """Lower (GT as is), upper (GT plus the non-seed trackers' unmatched boxes) and
    corrected IDF1 gaps at the Table 2 size: how far the seed moves each comparison."""
    b = bracket(size)
    groups = (('ocsort_matched - ctrv_gate_nsa', 'OC-SORT (0.5) $-$ C2'), ('ctrv_gate_nsa - baseline', 'C2 $-$ ByteTrack'))
    fig, ax = plt.subplots(figsize=(88 * MM, 52 * MM))
    ticks, labels, y = [], [], 0
    for pair, name in groups:
        for seg in reversed(SEGMENTS):
            g = b[seg]['comparisons'][pair]['gap']['idf1']
            lo, up = g['lower'], g['upper']
            mid, a, c = corrected(b, seg, pair)
            first = not ticks
            yb, yd = y - 0.15, y + 0.15
            ax.plot([lo, up], [yb, yb], color='0.65', linewidth=0.9, zorder=1)
            ax.plot(lo, yb, marker='o', markersize=3.6, markerfacecolor='white', markeredgecolor='black',
                    markeredgewidth=0.8, linestyle='none', zorder=2, label='GT as annotated' if first else None)
            ax.plot(up, yb, marker='s', markersize=3.6, markerfacecolor='white', markeredgecolor='#D55E00',
                    markeredgewidth=0.8, linestyle='none', zorder=2, label='GT + non-seed boxes' if first else None)
            ax.errorbar(mid, yd, xerr=[[mid - a], [c - mid]], marker='D', color='#0072B2', markersize=2.8,
                        markeredgecolor='white', markeredgewidth=0.25, capsize=2.2, capthick=0.8,
                        elinewidth=0.9, linestyle='none', zorder=4, label='Seed-corrected (95% CI)' if first else None)
            ticks.append(y); labels.append(SEG_LABEL[seg]); y += 1
        ax.text(-0.004, y - 0.30, name, fontsize=7, fontstyle='italic', ha='left', va='bottom')
        y += 1
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels)
    ax.set_ylim(-0.6, y - 0.25)
    ax.set_xlim(-0.005, None)
    ax.axvline(0, color='0.5', linewidth=0.6, zorder=0)
    ax.set_xlabel(f'IDF1 difference at {size} px')
    ax.legend(frameon=False, loc='lower center', bbox_to_anchor=(0.45, 1.0), ncol=3, handletextpad=0.25,
              columnspacing=0.6, fontsize=6.5)
    fig.savefig(OUT / 'fig_seed_bias.pdf')


def fig_operating():
    """Accuracy cost of the operating point (replaces the old Table 4): (a) C2's motorcycle
    IDF1 per segment and (b) ByteTrack's dense miss rate by ground-truth box height, over
    the real-time detector's input sizes; hollow markers at 1,280 px: reference detector."""
    ops = load('experiments/operating_point/operating_points.json')['operating_points']
    ref = ops['yolo11_vn_1280_roi']
    fig, axes = plt.subplots(1, 2, figsize=(84 * MM, 44 * MM))
    ax = axes[0]
    for seg in SEGMENTS:
        colour, marker = SEG_STYLE[seg]
        ax.plot(SIZES, [ops[f'best_{s}'][seg]['ctrv_gate_nsa']['motorcycle']['idf1'] for s in SIZES],
                color=colour, marker=marker, label=SEG_LABEL[seg])
        ax.plot(1280 * 1.06, ref[seg]['ctrv_gate_nsa']['motorcycle']['idf1'], marker=marker, linestyle='none',
                markerfacecolor='white', markeredgecolor=colour)
    size_axis(ax)
    ax.set_ylabel('C2 motorcycle IDF1')
    ax.set_ylim(0.45, 1.0)
    ax.set_title('(a)', loc='left')
    ax.legend(frameon=False, loc='upper left', handlelength=1.2)
    ax = axes[1]
    bins = (('0-20', 'height $<$20 px', 'o', '#000000', (800, 0.83)),
            ('20-40', '20$-$40 px', 's', '#0072B2', (760, 0.31)),
            ('40-10000', '$\geq$40 px', '^', '#D55E00', (760, 0.03)))
    for key, label, marker, colour, (tx, ty) in bins:
        ax.plot(SIZES, [ops[f'best_{s}']['dense']['baseline']['motorcycle']['miss_by_height'][key]['rate']
                        for s in SIZES], marker=marker, color=colour)
        ax.plot(1280 * 1.06, ref['dense']['baseline']['motorcycle']['miss_by_height'][key]['rate'], marker=marker,
                linestyle='none', markerfacecolor='white', markeredgecolor=colour)
        ax.text(tx, ty, label, color=colour, fontsize=6.5, ha='center')
    size_axis(ax)
    ax.set_ylim(0, 1.03)
    ax.set_ylabel('Dense miss rate')
    ax.set_title('(b)', loc='left')
    for a in axes:
        a.set_xlabel('Input size (px)', fontsize=7)
    fig.tight_layout(w_pad=0.8)
    fig.savefig(OUT / 'fig_operating.pdf')


def fig_ctrv_boundary():
    """Physical boundary of CTRV across an occlusion (Sect. 5.1):
    (a) Constant turn with a 12-frame occlusion: CTRV follows the arc and re-associates,
        whereas the linear KF coasts along the pre-occlusion tangent.
    (b) S-curve weaving with an occlusion starting immediately after turn-rate sign reversal
        (omega > 0 -> omega < 0): because omega requires second differences of position
        (Proposition 1), CTRV extrapolates the outdated turn rate in the wrong direction."""
    import numpy as np
    import sys
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from code.models.motorcycle_motion_model import CTRVMotionModel
    from code.tracker.vendor.kalman_filter import KalmanFilter
    from code.utils.synthetic import constant_turn, weaving, centers_to_bboxes

    wh = (40.0, 60.0)
    fig, axes = plt.subplots(1, 2, figsize=(148 * MM, 50 * MM))

    # Panel (a): Constant turn, 12-frame occlusion [80, 92)
    gt_a = constant_turn(v=6.0, omega=0.04, n=102, noise=0.0, start=(200.0, 200.0))
    obs_a = gt_a + np.random.default_rng(9).normal(0, 2.0, size=gt_a.shape)
    gap_a = set(range(80, 92))
    boxes_a = centers_to_bboxes(obs_a, wh)
    ctrv_a, kf_a = CTRVMotionModel(), KalmanFilter()
    ctrv_a.init(boxes_a[0])
    m_a, c_a = kf_a.initiate(np.array([obs_a[0, 0], obs_a[0, 1], wh[0] / wh[1], wh[1]]))
    c_arc, l_tan = [], []
    for k in range(1, 93):
        pb = ctrv_a.predict(lost=(k in gap_a))
        m_a, c_a = kf_a.predict(m_a, c_a)
        if k in gap_a or k == 79:
            c_arc.append([(pb[0] + pb[2]) / 2, (pb[1] + pb[3]) / 2])
            l_tan.append(m_a[:2].copy())
        if k not in gap_a:
            ctrv_a.update(boxes_a[k])
            m_a, c_a = kf_a.update(m_a, c_a, np.array([obs_a[k, 0], obs_a[k, 1], wh[0] / wh[1], wh[1]]))
    c_arc, l_tan = np.asarray(c_arc), np.asarray(l_tan)
    ax = axes[0]
    w0, w1 = 62, 98
    ax.plot(gt_a[w0:w1, 0], gt_a[w0:w1, 1], color='0.55', lw=1.0, label='True path')
    vis = [k for k in range(w0, w1) if k not in gap_a]
    ax.scatter(obs_a[vis, 0], obs_a[vis, 1], s=8, color='0.35', zorder=2, label='Detections')
    ax.plot(l_tan[:, 0], l_tan[:, 1], color='#D55E00', ls='--', marker='^', ms=3.0, markevery=3, lw=1.0, label='Linear KF coast')
    ax.plot(c_arc[:, 0], c_arc[:, 1], color='#0072B2', ls='-', marker='s', ms=3.0, markevery=3, lw=1.1, label='CTRV-EKF coast')
    ax.scatter([gt_a[92, 0]], [gt_a[92, 1]], s=36, marker='*', color='#009E73', zorder=5, label='Reappearance')
    ax.annotate('Occlusion starts ($t=80$)', xy=(gt_a[79, 0], gt_a[79, 1]), xytext=(12, -22),
                textcoords='offset points', fontsize=6.5, arrowprops=dict(arrowstyle='->', lw=0.5, color='0.3'))
    ax.set_title('(a) Constant turn ($\\omega>0$ steady)', loc='left')
    ax.set_xlabel('$x$ (px)'); ax.set_ylabel('$y$ (px)')
    ax.set_ylim(452, 535)
    ax.invert_yaxis()

    # Panel (b): S-curve weaving, occlusion right after omega changes sign (peak of heading sinusoid at k=15)
    # In weaving(period=60), heading(k) = amp*sin(2*pi*k/60), so omega(k) = d(heading)/dk ~ cos(2*pi*k/60),
    # which changes sign at k = 15 (and k = 45, 75). Start 10-frame occlusion at k = 46 (1 frame after sign flip).
    gt_b = weaving(v=6.0, amp=0.65, period=60.0, n=70, noise=0.0, start=(60.0, 260.0))
    obs_b = gt_b + np.random.default_rng(7).normal(0, 1.5, size=gt_b.shape)
    gap_b = set(range(46, 56))
    boxes_b = centers_to_bboxes(obs_b, wh)
    ctrv_b, kf_b = CTRVMotionModel(), KalmanFilter()
    ctrv_b.init(boxes_b[0])
    m_b, c_b = kf_b.initiate(np.array([obs_b[0, 0], obs_b[0, 1], wh[0] / wh[1], wh[1]]))
    c_arc_b, l_tan_b = [], []
    for k in range(1, 57):
        pb = ctrv_b.predict(lost=(k in gap_b))
        m_b, c_b = kf_b.predict(m_b, c_b)
        if k in gap_b or k == 45:
            c_arc_b.append([(pb[0] + pb[2]) / 2, (pb[1] + pb[3]) / 2])
            l_tan_b.append(m_b[:2].copy())
        if k not in gap_b:
            ctrv_b.update(boxes_b[k])
            m_b, c_b = kf_b.update(m_b, c_b, np.array([obs_b[k, 0], obs_b[k, 1], wh[0] / wh[1], wh[1]]))
    c_arc_b, l_tan_b = np.asarray(c_arc_b), np.asarray(l_tan_b)
    ax = axes[1]
    w0, w1 = 28, 62
    ax.plot(gt_b[w0:w1, 0], gt_b[w0:w1, 1], color='0.55', lw=1.0, label='True path')
    vis_b = [k for k in range(w0, w1) if k not in gap_b]
    ax.scatter(obs_b[vis_b, 0], obs_b[vis_b, 1], s=8, color='0.35', zorder=2, label='Detections')
    ax.plot(l_tan_b[:, 0], l_tan_b[:, 1], color='#D55E00', ls='--', marker='^', ms=3.0, markevery=3, lw=1.0, label='Linear KF coast')
    ax.plot(c_arc_b[:, 0], c_arc_b[:, 1], color='#0072B2', ls='-', marker='s', ms=3.0, markevery=3, lw=1.1, label='CTRV-EKF (outdated $\\hat\\omega$)')
    ax.scatter([gt_b[56, 0]], [gt_b[56, 1]], s=36, marker='*', color='#009E73', zorder=5, label='Reappearance')
    ax.annotate('$\\omega$ flips sign, then\nocclusion ($t=46$)', xy=(gt_b[45, 0], gt_b[45, 1]), xytext=(-72, 22),
                textcoords='offset points', fontsize=6.5, arrowprops=dict(arrowstyle='->', lw=0.5, color='0.3'))
    ax.set_title('(b) Weaving inflection ($\\omega$ changes sign before gap)', loc='left')
    ax.set_xlabel('$x$ (px)'); ax.set_ylabel('$y$ (px)')
    ax.set_ylim(242, 338)
    ax.invert_yaxis()
    axes[0].legend(frameon=False, loc='upper center', bbox_to_anchor=(0.48, 0.99), fontsize=6.0, handlelength=1.4, labelspacing=0.2)
    fig.tight_layout(w_pad=1.2)
    fig.savefig(OUT / 'fig_ctrv_boundary.pdf')


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    fig_sizes()
    fig_seed_bias()
    fig_operating()
    fig_ctrv_boundary()
    print('written:', sorted(p.name for p in OUT.glob('fig_*.pdf')))
