# External validation on the public FastTracker Benchmark

**Purpose.** Address Reviewer 2 Point 3 (external validity of the seed-bias
brackets and tracking conclusions on an independent public multi-class urban
CCTV dataset) and Reviewer 1 Point 4 (comparison on external sequences).

**Dataset.** The public FastTracker Benchmark (Hashempoor & Hwang, 2025;
12 urban CCTV sequences, 19,480 frames total at 15–30 fps, 760,739 ground-truth
bounding boxes across 15 raw labels, independently annotated without seeding
from our trackers). Cached detections from `YOLO11n` (640 px) and `YOLO11x`
(1,280 px) in `seed-bias-mot/data/processed/dets/fasttrack/`.

## 1. Two-wheeler scarcity in public urban CCTV benchmarks

Across all 12 sequences (`760,739` total boxes):
- `car`: 545,151 boxes (907 identities, 71.7% of all annotations)
- `person`: 141,546 boxes (216 identities)
- `motorbike` (class 8): **4,315 boxes (11 identities across 5 sequences, 0.57%
  of annotations; 7 sequences contain zero motorbikes)**
- `bike` (class 7): 11,209 boxes (10 identities; combined two-wheelers: 15,524
  boxes, 21 identities across 7 sequences, 2.04% of annotations)

This quantifies the benchmark gap noted in Section 2.2: public fixed-camera
multi-class traffic benchmarks contain under 1% motorcycle boxes and do not
exhibit dense motorcycle platoons.

## 2. Tracking comparison on identical detections (`YOLO11x` and `YOLO11n`)

Weighted by ground-truth box count across the sequences containing the evaluated
class (`motorbike` across 5 sequences, 4,315 boxes; all two-wheelers
`motorbike + bike` across 7 sequences, 15,524 boxes):

| Configuration | Motorbike MOTA | Motorbike IDF1 | Motorbike HOTA$_{0.5}$ | Motorbike IDSW | Two-wheeler MOTA | Two-wheeler IDF1 | Two-wheeler HOTA$_{0.5}$ | Two-wheeler IDSW |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Exp0: ByteTrack (baseline) | 0.178 | 0.286 | 0.189 | 3 | 0.172 | 0.325 | 0.342 | 18 |
| + CTRV | 0.182 | 0.293 | 0.205 | 3 | 0.173 | 0.327 | 0.346 | 18 |
| + CTRV, NSA | 0.182 | 0.294 | 0.205 | 3 | 0.174 | 0.328 | 0.347 | 18 |
| + CTRV, gate | 0.182 | 0.293 | 0.205 | 3 | 0.173 | 0.337 | 0.353 | 17 |
| Exp1: C2 (CTRV, gate, NSA) | 0.182 | 0.294 | 0.205 | 3 | 0.175 | 0.338 | 0.354 | 17 |
| OC-SORT, published | 0.050 | 0.058 | 0.042 | 5 | 0.213 | 0.241 | 0.264 | 22 |
| OC-SORT, matched 0.5 | **0.230** | 0.309 | **0.234** | 4 | 0.009 | 0.338 | **0.369** | 18 |
| OC-SORT, matched 0.6 | 0.191 | **0.312** | 0.227 | 2 | 0.088 | **0.345** | 0.368 | 15 |
| BoT-SORT, published | 0.051 | 0.093 | 0.064 | **1** | **0.216** | 0.304 | 0.320 | **7** |
| BoT-SORT, matched | 0.182 | 0.290 | 0.194 | 5 | 0.175 | 0.337 | 0.351 | 20 |

With `YOLO11n` at 640 px on the 1920×1080 views, 4,311 of 4,315 motorbike boxes
(99.9%) and 14,629 of 15,524 two-wheeler boxes (94.2%) are missed across all
trackers (`IDF1 <= 0.010` on motorbikes, `<= 0.068` on two-wheelers),
reproducing Section 6.2's finding that small-input real-time detection is
bounded by small-object recall.

## 3. External validation of seed bias and union bracketing

Because FastTracker Benchmark has an independent human-annotated ground truth
$G_{\mathrm{true}}$, we can directly test the box-membership seed-bias mechanism
of Section 5.3 by comparing $\Delta\mathrm{IDF1}$ on $G_{\mathrm{true}}$ against
single-seed ground truths $G_{\mathrm{C2}}$ and $G_{\mathrm{BT}}$ (which omit
true boxes missed by the seeding tracker but found by a non-seed tracker):

- **C2 vs ByteTrack (`ctrv_gate_nsa - baseline`):**
  - Motorbikes: $\Delta\mathrm{IDF1}(G_{\mathrm{true}}) = +0.008$, whereas
    $\Delta\mathrm{IDF1}(G_{\mathrm{C2}}) = +0.012$ and
    $\Delta\mathrm{IDF1}(G_{\mathrm{BT}}) = -0.005$ (flipping sign with the
    seeding tracker, as on our HCMC segments).
  - Two-wheelers: $\Delta\mathrm{IDF1}(G_{\mathrm{true}}) = +0.013$, against
    $+0.014$ on $G_{\mathrm{C2}}$ and $+0.008$ on $G_{\mathrm{BT}}$; the
    Section 5.3 bracket from $G_{\mathrm{C2}}$ is $[+0.013, +0.014]$
    ($0.80$-interpolated: $+0.013$).
- **OC-SORT (0.5) vs C2 (`ocsort_matched - ctrv_gate_nsa`):**
  - Motorbikes: on $G_{\mathrm{true}}$, OC-SORT leads C2 by $+0.016$ IDF1, but
    under $G_{\mathrm{C2}}$ the ranking reverses to $-0.021$ (reproducing the
    raw dense ranking reversal at 960–1,280 px in Section 5.3). Extending
    $G_{\mathrm{C2}}$ with OC-SORT's unmatched boxes yields the upper bound
    $+0.088$, so the bracket $[-0.021, +0.088]$ encloses the true unseeded
    difference $+0.016$ and restores the true sign.
