"""Average the per-piece jsonl results (mean over pieces, as in the paper) and print them next to the paper's numbers."""
import json, sys, numpy as np

PAPER = {  # Table I / VI, proposed system, P / R / F1 in %
    "frame": (88.71, 90.73, 89.62), "note": (98.17, 95.35, 96.72), "note_offset": (83.68, 81.32, 82.47),
    "note_offset_velocity": (82.10, 79.80, 80.92), "pedal_frame": (94.30, 94.42, 94.25), "pedal": (91.59, 92.41, 91.86),
    "pedal_offset": (86.36, 87.02, 86.58), "pedal_50ms": None, "pedal_offset_50ms": None}
PAPER_ONSET_TOL = {"0.1": 96.79, "0.05": 96.72, "0.02": 95.79, "0.01": 88.91, "0.005": 65.32, "0.002": 31.08}  # Table IV F1
PAPER_OFFSET_TOL = {"0.5": 91.88, "0.2": 86.90, "0.1": 83.25, "0.05": 79.06, "0.02": 68.94, "0.01": 51.28}  # Table V F1

rows = [json.loads(l) for l in open(sys.argv[1] if len(sys.argv) > 1 else "results/eval_pretrained_test.jsonl")]
print(f"{len(rows)} pieces, {sum(r['duration'] for r in rows) / 3600:.1f} h, {sum(r['n_ref_notes'] for r in rows)} reference notes\n")
print(f"{'metric':24s} {'reproduced P / R / F1':>26s}   {'paper P / R / F1':>22s}   {'dF1':>6s}")
for key, paper in PAPER.items():
    have = [r[key] for r in rows if key in r]
    if not have:
        continue
    p, r_, f = np.mean(have, axis=0) * 100
    ref = f"{paper[0]:.2f} / {paper[1]:.2f} / {paper[2]:.2f}" if paper else "n/a"
    d = f"{f - paper[2]:+.2f}" if paper else ""
    print(f"{key:24s} {p:8.2f} / {r_:5.2f} / {f:5.2f}   {ref:>22s}   {d:>6s}")
for title, sweep, paper in (("onset tolerance (note F1)", "onset_tolerance_sweep", PAPER_ONSET_TOL),
                            ("offset tolerance (note w/ offset F1, onset tol 50 ms)", "offset_tolerance_sweep", PAPER_OFFSET_TOL)):
    print(f"\n{title}")
    for tol, pf in paper.items():
        p, r_, f = np.mean([r[sweep][tol] for r in rows], axis=0) * 100
        print(f"  {float(tol) * 1000:5.0f} ms  reproduced {p:6.2f} / {r_:6.2f} / {f:6.2f}   paper F1 {pf:6.2f}   dF1 {f - pf:+.2f}")
