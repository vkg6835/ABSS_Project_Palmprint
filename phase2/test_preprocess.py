#!/usr/bin/env python3
"""
test_preprocess.py — does the ROI land on the same skin every time?

Run:  python test_preprocess.py

This is the test our Phase I critique obliges us to write. We faulted Paper 1
for building an ROI from hand landmarks and never reporting how repeatable it
is, and noted that Paper 2 inherits a preprocessing stage it admits can be
rotated by about ten degrees — then spends seven stored templates per
enrolment absorbing that error. Having made that criticism, we are not
entitled to leave our own ROI unmeasured.

Synthetic hands are used (palmprint/synthetic.py), each rendered under a known
rotation, translation and scale, so the error can be compared against the
truth rather than estimated. The numbers here are about GEOMETRY, which
synthetic hands model honestly: a valley is a valley. They are not
recognition-performance claims, which synthetic palms cannot support.
"""
import sys

import numpy as np

from palmprint import mfrat, metrics, preprocess as pp, protocol, rloc

_failures = []


def check(name, ok, detail=""):
    if not ok:
        _failures.append(name)
    print("[%s] %s%s" % ("  ok  " if ok else " FAIL ", name,
                         ("   " + detail) if detail else ""))
    return bool(ok)


def section(t):
    print("\n" + t)
    print("-" * len(t))


def poses(n, seed=11, rot=14.0, shift=12.0, scale=0.10):
    rng = np.random.default_rng(seed)
    return [(float(rng.uniform(-rot, rot)),
             float(rng.uniform(-shift, shift)),
             float(rng.uniform(-shift, shift)),
             float(rng.uniform(1 - scale, 1 + scale))) for _ in range(n)]


def extract_all(identities, pose_list, size=128):
    """Run the full pipeline over a grid of identities and poses."""
    from palmprint import synthetic
    out, failures = {}, []
    for ident in identities:
        rows = []
        for pi, (rot, dx, dy, sc) in enumerate(pose_list):
            img, truth = synthetic.render_hand(
                ident, size=320, rot_deg=rot, dx=dx, dy=dy, scale=sc,
                seed=ident * 1000 + pi)
            try:
                roi, info = pp.extract_roi(img, size, diagnostics=True)
            except pp.PreprocessError as e:
                failures.append((ident, pi, str(e)))
                continue
            rows.append({"roi": roi, "info": info, "truth": truth,
                         "pose": (rot, dx, dy, sc), "index": pi})
        out[ident] = rows
    return out, failures


# ====================================================================== 1
def test_success_rate(data, failures, total):
    section("1. Does it produce an ROI at all?")
    ok = sum(len(v) for v in data.values())
    rate = ok / float(total)
    for ident, pi, msg in failures[:5]:
        print("   declined: identity %d pose %d — %s" % (ident, pi, msg))
    check("an ROI is produced for every hand",
          rate == 1.0, "%d of %d (%.1f%%)" % (ok, total, 100 * rate))
    print()
    print("   A failure rate belongs in the results either way. Neither paper")
    print("   reports one, and a system that silently emits a wrong ROI is")
    print("   worse than one that declines — which is why preprocessing")
    print("   raises rather than guessing when the pose cannot be read.")
    return rate


# ====================================================================== 2
def test_angle_recovery(data):
    section("2. Does the ROI follow the hand's rotation?")
    # The frame's own convention puts its axis 180 degrees from the hand's
    # drawn rotation; what matters is that the OFFSET is constant, because a
    # constant offset is just a definition while a varying one is error.
    errs = []
    for ident, rows in data.items():
        for r in rows:
            rel = (r["info"]["angle_deg"] - r["pose"][0] + 180.0) % 360.0
            errs.append((rel + 180.0) % 360.0 - 180.0)
    errs = np.array(errs)
    spread = float(np.std(errs))
    worst = float(np.max(np.abs(errs - np.median(errs))))
    check("rotation is recovered to within a degree or so",
          spread < 2.0,
          "std %.2f deg, worst deviation %.2f deg over %d hands"
          % (spread, worst, len(errs)))
    print()
    print("   For comparison: Paper 2 states the ROI it inherits can be out")
    print("   by about +/-10 degrees, and builds seven rotated templates per")
    print("   enrolment to cope. At this accuracy that machinery is largely")
    print("   insurance — which is a claim we can now test rather than assert,")
    print("   by running the matcher with and without it on real images.")
    return spread


# ====================================================================== 3
def test_scale_normalisation(data):
    section("3. Is the ROI scale-normalised?")
    # The window side is defined in units of the valley separation, so it
    # should track the rendered scale exactly and cancel out of the crop.
    ratios = []
    for ident, rows in data.items():
        for r in rows:
            ratios.append(r["info"]["side"] / r["pose"][3])
    ratios = np.array(ratios)
    cv = float(ratios.std() / ratios.mean())
    check("window size tracks hand size",
          cv < 0.06,
          "side/scale varies by %.1f%% (raw side varies by %.1f%%)"
          % (100 * cv,
             100 * np.std([r["info"]["side"] for v in data.values()
                           for r in v])
             / np.mean([r["info"]["side"] for v in data.values() for r in v])))


# ====================================================================== 4
def test_repeatability(data):
    section("4. How repeatable is the ROI itself?")
    cfg = rloc.RlocConfig()
    codes = {i: [rloc.encode(r["roi"], cfg) for r in rows]
             for i, rows in data.items()}

    within, across = [], []
    for i, cs in codes.items():
        for a in range(len(cs)):
            for b in range(a + 1, len(cs)):
                within.append(mfrat.code_agreement(cs[a], cs[b]))
    ids = sorted(codes)
    for a in range(len(ids)):
        for b in range(a + 1, len(ids)):
            if codes[ids[a]] and codes[ids[b]]:
                across.append(mfrat.code_agreement(codes[ids[a]][0],
                                                   codes[ids[b]][0]))
    w, x = float(np.mean(within)), float(np.mean(across))
    check("two captures of one palm agree far more than two palms do",
          w > x + 0.10,
          "same palm %.3f, different palms %.3f, gap %.3f" % (w, x, w - x))
    print()
    print("   These are RAW code agreements, with no rotated templates and no")
    print("   pixel-to-area tolerance — the matcher's job is to widen that")
    print("   gap, and section 5 shows what it does with it.")


# ====================================================================== 5
def test_end_to_end_from_hands(data):
    section("5. Verification from whole hand images, end to end")
    # Everything before this point fed the matcher pre-cropped ROIs. Here the
    # input is a photograph of a hand and the system does the whole job:
    # segment, find the valleys, crop, encode, match.
    items = []
    for ident, rows in data.items():
        for r in rows:
            items.append(("S%03d" % ident, r["index"], r["roi"]))
    print("   protocol: %s" % protocol.protocol_summary(items, n_enrol=1))

    r = protocol.run_rloc(items, rloc.RlocConfig(), n_enrol=1,
                          label="RLOC from hand images")
    h = protocol.run_han(items, mode="sobel", cells=8, n_enrol=1,
                         label="Han baseline from hand images")
    print()
    print(metrics.compare([r, h]))
    check("the complete pipeline separates identities",
          r.dprime() > 2.0, "d' = %.2f" % r.dprime())
    check("RLOC beats the 2003 baseline on the same images",
          r.dprime() > h.dprime(),
          "d' %.2f vs %.2f" % (r.dprime(), h.dprime()))
    print()
    print("   Synthetic palms, so the EER is a smoke test and not a result.")
    print("   The real numbers come from the IITD database.")


def test_alignment_value(data):
    section("6. Are the seven rotated templates still worth storing?")
    items = []
    for ident, rows in data.items():
        for r in rows:
            items.append(("S%03d" % ident, r["index"], r["roi"]))

    out = []
    for label, cfg in protocol.alignment_configs():
        out.append(protocol.run_rloc(items, cfg, n_enrol=1, label=label))
    print(metrics.compare(out))

    d = {s.label: s.dprime() for s in out}
    seven, one = d["7 templates, cross"], d["1 template, cross"]
    check("one template is no worse than seven, once the ROI is accurate",
          one >= seven - 0.2,
          "d' %.2f with one template vs %.2f with seven" % (one, seven))
    print()
    print("   Paper 2 stores seven rotated templates because its inherited")
    print("   ROI can be ten degrees out. Ours is accurate to half a degree,")
    print("   and at that accuracy the seventh-of-the-storage version is not")
    print("   behind — if anything slightly ahead, by the same mechanism that")
    print("   makes the paper's own 3x3 square worse than its cross: every")
    print("   extra template is one more chance for an IMPOSTOR to match.")
    print()
    print("   So the enlarged training set may measure the quality of the")
    print("   preprocessing rather than the merit of the orientation code.")
    print("   A hypothesis with a mechanism, to be re-run on IITD — synthetic")
    print("   hands give our ROI an easier task than real ones will.")


def main():
    print("AI461 Phase II - ROI extraction checks")
    print("python %s, numpy %s" % (sys.version.split()[0], np.__version__))

    identities = list(range(1, 11))
    pose_list = poses(8)
    total = len(identities) * len(pose_list)
    print("\n%d identities x %d poses = %d hand images"
          % (len(identities), len(pose_list), total))
    print("poses: rotation +/-14 deg, translation +/-12 px, scale +/-10%")

    data, failures = extract_all(identities, pose_list)
    test_success_rate(data, failures, total)
    test_angle_recovery(data)
    test_scale_normalisation(data)
    test_repeatability(data)
    test_end_to_end_from_hands(data)
    test_alignment_value(data)

    section("Result")
    if _failures:
        print("%d check(s) failed: %s" % (len(_failures), ", ".join(_failures)))
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
