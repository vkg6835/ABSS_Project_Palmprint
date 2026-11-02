#!/usr/bin/env python3
"""
selftest.py — prove the pipeline is correct before any real data arrives.

Run:  python selftest.py

Each check below either asserts a property that must hold if the operator is
implemented as the papers describe, or measures something the papers predict
so the prediction can be compared against our own numbers. Nothing here is a
performance claim: the images are synthetic (see palmprint/synthetic.py) and
are far cleaner than real palms.
"""
import sys
import time

import numpy as np

from palmprint import han, mfrat, metrics, protocol, rloc, synthetic

PASS, FAIL = "  ok  ", " FAIL "
_failures = []


def check(name, condition, detail=""):
    ok = bool(condition)
    if not ok:
        _failures.append(name)
    print("[%s] %s%s" % (PASS if ok else FAIL, name,
                         ("   " + detail) if detail else ""))
    return ok


def section(title):
    print("\n" + title)
    print("-" * len(title))


# ====================================================================== 1
def test_masks():
    section("1. The MFRAT operator")
    masks, angles = mfrat.mfrat_masks(p=16, n_dir=6, width=4)

    counts = [int(m.sum()) for m in masks]
    check("every direction gets the same pixel count",
          len(set(counts)) == 1,
          "%d pixels per direction" % counts[0])

    # Change 2 in the paper: without this, the argmin is biased towards
    # whichever direction happens to collect the most lattice points.
    sym = []
    for m in masks:
        sym.append(np.allclose(m, m[::-1, ::-1]))
    check("every mask is symmetric about the lattice centre", all(sym))

    check("six distinct directions", len(angles) == 6
          and len({round(a, 6) for a in angles}) == 6,
          "0 to %.0f degrees" % np.rad2deg(angles[-1]))


# ====================================================================== 2
def test_argmin_finds_lines():
    section("2. Winner-take-all by argmin finds dark lines")
    n, p, step, tol = 96, 16, 4, 1.5

    # A single straight dark line at a known angle, nothing else. The code is
    # scored only at lattice centres that actually lie ON the line (within
    # `tol` pixels) and away from the border.
    #
    # That restriction is the point of the test, not a convenience. MFRAT
    # assigns a direction to EVERY point, including points with no line
    # through them at all; at such a centre the six sums are nearly equal and
    # the argmin is essentially arbitrary. So "the code is correct" can only
    # mean "correct where there is a line". What tells you whether to believe
    # a given code value is the directional contrast — which the paper
    # discards and we keep. This is also the strongest argument for the fusion
    # experiment we propose: magnitude is exactly the confidence measure the
    # orientation code lacks.
    results = []
    for k, ang in enumerate(np.pi * np.arange(6) / 6.0):
        t = (np.arange(n) + 0.5) / n - 0.5
        xx, yy = np.meshgrid(t, t)
        perp = -np.sin(ang) * xx + np.cos(ang) * yy
        img = 0.8 - 0.45 * np.exp(-(perp ** 2) / (2 * 0.012 ** 2))
        code, _ = mfrat.orientation_code(img, p=p, step=step)

        dist = perp[step // 2::step, step // 2::step] * n     # in pixels
        m0 = p // step
        sel = np.zeros(dist.shape, dtype=bool)
        sel[m0:-m0, m0:-m0] = True
        sel &= np.abs(dist) <= tol
        results.append(float(np.mean(code[sel] == k)))
    worst = min(results)
    check("the winning index is the true line direction",
          worst > 0.95,
          "agreement on the line: " +
          " ".join("%.0f%%" % (100 * r) for r in results))

    # The sign of the extremum is the whole point: palm lines are negative.
    t = (np.arange(n) + 0.5) / n - 0.5
    xx, yy = np.meshgrid(t, t)
    img = 0.8 - 0.45 * np.exp(-(yy ** 2) / (2 * 0.012 ** 2))   # horizontal
    resp = mfrat.mfrat_response(img, p=p)
    mid = resp[:, n // 2, n // 2]
    check("argmin picks the line, argmax would not",
          int(np.argmin(mid)) == 0 and int(np.argmax(mid)) != 0,
          "argmin=%d (direction 0 = horizontal), argmax=%d"
          % (int(np.argmin(mid)), int(np.argmax(mid))))


# ====================================================================== 3
def test_invariance():
    section("3. What the code is and is not invariant to")
    base = synthetic.sample(1, 0, size=128)
    cfg = rloc.RlocConfig()
    c0 = rloc.encode(base, cfg)

    # Illumination: an index is an order statistic, so a contrast and
    # brightness change should barely move it. This is the property the whole
    # design of Paper 2 is built to obtain.
    lit = np.clip(0.75 * base + 0.12, 0, 1)
    agree_light = mfrat.code_agreement(c0, rloc.encode(lit, cfg))
    check("robust to illumination change (gain 0.75, bias +0.12)",
          agree_light > 0.90, "%.1f%% of codes unchanged" % (100 * agree_light))

    # Paper 1's feature is a pooled magnitude, which scales with contrast...
    f0 = han.features(base, "sobel", cells=8)
    f1 = han.features(lit, "sobel", cells=8)
    shift = float(np.mean(np.abs(f1 - f0)) / (np.mean(np.abs(f0)) + 1e-12))
    check("the pooled Sobel magnitude does move under the same change",
          shift > 0.10, "mean change %.0f%%" % (100 * shift))
    # ...but Paper 1 matches with a Pearson correlation, which subtracts the
    # mean and divides by the spread, so a global gain and bias cancel in the
    # SCORE. The two methods reach lighting invariance by different routes —
    # RLOC in the code itself, Han in the matching — so lighting is not where
    # they differ, and nothing in our results should be read as saying so.
    other = synthetic.sample(1, 1, size=128)
    fo = han.features(other, "sobel", cells=8)
    s0, s1 = han.pearson(f0, fo), han.pearson(han.features(
        0.75 * base + 0.12, "sobel", cells=8), fo)
    check("but Han's Pearson score cancels a global gain and bias",
          abs(s1 - s0) < 1e-9, "score %.4f -> %.4f" % (s0, s1))

    # Rotation: it changes WHICH direction wins, i.e. the feature values
    # themselves. This is why enrolment stores rotated templates.
    agree_rot = mfrat.code_agreement(
        c0, rloc.encode(rloc.rotate_roi(base, 9.0), cfg))
    check("rotation by 9 degrees does disturb the code",
          agree_rot < 0.80,
          "%.1f%% unchanged - hence the seven rotated templates"
          % (100 * agree_rot))


# ====================================================================== 4
def test_affine_invariance_proof():
    section("3b. Why equal pixel counts matter (our own result)")
    # The paper justifies giving every direction the same number of pixels on
    # the grounds that the six sums are then "comparable". The sharper
    # statement, which the paper does not make, is that equal counts are the
    # exact condition under which the code is PROVABLY invariant to any
    # affine illumination change.
    #
    #   r_k = sum over L_k of f.      Let f -> a*f + b, with a > 0.
    #   Then r_k -> a*r_k + b*|L_k|.
    #
    # If every |L_k| is the same N, this is one strictly increasing map
    # applied identically to all six sums, so argmin is untouched: the code is
    # bit-for-bit unchanged. If the counts differ, the b*|L_k| term shifts the
    # directions by different amounts and the winner can flip. So requirement
    # 2 is not housekeeping — it is what buys the illumination robustness the
    # whole method is sold on.
    base = synthetic.sample(3, 1, size=128)
    cfg = rloc.RlocConfig()
    c0 = rloc.encode(base, cfg)

    exact = []
    for a, b in ((0.5, 0.20), (1.8, -0.30), (0.25, 0.50)):
        # deliberately NOT clipped to [0,1]: clipping is a non-affine
        # operation and would itself destroy the invariance
        exact.append(mfrat.code_agreement(c0, rloc.encode(a * base + b, cfg)))
    check("equal counts give EXACT invariance to gain and bias",
          all(abs(e - 1.0) < 1e-12 for e in exact),
          "agreement %s" % " ".join("%.4f" % e for e in exact))

    # The counter-demonstration: break requirement 2 and watch it fail.
    masks, _ = mfrat.mfrat_masks(cfg.p, cfg.n_dir, cfg.width)
    bad = [m.copy() for m in masks]
    ys, xs = np.nonzero(bad[0])
    bad[0][ys[:8], xs[:8]] = 0.0          # direction 0 now has 8 fewer pixels

    def code_with(masks_, img):
        resp = np.stack([mfrat._sparse_correlate(img, m) for m in masks_])
        off = cfg.step // 2
        return np.argmin(resp, axis=0)[off::cfg.step, off::cfg.step]

    b0 = code_with(bad, base)
    b1 = code_with(bad, 0.5 * base + 0.2)
    agree_bad = float(np.mean(b0 == b1))
    check("unequal counts break that invariance",
          agree_bad < 0.999,
          "one direction short by 8 px -> only %.1f%% of codes survive"
          % (100 * agree_bad))


def test_matching_rule():
    section("4. The pixel-to-area matching rule")
    a = np.array([[0, 1, 2], [3, 4, 5], [0, 1, 2]], dtype=np.uint8)

    check("a code matches itself perfectly",
          abs(rloc.area_match(a, a, rloc.NEIGHBOURHOODS["pixel"]) - 1.0) < 1e-12)

    # A one-pixel shift is the failure mode the rule exists to fix.
    shifted = np.roll(a, 1, axis=1)
    pix = rloc.area_match(a, shifted, rloc.NEIGHBOURHOODS["pixel"])
    crs = rloc.area_match(a, shifted, rloc.NEIGHBOURHOODS["cross"])
    check("the cross neighbourhood rescues a one-pixel shift",
          crs > pix, "pixel-to-pixel %.2f -> cross %.2f" % (pix, crs))

    # Tolerance rises monotonically with the neighbourhood, which is exactly
    # why the 3x3 square is WORSE: impostors benefit too.
    rng = np.random.default_rng(0)
    x = rng.integers(0, 6, (32, 32)).astype(np.uint8)
    y = rng.integers(0, 6, (32, 32)).astype(np.uint8)
    s = [rloc.area_match(x, y, rloc.NEIGHBOURHOODS[k])
         for k in ("pixel", "cross", "square")]
    check("unrelated codes also agree more as the area grows",
          s[0] < s[1] < s[2],
          "random pair: %.2f / %.2f / %.2f for 1, 5, 9 points"
          % tuple(s))

    # The fast batched scorer must be the SAME matcher, not an approximation
    # of it: every number in the results comes from it, while the demo's
    # matching panel is drawn by the original one.
    same = True
    for nb in ("pixel", "cross", "square"):
        for rots in ((0.0,), (0, 3, -3, 6, -6, 9, -9)):
            for shift, sym in ((2, True), (1, False)):
                cfg = rloc.RlocConfig(neighbourhood=nb, rotations=rots,
                                      shift_range=shift, symmetric=sym)
                g = rloc.Gallery(cfg)
                for i in range(6):
                    g.enrol("S%d" % i, synthetic.sample(i + 1, 0))
                probes = np.stack([rloc.encode(
                    synthetic.sample(1 + k % 6, 1 + k // 6), cfg)
                    for k in range(12)])
                ref = np.stack([g.scores(p) for p in probes])
                same &= bool(np.array_equal(ref, g.scores_batch(probes,
                                                                chunk=5)))
    check("the batched scorer is bit-for-bit the same matcher",
          same, "pixel / cross / square x 1 or 7 templates x as published "
                "or as first built, all identical")

    # Equation (9) is symmetric by construction: S(A,B) = S(B,A).
    cfg = rloc.RlocConfig()
    sab = rloc.match_codes(x, y, cfg)
    sba = rloc.match_codes(y, x, cfg)
    check("equation (9) gives the same score whichever code is the probe",
          sab == sba, "S(A,B) %.4f, S(B,A) %.4f" % (sab, sba))
    one = rloc.RlocConfig(symmetric=False)
    check("and is never below either direction on its own",
          sab >= rloc.match_codes(x, y, one)
          and sab >= rloc.match_codes(y, x, one))


# ====================================================================== 5
def test_end_to_end():
    section("5. End-to-end verification on synthetic identities")
    # Deliberately harsh settings: more rotation and translation than Paper 2
    # reports inheriting, and heavy noise. Even so, read the EER below as a
    # smoke test and nothing more — synthetic identities are far more
    # distinctive than real palms, so an EER measured here is not a
    # performance claim and will not appear in the presentation. What this
    # does establish is that enrolment, matching, the protocol and the metrics
    # are wired together correctly.
    t0 = time.time()
    items = synthetic.dataset(n_identities=40, n_samples=5, size=128,
                              noise_sd=0.09, rot_deg=14.0, shift_px=8.0)
    print("   generated %d images in %.1f s" % (len(items), time.time() - t0))
    print("   protocol: %s" % protocol.protocol_summary(items, n_enrol=1))

    cfg = rloc.RlocConfig()
    r = protocol.run_rloc(items, cfg, n_enrol=1, label="RLOC (cross, 7 rot)")
    print()
    print(r.report())
    print("   %.1f ms per probe (against %d identities)"
          % (1000 * r.elapsed / r.n_genuine, len(set(i for i, _, _ in items))))

    h = protocol.run_han(items, mode="sobel", cells=8,
                         label="Han Sobel 8x8 grid")
    print()
    print(h.report())

    m = protocol.run_han(items, mode="morph", cells=8,
                         label="Han morphology 8x8 grid")
    print()
    print(m.report())

    print()
    print(metrics.compare([r, h, m]))

    check("RLOC separates genuine from impostor", r.dprime() > 2.0,
          "d' = %.2f" % r.dprime())
    check("the Han baseline also works", h.dprime() > 1.0,
          "d' = %.2f" % h.dprime())
    check("genuine scores beat impostor scores on average",
          r.genuine.mean() > r.impostor.mean())
    return items


# ====================================================================== 6
def test_ablation(items):
    section("6. Paper 2's Table 4, re-run (the experiment it never published)")
    out = []
    for name, cfg in protocol.ablation_configs():
        ss = protocol.run_rloc(items, cfg, n_enrol=1, label=name)
        out.append(ss)
    print(metrics.compare(out))
    print()
    print("   Reading: the gap between Exp 1 and Exp 3 is what the enlarged")
    print("   training set and the cross neighbourhood are worth. Neither is")
    print("   specific to MFRAT, and neither was ever applied to the Gabor")
    print("   baseline the paper compares against.")
    print()
    print("   Ordering is checked on d' rather than EER: on synthetic data the")
    print("   EER of the stronger configurations hits the floor, so it cannot")
    print("   rank them. d' uses the whole distribution and still can.")

    d = {s.label.split()[1]: s.dprime() for s in out}
    check("the enlarged training set helps",
          d["2"] > d["1"], "d' %.2f -> %.2f" % (d["1"], d["2"]))
    check("the cross neighbourhood helps further",
          d["3"] > d["2"], "d' %.2f -> %.2f" % (d["2"], d["3"]))

    # The paper finds the 9-point square WORSE than the 5-point cross on real
    # data (0.27% against 0.16% EER): nine candidate positions give impostor
    # pixels nine chances to agree, so tolerance rises but discrimination
    # falls. We do not assert that ordering here, because it is not a property
    # of the operator — it depends on how much residual misalignment is left
    # after preprocessing. Our synthetic fixture carries 14 degrees of
    # rotation and 8 pixels of shift, which is more than Paper 2 inherits, and
    # at that level the square's extra tolerance still earns its keep. What
    # does reproduce, and is the substantive point, is that the returns
    # collapse once the cross is in place. Whether widening further actively
    # hurts is a question for the real database, and is on the experiment list.
    gain_cross = d["3"] - d["2"]
    gain_square = d["4"] - d["3"]
    check("returns collapse past the cross neighbourhood",
          gain_square < 0.25 * gain_cross,
          "cross step +%.2f d', square step %+.2f d'"
          % (gain_cross, gain_square))
    print()
    print("   Note: the paper reports the square as WORSE than the cross, and")
    print("   on Tongji it is (results/results.md, section 3). Here only the")
    print("   collapse of the returns is checked, since how much a wider")
    print("   neighbourhood hurts depends on the fixture's misalignment.")


def main():
    print("AI461 Phase II - palmprint pipeline self-test")
    print("python %s, numpy %s" % (sys.version.split()[0], np.__version__))
    test_masks()
    test_argmin_finds_lines()
    test_invariance()
    test_affine_invariance_proof()
    test_matching_rule()
    items = test_end_to_end()
    test_ablation(items)

    section("Result")
    if _failures:
        print("%d check(s) failed: %s" % (len(_failures), ", ".join(_failures)))
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
