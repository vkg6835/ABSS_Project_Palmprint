#!/usr/bin/env python3
"""
test_demo.py — everything behind the demo window, checked without a window.

Run:  python test_demo.py
      python test_demo.py --sheet views.png   (also save every view, to look at)

The demo window (demo.py) only displays what palmprint/engine.py decides and
palmprint/stage.py draws, so those two are what is tested here: the dataset
cache, every one-click example, identification, live enrolment, refusing a
bad image, and the drawn view at every window size from a small projector up
to full HD — including that nothing on it overlaps or leaves the frame.

`python demo.py --check` then rehearses the real window on the real laptop.
Everything is written to a temporary folder and deleted afterwards.
"""
import os
import shutil
import sys
import tempfile
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import make_demo_data                                   # noqa: E402
from palmprint import engine as E                       # noqa: E402
from palmprint import metrics, render as R, stage as S  # noqa: E402

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


def views(eng):
    """Every distinct thing the stage can show, as (name, spec)."""
    ex = eng.examples()
    out = [("welcome", S.build(eng, None, "verify", "rloc", "", 0, None))]

    def add(name, path, task, matcher, claim, step=5, note=None):
        tr = eng.process(path)
        truth = S.truth_of(eng, tr.source)
        res = None
        if note is None:
            res = (eng.verify(tr, claim, matcher) if task == "verify"
                   else eng.identify(tr, matcher))
        step = min(step, 2 if note is not None else S.max_step(tr))
        out.append((name, S.build(eng, tr, task, matcher, claim, step, res,
                                  truth, note)))

    g = ex["genuine"]
    for st in (1, 2, 3, 4, 5):
        add("genuine_step%d" % st, g["path"], "verify", "rloc", g["claim"],
            st)
    add("impostor", ex["impostor"]["path"], "verify", "rloc",
        ex["impostor"]["claim"])
    add("closest", ex["hardest"]["path"], "verify", "rloc",
        ex["hardest"]["claim"])
    add("unreadable", ex["unreadable"]["path"], "verify", "rloc", g["claim"])
    add("identify", g["path"], "identify", "rloc", g["claim"])
    add("han_verify", g["path"], "verify", "han", g["claim"])
    add("han_identify", g["path"], "identify", "han", g["claim"])
    add("note", g["path"], "verify", "rloc", g["claim"], 2,
        ["Enrolled somebody", "A note shown in place of a decision."])
    return out


def boxes(L):
    """The rectangles the layout promises not to overlap: (name, x0,y0,x1,y1)."""
    out = []
    for k, x in enumerate(L["panel_x"]):
        out.append(("panel%d" % k, x, L["panel_y"], x + L["P"],
                    L["panel_y"] + L["P"]))
        out.append(("caption%d" % k, x, L["cap_y"], x + L["P"],
                    L["cap_y"] + 3 * L["cap_lh"]))
    out.append(("plot", L["x0"], L["dec_y"], L["x0"] + L["plot_w"],
                L["dec_y"] + L["plot_h"]))
    out.append(("verdict", L["box_x"], L["dec_y"], L["box_x"] + L["box_w"],
                L["dec_y"] + L["box_h"]))
    out.append(("plot caption", L["x0"], L["pcap_y"],
                L["x0"] + L["plot_w"], L["pcap_y"] + 2 * L["cap_lh"]))
    return out


def main(argv):
    sheet = None
    if "--sheet" in argv:
        sheet = argv[argv.index("--sheet") + 1]
    print("AI461 Phase II - demo engine and display checks")
    print("python %s, numpy %s" % (sys.version.split()[0], np.__version__))
    tmp = tempfile.mkdtemp(prefix="palm_demo_test_")
    try:
        return run(tmp, sheet)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def run(tmp, sheet):
    data = os.path.join(tmp, "data")
    cache = os.path.join(tmp, "cache")

    # ------------------------------------------------------------ the data
    section("1. Demo dataset and cache")
    t0 = time.time()
    make_demo_data.make(data, people=12, samples=5)
    print("   dataset generated in %.1f s" % (time.time() - t0))
    t0 = time.time()
    eng = E.Engine(data, cache_dir=cache).prepare()
    cold = time.time() - t0
    check("identities are the person folders only",
          len(eng.identities) == 12 and not any(
              i.startswith("_") for i in eng.identities),
          "%d enrolled; _extra and _visitors ignored" % len(eng.identities))
    t0 = time.time()
    warm = E.Engine(data, cache_dir=cache).prepare()
    hot = time.time() - t0
    check("a second start-up loads from the cache",
          warm.cache_status == "loaded from cache" and hot < cold,
          "%.1f s cold, %.2f s from cache" % (cold, hot))
    same = all(np.array_equal(eng.score_matrix[m], warm.score_matrix[m],
                              equal_nan=True) for m in E.MATCHERS)
    check("cached scores are identical to computed ones", same)

    victim = eng.enrol_source[eng.identities[0]]
    st = os.stat(victim)
    os.utime(victim, ns=(st.st_atime_ns, st.st_mtime_ns + 10 ** 9))
    again = E.Engine(data, cache_dir=cache).prepare()
    check("changing an image invalidates the cache",
          again.cache_status == "computed")

    # ------------------------------------------------------- the examples
    section("2. The one-click examples")
    ex = eng.examples()
    check("all four examples exist",
          all(k in ex for k in ("genuine", "impostor", "hardest",
                                "unreadable")), ", ".join(sorted(ex)))
    for m in E.MATCHERS:
        tr = eng.process(ex["genuine"]["path"])
        r = eng.verify(tr, ex["genuine"]["claim"], m)
        check("%s accepts the genuine example" % m, r["decision"] == "accept",
              "score %.3f, threshold %.3f" % (r["score"], r["threshold"]))
        tr = eng.process(ex["impostor"]["path"])
        r = eng.verify(tr, ex["impostor"]["claim"], m)
        check("%s rejects the impostor example" % m,
              r["decision"] == "reject",
              "score %.3f, threshold %.3f" % (r["score"], r["threshold"]))
    tr = eng.process(ex["unreadable"]["path"])
    r = eng.verify(tr, eng.identities[0])
    check("the unreadable image is declined, with a reason",
          not tr.ok and r["decision"] == "declined" and tr.reason,
          tr.reason)

    tr = eng.process(ex["genuine"]["path"])
    truth = S.truth_of(eng, tr.source)
    agree = True
    for ident in eng.identities:
        v = eng.verify(tr, ident, "rloc")["score"]
        g = float(eng.rloc_gallery.scores(tr.code)[
            eng.identities.index(ident)])
        agree &= abs(v - g) < 1e-12
    check("the matching shown on screen gives the evaluated score", agree)

    # Verifying a person with the image they were enrolled from scores 1.000
    # by construction. The verdict must say so, or a live audience reads it
    # as evidence (it happened on the real laptop, with session1/00001.bmp).
    ident0 = eng.identities[0]
    tr0 = eng.process(eng.enrol_source[ident0])
    r0 = eng.verify(tr0, ident0, "rloc")
    lines0 = S.correctness_lines(eng, "verify", r0, S.truth_of(eng, tr0.source),
                                 tr0.source)
    check("the enrolment image itself is flagged, and only it",
          abs(r0["score"] - 1.0) < 1e-12
          and lines0 == ["correct", "its enrolment image"]
          and "its enrolment image" not in S.correctness_lines(
              eng, "verify", eng.verify(tr, truth, "rloc"), truth, tr.source),
          "score %.3f: %s" % (r0["score"], " / ".join(lines0)))
    r = eng.identify(tr, "rloc")
    check("identification names the right person",
          r["identity"] == truth, "%s, truth %s" % (r["identity"], truth))
    a = eng.process(ex["genuine"]["path"]).code
    b = eng.process(ex["genuine"]["path"]).code
    check("processing is deterministic", np.array_equal(a, b))

    # ---------------------------------------------------- live enrolment
    section("3. Live enrolment")
    vis = os.path.join(data, "_visitors")
    tr = eng.enrol("Visitor A", os.path.join(vis, "visitor_A_1.png"))
    check("a visitor can be enrolled from one image",
          tr.ok and "Visitor A" in eng.identities)
    p = eng.process(os.path.join(vis, "visitor_A_2.png"))
    r = eng.verify(p, "Visitor A")
    check("their second capture is then accepted", r["accept"],
          "score %.3f" % r["score"])
    r = eng.identify(eng.process(os.path.join(vis, "visitor_B_1.png")))
    check("an unenrolled visitor is identified as nobody", not r["accept"],
          "best %s at %.3f, threshold %.3f"
          % (r["candidate"], r["score"], r["threshold"]))
    n = len(eng.identities)
    tr = eng.enrol("Mitten", ex["unreadable"]["path"])
    check("an unreadable image is refused at enrolment",
          not tr.ok and len(eng.identities) == n
          and "Mitten" not in eng.identities)
    before = len(eng.rloc_templates["Visitor A"])
    eng.enrol("Visitor A", os.path.join(vis, "visitor_A_2.png"))
    check("enrolling a known name adds templates rather than a person",
          len(eng.rloc_templates["Visitor A"]) == 2 * before
          and len(eng.identities) == n)

    # ------------------------------------------------------- the display
    section("4. What the audience sees")
    V = views(eng)
    # stage sizes taken from real windows on the demo laptop (1920 x 1080 at
    # 125% scaling): 1024 x 768, 1280 x 720, 1366 x 768, the same with the
    # controls hidden, 1600 x 900, maximised — plus some awkward extremes
    sizes = [(730, 650), (990, 602), (1074, 650), (1340, 650), (1300, 782),
             (1630, 880), (1900, 1000), (640, 420), (1020, 590)]
    bad_size, bad_layout, cut, worst = [], [], [], 0.0
    for w, h in sizes:
        L = S.layout(w, h)
        bx = boxes(L)
        for name, x0, y0, x1, y1 in bx:
            if x0 < 0 or y0 < 0 or x1 > w + 1 or y1 > h + 1:
                bad_layout.append("%dx%d %s leaves the frame" % (w, h, name))
        for i in range(len(bx)):
            for j in range(i + 1, len(bx)):
                a, b = bx[i], bx[j]
                if a[1] < b[3] and b[1] < a[3] and a[2] < b[4] \
                        and b[2] < a[4]:
                    bad_layout.append("%dx%d %s overlaps %s"
                                      % (w, h, a[0], b[0]))
        for name, spec in V:
            t0 = time.perf_counter()
            im = S.compose(spec, w, h)
            worst = max(worst, time.perf_counter() - t0)
            if im.size != (w, h):
                bad_size.append("%s at %dx%d" % (name, w, h))
            dec = spec["decision"]
            if not dec.get("wide") and (w, h) != (640, 420):
                lost = R.verdict_overflow(dec["big"], dec["lines"],
                                          L["box_w"], L["box_h"],
                                          S.text_px(L))
                if lost:
                    cut.append("%s at %dx%d loses %d line(s)"
                               % (name, w, h, lost))
    check("every view is drawn at exactly the window size",
          not bad_size, "%d views x %d sizes" % (len(V), len(sizes))
          if not bad_size else "; ".join(bad_size[:4]))
    check("nothing overlaps or leaves the frame, at any size",
          not bad_layout, "; ".join(bad_layout[:4]))
    check("the verdict box never cuts off a line",
          not cut, "; ".join(cut[:4]) if cut else
          "every verdict, every real window size")

    # The views above only exercise the names in this dataset. The box must
    # also hold the longest thing it can ever be asked to say — a wrong-person
    # verdict naming someone enrolled live under a full name — because the
    # first real run lost exactly that last line on Tongji.
    long_name = "Vishal Gautam"
    worst_case = {
        "ACCEPT": ["score  0.846", "threshold  0.662", "FALSE ACCEPT",
                   "it is %s" % long_name],
        "REJECT": ["score  0.557", "threshold  0.662", "correct",
                   "it is %s" % long_name],
        "IDENTIFIED": [long_name, "score  0.870", "WRONG PERSON",
                       "it is %s" % long_name],
        "NO MATCH": ["best was %s" % long_name, "score  0.591", "correct",
                     "%s is not enrolled" % long_name],
    }
    # (a second IDENTIFIED case: a dict cannot hold the key twice)
    worst_case = list(worst_case.items()) + [
        ("IDENTIFIED", [long_name, "score  1.000", "correct",
                        "its enrolment image"])]
    lost = []
    for w, h in sizes:
        if (w, h) == (640, 420):
            continue
        L = S.layout(w, h)
        for big, lines in worst_case:
            n = R.verdict_overflow(big, lines, L["box_w"], L["box_h"],
                                   S.text_px(L))
            if n:
                lost.append("%s at %dx%d loses %d" % (big, w, h, n))
    check("even the longest possible verdict fits, at every real size",
          not lost, "; ".join(lost[:4]) if lost else
          "%d worst-case verdicts x %d sizes"
          % (len(worst_case), len(sizes) - 1))
    check("drawing is fast enough to step through live",
          worst < 0.6, "slowest view %.0f ms" % (1000 * worst))

    ss = metrics.ScoreSet([0.80, 0.90, 0.85], [0.10, 0.20, 0.30])
    e, t = ss.eer()
    check("with no overlap the threshold sits midway between the classes",
          e == 0.0 and abs(t - 0.55) < 1e-12, "threshold %.3f" % t)

    if sheet:
        from PIL import Image
        ims = [S.compose(spec, 1100, 650) for _, spec in V]
        cols = 3
        tw, th = 550, 325
        rows = (len(ims) + cols - 1) // cols
        out = Image.new("RGB", (cols * (tw + 10) + 10, rows * (th + 10) + 10),
                        (128, 128, 128))
        for k, im in enumerate(ims):
            out.paste(im.resize((tw, th)), (10 + (k % cols) * (tw + 10),
                                            10 + (k // cols) * (th + 10)))
        out.save(sheet)
        print("\n   saved every view to %s" % sheet)

    section("Result")
    if _failures:
        print("%d check(s) failed: %s" % (len(_failures), ", ".join(_failures)))
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
