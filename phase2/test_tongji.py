#!/usr/bin/env python3
"""
test_tongji.py — the Tongji loader, checked before the real files arrive.

    python test_tongji.py
    python test_tongji.py --data PATH     also check a real extraction

The first part writes a Tongji-shaped dataset out of synthetic palms — the
same session1/session2 folders, the same flat numbering, the same 128 x 128
8-bit BMPs — so the loader can be proven now rather than on the day. It
checks the grouping, that a truncated or gappy download is refused instead of
silently mis-grouped, and that pairing_check() tells the truth about a
correctly numbered set AND catches a deliberately scrambled one.

`--data` then runs the same pairing check on a real extraction, which is the
question that must be answered before any result from that database is
trusted.
"""
import os
import shutil
import sys
import tempfile

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from palmprint import protocol, synthetic, tongji   # noqa: E402

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


def write_tongji(path, palms=6, per_palm=4, sessions=("session1", "session2"),
                 scramble=False, drop=None, skip=None):
    """
    Write a dataset in the Tongji layout: 128 x 128 8-bit BMPs, named
    00001.bmp upwards in each session folder.

    `scramble` renumbers session 2 so palm k appears under another palm's
    number — the failure mode pairing_check() must catch. `drop` removes one
    file (truncated download); `skip` leaves a gap in the numbering.
    """
    os.makedirs(path, exist_ok=True)
    for s_i, sess in enumerate(sessions):
        folder = os.path.join(path, sess)
        os.makedirs(folder, exist_ok=True)
        for palm in range(1, palms + 1):
            for k in range(per_palm):
                number = (palm - 1) * per_palm + k + 1
                # each palm is a different identity seed; session and
                # capture give it a fresh pose, brightness and noise
                seed = palm if not (scramble and s_i == 1) else \
                    (palm % palms) + 1
                img = synthetic.render(
                    synthetic.palm_params(seed),
                    size=128, rot_deg=3.0 * (s_i + k), dx=4.0 * (s_i + 1),
                    dy=-3.0 * k, gain=0.9 + 0.1 * s_i, bias=0.02 * k,
                    noise=0.012, seed=seed * 100 + s_i * 10 + k)
                arr = (np.clip(img, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
                name = "%05d.bmp" % number
                if drop and (sess, name) == tuple(drop):
                    continue
                Image.fromarray(arr, mode="L").save(os.path.join(folder, name))
        if skip and sess == skip[0]:
            os.remove(os.path.join(folder, skip[1]))
    return path


def main(argv):
    real = None
    if "--data" in argv:
        real = argv[argv.index("--data") + 1]
    print("AI461 Phase II - Tongji loader checks")
    print("python %s, numpy %s" % (sys.version.split()[0], np.__version__))
    tmp = tempfile.mkdtemp(prefix="palm_tongji_test_")
    try:
        return run(tmp, real)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def run(tmp, real):
    good = os.path.join(tmp, "good")
    write_tongji(good, palms=6, per_palm=4)

    # ------------------------------------------------------------ grouping
    section("1. Does it group the flat numbering into palms?")
    items = tongji.index(good, per_palm=4)
    ids = sorted({t[0] for t in items})
    check("six palms are found", len(ids) == 6, ", ".join(ids))
    check("every palm has 4 + 4 captures from the two sessions",
          all(sum(1 for t in items if t[0] == i) == 8 for i in ids),
          "4 palms x 8 images = %d files" % len(items))
    first = [t for t in items if t[0] == "P001"]
    check("session 1 captures come first, so the first is enrolled",
          all(tongji.session_of(p) == "session1" for _, _, p in first[:4])
          and all(tongji.session_of(p) == "session2" for _, _, p in first[4:]),
          "ranks %s" % [r for _, r, _ in first])
    enrol, probe = protocol.split_enrol_probe(items, n_enrol=1)
    check("one capture is enrolled per palm; the other seven are probes",
          len(enrol) == 6 and len(probe) == 42,
          "enrol %d, probe %d" % (len(enrol), len(probe)))
    ident, _, path = items[0]
    check("the images are the 128 x 128 greyscale ROIs the papers assume",
          protocol.load_gray(path).shape == (128, 128))
    check("the session is read from the path",
          tongji.session_of(items[0][2]) == "session1"
          and tongji.session_of(items[-1][2]) == "session2")

    # ------------------------------------------------------ reading the disk
    section("2. Does it describe what is actually on disk?")
    d = tongji.describe(good, per_palm=4)
    check("the description counts both sessions",
          d["images"] == 48 and d["palms"] == 6,
          "%d images, %d palms" % (d["images"], d["palms"]))

    # ------------------------------------------------- refusing bad folders
    section("3. Does it refuse a download that is not what it claims?")
    # Each case also names the check that should catch it: a fault reported
    # by the wrong check gives a confusing message on a real download.
    def refuses(path, per_palm, want, name):
        try:
            tongji.index(path, per_palm=per_palm)
        except tongji.TongjiError as e:
            got = str(e)
            return check(name, want in got, got)
        return check(name, False, "accepted it")

    drop = os.path.join(tmp, "dropped")
    write_tongji(drop, palms=6, per_palm=4, drop=("session2", "00007.bmp"))
    refuses(drop, 4, "different numbers of images",
            "one session missing an image is refused")

    gap = os.path.join(tmp, "gap")
    write_tongji(gap, palms=6, per_palm=4, skip=("session1", "00005.bmp"))
    refuses(gap, 4, "different numbers of images",
            "a gap in one session is refused")

    odd = os.path.join(tmp, "odd")
    write_tongji(odd, palms=5, per_palm=4)          # 20 per session
    refuses(odd, 3, "not a whole number of palms",
            "a count that is not a whole number of palms is refused")

    # The subtle one: the same number missing from BOTH sessions leaves the
    # per-session counts equal, so only the gap check can catch it — and it
    # must say "incomplete", not "not a whole number of palms".
    both = os.path.join(tmp, "both")
    write_tongji(both, palms=6, per_palm=4)
    for sess in ("session1", "session2"):
        os.remove(os.path.join(both, sess, "00009.bmp"))
    refuses(both, 4, "incomplete",
            "a gap in BOTH sessions is caught and named as an incomplete "
            "download")

    # And one palm's worth missing entirely from the end of both sessions is
    # a legitimate smaller release — that must be accepted, not refused.
    smaller = os.path.join(tmp, "smaller")
    write_tongji(smaller, palms=5, per_palm=4)
    try:
        n = len(tongji.index(smaller, per_palm=4))
        ok = n == 40
        msg = "%d images from 5 palms" % n
    except tongji.TongjiError as e:
        ok = False
        msg = str(e)
    check("a smaller but complete release is accepted", ok, msg)

    try:
        tongji.index(os.path.join(tmp, "nothing-here"), per_palm=4)
        ok = False
        msg = "accepted a folder that does not exist"
    except tongji.TongjiError as e:
        ok = True
        msg = str(e)
    check("a missing folder gives a clear message", ok, msg)

    # ------------------------------------------- is the numbering believable?
    section("4. Does it check the numbering against the pixels?")
    res = tongji.pairing_check(tongji.index(good, per_palm=4), n_palms=6,
                               pairs_per_palm=3)
    check("correctly numbered captures of one palm agree more than strangers",
          res["ok"] and res["gap"] > 0,
          "same palm %.3f, different palm %.3f, gap %+.3f over %d + %d pairs"
          % (res["same_palm"], res["different_palm"], res["gap"],
             res["pairs"][0], res["pairs"][1]))

    bad = os.path.join(tmp, "scrambled")
    write_tongji(bad, palms=6, per_palm=4, scramble=True)
    res_bad = tongji.pairing_check(tongji.index(bad, per_palm=4), n_palms=6,
                                   pairs_per_palm=3)
    check("a scrambled numbering is caught", not res_bad["ok"],
          "same palm %.3f, different palm %.3f, gap %+.3f"
          % (res_bad["same_palm"], res_bad["different_palm"], res_bad["gap"]))

    # -------------------------------------- end to end, as the demo will run
    section("5. The workspace and the demo, end to end on a Tongji set")
    from palmprint import engine as E
    write_tongji(good, palms=8, per_palm=4)          # rewrite a bit larger
    tongji.write_manifest(good, per_palm=4)
    eng = E.Engine(good, cache_dir=os.path.join(tmp, "cache")).prepare()
    check("the manifest tells the engine to use pre-cropped ROIs",
          eng.kind == "roi" and eng.layout == "tongji",
          "kind %s, layout %s" % (eng.kind, eng.layout))
    check("every palm is enrolled and the rest are probes",
          len(eng.identities) == 8 and len(eng.probes) == 56,
          "%d enrolled, %d probes" % (len(eng.identities), len(eng.probes)))
    check("no image is declined (these are already cropped palms)",
          sum(1 for p in eng.probes if not p["ok"]) == 0)
    tr = eng.process(eng.probes[0]["path"])
    check("a probe goes straight to a code, with no segmentation of ours",
          tr.ok and tr.info is None and tr.code.shape == (32, 32),
          "code %s, no preprocessing info" % (tr.code.shape,))
    s = eng.summary()
    check("both matchers produce an evaluation with a threshold",
          all(s[m]["eer"] is not None for m in E.MATCHERS),
          "RLOC EER %.1f%%, Han EER %.1f%%"
          % (100 * s["rloc"]["eer"], 100 * s["han"]["eer"]))
    ex = eng.examples()
    r = eng.verify(eng.process(ex["genuine"]["path"]), ex["genuine"]["claim"])
    check("the genuine example the demo would offer is accepted", r["accept"],
          "score %.3f, threshold %.3f" % (r["score"], r["threshold"]))

    # ------------------------------------- what the screen says is right
    from palmprint import stage as S
    section("5b. Does the screen judge its own decisions correctly?")
    probe = next(p for p in eng.probes if p["path"].endswith(
        os.path.join("session2", "00005.bmp")))
    check("the true identity comes from the index, not a folder name",
          S.truth_of(eng, probe["path"]) == "P002",
          "session2/00005.bmp -> %s (palm 2 at 4 per palm)"
          % S.truth_of(eng, probe["path"]))
    tr = eng.process(probe["path"])
    check("probe names keep the session, so the two 00005.bmp differ",
          tr.name == "session2/00005.bmp", tr.name)

    def judged(path, claim):
        t = eng.process(path)
        res = eng.verify(t, claim)
        return res, S.correctness_lines(eng, "verify", res,
                                        S.truth_of(eng, path))

    res, lines = judged(ex["genuine"]["path"], ex["genuine"]["claim"])
    check("a correct genuine accept is called correct",
          res["accept"] and lines == ["correct"], "%s" % lines)
    res, lines = judged(ex["impostor"]["path"], ex["impostor"]["claim"])
    check("a correct impostor reject is called correct",
          not res["accept"] and lines[:1] == ["correct"], "%s" % lines)
    res, lines = judged(ex["genuine"]["path"],
                        next(i for i in eng.identities
                             if i != ex["genuine"]["claim"]))
    check("verdict lines are short enough for the box",
          all(len(x) <= 24 for x in lines), "%s" % lines)
    ident = eng.identify(eng.process(ex["genuine"]["path"]))
    check("identification is judged against the same truth",
          S.correctness_lines(eng, "identify", ident,
                              S.truth_of(eng, ex["genuine"]["path"]))
          == ["correct"], "identified %s" % ident["identity"])

    # ------------------------------------------------- the evaluation used
    section("5c. Is the evaluation the honest, cross-session one?")
    check("the evaluation is cross-session",
          eng.protocol == "cross-session", eng.protocol_detail)
    ss = eng.scoresets["rloc"]
    check("only session-2 probes count: 8 palms x 4 genuine, x 7 impostor",
          ss.n_genuine == 32 and ss.n_impostor == 32 * 7,
          "%d genuine, %d impostor" % (ss.n_genuine, ss.n_impostor))
    check("the same-session numbers are kept separately",
          "rloc" in eng.same_session
          and eng.same_session["rloc"].n_genuine == 8 * 3,
          "%d same-session genuine" % eng.same_session["rloc"].n_genuine)
    check("the genuine example is a later-session capture",
          tongji.session_of(ex["genuine"]["path"]) == "session2",
          ex["genuine"]["path"].replace(tmp, "…"))

    # ------------------------------------------- photographs from outside
    section("5d. Photographs from outside the database")
    rng = np.random.default_rng(3)
    sq = os.path.join(tmp, "my_palm_square.png")
    wide = os.path.join(tmp, "my_hand_wide.jpg")
    Image.fromarray((rng.random((300, 300)) * 255).astype(np.uint8)).save(sq)
    Image.fromarray((rng.random((300, 400)) * 255).astype(np.uint8)).save(wide)
    t_sq, t_wide = eng.process(sq), eng.process(wide)
    check("a square palm crop from outside is accepted as an ROI",
          t_sq.ok and S.truth_of(eng, sq) is None, "no truth claimed for it")
    check("a non-square photograph is declined, not squashed",
          not t_wide.ok and "square" in t_wide.reason, t_wide.reason)

    # --palms: the demo's way of opening a big database quickly
    section("5e. Opening a big database quickly (--palms)")
    from palmprint import engine as E2
    part = E2.Engine(good, cache_dir=os.path.join(tmp, "cache2")).prepare(
        max_palms=3)
    check("--palms keeps only that many people",
          len(part.identities) == 3 and len(part.subset) == 3,
          "%d enrolled of 8" % len(part.identities))
    check("a palm-limited run is cached separately from the full one",
          part.cache_status == "computed"
          and E2.Engine(good, cache_dir=os.path.join(tmp, "cache2")
                        ).prepare().cache_status == "computed",
          "the full set does not reuse the 3-palm cache")
    check("the limited run still evaluates both matchers",
          all(part.summary()[m]["eer"] is not None for m in E2.MATCHERS))

    # -------------------------------------------------------- a real folder
    if real:
        section("6. The real extraction")
        before = len(_failures)
        real = os.path.abspath(real)
        d = tongji.describe(real)
        print("   %s" % d)
        items = tongji.index(real)
        check("the whole release loads", len(items) > 0,
              "%d images, %d palms"
              % (len(items), len({t[0] for t in items})))

        # The ROIs are 8-bit paletted BMPs; a palette misread shows up as a
        # near-constant or saturated image. Real Tongji ROIs are low-contrast
        # (one in 240 spans under 0.2 of the grey range), so the floor is set
        # to catch a failure, not to grade the photography.
        sample = [t[2] for t in items[::max(1, len(items) // 240)]]
        shapes, means, spans = set(), [], []
        for path in sample:
            a = protocol.load_gray(path)
            shapes.add(a.shape)
            means.append(float(a.mean()))
            spans.append(float(a.max() - a.min()))
        check("every ROI is 128 x 128", shapes == {(128, 128)}, "%s" % (shapes,))
        check("the palette converts to usable greyscale",
              min(means) > 0.05 and max(means) < 0.95 and min(spans) > 0.05,
              "mean %.2f..%.2f, range %.2f..%.2f over %d images"
              % (min(means), max(means), min(spans), max(spans), len(sample)))

        res = tongji.pairing_check(items, n_palms=40, pairs_per_palm=2)
        check("the numbering means what we think it means", res["ok"],
              "same palm %.3f, different palm %.3f, gap %+.3f"
              % (res["same_palm"], res["different_palm"], res["gap"]))
        if len(_failures) == before:
            path = tongji.write_manifest(real)
            print("   wrote %s" % path)
            print("   the demo will now open it with:  python demo.py --data "
                  "%s --palms 40" % real)
        else:
            print("   manifest NOT written: fix the failures above first, so "
                  "the demo cannot run on a folder that failed its checks")

    section("Result")
    if _failures:
        print("%d check(s) failed: %s" % (len(_failures), ", ".join(_failures)))
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
