#!/usr/bin/env python3
"""
test_experiments.py — the experiment pipeline, checked on a small set.

    python test_experiments.py

Writes a small Tongji-shaped dataset of synthetic palms to a temporary
folder, runs run_experiments.py on it — including being interrupted by its
time budget and resumed — and checks that the numbers it reports come from
exactly the matchers the demo uses. Takes about a minute. Everything is
deleted afterwards.
"""
import json
import os
import shutil
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import run_experiments                                       # noqa: E402
import test_tongji                                           # noqa: E402
from palmprint import experiments as X                       # noqa: E402
from palmprint import gabor, han, metrics, rloc, tongji      # noqa: E402

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


def main():
    print("AI461 Phase II - experiment pipeline checks")
    print("python %s, numpy %s" % (sys.version.split()[0], np.__version__))
    tmp = tempfile.mkdtemp(prefix="palm_exp_test_")
    try:
        return run(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def run(tmp):
    data = os.path.join(tmp, "data")
    out = os.path.join(tmp, "results")
    test_tongji.write_tongji(data, palms=10, per_palm=4)
    tongji.write_manifest(data, per_palm=4)
    devnull = open(os.devnull, "w")

    # --------------------------------------------------------- the runner
    section("1. The runner, interrupted and resumed")
    old = sys.stdout
    sys.stdout = devnull
    try:
        rc1 = run_experiments.main(["--data", data, "--out", out,
                                    "--boot", "100", "--max-seconds", "0.01"])
        rc2 = run_experiments.main(["--data", data, "--out", out,
                                    "--boot", "100"])
    finally:
        sys.stdout = old
    check("a run out of time stops cleanly and says so", rc1 == 3,
          "exit code %s" % rc1)
    check("the next run picks up and finishes", rc2 == 0,
          "exit code %s" % rc2)
    with open(os.path.join(out, "results.json"), encoding="utf-8") as f:
        res = json.load(f)
    figs = sorted(os.listdir(os.path.join(out, "figures")))
    check("the report, the numbers and eight figures are written",
          os.path.isfile(os.path.join(out, "results.md")) and len(figs) == 8,
          ", ".join(figs))
    check("every configuration is evaluated",
          set(res["results"]) == set(X.LABELS),
          "%d configurations" % len(res["results"]))
    check("cross-session is the headline group",
          res["main_group"] == "cross"
          and res["groups"]["cross"]["genuine"] == 10 * 4,
          "%s probes" % res["groups"]["cross"]["genuine"])
    check("every EER has a 95% interval that contains it",
          all(r["cross"]["eer_ci95"][0] - 1e-9 <= r["cross"]["eer"]
              <= r["cross"]["eer_ci95"][1] + 1e-9
              for r in res["results"].values()))
    check("the error table and the confusion matrix count the same rejects",
          all(r["cross"]["false_rejects"]
              == r["cross"]["confusion_eer"]["genuine_rejected"]
              for r in res["results"].values()))
    check("each Gabor configuration got its own tuned wavelength",
          set(res["gabor_omega"]) == {"gab1", "gab2", "gab3"},
          ", ".join("%s %.2f" % kv for kv in sorted(res["gabor_omega"].items())))

    # ------------------------------------------- same numbers as the demo
    section("2. The numbers come from the demo's own matchers")
    split = X.make_split(data, "tongji", 4)
    probes = [rloc.encode(X.protocol.load_gray(p, 128)) for p in
              split.probe_paths]
    probes = np.stack(probes)
    enrol = np.stack([rloc.enrol_codes(X.protocol.load_gray(p, 128))
                      for p in split.enrol_paths])
    for key, nb, nt in (("rloc", "cross", 7), ("exp1", "pixel", 1),
                        ("exp4", "square", 7), ("first", "cross", 7)):
        cfg = rloc.RlocConfig(neighbourhood=nb, rotations=X.ROTATIONS[:nt],
                              **X.CFG_EXTRA.get(key, {}))
        S = X.rloc_scores(probes, enrol, nt, cfg)
        g = rloc.Gallery(cfg)
        for ident, codes in zip(split.identities, enrol):
            g.enrol_codes(ident, codes[:nt])
        ref = np.stack([g.scores(c) for c in probes]).astype(np.float32)
        check("%s: batched scores equal the demo's Gallery, bit for bit"
              % key, np.array_equal(S, ref))
    check("the published matcher is the default the demo uses",
          rloc.RlocConfig().shift_range == 2 and rloc.RlocConfig().symmetric)
    stored = res["results"]
    check("both details of the published matcher are measured",
          {"oneway", "shift1", "first"} <= set(stored))
    vecs_p = np.stack([han.features(X.protocol.load_gray(p, 128))
                       for p in split.probe_paths])
    vecs_e = np.stack([han.features(X.protocol.load_gray(p, 128))
                       for p in split.enrol_paths])
    hg = han.HanGallery()
    for ident, v in zip(split.identities, vecs_e):
        hg.enrol_vector(ident, v)
    href = np.stack([hg.scores(v) for v in vecs_p])
    check("Han scores equal the demo's HanGallery",
          np.allclose(X.han_scores(vecs_p, vecs_e), href, atol=1e-6))

    # ------------------------------------------------------- the statistics
    section("3. The statistics")
    rng = np.random.default_rng(1)
    gen = rng.normal(0.8, 0.05, 300)
    imp = rng.normal(0.55, 0.05, 30000)
    ss = metrics.ScoreSet(gen, imp)
    far, frr, _ = ss.det_rates(400)
    check("the DET curve reaches down to a single impostor",
          far[far > 0].min() <= 2.0 / imp.size,
          "lowest FAR plotted %.1e" % far[far > 0].min())
    check("and down to a single genuine attempt",
          frr[frr > 0].min() <= 2.0 / gen.size,
          "lowest FRR plotted %.1e" % frr[frr > 0].min())
    # binned EER, every palm once, against the exact EER on a fake split
    n_pal = 30
    owner = np.repeat(np.arange(n_pal), 10)
    S = rng.normal(0.55, 0.05, (owner.size, n_pal))
    S[np.arange(owner.size), owner] = rng.normal(0.75, 0.06, owner.size)

    class FakeSplit(object):
        identities = list(range(n_pal))
        probe_owner = owner
        probe_group = np.array(["cross"] * owner.size)
    fs = FakeSplit()
    exact = X.scoreset(S, fs, "cross")[0].eer()[0]
    binned = X.bootstrap_eers(S, fs, "cross", np.ones((1, n_pal)))[0]
    check("the bootstrap's binned EER matches the exact one",
          abs(exact - binned) < 2e-4,
          "exact %.4f%%, binned %.4f%%" % (100 * exact, 100 * binned))
    W = X.boot_weights(n_pal, 300)
    check("bootstrap weights resample palms, n draws each time",
          np.all(W.sum(axis=1) == n_pal))

    # ---------------------------------------------------------- the filter
    section("4. The Gabor filter bank")
    img = rng.random((128, 128))
    om = 0.4
    k = gabor.kernel(om, np.pi / 3)
    r = k.shape[0] // 2
    pad = np.pad(img, r, mode="edge")
    direct = np.sum(pad[10:10 + 2 * r + 1, 20:20 + 2 * r + 1] * k)
    check("FFT filtering equals the direct sum",
          abs(gabor.responses(img, om)[2][10, 20] - direct) < 1e-10)
    c0 = gabor.encode(img, om)
    check("the Gabor code is exactly invariant to brightness and contrast",
          all(np.array_equal(c0, gabor.encode(a * img + b, om))
              for a, b in ((0.5, 0.2), (1.7, -0.3))))
    check("it has the same shape as the MFRAT code",
          c0.shape == rloc.encode(img).shape)
    devnull.close()

    section("Result")
    if _failures:
        print("%d check(s) failed: %s" % (len(_failures), ", ".join(_failures)))
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
