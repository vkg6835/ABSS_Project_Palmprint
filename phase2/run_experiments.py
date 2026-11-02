#!/usr/bin/env python3
"""
run_experiments.py — the Phase II results: every table and figure, from the
images.

    python run_experiments.py                         # Tongji, all palms
    python run_experiments.py --data PATH             # another dataset
    python run_experiments.py --palms 50              # a quick look
    python run_experiments.py --max-seconds 150       # stop and resume later

Writes results/results.md (the tables, ready to read), results/results.json
(every number), and results/figures/*.png. Work is saved as it is done under
results/cache/, so a second run — or a run after an interruption — picks up
where the first stopped, and only redraws what it must.

What it measures (all on cross-session probes unless marked):
  1. RLOC against the Han et al. 2003 baseline: EER, FRR at fixed FARs,
     confusion matrices, rank-1 identification, DET and FAR/FRR curves.
  2. The same against same-session probes: how much the easy case flatters.
  3. Paper 2's Table 4, re-run on our data.
  4. MFRAT against Gabor with everything else identical — the experiment
     our Phase I critique said would settle what MFRAT is worth.
  5. One enrolment template against seven, on device-aligned ROIs.
  6. Paper 1's two feature types, Sobel and morphology.
  7. What each costs: extraction and matching time, template size.
  8. Two details of Paper 2's matcher that our first version left out —
     equation (9)'s maximum over both directions and the +/-2 shift search —
     and what each is worth.

Every EER comes with a 95% interval from a bootstrap over palms, and every
comparison with a paired interval for the difference — because, as we found,
an EER measured on a few dozen people can swing by a factor of five.
"""
import argparse
import datetime
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from palmprint import engine, experiments as X, plots   # noqa: E402

DEFAULT_DATA = os.path.join(HERE, "data", "tongji", "ROI")
DEFAULT_OUT = os.path.join(HERE, "results")

# Paper 2's own figures (PolyU database), as transcribed in our Phase I deck:
# Table 4 and the Competitive Code baseline. A different database and
# protocol, so they are context for the shape of the results, not targets.
PAPER2 = {"exp1": 0.0040, "exp2": 0.0036, "rloc": 0.0016, "exp4": 0.0027,
          "compcode": 0.0047}

# The comparisons reported with paired bootstrap intervals: (a, b, meaning).
PAIRS = [
    ("rloc", "han", "RLOC against the 2003 baseline"),
    ("exp2", "exp1", "adding the enlarged training set (7 templates)"),
    ("rloc", "exp2", "adding the cross neighbourhood"),
    ("exp4", "rloc", "3x3 square instead of the cross"),
    ("exp1", "gab1", "MFRAT instead of Gabor - no tricks"),
    ("exp2", "gab2", "MFRAT instead of Gabor - with 7 templates"),
    ("rloc", "gab3", "MFRAT instead of Gabor - full pipeline"),
    ("gab3", "gab1", "both tricks, applied to the Gabor code"),
    ("rloc", "exp1", "both tricks, applied to the MFRAT code"),
    ("align1", "rloc", "1 template instead of 7"),
    ("han_morph", "han", "Paper 1: morphology instead of Sobel"),
    ("rloc", "oneway", "eq. 9's maximum over both directions"),
    ("rloc", "shift1", "+/-2 shift search instead of +/-1"),
    ("rloc", "first", "the matcher as published, against our first version"),
]


def log(msg):
    print("%s  %s" % (datetime.datetime.now().strftime("%H:%M:%S"), msg),
          flush=True)


def pct(v, d=2):
    return "%.*f%%" % (d, 100.0 * v)


def pp(v):
    """A difference in EER, as signed percentage points."""
    return "%+.2f pt" % (100.0 * v)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--palms", type=int, default=None)
    ap.add_argument("--max-seconds", type=float, default=None,
                    help="stop cleanly after this long; run again to resume")
    ap.add_argument("--boot", type=int, default=1000,
                    help="bootstrap replicates (default 1000)")
    a = ap.parse_args(argv)
    clock = X.Clock(a.max_seconds)          # the budget counts from here

    root = os.path.abspath(a.data)
    man = engine.read_manifest(root)
    layout = man.get("layout", "subdirs")
    per_palm = int(man.get("per_palm", 10))
    name = man.get("name", os.path.basename(root))
    split = X.make_split(root, layout, per_palm, a.palms)
    key = X.cache_key(split, extra="palms=%s" % a.palms)
    store = X.Store(os.path.join(a.out, "cache", key))
    log("%s: %d palms, %d enrolment images, %d probes (%s)"
        % (name, len(split.identities), len(split.enrol_paths),
           len(split.probe_paths),
           ", ".join("%s %d" % (g, np.sum(split.probe_group == g))
                     for g in split.groups)))

    try:
        omegas, tune_table, tune_n = X.tune_gabor(split, store, clock, log)
        log("Gabor wavelengths chosen on session-1 data: %s"
            % ", ".join("%s %.2f" % kv for kv in sorted(omegas.items())))
        scores, secs = X.score_all(split, store, clock, log, omegas)
        main_group = "cross" if "cross" in split.groups else "all"
        W = X.boot_weights(len(split.identities), a.boot)
        res, boot = {}, {}
        for k in scores:
            t0 = time.time()
            res[k], boot[k] = X.metrics_unit(store, clock, k, omegas,
                                             scores[k], split, W, main_group)
            if time.time() - t0 > 1:
                log("  metrics and %d bootstrap replicates: %s (%.0f s)"
                    % (a.boot, X.LABELS[k], time.time() - t0))
    except X.Deadline:
        log("time budget reached - progress is saved; run again to continue")
        return 3

    pairs = []
    for x, y, meaning in PAIRS:
        if x in boot and y in boot:
            dlt = boot[x] - boot[y]
            pairs.append({"a": x, "b": y, "meaning": meaning,
                          "diff": res[x][main_group]["eer"]
                          - res[y][main_group]["eer"],
                          "ci95": X.ci(dlt),
                          "share_a_better": float(np.mean(dlt < 0))})

    log("measuring costs")
    cost = X.costs(split, omegas.get("gab3", 0.5))
    for k, s in secs.items():
        cost["batch_seconds_%s" % k] = s

    out_dir = os.path.abspath(a.out)
    fig_dir = os.path.join(out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    log("drawing figures")
    figs = figures(res, main_group, fig_dir, scores, split)

    rel_root = os.path.relpath(root, HERE).replace("\\", "/")
    summary = {
        "dataset": name,
        "root": rel_root if not rel_root.startswith("..") else root,
        "layout": layout,
        "generated": datetime.datetime.now().isoformat(timespec="seconds"),
        "palms": len(split.identities), "probes": len(split.probe_paths),
        "groups": {g: split.counts(g) for g in split.groups},
        "main_group": main_group, "gabor_omega": omegas,
        "gabor_tuning": [{"omega": o,
                          "session1": {k: {"eer": v[0], "dprime": v[1]}
                                       for k, v in vals.items()}}
                         for o, vals in tune_table],
        "gabor_tuning_probes": tune_n,
        "labels": X.LABELS, "results": res, "pairs": pairs, "costs": cost,
        "paper2_reference": PAPER2, "bootstrap_replicates": a.boot,
        "figures": figs,
    }
    X.write_json(os.path.join(out_dir, "results.json"), summary)
    md = report(summary)
    with open(os.path.join(out_dir, "results.md"), "w", encoding="utf-8") as f:
        f.write(md)
    log("done in %.0f s - results in %s" % (clock.elapsed(), out_dir))
    print()
    print(headline(summary))
    return 0


# ================================================================ figures
# Shorter names for figures, where the tables' full labels would crowd out
# the curves.
FIG_LABELS = {
    "rloc": "RLOC: MFRAT + 7 templates + cross",
    "han": "Han et al. 2003 (Sobel)",
    "han_morph": "Han et al. 2003 (morphology)",
    "exp1": "Exp 1  MFRAT alone",
    "exp2": "Exp 2  + 7 rotated templates",
    "exp4": "Exp 4  square instead of cross",
    "align1": "RLOC with 1 template",
    "gab1": "Gabor alone",
    "gab2": "Gabor + 7 rotated templates",
    "gab3": "Gabor + 7 templates + cross",
    "oneway": "RLOC, one direction only",
    "shift1": "RLOC, +/-1 shift",
    "first": "RLOC as first built (+/-1, one way)",
}


def figures(res, g, fig_dir, scores, split):
    out = {}

    def det(keys_styles, title, fname):
        curves = []
        for k, style in keys_styles:
            if k in res:
                far, frr = res[k][g]["_det"]
                curves.append((FIG_LABELS[k], far, frr, style,
                               res[k][g]["eer"]))
        if curves:
            out[fname] = plots.det_figure(os.path.join(fig_dir, fname),
                                          curves, title)

    det([("rloc", "primary"), ("han", "baseline")],
        "RLOC against Han et al. 2003 - cross-session, all palms",
        "det_main.png")
    det([("exp1", "dot"), ("exp2", "dash"), ("rloc", "primary"),
         ("exp4", "grey_dash")],
        "Paper 2's Table 4, re-run on our data", "det_ablation.png")
    det([("exp1", "dash"), ("gab1", "grey_dash"), ("rloc", "primary"),
         ("gab3", "baseline")],
        "MFRAT against Gabor, alone and with both matching tricks",
        "det_filter.png")
    det([("first", "grey_dash"), ("shift1", "dash"), ("oneway", "dot"),
         ("rloc", "primary")],
        "Paper 2's matcher as published, against our first version",
        "det_matcher.png")
    for k, fname in (("rloc", "far_frr_rloc.png"), ("han", "far_frr_han.png")):
        if k in res:
            t, far, frr = res[k][g]["_farfrr"]
            out[fname] = plots.far_frr_figure(
                os.path.join(fig_dir, fname), t, far, frr, res[k][g]["eer"],
                res[k][g]["threshold"],
                "FAR and FRR against the threshold - %s" % FIG_LABELS[k])
    if "rloc" in res and "han" in res:
        panels = []
        for k in ("rloc", "han"):
            ss = X.scoreset(scores[k], split, g)[0]
            panels.append((FIG_LABELS[k], ss.genuine, ss.impostor,
                           res[k][g]["threshold"]))
        out["hist_main.png"] = plots.hist_figure(
            os.path.join(fig_dir, "hist_main.png"), panels,
            "Score distributions, cross-session")
    order = ["rloc", "oneway", "shift1", "first", "align1", "exp4", "exp2",
             "exp1", "gab3", "gab2", "gab1", "han", "han_morph"]
    rows = []
    for k in order:
        if k in res:
            style = "primary" if k in ("rloc", "exp1", "exp2", "exp4",
                                       "align1", "oneway", "shift1",
                                       "first") else "baseline"
            rows.append((FIG_LABELS[k], res[k][g]["eer"],
                         res[k][g]["eer_ci95"], style))
    if rows:
        out["eer_all.png"] = plots.bar_figure(
            os.path.join(fig_dir, "eer_all.png"), rows,
            "Equal error rate of every configuration, with 95% intervals",
            note="Bootstrap over palms. Dark: MFRAT code. Grey: Gabor code "
                 "and the 2003 baseline.")
    return {k: os.path.basename(v) for k, v in out.items()}


# ================================================================= report
def headline(s):
    g = s["main_group"]
    r = s["results"]
    lines = ["Headline (%s probes):" % g]
    for k in ("rloc", "han"):
        if k in r:
            e = r[k][g]
            lo, hi = e["eer_ci95"]
            lines.append("  %-38s EER %s  [%s - %s]   FRR at FAR 0.1%%: %s"
                         % (s["labels"][k], pct(e["eer"]), pct(lo), pct(hi),
                            pct(e["frr_at_far_0.001"])))
    return "\n".join(lines)


def _row(s, k, g, paper=None):
    e = s["results"][k][g]
    lo, hi = e.get("eer_ci95", (float("nan"), float("nan")))
    cells = [s["labels"][k], pct(e["eer"]),
             ("%s - %s" % (pct(lo), pct(hi))) if "eer_ci95" in e else "",
             "%.2f" % e["dprime"], pct(e["frr_at_far_0.01"]),
             pct(e["frr_at_far_0.001"]), pct(e["frr_at_far_0.0001"]),
             pct(e["rank1"])]
    if paper is not None:
        cells.append(pct(paper) if paper == paper else "-")
    return "| " + " | ".join(cells) + " |"


HEAD = ("| configuration | EER | 95% interval | d' | FRR @ FAR 1% "
        "| FRR @ FAR 0.1% | FRR @ FAR 0.01% | rank-1 |")
RULE = "|---|---|---|---|---|---|---|---|"


def report(s):
    g = s["main_group"]
    r = s["results"]
    L = []
    w = L.append
    w("# Phase II results - %s" % s["dataset"])
    w("")
    w("Generated %s by `run_experiments.py`. Every number below is "
      "recomputed from the images; nothing is copied from a paper except "
      "where a column says so." % s["generated"].replace("T", " "))
    w("")
    w("## Protocol")
    w("")
    w("%d palms. Each is enrolled from one image, its first session-1 "
      "capture; every other image of it is a probe and is compared with every "
      "enrolled palm." % s["palms"])
    w("")
    w("| probe group | probes | genuine comparisons | impostor comparisons |")
    w("|---|---|---|---|")
    for grp, c in s["groups"].items():
        w("| %s | %s | %s | %s |" % (grp, format(c["probes"], ","),
                                      format(c["genuine"], ","),
                                      format(c["impostor"], ",")))
    w("")
    w("**cross** = taken in a different session from the enrolment; "
      "**same** = taken in the same sitting. Every headline figure uses "
      "%s probes. The 95%% intervals come from %d bootstrap replicates "
      "that resample palms, shared by every configuration so that "
      "differences are paired." % (g, s["bootstrap_replicates"]))
    w("")

    # 1 ---------------------------------------------------------------
    w("## 1. RLOC against the 2003 baseline")
    w("")
    w(HEAD)
    w(RULE)
    for k in ("rloc", "han"):
        if k in r:
            w(_row(s, k, g))
    w("")
    for p in s["pairs"]:
        if (p["a"], p["b"]) == ("rloc", "han"):
            w("Difference in EER: %s, 95%% interval %s to %s; RLOC is better "
              "in %.1f%% of bootstrap replicates."
              % (pp(p["diff"]), pp(p["ci95"][0]), pp(p["ci95"][1]),
                 100 * p["share_a_better"]))
            w("")
    w("### Confusion matrices at the equal-error threshold")
    w("")
    for k in ("rloc", "han"):
        if k not in r:
            continue
        c = r[k][g]["confusion_eer"]
        w("**%s** - threshold %.3f" % (s["labels"][k], c["threshold"]))
        w("")
        w("| truth \\ decision | accept | reject | rate |")
        w("|---|---|---|---|")
        w("| genuine (%s) | %s | %s | FRR %s |"
          % (format(c["genuine_accepted"] + c["genuine_rejected"], ","),
             format(c["genuine_accepted"], ","),
             format(c["genuine_rejected"], ","), pct(c["frr"], 3)))
        w("| impostor (%s) | %s | %s | FAR %s |"
          % (format(c["impostor_accepted"] + c["impostor_rejected"], ","),
             format(c["impostor_accepted"], ","),
             format(c["impostor_rejected"], ","), pct(c["far"], 3)))
        w("")
    w("### Where the errors come from")
    w("")
    w("| configuration | false rejects at the EER threshold | palms they "
      "come from (of %d) | worst palms |" % s["palms"])
    w("|---|---|---|---|")
    for k in ("rloc", "han"):
        if k in r:
            e = r[k][g]
            w("| %s | %s | %d | %s |"
              % (s["labels"][k], format(e["false_rejects"], ","),
                 e["palms_with_false_reject"],
                 ", ".join("%s (%d)" % (p, n) for p, n in e["worst_palms"])))
    w("")
    w("When a method's few errors sit on a handful of palms, its EER moves a "
      "lot with which palms happen to be in the test - which is why RLOC's "
      "interval is wide relative to its EER.")
    w("")
    w("### At a stricter operating point: FAR = 0.1%")
    w("")
    w("| configuration | threshold | genuine rejected | impostors accepted |")
    w("|---|---|---|---|")
    for k in ("rloc", "han"):
        if k in r:
            c = r[k][g]["confusion_far_0.001"]
            w("| %s | %.3f | %s of %s (%s) | %s of %s (%s) |"
              % (s["labels"][k], c["threshold"],
                 format(c["genuine_rejected"], ","),
                 format(c["genuine_accepted"] + c["genuine_rejected"], ","),
                 pct(c["frr"], 2), format(c["impostor_accepted"], ","),
                 format(c["impostor_accepted"] + c["impostor_rejected"], ","),
                 pct(c["far"], 3)))
    w("")

    # 2 ---------------------------------------------------------------
    if "same" in s["groups"]:
        w("## 2. Same session against cross-session")
        w("")
        w("| configuration | EER, same session | EER, cross-session |")
        w("|---|---|---|")
        for k in ("rloc", "han", "exp1", "gab3"):
            if k in r:
                w("| %s | %s | %s |" % (s["labels"][k],
                                        pct(r[k]["same"]["eer"]),
                                        pct(r[k]["cross"]["eer"])))
        w("")
        w("Probes taken in the enrolment's own sitting are far easier. "
          "Reporting them, or mixing them in, flatters every method.")
        w("")

    def pair_lines(keys):
        for p in s["pairs"]:
            if (p["a"], p["b"]) in keys:
                w("- **%s**: %s (95%% interval %s to %s; better in %.1f%% of "
                  "replicates)" % (p["meaning"], pp(p["diff"]),
                                   pp(p["ci95"][0]), pp(p["ci95"][1]),
                                   100 * p["share_a_better"]))
        w("")

    # 3 ---------------------------------------------------------------
    w("## 3. Paper 2's Table 4, re-run")
    w("")
    w(HEAD.rstrip(" |") + " | Paper 2 (PolyU) |")
    w(RULE + "---|")
    for k in ("exp1", "exp2", "rloc", "exp4"):
        if k in r:
            w(_row(s, k, g, PAPER2.get(k, float("nan"))))
    w("")
    pair_lines([("exp2", "exp1"), ("rloc", "exp2"), ("exp4", "rloc")])

    # 4 ---------------------------------------------------------------
    w("## 4. MFRAT against Gabor, everything else identical")
    w("")
    w("The Gabor code is a Competitive-Code-style filter bank feeding exactly "
      "the same lattice, rotated templates, matching rule and scorer as "
      "MFRAT; the filter is the only difference. Each Gabor configuration's "
      "wavelength was chosen with that configuration's own matching, on "
      "session-1 data only (%d probes) - never on the session-2 data "
      "reported here. MFRAT's parameters are its paper's, untuned, so if "
      "anything the Gabor arm is favoured." % s["gabor_tuning_probes"])
    w("")
    stages = [("gab1", "pixel, 1 template"), ("gab2", "pixel, 7 templates"),
              ("gab3", "cross, 7 templates")]
    w("| Gabor omega | " + " | ".join("session-1 EER (d'), %s" % lab
                                       for _, lab in stages) + " |")
    w("|---|" + "---|" * len(stages))
    for t in s["gabor_tuning"]:
        cells = []
        for k, _ in stages:
            v = t["session1"][k]
            mark = " **chosen**" if t["omega"] == s["gabor_omega"][k] else ""
            cells.append("%s (%.2f)%s" % (pct(v["eer"], 3), v["dprime"], mark))
        w("| %.2f | %s |" % (t["omega"], " | ".join(cells)))
    w("")
    w("Lowest session-1 EER wins; ties go to the larger d'.")
    w("")
    w(HEAD)
    w(RULE)
    for k in ("exp1", "gab1", "exp2", "gab2", "rloc", "gab3"):
        if k in r:
            w(_row(s, k, g))
    w("")
    pair_lines([("exp1", "gab1"), ("exp2", "gab2"), ("rloc", "gab3"),
                ("gab3", "gab1"), ("rloc", "exp1")])

    # 5 ---------------------------------------------------------------
    w("## 5. One enrolment template against seven")
    w("")
    w(HEAD)
    w(RULE)
    for k in ("rloc", "align1", "exp2", "exp1"):
        if k in r:
            w(_row(s, k, g))
    w("")
    pair_lines([("align1", "rloc"), ("exp2", "exp1")])

    # 6 ---------------------------------------------------------------
    w("## 6. Paper 1's two feature types")
    w("")
    w(HEAD)
    w(RULE)
    for k in ("han", "han_morph"):
        if k in r:
            w(_row(s, k, g))
    w("")
    pair_lines([("han_morph", "han")])

    # 7 ---------------------------------------------------------------
    c = s["costs"]
    w("## 7. What it costs (this computer)")
    w("")
    w("| step | time |")
    w("|---|---|")
    w("| MFRAT code, one ROI | %.1f ms |" % c["mfrat_ms"])
    w("| Gabor code, one ROI (FFT, omega %.2f) | %.1f ms |"
      % (c["gabor_omega"], c["gabor_ms"]))
    w("| MFRAT enrolment, 7 rotated codes | %.1f ms |" % c["enrol7_ms"])
    w("| Paper 1 Sobel features, one ROI | %.1f ms |" % c["sobel_ms"])
    w("| Paper 1 morphology features, one ROI | %.1f ms |" % c["morph_ms"])
    w("| RLOC match, one probe against one palm (7 templates) | %.2f ms |"
      % c["match_one_palm_ms"])
    w("| Han match, one probe against one palm | %.3f ms |"
      % c["han_match_ms"])
    if "batch_seconds_rloc" in c:
        # stored with the cached scores, so measured on whichever computer
        # first computed them — not necessarily this one
        w("| RLOC, every probe against every palm, batched (timed when the "
          "scores were first computed) | %.0f s |" % c["batch_seconds_rloc"])
    w("")
    w("Template size: RLOC %d bytes per palm (7 codes of 32 x 32 values from "
      "6 directions, %d bytes each if packed); Han %d bytes (64 floats)."
      % (c["rloc_template_bytes"], c["rloc_template_bytes_one"],
         c["han_template_bytes"]))
    w("")

    # 8 ---------------------------------------------------------------
    if "first" in r:
        w("## 8. The matcher as published, against our first version")
        w("")
        w("Paper 2 scores a pair as the larger of s(probe, template) and "
          "s(template, probe) - equation (9) - and searches +/-2 code pixels "
          "of translation. Our first version kept one direction and searched "
          "+/-1, to save matching time. Every other configuration in this "
          "report matches as the paper does; these rows show what the two "
          "details are worth.")
        w("")
        w(HEAD)
        w(RULE)
        for k in ("rloc", "oneway", "shift1", "first"):
            if k in r:
                w(_row(s, k, g))
        w("")
        pair_lines([("rloc", "oneway"), ("rloc", "shift1"),
                    ("rloc", "first")])

    w("## Figures")
    w("")
    for k, v in s["figures"].items():
        w("- `figures/%s`" % v)
    w("")
    return "\n".join(L)


if __name__ == "__main__":
    sys.exit(main())
