"""
stage.py — what the audience sees, composed as a single image.

The demo window is a column of ordinary controls beside one picture, and
this module draws the picture: four pipeline panels that are revealed one
step at a time, a caption under each saying what that step did, and a
decision row with the verdict and where the score fell.

Drawing the whole view with Pillow, rather than as dozens of Tk widgets, is a
deliberate robustness choice. It means the exact image the class will see
can be rendered and checked on any machine without opening a window
(test_demo.py does this at several projector resolutions), it looks identical
on every platform, and the window code that remains is small.

  build(...)    describes the view for the current state: titles, captions,
                and a drawing function per panel
  compose(...)  lays that description out at a given pixel size
"""
from PIL import Image, ImageDraw

from . import render as R

N_STEPS = 5

MATCHER_SHORT = {"rloc": "RLOC (2008)", "han": "Han et al. (2003)"}


def truth_of(eng, path):
    """
    The true identity of a probe file, when it can be known — from the
    dataset's own index (Engine.identity_of), never from a folder name.
    Files the index does not cover — a phone photograph, the visitors
    folder — have no known truth and are shown without one.

    (The first version took the first folder under the dataset as the
    person. That is right on the synthetic set and wrong on Tongji, where it
    is `session1`: every correct accept was labelled FALSE ACCEPT.)
    """
    return eng.identity_of(path) if path else None


def max_step(trace):
    if trace is None:
        return 0
    return N_STEPS if trace.ok else 2


def correctness_lines(eng, task, result, truth, source=None):
    """
    Whether the decision was right — when we know — as at most two short
    lines for the verdict box: the judgement, then who it really was. Kept
    short on purpose: a long single line wraps, and the verdict box is the
    one place where losing the end of a sentence changes its meaning.

    `source` is the probe's file. If it is the very image the matched
    person was enrolled from, a correct accept says so: its score of 1.000
    is guaranteed by construction, and an audience should not take it for
    evidence.
    """
    if truth is None or result is None or result.get("score") is None:
        return []
    own = eng.enrolled_from(source) if source else None
    if task == "verify":
        genuine = truth == result["claimed"]
        if result["accept"]:
            if not genuine:
                return ["FALSE ACCEPT", "it is %s" % truth]
            return (["correct", "its enrolment image"]
                    if own == result["claimed"] else ["correct"])
        return ["FALSE REJECT"] if genuine else ["correct",
                                                 "it is %s" % truth]
    enrolled = truth in eng.identities
    if result["accept"]:
        if result["identity"] == truth:
            return (["correct", "its enrolment image"]
                    if own == truth else ["correct"])
        return (["WRONG PERSON", "it is %s" % truth] if enrolled
                else ["FALSE ACCEPT", "%s is not enrolled" % truth])
    return (["MISSED", "%s is enrolled" % truth] if enrolled
            else ["correct", "%s is not enrolled" % truth])


def correctness(eng, task, result, truth, source=None):
    """The same judgement as one line, for reports."""
    lines = correctness_lines(eng, task, result, truth, source)
    return " - ".join(lines) if lines else None


def build(eng, trace, task, matcher, claim, step, result, truth=None,
          note=None):
    """
    Describe the view. `step` counts how many stages are revealed (1-5);
    `note`, if given, replaces the decision row with a card of text — used
    after a live enrolment, when there is no decision to show.
    """
    rl = matcher == "rloc"
    titles = ["1  Input", "2  Region of interest",
              "3  " + ("Orientation code" if rl else "Line strength"),
              "4  " + ("Matching" if rl else "Pooled features")]
    spec = {"panels": [], "titles": titles}

    # ---------------------------------------------------- nothing loaded yet
    if trace is None:
        spec["headline"] = "No probe loaded"
        spec["step_label"] = ""
        for k, ttl in enumerate(titles):
            spec["panels"].append({
                "draw": (lambda P, k=k, t=ttl: R.placeholder(
                    P, k + 1, t.split("  ", 1)[1])),
                "caption": "", "active": False})
        lines = note or [
            "Choose a probe to begin",
            "G  genuine attempt   ·   I  impostor   ·   C  closest impostor"
            "   ·   U  unreadable image",
            "O  open any image   ·   E  enrol a new person",
            "Space  next step   ·   T  verify or identify   ·   M  switch "
            "matcher   ·   H  hide the controls",
        ]
        spec["decision"] = {"kind": "pending", "big": "", "lines": [],
                            "wide": True,
                            "plot": lambda w, h, tp, L=lines: R.text_card(
                                L, w, h, text_px=tp),
                            "plot_caption": ""}
        return spec

    # -------------------------------------------------------- the headline
    if note is not None:
        head = "Enrolment image  %s" % trace.name
    else:
        head = "Probe  %s" % trace.name
        if truth:
            head += "   ·   true identity %s" % truth
        if task == "verify":
            head += "   ·   claims to be %s" % claim
        else:
            head += "   ·   who is this?"
    head += "   ·   %s" % MATCHER_SHORT[matcher]
    spec["headline"] = head
    top = 2 if note is not None else max_step(trace)
    spec["step_label"] = "step %d of %d" % (step, top)
    # "press Space" only where Space actually leads somewhere
    hint = "press Space" if (trace.ok and note is None) else ""

    info = trace.info
    is_roi_db = eng.kind == "roi"
    from_db = truth is not None or eng.identity_of(trace.source) is not None

    # ------------------------------------------------------- panel 1: input
    if is_roi_db and trace.image is not None:
        # letterboxed, never stretched: a declined non-square photograph
        # must be shown as it really is
        draw1 = lambda P: R.hand_panel(trace, P, overlays=False)
        w_in, h_in = trace.image.shape[1], trace.image.shape[0]
        if from_db:
            cap1 = "Pre-cropped by the database: %d × %d." % (w_in, h_in)
        else:
            cap1 = ("An image from outside the database, %d × %d — used as "
                    "a crop of the palm centre." % (w_in, h_in))
    else:
        draw1 = lambda P: R.hand_panel(trace, P, overlays=step >= 2)
        if trace.image is None:
            cap1 = "The file could not be read as an image."
        elif step < 2:
            cap1 = ("As captured: %d × %d pixels."
                    % (trace.image.shape[1], trace.image.shape[0]))
        elif trace.ok:
            cap1 = ("Otsu threshold, outline, finger valleys, then the ROI "
                    "frame. Hand turned %+.1f°." % trace.rotation_deg)
        else:
            cap1 = "No usable pair of finger valleys on this outline."
    spec["panels"].append({"draw": draw1, "caption": cap1, "active": True})

    # --------------------------------------------------------- panel 2: ROI
    if step < 2:
        draw2 = lambda P: R.placeholder(P, 2, "Region of interest", hint)
        cap2 = ""
    elif trace.ok:
        draw2 = lambda P: R.roi_panel(trace.roi, P)
        if is_roi_db or not info:
            h_in, w_in = trace.image.shape
            if (h_in, w_in) == trace.roi.shape:
                cap2 = ("Used exactly as supplied: already %d × %d, so "
                        "nothing of ours touches it." % (w_in, h_in))
            else:
                cap2 = ("Resized from %d × %d to %d × %d for the feature "
                        "extractor." % (w_in, h_in, trace.roi.shape[1],
                                        trace.roi.shape[0]))
        else:
            cap2 = ("128 × 128, scale-normalised: the side is 1.35 × the "
                    "valley spacing — %d px in this image." % info["side"])
    else:
        draw2 = lambda P: R.declined_panel(P, trace.reason)
        cap2 = "A wrong ROI is worse than none, so nothing is guessed."
    spec["panels"].append({"draw": draw2, "caption": cap2,
                           "active": step >= 2})

    # ----------------------------------------------- panels 3, 4 and decision
    detail = (result or {}).get("detail") or {}
    if task == "verify":
        against = claim
    else:
        against = (result or {}).get("candidate")

    if step < 3 or not trace.ok:
        draw3 = lambda P: R.placeholder(P, 3, titles[2].split("  ", 1)[1],
                                        hint)
        cap3 = ""
    elif rl:
        draw3 = lambda P: R.code_panel(trace.code, trace.contrast, P,
                                       eng.cfg.n_dir)
        cap3 = ("MFRAT sums along %d directions; the argmin is stored. "
                "32 × 32 integers. Faint strokes: no clear winner."
                % eng.cfg.n_dir)
    else:
        draw3 = lambda P: R.sobel_panel(trace.sobel, P)
        cap3 = ("Equation 1: maximum over four Sobel masks. Keeps how strong "
                "a line is, discards which way it runs.")
    spec["panels"].append({"draw": draw3, "caption": cap3,
                           "active": step >= 3 and trace.ok})

    if step < 4 or not trace.ok or not detail:
        draw4 = lambda P: R.placeholder(P, 4, titles[3].split("  ", 1)[1],
                                        hint)
        cap4 = ""
    elif rl:
        hm = detail["hit_map"]
        draw4 = lambda P: R.match_panel(hm, P)
        if detail.get("direction", "probe") == "probe":
            found = "probe cell's direction found in %s's template" % against
        else:
            found = "%s's template cell found in the probe" % against
        cap4 = ("Dark = %s (cross neighbourhood). Best of %d rotations: "
                "%+.0f°, shift (%+d, %+d)."
                % (found, len(eng.cfg.rotations), detail["rotation"],
                   detail["shift"][0], detail["shift"][1]))
    else:
        pg, tg = detail["probe_grid"], detail["template_grid"]
        draw4 = lambda P: R.grid_panel(pg, tg, P)
        cap4 = ("8 × 8 cell means against %s — the lines averaged away — "
                "compared by Pearson correlation." % against)
    spec["panels"].append({"draw": draw4, "caption": cap4,
                           "active": step >= 4 and trace.ok})

    spec["decision"] = _decision(eng, trace, task, matcher, step, result,
                                 truth, note)
    return spec


def _decision(eng, trace, task, matcher, step, result, truth, note):
    """
    The decision row. Every "plot" here is a function of (width, height,
    text_px): the stage passes one text size to all of them, so labels in the
    plot, the verdict and the captions stay in proportion at any window shape.
    """
    ss = eng.scoresets.get(matcher)
    thr = eng.thresholds.get(matcher)
    if ss is not None and eng.protocol == "cross-session":
        population = ("Cross-session comparisons only (enrolled in one "
                      "session, tested in another): %s genuine (dark), %s "
                      "impostor (light)"
                      % (format(ss.n_genuine, ","), format(ss.n_impostor, ",")))
    elif ss is not None:
        population = ("Every comparison in this dataset: %s genuine (dark) "
                      "and %s impostor (light)"
                      % (format(ss.n_genuine, ","), format(ss.n_impostor, ",")))
    else:
        population = ""

    if note is not None:
        return {"kind": "pending", "big": "", "lines": [], "wide": True,
                "plot": lambda w, h, tp: R.text_card(note, w, h, text_px=tp),
                "plot_caption": ""}

    if not trace.ok and step >= 2:
        def plot(w, h, tp):
            if ss is None:
                return R.blank(w, h, "no evaluation available", text_px=tp)
            return R.score_plot(ss.genuine, ss.impostor, None, thr, w, h,
                                text_px=tp)
        if population:
            reference = population[0].lower() + population[1:]
            cap = ("No score: the image never reached the matcher. Shown "
                   "for reference — %s." % reference)
        else:
            cap = "No score: the image never reached the matcher."
        return {"kind": "declined", "big": "DECLINED",
                "lines": ["no ROI could be found", "nothing is guessed"],
                "plot": plot, "plot_caption": cap}

    if step < N_STEPS or result is None:
        return {"kind": "pending", "big": "",
                "lines": ["step 5: the decision"],
                "plot": lambda w, h, tp: R.blank(
                    w, h, "step 5: where this score falls", text_px=tp),
                "plot_caption": ""}

    score = result["score"]
    right = correctness_lines(eng, task, result, truth,
                              trace.source if trace is not None else None)
    if task == "verify":
        # one fact per line, short enough not to wrap in the box
        lines = ["score  %.3f" % score, "threshold  %.3f" % thr] + right

        def plot(w, h, tp):
            if ss is None:
                return R.blank(w, h, "no evaluation available", text_px=tp)
            return R.score_plot(ss.genuine, ss.impostor, score, thr, w, h,
                                text_px=tp)
        if ss is not None:
            cap = ("%s, each histogram scaled to its own height. Threshold: "
                   "their equal-error point." % population)
        else:
            cap = ""
        return {"kind": "accept" if result["accept"] else "reject",
                "big": "ACCEPT" if result["accept"] else "REJECT",
                "lines": lines, "plot": plot, "plot_caption": cap}

    ranked = result["ranked"]
    # the threshold is drawn on the bar chart beside this, so the box keeps
    # to who, how well, and whether that was right
    if result["accept"]:
        lines = [result["identity"], "score  %.3f" % score]
    else:
        lines = ["best was %s" % result["candidate"], "score  %.3f" % score]
    lines += right
    return {"kind": "accept" if result["accept"] else "reject",
            "big": "IDENTIFIED" if result["accept"] else "NO MATCH",
            "lines": lines,
            "plot": lambda w, h, tp: R.ranked_bars(ranked, thr, truth, w, h,
                                                   text_px=tp),
            "plot_caption": ("The %d best of %d enrolled palms by %s score. "
                             "The dashed line is the same threshold used "
                             "for verification."
                             % (len(ranked), len(eng.identities),
                                MATCHER_SHORT[matcher]))}


# =================================================================== layout
def layout(width, height):
    """
    Every position on the stage, from its pixel size alone. Separate from
    compose() so a test can assert that nothing overlaps or leaves the frame.
    """
    W, H = int(width), int(height)
    b = int(max(12, min(26, round(min(H / 36.0, W / 70.0)))))
    m = g = b
    head_h = int(round(b * 1.8))
    title_h = int(round(b * 1.5))
    cap_fs = b * 0.80
    cap_lh = int(round(cap_fs * 1.32))
    cap_h = 3 * cap_lh + int(round(b * 0.35))
    pc_h = 2 * cap_lh + int(round(b * 0.3))

    P_w = (W - 2 * m - 3 * g) / 4.0
    avail = H - 2 * m - head_h - title_h - cap_h - g
    # The decision row must hold a verdict box of at least 9 text lines'
    # height — the big word plus three lines — whatever the window shape.
    # Without that floor a wide, short window (controls hidden on a laptop)
    # gives every pixel to the panels and the verdict silently loses lines.
    P_h = min((avail - pc_h) / 1.40, avail - pc_h - 9 * b)
    P = int(max(48, min(P_w, P_h)))
    # When the panels are limited by width (a narrow window), some of the
    # height they leave unused goes to the decision row — but only up to a
    # plot 1.3 panels tall. Beyond that the plot stops reading as part of
    # the same picture, and plain space at the bottom is the better use.
    D = int(max(0, min(avail - P, 1.3 * P + pc_h)))

    block_w = 4 * P + 3 * g
    x0 = int((W - block_w) / 2)
    y_title = m + head_h
    y_panel = y_title + title_h
    y_cap = y_panel + P + int(round(b * 0.35))
    y_dec = y_panel + P + cap_h + g
    plot_w = 3 * P + 2 * g
    box_h = max(1, D - pc_h)
    return {"W": W, "H": H, "b": b, "m": m, "g": g, "P": P,
            "x0": x0, "block_w": block_w, "head_y": m, "head_h": head_h,
            "title_y": y_title, "title_h": title_h,
            "panel_y": y_panel, "cap_y": y_cap, "cap_fs": cap_fs,
            "cap_lh": cap_lh, "cap_h": cap_h,
            "dec_y": y_dec, "D": D, "plot_w": plot_w, "plot_h": box_h,
            "box_x": x0 + 3 * (P + g), "box_w": P, "box_h": box_h,
            "pcap_y": y_dec + box_h + int(round(b * 0.3)), "pc_h": pc_h,
            "panel_x": [x0 + k * (P + g) for k in range(4)]}


def text_px(L):
    """The one text size of the decision row — plot labels, verdict lines,
    notes — derived from the stage's base size so it scales with the rest."""
    return max(10, int(round(L["b"] * 0.92)))


def compose(spec, width, height):
    """Draw the described view at exactly width x height pixels."""
    L = layout(width, height)
    W, H, b, P = L["W"], L["H"], L["b"], L["P"]
    im = Image.new("RGB", (W, H), R.WHITE)
    d = ImageDraw.Draw(im)
    x0 = L["x0"]

    # headline, with the step counter at the right
    f_step = R.font(b * 0.9)
    step = spec.get("step_label", "")
    sw = d.textlength(step, font=f_step) if step else 0
    f_head = R.font(b * 1.0, True)
    head = R.wrap_lines(d, spec["headline"], f_head,
                        L["block_w"] - sw - 2 * b, max_lines=1)[0]
    d.text((x0, L["head_y"]), head, font=f_head, fill=R.INK)
    if step:
        d.text((x0 + L["block_w"] - sw, L["head_y"] + b * 0.08), step,
               font=f_step, fill=R.MUTED)
    d.line([(x0, L["head_y"] + L["head_h"] - b * 0.45),
            (x0 + L["block_w"], L["head_y"] + L["head_h"] - b * 0.45)],
           fill=R.RULE, width=1)

    # the four panels
    f_title = R.font(b * 0.95, True)
    f_cap = R.font(L["cap_fs"])
    for k, pan in enumerate(spec["panels"]):
        x = L["panel_x"][k]
        title = R.wrap_lines(d, spec["titles"][k], f_title, P,
                             max_lines=1)[0]
        d.text((x, L["title_y"]), title, font=f_title,
               fill=R.INK if pan["active"] else R.FAINT)
        im.paste(pan["draw"](P), (x, L["panel_y"]))
        y = L["cap_y"]
        for line in R.wrap_lines(d, pan["caption"], f_cap, P, max_lines=3):
            d.text((x, y), line, font=f_cap, fill=R.SLATE)
            y += L["cap_lh"]

    # the decision row — or, when there is no decision to show, a note
    # across the full width. One text size for all of it, from the stage.
    dec = spec["decision"]
    tp = text_px(L)
    if L["plot_h"] >= 24:
        if dec.get("wide"):
            im.paste(dec["plot"](L["block_w"], L["plot_h"], tp),
                     (x0, L["dec_y"]))
        else:
            im.paste(dec["plot"](L["plot_w"], L["plot_h"], tp),
                     (x0, L["dec_y"]))
            im.paste(R.verdict_box(dec["kind"], dec["big"], dec["lines"],
                                   L["box_w"], L["box_h"], text_px=tp),
                     (L["box_x"], L["dec_y"]))
        y = L["pcap_y"]
        for line in R.wrap_lines(d, dec.get("plot_caption", ""), f_cap,
                                 L["plot_w"], max_lines=2):
            d.text((x0, y), line, font=f_cap, fill=R.MUTED)
            y += L["cap_lh"]
    return im
