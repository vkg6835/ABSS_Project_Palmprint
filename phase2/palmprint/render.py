"""
render.py — every picture the demo shows, drawn with Pillow alone.

Kept apart from the window code so that each panel can be produced and
checked without a screen (test_demo.py renders all of them), and so the same
drawings can serve as figures in the presentation.

The palette follows the Phase I deck: near-black ink, one muted slate accent,
greys for everything else. The only other colours are a muted green and red
for the accept/reject verdict, where colour carries meaning rather than
decoration.

Every function takes the size it should fill in pixels and returns an RGB
image of exactly that size. Lines and text are drawn at twice the size and
reduced, which is the cheapest way to get anti-aliasing out of Pillow.
"""
import math
import re
from functools import lru_cache

import numpy as np
from PIL import Image, ImageDraw, ImageFont

INK = (31, 41, 51)
SLATE = (74, 85, 96)
MUTED = (107, 114, 128)
FAINT = (154, 161, 169)
ACCENT = (53, 86, 107)
RULE = (213, 218, 224)
LIGHT = (237, 239, 242)
PALE = (246, 247, 248)
WHITE = (255, 255, 255)
ACCEPT = (46, 106, 79)
REJECT = (155, 44, 44)
UNMATCHED = (225, 229, 234)
WEAK_LINE = (214, 219, 225)
IMPOSTOR_BAR = (196, 202, 209)

SS = 2                                  # supersampling factor

_FONT_FILES = {
    False: ("segoeui.ttf", "calibri.ttf", "arial.ttf", "DejaVuSans.ttf",
            "LiberationSans-Regular.ttf"),
    True: ("segoeuib.ttf", "calibrib.ttf", "arialbd.ttf",
           "DejaVuSans-Bold.ttf", "LiberationSans-Bold.ttf"),
}


@lru_cache(maxsize=64)
def font(px, bold=False):
    """A TrueType font at `px` pixels, falling back gracefully: Segoe UI or
    Calibri on Windows, DejaVu on Linux, Pillow's built-in font if all else
    fails."""
    px = max(6, int(round(px)))
    for name in _FONT_FILES[bool(bold)]:
        try:
            return ImageFont.truetype(name, px)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size=px)
    except TypeError:                   # Pillow older than 10.1
        return ImageFont.load_default()


# ------------------------------------------------------------------ helpers
def _finish(im, size):
    """Reduce a supersampled drawing to its display size."""
    if isinstance(size, tuple):
        return im.resize(size, Image.Resampling.LANCZOS)
    return im.resize((size, size), Image.Resampling.LANCZOS)


def _to_l(arr):
    a = np.clip(np.asarray(arr, dtype=np.float64), 0.0, 1.0)
    return Image.fromarray((a * 255.0 + 0.5).astype(np.uint8))


def _mix(a, b, t):
    t = float(min(1.0, max(0.0, t)))
    return tuple(int(round(a[i] + (b[i] - a[i]) * t)) for i in range(3))


def _text_size(draw, text, f):
    l, t, r, b = draw.textbbox((0, 0), text, font=f)
    return r - l, b - t, l, t


def _text_center(draw, cx, cy, text, f, fill):
    w, h, l, t = _text_size(draw, text, f)
    draw.text((cx - w / 2.0 - l, cy - h / 2.0 - t), text, font=f, fill=fill)


def _border(draw, S, colour=RULE):
    draw.rectangle([0, 0, S - 1, S - 1], outline=colour, width=SS)


def _nice_step(raw):
    if raw <= 0:
        return 1.0
    p = 10.0 ** math.floor(math.log10(raw))
    for m in (1.0, 2.0, 2.5, 5.0, 10.0):
        if raw <= m * p:
            return m * p
    return 10.0 * p


# ------------------------------------------------------------- placeholders
def placeholder(size, step, title, note=""):
    """An empty stage: what the audience sees before a step is revealed."""
    S = size * SS
    im = Image.new("RGB", (S, S), PALE)
    d = ImageDraw.Draw(im)
    _border(d, S)
    _text_center(d, S / 2.0, S * 0.42, str(step), font(S * 0.17, True), RULE)
    _text_center(d, S / 2.0, S * 0.60, title, font(S * 0.062), FAINT)
    if note:
        _text_center(d, S / 2.0, S * 0.69, note, font(S * 0.050), FAINT)
    return _finish(im, size)


def blank(width, height, text="", text_px=None):
    S = SS
    im = Image.new("RGB", (width * S, height * S), PALE)
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, width * S - 1, height * S - 1], outline=RULE, width=S)
    if text:
        size = (text_px * S if text_px
                else min(height * S * 0.085, width * S * 0.03))
        _text_center(d, width * S / 2.0, height * S / 2.0, text, font(size),
                     FAINT)
    return _finish(im, (width, height))


def wrap_lines(draw, text, f, max_w, max_lines=None):
    """
    Greedy word wrap to `max_w` pixels. Runs of several spaces inside a line
    are kept as written, so text can be laid out in loose columns, but are
    dropped where a line breaks. An overflowing last line ends in an ellipsis
    rather than running off the edge.
    """
    parts = re.split(r"( +)", str(text).strip())
    words, seps = parts[0::2], [""] + parts[1::2]
    lines, cur = [], ""
    for sep, w in zip(seps, words):
        if not w:
            continue
        t = (cur + sep + w) if cur else w
        if not cur or draw.textlength(t, font=f) <= max_w:
            cur = t
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    if max_lines is not None and len(lines) > max_lines:
        lines = lines[:max_lines]
        last = lines[-1]
        while last and draw.textlength(last + "…", font=f) > max_w:
            last = last[:-1].rstrip()
        lines[-1] = last + "…"
    return lines


def declined_panel(size, reason):
    """Stage 2 when preprocessing refuses the input: say so, and say why."""
    S = size * SS
    im = Image.new("RGB", (S, S), PALE)
    d = ImageDraw.Draw(im)
    _border(d, S)
    pad = S * 0.09
    f_big = font(S * 0.105, True)
    _text_center(d, S / 2.0, S * 0.27, "No ROI", f_big, INK)
    f = font(S * 0.056)
    y = S * 0.42
    for line in wrap_lines(d, reason, f, S - 2 * pad, max_lines=4):
        _text_center(d, S / 2.0, y, line, f, SLATE)
        y += S * 0.075
    _text_center(d, S / 2.0, S * 0.86, "declined, not guessed",
                 font(S * 0.05), FAINT)
    return _finish(im, size)


def text_card(lines, width, height, text_px=None):
    """A plain card of instructions or notes; the first line is bold.

    `text_px` sets the type size in display pixels. The stage always passes
    it, so every piece of text on screen is sized from one base and the card
    cannot drift out of proportion with the captions around it when the
    window changes shape."""
    W, H = width * SS, height * SS
    im = Image.new("RGB", (W, H), WHITE)
    d = ImageDraw.Draw(im)
    d.rounded_rectangle([0, 0, W - 1, H - 1], radius=int(min(W, H) * 0.04),
                        fill=PALE, outline=RULE, width=SS)
    pad = min(W, H) * 0.08
    if text_px:
        fs = min(text_px * SS, H * 0.12)
    else:
        fs = max(9.0 * SS, min(H * 0.085, W * 0.028))
    lh = fs * 1.38
    wrapped = []
    for k, line in enumerate(lines):
        f = font(fs, k == 0)
        for wl in wrap_lines(d, line, f, W - 2 * pad):
            wrapped.append((wl, f, INK if k == 0 else SLATE))
        if k == 0:
            wrapped.append(("", f, INK))
    room = int((H - 2 * pad) // lh)
    wrapped = wrapped[:max(1, room)]
    y = (H - len(wrapped) * lh) / 2.0
    for text, f, col in wrapped:
        if text:
            d.text((pad, y), text, font=f, fill=col)
        y += lh
    return _finish(im, (width, height))


VERDICT_FILL = {"accept": ACCEPT, "reject": REJECT, "declined": INK}


def verdict_box(kind, big, lines, width, height, text_px=None):
    """
    The decision, in the one place on screen where colour carries meaning:
    green to accept, red to reject, ink when the input was declined, and an
    empty outline while the decision is still to come.

    With `text_px`, the small lines use the stage's text size and the big
    word is held to about three times it, so the box reads at the same scale
    as everything around it whatever its own proportions.
    """
    W, H = width * SS, height * SS
    im = Image.new("RGB", (W, H), WHITE)
    d = ImageDraw.Draw(im)
    r = int(min(W, H) * 0.06)
    if kind in VERDICT_FILL:
        d.rounded_rectangle([0, 0, W - 1, H - 1], radius=r,
                            fill=VERDICT_FILL[kind])
        fg, fg2 = WHITE, (236, 239, 242)
    else:
        d.rounded_rectangle([0, 0, W - 1, H - 1], radius=r, fill=PALE,
                            outline=RULE, width=SS)
        fg, fg2 = FAINT, FAINT
    big_spec, placed, f2, _ = _verdict_layout(d, big, lines, W, H, text_px)
    if big_spec:
        x, y, text, f = big_spec
        d.text((x, y), text, font=f, fill=fg)
    for x, y, text in placed:
        d.text((x, y), text, font=f2, fill=fg2)
    return _finish(im, (width, height))


def _verdict_layout(d, big, lines, W, H, text_px):
    """Where everything in a verdict box goes, at supersampled size.
    Returns (big word placement or None, [(x, y, line)], line font,
    number of wrapped lines that did not fit).

    If the lines do not fit at the stage's text size, the box shrinks its
    own text — down to 70% — before it drops anything. Dropping is the last
    resort, and test_demo.py checks it never happens at a real window size:
    the last line of a verdict is the one that says FALSE ACCEPT."""
    pad = W * 0.08
    y0 = H * 0.11
    big_spec = None
    if big:
        size = min(H * 0.25, W * 0.19)
        if text_px:
            size = min(size, 3.1 * text_px * SS)
        f = font(size, True)
        while d.textlength(big, font=f) > W - 2 * pad and size > 8:
            size *= 0.92
            f = font(size, True)
        bw, bh, l, t = _text_size(d, big, f)
        big_spec = (pad - l, y0 - t, big, f)
        y0 += bh + H * 0.09
    if text_px:
        fs0 = min(text_px * SS, H * 0.13)
    else:
        fs0 = max(8.0 * SS, min(H * 0.082, W * 0.066))

    best = None
    for shrink in (1.0, 0.92, 0.85, 0.78, 0.70):
        fs = fs0 * shrink
        f2 = font(fs)
        lh = fs * 1.3
        y = y0
        placed, dropped = [], 0
        for line in lines:
            for wl in wrap_lines(d, line, f2, W - 2 * pad):
                if y + lh > H - H * 0.05:
                    dropped += 1
                    continue
                placed.append((pad, y, wl))
                y += lh
            y += lh * 0.3
        best = (placed, f2, dropped)
        if dropped == 0:
            break
    placed, f2, dropped = best
    return big_spec, placed, f2, dropped


def verdict_overflow(big, lines, width, height, text_px=None):
    """How many lines of a verdict box would be cut off — 0 if it all fits.
    A box that silently loses its last line (say, 'FALSE ACCEPT') is worse
    than useless, so the tests check this at every window size."""
    d = ImageDraw.Draw(Image.new("RGB", (8, 8)))
    return _verdict_layout(d, big, lines, width * SS, height * SS,
                           text_px)[3]


def ranked_bars(ranked, threshold, truth, width, height, text_px=None):
    """
    Identification: the best-scoring enrolled palms, best first, against the
    threshold. Bars start from zero, so their lengths are honest; the top bar
    takes the verdict colour, and the true identity, where known, is marked.
    """
    W, H = width * SS, height * SS
    im = Image.new("RGB", (W, H), WHITE)
    d = ImageDraw.Draw(im)
    n = max(1, len(ranked))
    if text_px:
        # the stage's size, unless the rows are too tight to hold it
        fs = min(text_px * SS, H / (n * 1.45 + 4.5))
    else:
        fs = max(9.0 * SS, min(H * 0.075, W * 0.026))
    f, fb = font(fs), font(fs, True)
    top, bottom = fs * 2.6, H - fs * 1.9
    row = (bottom - top) / float(n)

    def name_of(nm):
        return nm + ("  (true)" if truth is not None and nm == truth else "")

    name_w = max(d.textlength(name_of(nm), font=fb) for nm, _ in ranked) \
        if ranked else fs * 4
    x0 = W * 0.02 + name_w + fs * 0.8
    x1 = W - fs * 4.2
    vals = [s for _, s in ranked] + [threshold]
    lo = min(0.0, min(vals))
    hi = max(1.0, max(vals))

    def X(v):
        return x0 + (v - lo) / (hi - lo) * (x1 - x0)

    for k, (nm, s) in enumerate(ranked):
        yc = top + (k + 0.5) * row
        bh = min(row * 0.62, fs * 1.6)
        if k == 0:
            col = ACCEPT if s >= threshold else REJECT
        else:
            col = IMPOSTOR_BAR
        is_truth = truth is not None and nm == truth
        nf = fb if (k == 0 or is_truth) else f
        tw, th, l, t = _text_size(d, name_of(nm), nf)
        d.text((x0 - fs * 0.8 - tw - l, yc - th / 2.0 - t), name_of(nm),
               font=nf, fill=INK if (k == 0 or is_truth) else SLATE)
        d.rectangle([X(0.0 if lo < 0 else lo), yc - bh / 2.0,
                     max(X(0.0 if lo < 0 else lo) + SS, X(max(s, lo))),
                     yc + bh / 2.0], fill=col)
        vw, vh, vl, vt = _text_size(d, "%.3f" % s, nf)
        d.text((X(max(s, lo)) + fs * 0.4 - vl, yc - vh / 2.0 - vt),
               "%.3f" % s, font=nf, fill=INK)

    xt = X(threshold)
    y = top - fs * 0.6
    dash = fs * 0.45
    while y < bottom:
        d.line([(xt, y), (xt, min(y + dash, bottom))], fill=INK, width=SS)
        y += dash * 1.9
    lab = "threshold %.3f" % threshold
    tw, th, l, t = _text_size(d, lab, f)
    tx = xt + fs * 0.35 if xt + fs * 0.35 + tw < W - 2 else xt - fs * 0.35 - tw
    d.text((tx - l, top - fs * 1.4 - t), lab, font=f, fill=INK)

    d.line([(X(lo), bottom + fs * 0.3), (x1, bottom + fs * 0.3)],
           fill=RULE, width=SS)
    for v in (0.0, 0.25, 0.5, 0.75, 1.0):
        if lo <= v <= hi:
            _text_center(d, X(v), bottom + fs * 1.0, "%.2f" % v,
                         font(fs * 0.85), MUTED)
    d.rectangle([0, 0, W - 1, H - 1], outline=RULE, width=SS)
    return _finish(im, (width, height))


# ----------------------------------------------------------- stage 1: input
def hand_panel(trace, size, overlays=True):
    """
    The captured image, letterboxed into a square, with what preprocessing
    found drawn over it: the outline, every candidate valley (small), the two
    valleys the frame was built from (large), and the ROI window.

    Overlays are drawn as a thin slate line on a wider white one, so they
    stay visible over both the dark backdrop and the bright palm. With
    overlays=False the image is shown exactly as captured — step 1 of the
    demo, before anything has been computed from it.
    """
    S = size * SS
    canvas = Image.new("RGB", (S, S), PALE)
    d = ImageDraw.Draw(canvas)
    if trace.image is None:
        _border(d, S)
        _text_center(d, S / 2.0, S / 2.0, "could not read the image",
                     font(S * 0.06), MUTED)
        return _finish(canvas, size)

    img = trace.image
    h, w = img.shape
    k = S / float(max(h, w))
    nw, nh = max(1, int(round(w * k))), max(1, int(round(h * k)))
    pic = _to_l(img).resize((nw, nh), Image.Resampling.LANCZOS)
    ox, oy = (S - nw) // 2, (S - nh) // 2
    canvas.paste(pic.convert("RGB"), (ox, oy))
    d = ImageDraw.Draw(canvas)

    def P(rc):
        return (ox + (rc[1] + 0.5) * k, oy + (rc[0] + 0.5) * k)

    lw = max(1, int(round(S / 240.0)))
    info = trace.info if overlays else None
    if info:
        c = info["contour"]
        step = max(1, len(c) // 700)
        pts = [P(p) for p in c[::step]]
        pts.append(pts[0])
        d.line(pts, fill=WHITE, width=3 * lw, joint="curve")
        d.line(pts, fill=ACCENT, width=lw, joint="curve")

        quad = [P(q) for q in info["corners"]]
        quad.append(quad[0])
        d.line(quad, fill=WHITE, width=5 * lw, joint="curve")
        d.line(quad, fill=ACCENT, width=2 * lw, joint="curve")

        def dot(rc, r, fill, outline):
            x, y = P(rc)
            d.ellipse([x - r, y - r, x + r, y + r], fill=fill,
                      outline=outline, width=lw)

        for v in info["valleys"]:
            dot(v, 3 * lw, WHITE, ACCENT)
        for v in info["roi_valleys"]:
            dot(v, 6 * lw, ACCENT, WHITE)
        cx, cy = P(info["centre"])
        a = 5 * lw
        d.line([(cx - a, cy), (cx + a, cy)], fill=WHITE, width=3 * lw)
        d.line([(cx, cy - a), (cx, cy + a)], fill=WHITE, width=3 * lw)
        d.line([(cx - a, cy), (cx + a, cy)], fill=ACCENT, width=lw)
        d.line([(cx, cy - a), (cx, cy + a)], fill=ACCENT, width=lw)

    if overlays and not trace.ok:
        band = S * 0.13
        d.rectangle([0, S - band, S, S], fill=INK)
        _text_center(d, S / 2.0, S - band / 2.0, "declined - no ROI",
                     font(band * 0.42, True), WHITE)
    _border(d, S)
    return _finish(canvas, size)


# ------------------------------------------------------------- stage 2: ROI
def roi_panel(roi, size):
    S = size * SS
    im = _to_l(roi).resize((S, S), Image.Resampling.BICUBIC).convert("RGB")
    _border(ImageDraw.Draw(im), S)
    return _finish(im, size)


# ------------------------------------------- stage 3 (RLOC): the code itself
def code_panel(code, contrast, size, n_dir=6):
    """
    The orientation code as a field of short strokes: each of the 32 x 32
    cells shows the direction that won the argmin at that point.

    Stroke darkness is the directional contrast — which the code itself
    throws away. Faint strokes are points where no direction clearly won;
    the code still assigns them a label, and that label is close to
    arbitrary. Showing it is the honest way to draw the representation.
    """
    S = size * SS
    im = Image.new("RGB", (S, S), WHITE)
    d = ImageDraw.Draw(im)
    code = np.asarray(code)
    h, w = code.shape
    cell = S / float(max(h, w))
    c = np.asarray(contrast, dtype=np.float64)
    hi = float(np.percentile(c, 95)) if c.size else 1.0
    q = np.clip(c / (hi if hi > 0 else 1.0), 0.0, 1.0)
    half = 0.43 * cell
    width = max(1, int(round(cell * 0.20)))
    for i in range(h):
        for j in range(w):
            th = math.pi * int(code[i, j]) / float(n_dir)
            dx, dy = math.cos(th) * half, math.sin(th) * half
            cx, cy = (j + 0.5) * cell, (i + 0.5) * cell
            col = _mix(WEAK_LINE, ACCENT, q[i, j] ** 0.8)
            d.line([(cx - dx, cy - dy), (cx + dx, cy + dy)], fill=col,
                   width=width)
    _border(d, S)
    return _finish(im, size)


# --------------------------------------------- stage 4 (RLOC): the matching
def match_panel(hit_map, size):
    """
    Equations (7)-(9) as a picture: one square per code cell, slate where the
    probe's direction was found in the template's cross-shaped neighbourhood,
    pale grey where it was not, blank outside the region the two codes
    overlap after the best shift. The score is the slate fraction.
    """
    S = size * SS
    im = Image.new("RGB", (S, S), WHITE)
    d = ImageDraw.Draw(im)
    hm = np.asarray(hit_map)
    h, w = hm.shape
    cell = S / float(max(h, w))
    g = max(1.0, cell * 0.10)
    for i in range(h):
        for j in range(w):
            v = hm[i, j]
            if v < 0:
                continue
            d.rectangle([j * cell + g / 2.0, i * cell + g / 2.0,
                         (j + 1) * cell - g / 2.0, (i + 1) * cell - g / 2.0],
                        fill=ACCENT if v == 1 else UNMATCHED)
    _border(d, S)
    return _finish(im, size)


# ---------------------------------------------- stage 3 (Han): Sobel energy
def sobel_panel(sobel, size):
    """Equation (1) of Paper 1, drawn dark-on-light like ink on paper."""
    m = np.asarray(sobel, dtype=np.float64)
    hi = float(np.percentile(m, 99)) or 1.0
    S = size * SS
    im = _to_l(1.0 - np.clip(m / hi, 0.0, 1.0)).resize(
        (S, S), Image.Resampling.BICUBIC).convert("RGB")
    _border(ImageDraw.Draw(im), S)
    return _finish(im, size)


# ----------------------------------------- stage 4 (Han): pooled features
def grid_panel(probe_grid, template_grid, size):
    """
    Paper 1's actual feature vector: the 8 x 8 cell means of the Sobel map,
    for the probe and for the enrolled template, side by side. Each grid is
    scaled to its own range — legitimate here because the Pearson correlation
    that compares them is blind to exactly that scaling.

    This is the panel that makes our Phase I criticism visible: the line
    structure of stage 3 is gone, averaged into a coarse map of edge energy.
    """
    S = size * SS
    im = Image.new("RGB", (S, S), WHITE)
    d = ImageDraw.Draw(im)
    gap = 0.045 * S
    gw = (S - 3.0 * gap) / 2.0
    y0 = (S - gw) / 2.0 - 0.05 * S
    for idx, (grid, label) in enumerate(((probe_grid, "probe"),
                                         (template_grid, "enrolled"))):
        g = np.asarray(grid, dtype=np.float64)
        lo, hi = float(g.min()), float(g.max())
        n = (g - lo) / (hi - lo) if hi > lo else np.zeros_like(g)
        x0 = gap + idx * (gw + gap)
        rows, cols = g.shape
        cw, ch = gw / cols, gw / rows
        for i in range(rows):
            for j in range(cols):
                v = int(round(255 * (1.0 - 0.85 * n[i, j])))
                d.rectangle([x0 + j * cw, y0 + i * ch,
                             x0 + (j + 1) * cw, y0 + (i + 1) * ch],
                            fill=(v, v, v))
        d.rectangle([x0, y0, x0 + gw, y0 + gw], outline=RULE, width=SS)
        _text_center(d, x0 + gw / 2.0, y0 + gw + 0.07 * S, label,
                     font(S * 0.058), MUTED)
    _border(d, S)
    return _finish(im, size)


# ------------------------------------------------------ the decision plot
def score_plot(genuine, impostor, score, threshold, width, height,
               marker_label="this probe", text_px=None):
    """
    Where this attempt's score lands among every genuine and impostor
    attempt in the evaluation.

    The two histograms are each scaled to their own peak — there are tens of
    impostor comparisons for every genuine one, so on a common scale the
    genuine distribution would be a flat line. The caption under the plot in
    the demo says so, because an unlabelled rescaling is a way to mislead.
    """
    W, H = width * SS, height * SS
    im = Image.new("RGB", (W, H), WHITE)
    if text_px:
        # the stage's text size, unless the plot is too short to hold the
        # labels above it and still leave room for the histograms
        fs = min(text_px * SS, H / 8.5)
    else:
        fs = max(9.0 * SS, H * 0.072)
    f_lab, f_tick = font(fs), font(fs * 0.92)
    ml, mr = W * 0.025, W * 0.025
    mt, mb = fs * 3.0, fs * 2.2
    x0, x1 = ml, W - mr
    ybase, ytop = H - mb, mt

    g = np.asarray(genuine, dtype=np.float64)
    i = np.asarray(impostor, dtype=np.float64)
    vals = [threshold]
    for arr in (g, i):
        if arr.size:
            vals += [float(np.percentile(arr, 0.5)),
                     float(np.percentile(arr, 99.5))]
    if score is not None:
        vals.append(float(score))
    lo, hi = min(vals), max(vals)
    span = (hi - lo) or 1.0
    lo, hi = lo - 0.06 * span, hi + 0.06 * span

    def X(v):
        return x0 + (v - lo) / (hi - lo) * (x1 - x0)

    bins = 44
    edges = np.linspace(lo, hi, bins + 1)
    for arr, colour, alpha in ((i, IMPOSTOR_BAR, 255), (g, ACCENT, 205)):
        if not arr.size:
            continue
        counts, _ = np.histogram(np.clip(arr, lo, hi), edges)
        if counts.max() == 0:
            continue
        layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        ld = ImageDraw.Draw(layer)
        for b in range(bins):
            if counts[b] == 0:
                continue
            top = ybase - counts[b] / float(counts.max()) * (ybase - ytop)
            ld.rectangle([X(edges[b]) + SS * 0.5, top,
                          max(X(edges[b]) + SS * 0.5,
                              X(edges[b + 1]) - SS * 0.5), ybase],
                         fill=colour + (alpha,))
        im = Image.alpha_composite(im.convert("RGBA"), layer).convert("RGB")

    d = ImageDraw.Draw(im)
    d.line([(x0, ybase), (x1, ybase)], fill=SLATE, width=SS)
    step = _nice_step((hi - lo) / 6.0)
    t = math.ceil(lo / step) * step
    while t <= hi + 1e-12:
        x = X(t)
        d.line([(x, ybase), (x, ybase + fs * 0.35)], fill=SLATE, width=SS)
        _text_center(d, x, ybase + fs * 1.15, "%.2f" % t, f_tick, MUTED)
        t += step

    def label(x, y, text, fill, f, prefer_right=True, pad=None):
        """Put a label beside its line: the preferred side if it fits, the
        other side if not, and only as a last resort centred on it. `pad`
        is the clear gap from the line, wider where a marker sits on it."""
        tw, th, l, tt = _text_size(d, text, f)
        pad = fs * 0.35 if pad is None else pad
        right_ok = x + pad + tw < W - 2
        left_ok = x - pad - tw > 2
        if prefer_right and right_ok:
            tx = x + pad
        elif not prefer_right and left_ok:
            tx = x - pad - tw
        elif right_ok:
            tx = x + pad
        elif left_ok:
            tx = x - pad - tw
        else:
            tx = min(max(2.0, x - tw / 2.0), W - tw - 2)
        d.text((tx - l, y - th / 2.0 - tt), text, font=f, fill=fill)

    # threshold: dashed, ink
    xt = X(threshold)
    y = ytop - fs * 0.6
    dash = fs * 0.45
    while y < ybase:
        d.line([(xt, y), (xt, min(y + dash, ybase))], fill=INK, width=SS)
        y += dash * 1.9
    label(xt, ytop - fs * 1.05, "threshold %.3f" % threshold, INK, f_lab,
          prefer_right=(score is None or score < threshold))

    # this attempt: solid, in the colour of the verdict
    if score is not None:
        col = ACCEPT if score >= threshold else REJECT
        xs = X(score)
        d.line([(xs, ytop - fs * 1.9), (xs, ybase)], fill=col, width=2 * SS)
        tri = fs * 0.45
        d.polygon([(xs - tri, ytop - fs * 2.45), (xs + tri, ytop - fs * 2.45),
                   (xs, ytop - fs * 1.9)], fill=col)
        # clear of the triangle, which is wider than the line beneath it
        label(xs, ytop - fs * 2.25, "%s %.3f" % (marker_label, score), col,
              font(fs, True), prefer_right=(score >= threshold),
              pad=tri + fs * 0.3)

    d.rectangle([0, 0, W - 1, H - 1], outline=RULE, width=SS)
    return _finish(im, (width, height))
