"""
plots.py — the result figures, drawn with Pillow.

Same palette and type as the demo (render.py), so the deck's figures and the
live demo look like one piece of work, and no plotting library is needed on
the presenting laptop. Each function writes a PNG and returns its path.

    det_figure      FRR against FAR, both on log axes — the standard
                    verification plot; lower-left is better
    far_frr_figure  FAR and FRR against the threshold, crossing at the EER
    hist_figure     genuine and impostor score distributions
    bar_figure      EERs with 95% confidence intervals
"""
import math

import numpy as np
from PIL import Image, ImageDraw

from . import render as R

SS = 2
GREY = (128, 133, 140)
DARK_GREY = (80, 86, 94)

# (colour, line width, dash pattern in px or None)
STYLES = {
    "primary": (R.ACCENT, 3.0, None),
    "baseline": (GREY, 3.0, None),
    "dash": (R.ACCENT, 2.2, (10, 6)),
    "dot": (R.ACCENT, 2.2, (2, 5)),
    "grey_dash": (DARK_GREY, 2.2, (10, 6)),
    "grey_dot": (DARK_GREY, 2.2, (2, 5)),
    "dark": (R.INK, 2.4, None),
}


def _poly(d, pts, colour, width, dash):
    if len(pts) < 2:
        return
    if not dash:
        d.line(pts, fill=colour, width=int(round(width)), joint="curve")
        return
    on, off = dash
    acc, drawing = 0.0, True
    seg = [pts[0]]
    for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:]):
        L = math.hypot(x1 - x0, y1 - y0)
        pos = 0.0
        while pos < L:
            left = (on if drawing else off) - acc
            step = min(left, L - pos)
            t = (pos + step) / L if L else 1.0
            p = (x0 + (x1 - x0) * t, y0 + (y1 - y0) * t)
            if drawing:
                seg.append(p)
            pos += step
            acc += step
            if acc >= (on if drawing else off) - 1e-9:
                if drawing and len(seg) > 1:
                    d.line(seg, fill=colour, width=int(round(width)))
                drawing = not drawing
                acc = 0.0
                seg = [p]
        if drawing:
            seg.append((x1, y1))
    if drawing and len(seg) > 1:
        d.line(seg, fill=colour, width=int(round(width)))


def _fit(d, text, size, max_w, bold=True):
    """The largest font, up to `size`, at which `text` fits in `max_w`."""
    f = R.font(size, bold)
    while d.textlength(text, font=f) > max_w and size > 8:
        size *= 0.94
        f = R.font(size, bold)
    return f


def _boxed(d, xy, text, f, fill=R.INK, pad=None):
    """Text on a white box, so it stays legible over bars or curves."""
    x, y = xy
    l, t, r, b = d.textbbox((x, y), text, font=f)
    p = pad if pad is not None else (b - t) * 0.25
    d.rectangle([l - p, t - p, r + p, b + p], fill=R.WHITE)
    d.text((x, y), text, font=f, fill=fill)


class Axes(object):
    """A plotting frame with linear or log axes, in supersampled pixels."""

    def __init__(self, width, height, xlim, ylim, xlog=False, ylog=False,
                 title="", xlabel="", ylabel="", margins=None):
        self.W, self.H = width * SS, height * SS
        self.size = (width, height)
        self.im = Image.new("RGB", (self.W, self.H), R.WHITE)
        self.d = ImageDraw.Draw(self.im)
        self.fs = self.H * 0.034
        self.f, self.fb = R.font(self.fs), R.font(self.fs, True)
        ml, mr, mt, mb = margins or (0.11, 0.04, 0.10, 0.13)
        self.x0, self.x1 = self.W * ml, self.W * (1 - mr)
        self.y0, self.y1 = self.H * mt, self.H * (1 - mb)
        self.ft = _fit(self.d, title, self.fs * 1.25, self.x1 - self.x0)
        self.xlim, self.ylim, self.xlog, self.ylog = xlim, ylim, xlog, ylog
        if title:
            self.d.text((self.x0, self.H * 0.025), title, font=self.ft,
                        fill=R.INK)
        if xlabel:
            w = self.d.textlength(xlabel, font=self.f)
            self.d.text(((self.x0 + self.x1) / 2 - w / 2,
                         self.H - self.fs * 1.6), xlabel, font=self.f,
                        fill=R.SLATE)
        if ylabel:
            lab = Image.new("RGBA", (int(self.d.textlength(ylabel, font=self.f))
                                     + 4, int(self.fs * 1.4)), (0, 0, 0, 0))
            ImageDraw.Draw(lab).text((2, 0), ylabel, font=self.f,
                                     fill=R.SLATE)
            lab = lab.rotate(90, expand=True)
            self.im.paste(lab, (int(self.fs * 0.3),
                                int((self.y0 + self.y1) / 2 - lab.size[1] / 2)),
                          lab)

    def _t(self, v, lim, log):
        a, b = lim
        if log:
            v = max(v, a)
            return (math.log10(v) - math.log10(a)) / (
                math.log10(b) - math.log10(a))
        return (v - a) / (b - a)

    def X(self, v):
        return self.x0 + self._t(v, self.xlim, self.xlog) * (self.x1 - self.x0)

    def Y(self, v):
        return self.y1 - self._t(v, self.ylim, self.ylog) * (self.y1 - self.y0)

    def grid(self, xticks, yticks, xfmt, yfmt):
        d = self.d
        for v in xticks:
            x = self.X(v)
            d.line([(x, self.y0), (x, self.y1)], fill=R.LIGHT, width=SS)
            lab = xfmt(v)
            w = d.textlength(lab, font=self.f)
            d.text((x - w / 2, self.y1 + self.fs * 0.5), lab, font=self.f,
                   fill=R.MUTED)
        for v in yticks:
            y = self.Y(v)
            d.line([(self.x0, y), (self.x1, y)], fill=R.LIGHT, width=SS)
            lab = yfmt(v)
            w = d.textlength(lab, font=self.f)
            d.text((self.x0 - w - self.fs * 0.5, y - self.fs * 0.6), lab,
                   font=self.f, fill=R.MUTED)
        d.rectangle([self.x0, self.y0, self.x1, self.y1], outline=R.RULE,
                    width=SS)

    def clip_line(self, xs, ys, style):
        colour, width, dash = STYLES[style]
        pts = []
        for x, y in zip(xs, ys):
            if not (np.isfinite(x) and np.isfinite(y)):
                continue
            px = min(max(self.X(x), self.x0), self.x1)
            py = min(max(self.Y(y), self.y0), self.y1)
            pts.append((px, py))
        _poly(self.d, pts, colour, width * SS,
              tuple(v * SS for v in dash) if dash else None)

    def legend_box(self, entries, corner="upper right"):
        """Where `legend` would put its box: (x, y, width, height)."""
        d, fs = self.d, self.fs
        row = fs * 1.55
        w = max(d.textlength(lab, font=self.f) for lab, _ in entries)
        bw = w + fs * 4.2
        bh = row * len(entries) + fs * 0.8
        if isinstance(corner, tuple):
            bx = self.x0 + corner[0] * (self.x1 - self.x0)
            by = self.y0 + corner[1] * (self.y1 - self.y0)
        elif corner == "upper right":
            bx, by = self.x1 - bw - fs * 0.6, self.y0 + fs * 0.6
        elif corner == "lower left":
            bx, by = self.x0 + fs * 0.6, self.y1 - bh - fs * 0.6
        elif corner == "lower right":
            bx, by = self.x1 - bw - fs * 0.6, self.y1 - bh - fs * 0.6
        else:
            bx, by = self.x0 + fs * 0.6, self.y0 + fs * 0.6
        return bx, by, bw, bh

    def legend(self, entries, corner="upper right"):
        """corner: "upper right", "lower left", "upper left", "lower right",
        or an (x, y) pair of fractions of the plotting frame for the box's
        top-left."""
        d, fs = self.d, self.fs
        row = fs * 1.55
        bx, by, bw, bh = self.legend_box(entries, corner)
        d.rectangle([bx, by, bx + bw, by + bh], fill=R.WHITE, outline=R.RULE,
                    width=SS)
        for k, (lab, style) in enumerate(entries):
            y = by + fs * 0.4 + row * k + row / 2
            colour, width, dash = STYLES[style]
            _poly(d, [(bx + fs * 0.6, y), (bx + fs * 3.2, y)], colour,
                  width * SS, tuple(v * SS for v in dash) if dash else None)
            d.text((bx + fs * 3.7, y - fs * 0.62), lab, font=self.f,
                   fill=R.INK)

    def legend_row(self, entries):
        """The legend as one row just above the frame, out of the data."""
        d, fs = self.d, self.fs
        x = self.x0
        y = self.y0 - fs * 1.15
        for lab, style in entries:
            colour, width, dash = STYLES[style]
            _poly(d, [(x, y), (x + fs * 2.4, y)], colour, width * SS,
                  tuple(v * SS for v in dash) if dash else None)
            d.text((x + fs * 2.9, y - fs * 0.62), lab, font=self.f,
                   fill=R.INK)
            x += fs * 2.9 + d.textlength(lab, font=self.f) + fs * 2.0

    def save(self, path):
        self.im.resize(self.size, Image.Resampling.LANCZOS).save(path)
        return path


def _pct(v, places=None):
    p = 100.0 * v
    if places is not None:
        return "%.*f%%" % (places, p)
    if p >= 1:
        return "%g%%" % round(p, 3)
    return ("%.4f" % p).rstrip("0").rstrip(".") + "%"


def det_figure(path, curves, title, width=1600, height=1000,
               xlim=(1e-6, 1.0), ylim=(1e-4, 1.0), mark_eer=True,
               legend="auto"):
    """
    curves: [(label, far, frr, style, eer or None)]. Thresholds at which
    FAR or FRR is exactly zero are left out: a log axis has no place for
    zero, and pinning them to the frame draws lines along its edges that
    look like data.

    legend="auto" puts the legend in whichever corner covers the least of
    the curves — a fixed corner hid the very curve the figure is about once
    the results improved and it moved into that corner.
    """
    ax = Axes(width, height, xlim, ylim, xlog=True, ylog=True, title=title,
              xlabel="False accept rate (FAR) - impostors let in",
              ylabel="False reject rate (FRR) - genuine users turned away")
    xt = [10.0 ** k for k in range(int(math.log10(xlim[0])), 1)]
    yt = [10.0 ** k for k in range(int(math.log10(ylim[0])), 1)]
    ax.grid(xt, yt, _pct, _pct)
    lo = max(xlim[0], ylim[0])
    ax.clip_line([lo, 1.0], [lo, 1.0], "grey_dot")
    drawn, marks = [], []
    for label, far, frr, style, eer in curves:
        far, frr = np.asarray(far), np.asarray(frr)
        keep = (far > 0) & (frr > 0)
        order = np.argsort(far[keep])
        xs, ys = far[keep][order], frr[keep][order]
        ax.clip_line(xs, ys, style)
        drawn.append((xs, ys))
        if mark_eer and eer is not None and eer > 0:
            x, y = ax.X(eer), ax.Y(eer)
            marks.append((x, y))
            r = ax.fs * 0.32
            colour = STYLES[style][0]
            ax.d.ellipse([x - r, y - r, x + r, y + r], fill=R.WHITE,
                         outline=colour, width=SS * 2)
    entries = ([(c[0], c[3]) for c in curves]
               + [("FAR = FRR  (circles: each EER)", "grey_dot")])
    if legend == "auto":
        legend = _clearest_corner(ax, entries, drawn, marks)
    ax.legend(entries, legend)
    return ax.save(path)


def _clearest_corner(ax, entries, drawn, marks):
    """The legend corner that hides the least ink: every curve sampled
    densely along its length in pixels, each EER circle counted heavily."""
    pts = []
    for xs, ys in drawn:
        px = [min(max(ax.X(x), ax.x0), ax.x1) for x in xs]
        py = [min(max(ax.Y(y), ax.y0), ax.y1) for y in ys]
        for (x0, y0), (x1, y1) in zip(zip(px, py), zip(px[1:], py[1:])):
            n = max(1, int(math.hypot(x1 - x0, y1 - y0) / ax.fs))
            for k in range(n):
                f = k / float(n)
                pts.append((x0 + f * (x1 - x0), y0 + f * (y1 - y0)))
    best, best_cost = "upper right", None
    fw, fh = float(ax.x1 - ax.x0), float(ax.y1 - ax.y0)
    _, _, bw0, bh0 = ax.legend_box(entries, "upper right")
    m = ax.fs * 0.6
    # the four corners first (preferred on a tie), then the left and right
    # edges at every twentieth of the height
    places = ["upper right", "lower left", "lower right", "upper left"]
    for k in range(1, 20):
        top = (m + k * (fh - bh0 - 2 * m) / 20.0) / fh
        places += [(m / fw, top), ((fw - bw0 - m) / fw, top)]
    for corner in places:
        bx, by, bw, bh = ax.legend_box(entries, corner)
        pad = ax.fs * 0.5
        inside = lambda x, y: (bx - pad <= x <= bx + bw + pad
                               and by - pad <= y <= by + bh + pad)
        cost = sum(1 for x, y in pts if inside(x, y))
        cost += 1000 * sum(1 for x, y in marks if inside(x, y))
        if best_cost is None or cost < best_cost:
            best, best_cost = corner, cost
    return best


def far_frr_figure(path, thresholds, far, frr, eer, t_eer, title,
                   width=1600, height=1000):
    t = np.asarray(thresholds)
    lo, hi = float(np.min(t)), float(np.max(t))
    ax = Axes(width, height, (lo, hi), (0.0, 1.0), title=title,
              xlabel="Decision threshold (accept if score >= threshold)",
              ylabel="Error rate", margins=(0.11, 0.04, 0.15, 0.13))
    step = _nice((hi - lo) / 6)
    xt = list(np.arange(math.ceil(lo / step) * step, hi + 1e-9, step))
    ax.grid(xt, [0, 0.25, 0.5, 0.75, 1.0], lambda v: "%.2f" % v,
            lambda v: "%d%%" % round(100 * v))
    ax.clip_line(t, far, "baseline")
    ax.clip_line(t, frr, "primary")
    x = ax.X(t_eer)
    ax.d.line([(x, ax.y0), (x, ax.y1)], fill=R.INK, width=SS)
    # Mid-height on the threshold line lies between the falling FAR curve
    # and the rising FRR curve, so it is empty; keep the label in the frame.
    lab = "EER %s at threshold %.3f" % (_pct(eer, 2), t_eer)
    tw = ax.d.textlength(lab, font=ax.fb)
    lx = min(max(x - tw / 2.0, ax.x0 + ax.fs * 0.5),
             ax.x1 - tw - ax.fs * 0.5)
    _boxed(ax.d, (lx, ax.Y(0.5) - ax.fs * 0.6), lab, ax.fb)
    ax.legend_row([("FAR - impostors accepted", "baseline"),
                   ("FRR - genuine users rejected", "primary")])
    return ax.save(path)


def _nice(raw):
    p = 10.0 ** math.floor(math.log10(raw))
    for m in (1, 2, 2.5, 5, 10):
        if raw <= m * p:
            return m * p
    return 10 * p


def hist_figure(path, panels, title, width=1600, height=1000):
    """
    panels: [(label, genuine, impostor, threshold)], drawn one above the
    other on the same score axis, each distribution scaled to its own area
    (density), since impostors outnumber genuine comparisons ~600 to 1.
    """
    W, H = width * SS, height * SS
    im = Image.new("RGB", (W, H), R.WHITE)
    d = ImageDraw.Draw(im)
    fs = H * 0.034
    f, fb, ft = R.font(fs), R.font(fs, True), R.font(fs * 1.25, True)
    d.text((W * 0.06, H * 0.025), title, font=ft, fill=R.INK)
    n = len(panels)
    top, bottom = H * 0.11, H * 0.90
    gap = fs * 1.5
    ph = (bottom - top - gap * (n - 1)) / n
    x0, x1 = W * 0.06, W * 0.97
    for k, (label, gen, imp, thr) in enumerate(panels):
        lo = min(float(np.percentile(imp, 0.01)), float(np.min(gen)))
        hi = max(float(np.max(gen)), float(np.percentile(imp, 99.99)))
        pad = 0.03 * (hi - lo)
        lo, hi = lo - pad, hi + pad
        py0 = top + k * (ph + gap)
        py1 = py0 + ph
        edges = np.linspace(lo, hi, 81)
        hg = np.histogram(np.clip(gen, lo, hi), edges, density=True)[0]
        hi_ = np.histogram(np.clip(imp, lo, hi), edges, density=True)[0]
        peak = max(hg.max(), hi_.max())

        def X(v):
            return x0 + (v - lo) / (hi - lo) * (x1 - x0)

        for h, colour in ((hi_, (196, 202, 209)), (hg, R.ACCENT)):
            layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
            ld = ImageDraw.Draw(layer)
            for b in range(len(h)):
                if h[b] <= 0:
                    continue
                yt = py1 - h[b] / peak * (ph - fs * 1.6)
                ld.rectangle([X(edges[b]) + SS, yt, X(edges[b + 1]) - SS, py1],
                             fill=colour + (215,))
            im = Image.alpha_composite(im.convert("RGBA"), layer).convert("RGB")
            d = ImageDraw.Draw(im)
        d.line([(x0, py1), (x1, py1)], fill=R.SLATE, width=SS)
        step = _nice((hi - lo) / 8)
        v = math.ceil(lo / step) * step
        while v <= hi:
            d.line([(X(v), py1), (X(v), py1 + fs * 0.3)], fill=R.SLATE,
                   width=SS)
            lab = "%.2f" % v
            d.text((X(v) - d.textlength(lab, font=f) / 2, py1 + fs * 0.4),
                   lab, font=f, fill=R.MUTED)
            v += step
        xt = X(thr)
        y = py0 + fs * 1.4
        while y < py1:
            d.line([(xt, y), (xt, min(y + fs * 0.4, py1))], fill=R.INK,
                   width=SS)
            y += fs * 0.75
        d.text((x0, py0), label, font=fb, fill=R.INK)
        lab = "threshold %.3f" % thr
        _boxed(d, (xt + fs * 0.4, py0 + fs * 1.3), lab, f)
    # legend
    lx, ly = x1 - fs * 16, H * 0.03
    d.rectangle([lx, ly + fs * 0.25, lx + fs * 1.2, ly + fs * 1.05],
                fill=R.ACCENT)
    d.text((lx + fs * 1.6, ly), "genuine", font=f, fill=R.INK)
    d.rectangle([lx + fs * 7, ly + fs * 0.25, lx + fs * 8.2, ly + fs * 1.05],
                fill=(196, 202, 209))
    d.text((lx + fs * 8.6, ly), "impostor", font=f, fill=R.INK)
    im.resize((width, height), Image.Resampling.LANCZOS).save(path)
    return path


def bar_figure(path, rows, title, width=1600, height=None, note=""):
    """
    rows: [(label, eer, (lo, hi), style)] — horizontal bars of EER with
    95% bootstrap intervals. style "primary" or "baseline" sets the colour.
    """
    n = len(rows)
    height = height or int(260 + 92 * n)
    W, H = width * SS, height * SS
    im = Image.new("RGB", (W, H), R.WHITE)
    d = ImageDraw.Draw(im)
    fs = 30 * SS * 0.95
    f, fb, ft = R.font(fs), R.font(fs, True), R.font(fs * 1.25, True)
    d.text((W * 0.03, fs * 0.8), title, font=ft, fill=R.INK)
    lw = max(d.textlength(r[0], font=f) for r in rows) + fs
    x0 = W * 0.03 + lw
    x1 = W * 0.97 - fs * 7
    top = fs * 3.4
    row_h = (H - top - fs * 3.2) / n
    vmax = max(r[2][1] for r in rows) * 1.08

    def X(v):
        return x0 + v / vmax * (x1 - x0)

    step = _nice(vmax / 6)
    v = 0.0
    while v <= vmax:
        d.line([(X(v), top - fs * 0.3), (X(v), top + row_h * n)],
               fill=R.LIGHT, width=SS)
        lab = _pct(v)
        d.text((X(v) - d.textlength(lab, font=f) / 2, top + row_h * n
                + fs * 0.3), lab, font=f, fill=R.MUTED)
        v += step
    for k, (label, eer, (lo, hi), style) in enumerate(rows):
        yc = top + row_h * (k + 0.5)
        bh = row_h * 0.52
        colour = STYLES[style][0]
        d.rectangle([x0, yc - bh / 2, X(eer), yc + bh / 2], fill=colour)
        d.line([(X(lo), yc), (X(hi), yc)], fill=R.INK, width=SS * 2)
        for xv in (X(lo), X(hi)):
            d.line([(xv, yc - bh * 0.35), (xv, yc + bh * 0.35)], fill=R.INK,
                   width=SS * 2)
        d.text((W * 0.03, yc - fs * 0.62), label, font=f, fill=R.INK)
        val = _pct(eer, 2)
        d.text((X(hi) + fs * 0.5, yc - fs * 0.62), val, font=fb, fill=R.INK)
    if note:
        d.text((W * 0.03, H - fs * 1.6), note, font=f, fill=R.MUTED)
    im.resize((width, height), Image.Resampling.LANCZOS).save(path)
    return path
