"""
rloc.py — Robust Line Orientation Code: enrolment, matching, verification.

This is the back end of Paper 2: the answers to "how are the features trained"
and "what is the classification scheme".

Training
--------
There is none. No classifier is fitted, no parameters are learned from data.
What the paper calls an "enlarged training set" is pure enrolment-side data
augmentation, and it exists to fix an inherited defect: the ROI produced by
the preprocessing stage can be rotated by up to about 10 degrees, and rotation
is worse for an orientation code than for a magnitude, because it changes
*which* direction wins — the feature values themselves, not just their scale.

The paper considers four remedies and argues three of them down: rotation-
invariant features would discard the orientation that is the entire signal;
estimating and undoing the angle is unreliable at this resolution; rotating
the probe at match time pays the cost on every one of millions of
comparisons. What is left is to pay once, at enrolment: store the template at
0, +/-3, +/-6 and +/-9 degrees — seven codes — and take the best of seven at
match time.

Classification
--------------
Also none: a similarity and a threshold. The interesting part is the
similarity. Comparing two code images pixel against pixel is brittle, because
a residual one-pixel misalignment makes a genuine pair disagree almost
everywhere. So the paper redefines a match: a pixel of A counts as matched if
its code equals the code of B at ANY position in a small neighbourhood.

    s(A,B) = |{ (i,j) : A(i,j) == B(i',j') for some (i',j') in N(i,j) }| / total
                                                                    (7)-(8)
    S(A,B) = max( s(A,B), s(B,A) )                                     (9)

maximised over a translation search of +/-2 code pixels and, here, over the
seven rotated templates.

(Reimplementer's warning: the union symbol in equation 7 of the paper is not
set union — the text redefines it to mean "equals at least one of".)

Three neighbourhoods are reported. The 5-point cross is adopted. The 9-point
3x3 square is worse — 0.27% EER against 0.16% — because nine candidate
positions give impostor pixels nine chances to agree as well: tolerance up,
discrimination down. All three are implemented here so that result can be
reproduced rather than quoted.

Note the contrast with Paper 1, which buys the same alignment tolerance by
averaging its features into block means — destroying the line structure it
just extracted. Here the full 32x32 code is kept and the *metric* is relaxed
instead. Same goal, opposite method, and only one of them throws information
away.

All scores returned by this module are SIMILARITIES in [0, 1]: higher means
more alike. Thresholding is "accept if score >= t".
"""
import numpy as np
from PIL import Image

from . import mfrat

# The three matching neighbourhoods of Section 4.3 / Table 4.
NEIGHBOURHOODS = {
    "pixel": ((0, 0),),
    "cross": ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)),
    "square": tuple((dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)),
}

# Code values are 0 .. n_dir-1, so any value above that never compares equal.
_OUTSIDE = np.uint8(255)


class RlocConfig(object):
    """
    Every knob in one place, so an experiment is a config and a result.

    p, n_dir, width   the MFRAT operator (paper: 16, 6, 4)
    step              code sub-sampling; None means `width`, the paper's 16:1
    rotations         enrolment angles in degrees (paper: 0, +/-3, +/-6, +/-9).
                      Use (0,) to switch the enlarged training set off, which
                      is Experiment 1 of the paper's ablation.
    neighbourhood     "pixel", "cross" (paper's choice) or "square"
    shift_range       translation search on the CODE grid, in code pixels.
                      1 code pixel = `step` ROI pixels. The paper searches
                      +/-2 (25 shifts, +/-8 ROI pixels), and so do we.
    symmetric         equation (9): score a pair as max(s(A,B), s(B,A)) —
                      A's cells looked up in B's neighbourhoods, and the other
                      way round — as the paper does. False keeps only
                      s(probe, template).

    Our first version used shift_range=1 and symmetric=False, to save
    matching time. On Tongji that cost a great deal — cross-session EER
    0.69% against 0.11% as published — and the batched matcher has since
    made the saving irrelevant. Both settings stay available so the
    difference can be measured rather than asserted (run_experiments.py,
    section 8).
    """

    def __init__(self, p=mfrat.P_DEFAULT, n_dir=mfrat.NDIR_DEFAULT,
                 width=mfrat.WIDTH_DEFAULT, step=None,
                 rotations=(0.0, 3.0, -3.0, 6.0, -6.0, 9.0, -9.0),
                 neighbourhood="cross", shift_range=2, symmetric=True):
        self.p = int(p)
        self.n_dir = int(n_dir)
        self.width = int(width)
        self.step = int(width if step is None else step)
        self.rotations = tuple(float(r) for r in rotations)
        if neighbourhood not in NEIGHBOURHOODS:
            raise ValueError("neighbourhood must be one of %s"
                             % sorted(NEIGHBOURHOODS))
        self.neighbourhood = neighbourhood
        self.shift_range = int(shift_range)
        self.symmetric = bool(symmetric)

    @property
    def offsets(self):
        return NEIGHBOURHOODS[self.neighbourhood]

    def describe(self):
        return ("MFRAT p=%d dirs=%d width=%d step=%d | %d enrol rotation(s) "
                "| %s neighbourhood (%d pt) | shift +/-%d code px | %s"
                % (self.p, self.n_dir, self.width, self.step,
                   len(self.rotations), self.neighbourhood,
                   len(self.offsets), self.shift_range,
                   "both directions (eq. 9)" if self.symmetric
                   else "probe-to-template only"))

    def __repr__(self):
        return "RlocConfig(%s)" % self.describe()


# --------------------------------------------------------------------- encode
def encode(roi, cfg=None):
    """ROI image (float, any size) -> orientation code image (uint8)."""
    cfg = cfg or RlocConfig()
    code, _ = mfrat.orientation_code(roi, p=cfg.p, n_dir=cfg.n_dir,
                                     width=cfg.width, step=cfg.step)
    return code


def encode_with_contrast(roi, cfg=None):
    """As `encode`, but also returns the directional contrast map."""
    cfg = cfg or RlocConfig()
    return mfrat.orientation_code(roi, p=cfg.p, n_dir=cfg.n_dir,
                                  width=cfg.width, step=cfg.step)


def rotate_roi(roi, degrees):
    """
    Rotate an ROI about its centre, keeping the same size.

    The ROI is reflect-padded before rotating and centre-cropped afterwards.
    Rotating it directly would pull the fill colour into the corners, and a
    black fill would read as a strong dark line to an operator whose whole job
    is finding dark lines — fabricating orientations at the border.
    """
    if abs(degrees) < 1e-9:
        return np.asarray(roi, dtype=np.float64)
    arr = np.asarray(roi, dtype=np.float32)
    h, w = arr.shape
    pad = int(np.ceil(max(h, w) * (np.sqrt(2.0) - 1.0) / 2.0)) + 1
    big = np.pad(arr, pad, mode="reflect")
    # float32 2-D infers Pillow mode "F"; passing mode= explicitly is
    # deprecated in recent Pillow and scheduled for removal
    im = Image.fromarray(big).rotate(
        float(degrees), resample=Image.Resampling.BICUBIC)
    out = np.asarray(im, dtype=np.float64)
    return out[pad:pad + h, pad:pad + w]


def enrol_codes(roi, cfg=None):
    """
    Build the enrolment template: one code image per configured rotation.
    Returns a (n_rot, h, w) uint8 array.
    """
    cfg = cfg or RlocConfig()
    return np.stack([encode(rotate_roi(roi, a), cfg) for a in cfg.rotations])


# ---------------------------------------------------------------- the metric
def _shift(arr, dy, dx, fill=_OUTSIDE):
    """dest[i, j] = arr[i + dy, j + dx], out-of-range filled with `fill`."""
    out = np.full_like(arr, fill)
    h, w = arr.shape[-2], arr.shape[-1]
    dy0, dy1 = max(0, -dy), min(h, h - dy)
    sy0, sy1 = max(0, dy), min(h, h + dy)
    dx0, dx1 = max(0, -dx), min(w, w - dx)
    sx0, sx1 = max(0, dx), min(w, w + dx)
    if dy1 > dy0 and dx1 > dx0:
        out[..., dy0:dy1, dx0:dx1] = arr[..., sy0:sy1, sx0:sx1]
    return out


def area_match(a, b, offsets):
    """
    Equations (7)-(9) for one pair, or for a stack.

    Either `a` or `b` may be a stack (n, h, w), the other (h, w). Returns a
    scalar or an (n,) array: s(a, b), the fraction of positions of `a` whose
    code appears somewhere in the corresponding neighbourhood of `b`.
    """
    a = np.asarray(a)
    b = np.asarray(b)
    hit = np.zeros(np.broadcast(a, b).shape, dtype=bool)
    for dy, dx in offsets:
        # Shifting b and comparing against a is equivalent to looking up a's
        # neighbourhood in b, and it vectorises over a whole gallery stack.
        np.logical_or(hit, a == _shift(b, dy, dx), out=hit)
    return hit.mean(axis=(-2, -1))


def _pair_scores(pa, gb, cfg):
    """Equation (9) at one alignment: s(probe, template), and with
    cfg.symmetric the larger of that and s(template, probe)."""
    s = area_match(pa, gb, cfg.offsets)
    if cfg.symmetric:
        s = np.maximum(s, area_match(gb, pa, cfg.offsets))
    return s


def _crop_pair(a, b, sy, sx):
    """Align a against b under a (sy, sx) translation, keeping the overlap."""
    h, w = a.shape[-2], a.shape[-1]
    ay0, ay1 = max(0, -sy), min(h, h - sy)
    by0, by1 = max(0, sy), min(h, h + sy)
    ax0, ax1 = max(0, -sx), min(w, w - sx)
    bx0, bx1 = max(0, sx), min(w, w + sx)
    return a[..., ay0:ay1, ax0:ax1], b[..., by0:by1, bx0:bx1]


def match_codes(probe, gallery, cfg=None):
    """
    Score a probe code against one or many gallery codes.

    probe    (h, w) uint8
    gallery  (h, w) or (n, h, w) uint8
    returns  scalar or (n,) similarity in [0, 1], maximised over the
             translation search.

    Vectorised over the gallery axis: an entire gallery of templates is scored
    in a handful of array operations rather than one Python loop iteration per
    template, which is what makes a full all-against-all experiment finish in
    minutes instead of hours.
    """
    cfg = cfg or RlocConfig()
    probe = np.asarray(probe, dtype=np.uint8)
    gallery = np.asarray(gallery, dtype=np.uint8)
    r = cfg.shift_range

    best = None
    for sy in range(-r, r + 1):
        for sx in range(-r, r + 1):
            pa, gb = _crop_pair(probe, gallery, sy, sx)
            s = _pair_scores(pa, gb, cfg)
            best = s if best is None else np.maximum(best, s)
    return best


def explain_match(probe, templates, cfg=None):
    """
    The same score as `match_codes(probe, templates).max()`, plus where it
    came from — so a match can be shown, not just reported.

    templates  (n, h, w) uint8, e.g. one identity's seven rotated codes

    Returns a dict:
      score      the best similarity over templates and shifts
      template   index of the winning template (with the default enrolment,
                 index k means cfg.rotations[k] degrees)
      rotation   that template's enrolment rotation in degrees
      shift      (sy, sx) winning translation, in CODE pixels
      direction  "probe" if the winning count was s(probe, template) — the
                 probe's cells looked up in the template — or "template"
                 if it was s(template, probe); equation (9) keeps the larger
      hit_map    (h, w) int8: 1 where a cell of the code named by
                 `direction` found an equal code in the other one's
                 neighbourhood, 0 where it did not, -1 outside the overlap.
                 Drawn in that code's own coordinates. The score is exactly
                 the mean over the non-negative cells, which is equations
                 (7)-(9) drawn as a picture.
    """
    cfg = cfg or RlocConfig()
    probe = np.asarray(probe, dtype=np.uint8)
    templates = np.asarray(templates, dtype=np.uint8)
    if templates.ndim == 2:
        templates = templates[None]
    offsets = cfg.offsets
    r = cfg.shift_range

    # ties go to the probe's direction, then to the earlier template and
    # shift — the same winner whichever way the loops run
    best = (-1.0, 0, (0, 0), "probe")
    for sy in range(-r, r + 1):
        for sx in range(-r, r + 1):
            pa, gb = _crop_pair(probe, templates, sy, sx)
            ways = [("probe", area_match(pa, gb, offsets))]
            if cfg.symmetric:
                ways.append(("template", area_match(gb, pa, offsets)))
            for way, s in ways:
                k = int(np.argmax(s))
                if s[k] > best[0]:
                    best = (float(s[k]), k, (sy, sx), way)

    best_score, best_k, (sy, sx), way = best
    pa, gb = _crop_pair(probe, templates[best_k], sy, sx)
    a, b = (pa, gb) if way == "probe" else (gb, pa)
    hit = np.zeros(a.shape, dtype=bool)
    for dy, dx in offsets:
        np.logical_or(hit, a == _shift(b, dy, dx), out=hit)

    h, w = probe.shape
    full = np.full((h, w), -1, dtype=np.int8)
    if way == "probe":
        full[max(0, -sy):min(h, h - sy), max(0, -sx):min(w, w - sx)] = hit
    else:
        full[max(0, sy):min(h, h + sy), max(0, sx):min(w, w + sx)] = hit
    rotation = (cfg.rotations[best_k] if best_k < len(cfg.rotations)
                else 0.0)
    return {"score": best_score, "template": best_k, "rotation": rotation,
            "shift": (int(sy), int(sx)), "direction": way, "hit_map": full}


# ------------------------------------------------------- many probes at once
def _neighbourhood_planes(templates, sy, sx, offsets, n_dir):
    """
    For one translation (sy, sx): a (M, n_dir*h*w) float32 matrix whose entry
    [m, (d, y, x)] is 1 when direction d occurs anywhere in template m's
    neighbourhood of probe position (y, x) — with exactly the boundary rules
    of `match_codes`: only the overlap of probe and template counts, and a
    neighbour outside the overlap never matches. Also returns the overlap's
    area, the denominator of equation (9).
    """
    if n_dir > 8:
        raise ValueError("the bit-mask formulation supports up to 8 directions")
    M, h, w = templates.shape
    ay0, ay1 = max(0, -sy), min(h, h - sy)
    ax0, ax1 = max(0, -sx), min(w, w - sx)
    by0, by1 = max(0, sy), min(h, h + sy)
    bx0, bx1 = max(0, sx), min(w, w + sx)
    # the template, moved into probe coordinates; "outside" everywhere else
    g = np.full((M, h, w), _OUTSIDE, dtype=np.uint8)
    g[:, ay0:ay1, ax0:ax1] = templates[:, by0:by1, bx0:bx1]
    # which directions occur in each neighbourhood, as a bit mask
    bit = np.zeros(256, dtype=np.uint8)
    bit[:n_dir] = (1 << np.arange(n_dir)).astype(np.uint8)
    present = np.zeros((M, h, w), dtype=np.uint8)
    for dy, dx in offsets:
        np.bitwise_or(present, bit[_shift(g, dy, dx)], out=present)
    # probe positions outside the overlap are not counted at all
    present[:, :ay0] = 0
    present[:, ay1:] = 0
    present[:, :, :ax0] = 0
    present[:, :, ax1:] = 0
    planes = (present[:, None] >> np.arange(n_dir, dtype=np.uint8)[
        None, :, None, None]) & 1
    return (planes.reshape(M, n_dir * h * w).astype(np.float32),
            (ay1 - ay0) * (ax1 - ax0))


def _one_hot(codes, n_dir):
    """(B, h, w) codes -> (B, n_dir*h*w) float32 one-hot rows."""
    b, h, w = codes.shape
    oh = codes[:, None] == np.arange(n_dir, dtype=np.uint8)[None, :, None, None]
    return oh.reshape(b, n_dir * h * w).astype(np.float32)


def scores_batch(probe_codes, templates, owner, n_identities, cfg=None,
                 chunk=512, progress=None):
    """
    Every probe against every identity: a (B, n_identities) array, IDENTICAL
    to calling Gallery.scores once per probe — the same equations (7)-(9),
    the same translation search, the same best-of-templates — but computed
    as matrix products.

    Why that is possible: a probe cell holds exactly one direction, so "does
    the probe's direction occur in the template's neighbourhood here?" is
    the dot product of the probe's one-hot code with a 0/1 mask of the
    directions present in that neighbourhood. Summed over the code, the
    numerator of equation (9) for every probe-template pair at one
    translation is a single matrix product, which numpy hands to its BLAS
    library. The counts are small integers, exact in float32, and the score
    is count / area in float64 exactly as before — so the result is
    bit-for-bit the same, which selftest.py checks.

    Equation (9)'s other direction, s(template, probe), is the same product
    with the roles swapped: the template's one-hot code against a mask of
    the directions present in the probe's neighbourhoods, built in the
    template's coordinates (the probe moved by the opposite translation).
    Both directions share the overlap, so they share the denominator.

    On the full Tongji evaluation (11,400 probes x 4,200 templates x 25
    translations x 2 directions) this is a few minutes instead of hours.
    """
    cfg = cfg or RlocConfig()
    probes = np.asarray(probe_codes, dtype=np.uint8)
    templates = np.asarray(templates, dtype=np.uint8)
    owner = np.asarray(owner, dtype=np.int64)
    n_dir = cfg.n_dir
    B = probes.shape[0]
    out = np.zeros((B, int(n_identities)), dtype=np.float64)
    if B == 0 or templates.shape[0] == 0:
        return out

    # group each identity's templates together, so the best-of-templates
    # becomes one reduceat per block
    order = np.argsort(owner, kind="stable")
    templates = templates[order]
    owner_sorted = owner[order]
    starts = np.flatnonzero(np.r_[True, owner_sorted[1:] != owner_sorted[:-1]])
    present_ids = owner_sorted[starts]
    t_hot = _one_hot(templates, n_dir) if cfg.symmetric else None

    r = cfg.shift_range
    shifts = [(sy, sx) for sy in range(-r, r + 1) for sx in range(-r, r + 1)]
    steps = len(shifts) * ((B + chunk - 1) // chunk)
    done = 0
    for sy, sx in shifts:
        planes, area = _neighbourhood_planes(templates, sy, sx, cfg.offsets,
                                             n_dir)
        for c0 in range(0, B, chunk):
            c1 = min(B, c0 + chunk)
            counts = _one_hot(probes[c0:c1], n_dir) @ planes.T
            best = np.maximum.reduceat(counts, starts, axis=1)
            if t_hot is not None:
                p_planes, p_area = _neighbourhood_planes(
                    probes[c0:c1], -sy, -sx, cfg.offsets, n_dir)
                assert p_area == area
                back = (t_hot @ p_planes.T).T
                best = np.maximum(best,
                                  np.maximum.reduceat(back, starts, axis=1))
            sc = best.astype(np.float64) / float(area)
            block = out[c0:c1, present_ids]
            out[c0:c1, present_ids] = np.maximum(block, sc)
            done += 1
            if progress:
                progress(done, steps)
    return out


# ------------------------------------------------------------------- gallery
class Gallery(object):
    """
    An enrolled population, stored as one stacked array so that a probe can
    be scored against everybody at once.

    `codes`  (M, h, w) uint8 — every rotated template of every identity
    `owner`  (M,) int        — index into `identities` for each row
    """

    def __init__(self, cfg=None):
        self.cfg = cfg or RlocConfig()
        self.identities = []
        self._codes = []
        self._owner = []
        self._stacked = None

    def enrol(self, identity, roi):
        """Enrol one ROI image under `identity`."""
        return self.enrol_codes(identity, enrol_codes(roi, self.cfg))

    def enrol_codes(self, identity, codes):
        """Enrol a precomputed (n_rot, h, w) template stack."""
        codes = np.asarray(codes, dtype=np.uint8)
        if codes.ndim == 2:
            codes = codes[None]
        if identity in self.identities:
            idx = self.identities.index(identity)
        else:
            self.identities.append(identity)
            idx = len(self.identities) - 1
        for c in codes:
            self._codes.append(c)
            self._owner.append(idx)
        self._stacked = None
        return idx

    @property
    def codes(self):
        if self._stacked is None:
            if not self._codes:
                raise ValueError("gallery is empty")
            self._stacked = np.stack(self._codes)
        return self._stacked

    @property
    def owner(self):
        return np.asarray(self._owner, dtype=np.int64)

    def __len__(self):
        return len(self.identities)

    def scores(self, probe_code):
        """Similarity of the probe against every identity: an (n_id,) array.

        Reduced by max over each identity's templates, which is the paper's
        "best of seven".
        """
        raw = match_codes(probe_code, self.codes, self.cfg)
        owner = self.owner
        out = np.zeros(len(self.identities), dtype=np.float64)
        np.maximum.at(out, owner, raw)
        return out

    def scores_batch(self, probe_codes, chunk=512, progress=None):
        """`scores` for many probes at once: (B, n_id), identical values,
        far faster. See rloc.scores_batch."""
        return scores_batch(probe_codes, self.codes, self.owner,
                            len(self.identities), self.cfg, chunk, progress)

    def verify(self, probe_code, claimed_identity):
        """One-to-one: the score for a specific claimed identity."""
        idx = self.identities.index(claimed_identity)
        return float(self.scores(probe_code)[idx])

    def identify(self, probe_code, top=5):
        """One-to-many: the ranked candidate list, best first."""
        s = self.scores(probe_code)
        order = np.argsort(-s, kind="stable")[:top]
        return [(self.identities[i], float(s[i])) for i in order]
