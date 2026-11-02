"""
han.py — the comparison technique: Han, Cheng, Lin & Fan (2003).

The brief requires at least two techniques compared. Rather than quote Paper
1's published accuracy — which is not comparable with anything, since it is a
single operating point on 50 self-collected subjects with no EER, no ROC and
no baseline — this module implements it, so both techniques can be measured on
one database under one protocol. That also lets us run the apples-to-apples
experiment we argued in Phase I that Paper 2 never published.

Scope: this is a faithful implementation of Paper 1's FEATURE and TEMPLATE
MATCHING path, deliberately not of its per-user neural network. The BPNN needs
roughly ten enrolled samples per person plus negative examples drawn from
everyone else, trains one network per enrolled identity, and cannot be
operated at all from a single enrolled sample. It is excluded as a design
decision, not an oversight, and the presentation says so.

The pipeline
------------
1. Two parallel feature families on the same ROI:

   Sobel, equation (1):      f*S = max(f*S0, f*S45, f*S90, f*S135)

   Note what that maximum keeps and what it throws away. It keeps the
   magnitude — how strong the strongest oriented response was — and discards
   which mask won, i.e. the orientation of the line. Paper 2 keeps precisely
   the opposite half (see mfrat.py). The two methods differ at one line of
   code: max of the value, or argmin of the index.

   Grey-scale morphology, equations (2)-(9): oriented openings and closings
   with short line structuring elements at the same four angles, combined
   across directions (max for the algebraic opening, min for the algebraic
   closing — the same compete-across-directions idea as equation 1 in
   different clothes), cascaded into an alternating sequential filter, with
   the feature taken as the residue between two stages.

2. Grid-mean pooling into a short fixed-length vector. This is the step we
   criticise: the lines are averaged away, so what reaches the classifier is a
   coarse texture-energy descriptor, not a line descriptor. It is implemented
   faithfully, because the point is to measure the cost rather than assert it.

3. Multi-template matching by Pearson correlation, equation (10), averaged
   over the enrolled templates.

Two defects in the paper as printed, found while reimplementing it:
  - equation (10) has no 1/n normalisation, so R as written is not bounded by
    1 and cannot be thresholded the way the text describes;
  - E2 is called a contribution to the false reject rate where it is plainly
    the false accept rate.
Both are typos; both cost time. The standard Pearson definition is used here.

Scores returned by this module are SIMILARITIES: the mean Pearson correlation
over an identity's templates, in [-1, 1]. The paper's form is a distance,
1 - R; the ordering is the same.
"""
import numpy as np

# Paper 1, figure 5(a): the four 3x3 Sobel masks.
S0 = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float64)
S45 = np.array([[0, 1, 2], [-1, 0, 1], [-2, -1, 0]], dtype=np.float64)
S90 = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=np.float64)
S135 = np.array([[2, 1, 0], [1, 0, -1], [0, -1, -2]], dtype=np.float64)
SOBEL_MASKS = (S0, S45, S90, S135)

# The four angles used for both the Sobel bank and the morphological
# structuring elements.
ANGLES_DEG = (0.0, 45.0, 90.0, 135.0)


# ------------------------------------------------------------------ helpers
def _correlate3(img, k):
    """Valid 3x3 correlation, edge-padded back to the input size."""
    img = np.asarray(img, dtype=np.float64)
    w = np.lib.stride_tricks.sliding_window_view(img, k.shape)
    out = np.tensordot(w, k, axes=((2, 3), (0, 1)))
    return np.pad(out, 1, mode="edge")


def line_offsets(angle_deg, length=5):
    """
    The offsets of a line structuring element: `length` pixels centred on the
    origin at `angle_deg`. Duplicates from rounding are removed, so the
    element stays a set of distinct lattice points.
    """
    th = np.deg2rad(angle_deg)
    half = (length - 1) / 2.0
    seen, offs = set(), []
    for t in np.linspace(-half, half, length):
        dy = int(round(t * np.sin(th)))
        dx = int(round(t * np.cos(th)))
        if (dy, dx) not in seen:
            seen.add((dy, dx))
            offs.append((dy, dx))
    return tuple(offs)


def _morph(img, offsets, reduce_op):
    """
    Grey-scale dilation (reduce_op=np.maximum) or erosion (np.minimum).

    Edge padding is used so the border does not invent structure: a zero fill
    would read as a strong dark line to an operator looking for dark lines.
    The line elements here are symmetric about the origin, so reflecting the
    element — the formal difference between dilation and erosion supports —
    changes nothing.
    """
    img = np.asarray(img, dtype=np.float64)
    pad = max(max(abs(dy), abs(dx)) for dy, dx in offsets)
    p = np.pad(img, pad, mode="edge")
    h, w = img.shape
    out = None
    for dy, dx in offsets:
        v = p[pad + dy:pad + dy + h, pad + dx:pad + dx + w]
        out = v.copy() if out is None else reduce_op(out, v)
    return out


def dilate(img, offsets):
    return _morph(img, offsets, np.maximum)


def erode(img, offsets):
    return _morph(img, offsets, np.minimum)


def opening(img, offsets):
    return dilate(erode(img, offsets), offsets)


def closing(img, offsets):
    return erode(dilate(img, offsets), offsets)


# ------------------------------------------------------------ feature maps
def sobel_max(roi):
    """Equation (1): the maximum response over the four directions."""
    r = [np.abs(_correlate3(roi, k)) for k in SOBEL_MASKS]
    return np.max(np.stack(r), axis=0)


def algebraic_opening(img, length=5, angles=ANGLES_DEG):
    """Equation (4)-ish: the maximum over the four oriented openings."""
    return np.max(np.stack([opening(img, line_offsets(a, length))
                            for a in angles]), axis=0)


def algebraic_closing(img, length=5, angles=ANGLES_DEG):
    """Equation (5)-ish: the minimum over the four oriented closings."""
    return np.min(np.stack([closing(img, line_offsets(a, length))
                            for a in angles]), axis=0)


def morph_lines(roi, length=5, angles=ANGLES_DEG):
    """
    The morphological feature map, equations (6)-(9).

    Honest note on fidelity: the paper's alternating sequential filter is
    specified as a cascade ending in a residue fm = f2 - f1, but the printed
    equations do not pin down the cascade unambiguously. Implemented here as
    the oriented black top-hat — an algebraic closing, which fills in dark
    line structures, minus the original image — which is what that residue
    amounts to for negative (darker-than-skin) lines, and which is the
    operation the surrounding text describes. Treated as a baseline, not as a
    bit-exact replication, and the presentation states this.
    """
    roi = np.asarray(roi, dtype=np.float64)
    closed = algebraic_closing(roi, length, angles)
    return closed - roi


FEATURE_MAPS = {"sobel": sobel_max, "morph": morph_lines}


# ------------------------------------------------------------------ pooling
def grid_mean(img, cells):
    """
    Mean value per cell of a cells x cells grid — Paper 1's feature vector.

    The paper labels this inconsistently (Section 3 calls 32x32 a grid size,
    Section 5.2 a window size, equation 10 the length of the feature vector),
    which cannot all be true at once. Parametrised here by the number of
    OUTPUT cells, which is unambiguous and scale-independent: `cells=8` on a
    128x128 ROI means 16x16-pixel windows and a 64-element vector, matching
    the paper's Section 4.2 count.
    """
    img = np.asarray(img, dtype=np.float64)
    h, w = img.shape
    ch, cw = h // cells, w // cells
    if ch < 1 or cw < 1:
        raise ValueError("%d cells is finer than the %dx%d map" % (cells, h, w))
    crop = img[:ch * cells, :cw * cells]
    return crop.reshape(cells, ch, cells, cw).mean(axis=(1, 3))


def features(roi, mode="sobel", cells=8, strips=False, length=5):
    """
    The complete Paper 1 feature vector for one ROI.

    mode    "sobel" or "morph"
    cells   grid resolution; the paper reports 8, 16 and 32
    strips  append the row and column profile means. The paper feeds these to
            the neural network but not to template matching, so the two
            verifiers are not in fact competing on equal terms — another
            reason its headline "the network is better" claim is confounded.
    """
    if mode not in FEATURE_MAPS:
        raise ValueError("mode must be 'sobel' or 'morph'")
    fmap = FEATURE_MAPS[mode](roi) if mode == "sobel" \
        else FEATURE_MAPS[mode](roi, length)
    pooled = grid_mean(fmap, cells)
    vec = [pooled.ravel()]
    if strips:
        vec.append(pooled.mean(axis=1))
        vec.append(pooled.mean(axis=0))
    return np.concatenate(vec)


# ----------------------------------------------------------------- matching
def _zscore(v, eps=1e-12):
    """Centre and scale so a dot product becomes a Pearson correlation."""
    v = np.asarray(v, dtype=np.float64)
    m = v.mean(axis=-1, keepdims=True)
    s = v.std(axis=-1, keepdims=True)
    return (v - m) / np.maximum(s, eps)


def pearson(a, b):
    """Equation (10), with the normalisation the printed version omits."""
    az, bz = _zscore(a), _zscore(b)
    return float(np.mean(az * bz))


class HanGallery(object):
    """
    An enrolled population of Paper 1 feature vectors.

    Scoring is vectorised: all templates are stored z-scored in one (M, D)
    array, so a probe is matched against the entire population with a single
    matrix-vector product, and the Pearson correlation falls out as a dot
    product divided by the vector length.

    The per-identity score is the MEAN over that identity's templates, which
    is the paper's multi-template rule (it averages the distances). Note the
    contrast with RLOC, which takes the best of its rotated templates — there
    the templates are deliberate variants of one sample, here they are
    separate enrolment samples.
    """

    def __init__(self, mode="sobel", cells=8, strips=False, length=5):
        self.mode = mode
        self.cells = int(cells)
        self.strips = bool(strips)
        self.length = int(length)
        self.identities = []
        self._rows = []
        self._owner = []
        self._z = None

    def describe(self):
        return ("Han %s features, %dx%d grid (%d dims)%s, Pearson "
                "multi-template matching"
                % (self.mode, self.cells, self.cells,
                   self.cells * self.cells + (2 * self.cells if self.strips
                                              else 0),
                   ", with row/col strips" if self.strips else ""))

    def extract(self, roi):
        return features(roi, self.mode, self.cells, self.strips, self.length)

    def enrol(self, identity, roi):
        return self.enrol_vector(identity, self.extract(roi))

    def enrol_vector(self, identity, vec):
        if identity in self.identities:
            idx = self.identities.index(identity)
        else:
            self.identities.append(identity)
            idx = len(self.identities) - 1
        self._rows.append(np.asarray(vec, dtype=np.float64))
        self._owner.append(idx)
        self._z = None
        return idx

    @property
    def z(self):
        if self._z is None:
            if not self._rows:
                raise ValueError("gallery is empty")
            self._z = _zscore(np.stack(self._rows))
        return self._z

    def __len__(self):
        return len(self.identities)

    def scores(self, probe_vec):
        """Mean Pearson correlation per identity: an (n_id,) array."""
        pz = _zscore(np.asarray(probe_vec, dtype=np.float64))
        raw = self.z @ pz / pz.shape[-1]
        owner = np.asarray(self._owner, dtype=np.int64)
        tot = np.zeros(len(self.identities), dtype=np.float64)
        cnt = np.zeros(len(self.identities), dtype=np.float64)
        np.add.at(tot, owner, raw)
        np.add.at(cnt, owner, 1.0)
        return tot / np.maximum(cnt, 1.0)

    def verify(self, probe_vec, claimed_identity):
        idx = self.identities.index(claimed_identity)
        return float(self.scores(probe_vec)[idx])

    def identify(self, probe_vec, top=5):
        s = self.scores(probe_vec)
        order = np.argsort(-s, kind="stable")[:top]
        return [(self.identities[i], float(s[i])) for i in order]
