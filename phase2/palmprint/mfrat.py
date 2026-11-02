"""
mfrat.py — the Modified Finite Radon Transform and the orientation code.

This is the feature extractor of Paper 2 (Jia, Huang & Zhang 2008), i.e. the
answer to "what features are extracted" and "how are they extracted".

The operator
------------
The finite Radon transform (FRAT, Matus & Flusser) sums image values along
straight lines indexed by a slope k and an intercept l, modulo a prime p:

    r[k, l] = (1/sqrt(p)) * sum over (i, j) in L(k, l) of f[i, j]        (1-2)

MFRAT changes five things (Section 3 of the paper):

  1. the intercept l is dropped — every line passes through one chosen centre,
     so a line is indexed only by its direction;
  2. every direction is given the SAME number of pixels, so the responses are
     directly comparable (under FRAT they are not, because of the wrap-around
     effect);
  3. the 1/sqrt(p) scaling and the primality of p are dropped — both exist to
     make the transform invertible, and MFRAT is not invertible anyway;
  4. the line width becomes a free parameter (the paper uses 4 pixels), so the
     operator can be tuned to thick principal lines or thin wrinkles;
  5. the number of directions is untied from p (the paper uses 6).

    r[L_k] = sum over (i, j) in L_k of f[i, j]                           (3-4)

The code
--------
Winner-take-all, by ARGMIN, not argmax:

    direction(centre) = argmin_k r[L_k]                                  (5)

Palm lines are darker than the surrounding skin — the paper calls them
"negative lines" — so the direction of the line through a point is the one
whose sum along it is smallest. Taking argmax would find the brightest
direction, which is not a line at all.

What survives is one small integer per centre: the index of the winning
direction. The magnitude is thrown away. That is the whole point — an index is
an order statistic, so rescaling the image contrast cannot change it, which is
where the method's illumination robustness comes from.

Sub-sampling for free
---------------------
Because the lines are `width` pixels wide, moving the lattice centre anywhere
inside a width x width block gives the same winner. So the operator only has
to be evaluated on a grid of step = width, and a 128x128 ROI collapses to a
32x32 code image with no separate pooling step. Contrast Paper 1, which
compresses by averaging and destroys the line structure it just measured.
"""
import numpy as np

# Defaults from the paper: 16x16 lattice, 6 directions, 4-pixel-wide lines.
P_DEFAULT = 16
NDIR_DEFAULT = 6
WIDTH_DEFAULT = 4

_MASK_CACHE = {}


def mfrat_masks(p=P_DEFAULT, n_dir=NDIR_DEFAULT, width=WIDTH_DEFAULT):
    """
    One binary mask per direction: the set L_k of lattice points lying on the
    line of direction k through the centre, `width` pixels wide.

    Change 2 above (equal pixel count per direction) is enforced explicitly.
    A naive `perpendicular distance <= width/2` test gives different pixel
    counts for different angles — diagonal lines pick up more lattice points
    than axis-aligned ones — which would make the six sums incomparable and
    bias the argmin towards whichever direction happens to collect the most
    pixels. So each mask is trimmed to the smallest count among the six,
    keeping the pixels closest to the line and, where distances tie, those
    closest to the centre, so the support stays symmetric about the centre.

    Returns (masks, angles) with masks a list of (p, p) float arrays of 0.0/1.0
    and angles the direction in radians, 0 .. pi.
    """
    key = (p, n_dir, width)
    if key in _MASK_CACHE:
        return _MASK_CACHE[key]

    c = (p - 1) / 2.0
    yy, xx = np.mgrid[0:p, 0:p].astype(np.float64)

    cand, angles = [], []
    for i in range(n_dir):
        th = np.pi * i / n_dir
        # distance from the line through the centre at angle th
        perp = np.abs(-np.sin(th) * (xx - c) + np.cos(th) * (yy - c))
        # distance along that line — used only to break ties symmetrically
        along = np.abs(np.cos(th) * (xx - c) + np.sin(th) * (yy - c))
        cand.append((perp, along, perp <= width / 2.0))
        angles.append(th)

    target = int(min(m.sum() for _, _, m in cand))

    masks = []
    for perp, along, m in cand:
        # rank inside the band: perpendicular distance first, then centrality
        key_arr = np.where(m, perp * 1000.0 + along, np.inf)
        idx = np.argsort(key_arr.ravel(), kind="stable")[:target]
        k = np.zeros(p * p, dtype=np.float64)
        k[idx] = 1.0
        masks.append(k.reshape(p, p))

    _MASK_CACHE[key] = (masks, angles)
    return masks, angles


def _sparse_correlate(img, mask, pad_mode="edge"):
    """
    Correlate `img` with a sparse binary `mask`, same output size as `img`.

    A line mask holds only ~p*width non-zero lattice points out of p*p, so
    summing `nnz` shifted views of the padded image is far cheaper than a
    dense convolution and uses a fraction of the memory — it never
    materialises the (H, W, p, p) sliding-window array. With p=16 and width=4
    that is about 50 array additions per direction.
    """
    p = mask.shape[0]
    top = p // 2
    bot = p - 1 - top
    pad = np.pad(img, ((top, bot), (top, bot)), mode=pad_mode)
    h, w = img.shape
    out = np.zeros((h, w), dtype=np.float64)
    ys, xs = np.nonzero(mask)
    for dy, dx in zip(ys, xs):
        out += pad[dy:dy + h, dx:dx + w]
    return out


def mfrat_response(img, p=P_DEFAULT, n_dir=NDIR_DEFAULT, width=WIDTH_DEFAULT):
    """
    Equations (3)-(4): the sum along each of the n_dir lines, at every pixel.
    Returns an (n_dir, H, W) array.
    """
    img = np.asarray(img, dtype=np.float64)
    masks, _ = mfrat_masks(p, n_dir, width)
    return np.stack([_sparse_correlate(img, m) for m in masks])


def orientation_code(img, p=P_DEFAULT, n_dir=NDIR_DEFAULT,
                     width=WIDTH_DEFAULT, step=None):
    """
    The full feature extractor: equations (3)-(5) plus the operator's own
    sub-sampling.

    Returns (code, contrast):
      code     (H//step, W//step) uint8, values 0 .. n_dir-1, the winning
               direction index — this IS the template;
      contrast same shape, float, the spread between the largest and smallest
               directional sum at that point. NOT part of the template. The
               paper discards it; we keep it because it is a natural
               confidence measure, it is what the demo shades its orientation
               field with, and fusing it back in is the experiment we argue in
               Phase I that neither paper ran.

    `step` defaults to `width`, which is the paper's own 16:1 sub-sampling:
    a 128x128 ROI with 4-pixel lines gives a 32x32 code.
    """
    if step is None:
        step = width
    resp = mfrat_response(img, p, n_dir, width)
    code_full = np.argmin(resp, axis=0).astype(np.uint8)       # eq. (5): MIN
    contrast_full = np.ptp(resp, axis=0)
    off = step // 2                                            # block centres
    return code_full[off::step, off::step], contrast_full[off::step, off::step]


def code_agreement(a, b):
    """Fraction of positions where two code images carry the same label."""
    a = np.asarray(a)
    b = np.asarray(b)
    if a.shape != b.shape:
        raise ValueError("code shape mismatch: %s vs %s" % (a.shape, b.shape))
    return float(np.mean(a == b))
