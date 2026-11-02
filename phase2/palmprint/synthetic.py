"""
synthetic.py — parametric synthetic palms, for testing before data arrives.

Every public palmprint database requires a signed agreement and a wait of
days to weeks. This module exists so that none of that time is dead: it
generates ROI images of distinct synthetic "identities", each sampled with
the kinds of variation a real second session introduces, so the operator, the
matcher, the protocol and the metrics can all be built and validated against
a known ground truth and only then pointed at real images.

It is a TEST FIXTURE, not a result. Nothing produced here goes in the
presentation as a performance number: synthetic palms are far cleaner than
real ones and any EER measured on them is meaningless as a claim. What they
do establish is that the pipeline is correct — genuine pairs score high,
impostor pairs score low, and the degradations behave as the papers predict.

Identity model
--------------
A palm is a handful of dark curved strokes on a lighter field, which is a
reasonable caricature of principal lines and wrinkles, and exactly the
structure both papers' operators are built to find. An identity is a frozen
set of stroke parameters drawn from a seeded generator; a sample of that
identity re-renders those strokes under a fresh rotation, translation,
illumination and noise draw.

Session variation is deliberately set to mirror what the papers complain
about: rotation up to about ten degrees (the misalignment Paper 2 inherits
from its preprocessing and spends seven templates absorbing), a few pixels of
translation, and a contrast and brightness change (the illumination problem
that motivates throwing magnitude away in favour of an orientation index).

Geometry is applied by transforming the sampling grid rather than by rotating
a rendered bitmap, so there is no interpolation blur and no border fill to
confuse an operator whose job is to find dark lines.
"""
import numpy as np

# Session-to-session variation. Defaults chosen to match the papers' own
# statements about their preprocessing error, not to flatter the result.
ROT_DEG = 10.0      # Paper 2 states its inherited ROI error is about +/-10 deg
SHIFT_PX = 4.0
GAIN_RANGE = (0.75, 1.25)
BIAS_RANGE = (-0.10, 0.10)
NOISE_SD = 0.012


def palm_params(identity_seed, n_principal=4, n_crease=9):
    """
    Freeze one identity: a list of strokes, each a dict with
      x0, y0   a point the stroke passes through, in ROI-fraction units
      angle    radians
      curve    curvature, in inverse ROI widths
      width    Gaussian half-width in ROI fractions
      depth    how much darker than the field
      extent   stroke length in ROI fractions
    """
    rng = np.random.default_rng(int(identity_seed) * 7919 + 13)
    strokes = []

    # Principal lines: long, deep, well separated in angle so that the
    # orientation code of two identities actually differs.
    base = rng.permutation(np.linspace(0.0, np.pi, n_principal, endpoint=False))
    for i in range(n_principal):
        strokes.append({
            "x0": rng.uniform(0.25, 0.75),
            "y0": rng.uniform(0.25, 0.75),
            "angle": float(base[i] + rng.normal(0.0, 0.16)),
            "curve": float(rng.normal(0.0, 0.9)),
            "width": float(rng.uniform(0.016, 0.026)),
            "depth": float(rng.uniform(0.26, 0.40)),
            "extent": float(rng.uniform(0.55, 0.95)),
        })

    # Secondary creases: shorter, shallower, arbitrary angles. These are what
    # fill the lattice so most 16x16 windows contain some line to vote on.
    for _ in range(n_crease):
        strokes.append({
            "x0": float(rng.uniform(0.10, 0.90)),
            "y0": float(rng.uniform(0.10, 0.90)),
            "angle": float(rng.uniform(0.0, np.pi)),
            "curve": float(rng.normal(0.0, 1.4)),
            "width": float(rng.uniform(0.010, 0.017)),
            "depth": float(rng.uniform(0.10, 0.20)),
            "extent": float(rng.uniform(0.20, 0.45)),
        })
    return strokes


def render(strokes, size=128, rot_deg=0.0, dx=0.0, dy=0.0,
           gain=1.0, bias=0.0, noise=0.0, light_angle=None, seed=0):
    """
    Render one sample. `rot_deg`, `dx`, `dy` are the misalignment of the ROI
    window relative to the palm; gain and bias are the illumination change.
    """
    rng = np.random.default_rng(seed)
    n = int(size)

    # Sampling grid in ROI-fraction units, centred on the ROI.
    t = (np.arange(n) + 0.5) / n - 0.5
    xx, yy = np.meshgrid(t, t)

    # Undo the window's own rotation/translation to land in palm coordinates.
    th = np.deg2rad(rot_deg)
    ct, st = np.cos(th), np.sin(th)
    xs = ct * xx + st * yy - dx / n
    ys = -st * xx + ct * yy - dy / n
    xs = xs + 0.5
    ys = ys + 0.5

    img = np.full((n, n), 0.78, dtype=np.float64)

    for s in strokes:
        px = xs - s["x0"]
        py = ys - s["y0"]
        ca, sa = np.cos(s["angle"]), np.sin(s["angle"])
        along = ca * px + sa * py
        perp = -sa * px + ca * py
        perp = perp - s["curve"] * along * along          # gentle curvature
        core = np.exp(-(perp ** 2) / (2.0 * s["width"] ** 2))
        # soft finite length: flat in the middle, falling off past `extent`
        window = np.exp(-((along / s["extent"]) ** 4))
        img -= s["depth"] * core * window

    # Smooth illumination gradient, direction varying per sample.
    if light_angle is None:
        light_angle = rng.uniform(0.0, 2.0 * np.pi)
    img += 0.07 * (np.cos(light_angle) * (xx) + np.sin(light_angle) * (yy))

    img = gain * img + bias
    if noise > 0:
        img = img + rng.normal(0.0, noise, img.shape)
    return np.clip(img, 0.0, 1.0)


def sample(identity_seed, sample_index, size=128, vary=True,
           rot_deg=ROT_DEG, shift_px=SHIFT_PX, noise_sd=NOISE_SD):
    """
    One image of one identity. sample_index 0 is the clean, centred,
    nominally-lit view — a sensible thing to enrol — and the rest carry fresh
    session variation.
    """
    strokes = palm_params(identity_seed)
    seed = int(identity_seed) * 100003 + int(sample_index)
    if not vary or sample_index == 0:
        return render(strokes, size, noise=noise_sd, light_angle=0.0,
                      seed=seed)
    rng = np.random.default_rng(seed)
    return render(
        strokes, size,
        rot_deg=float(rng.uniform(-rot_deg, rot_deg)),
        dx=float(rng.uniform(-shift_px, shift_px)),
        dy=float(rng.uniform(-shift_px, shift_px)),
        gain=float(rng.uniform(*GAIN_RANGE)),
        bias=float(rng.uniform(*BIAS_RANGE)),
        noise=noise_sd, seed=seed)


def dataset(n_identities=20, n_samples=6, size=128, **kw):
    """
    A whole synthetic corpus as a list of (identity, sample_index, image),
    with identities named like real ones so protocol code is exercised the
    same way it will be on the IITD set.
    """
    out = []
    for i in range(n_identities):
        name = "S%03d" % (i + 1)
        for j in range(n_samples):
            out.append((name, j, sample(i + 1, j, size=size, **kw)))
    return out


# ======================================================================
# Synthetic HANDS, for testing ROI extraction
# ======================================================================
# Everything above renders an ROI directly, which is the right fixture for
# the operator and the matcher. It is the wrong fixture for preprocessing,
# which has to find the ROI in the first place: segment the hand, locate the
# finger valleys, and place a window that lands on the same patch of skin
# every time.
#
# So these functions render a whole hand — palm, five fingers, wrist — on a
# dark background, under a known rotation, translation and scale, and return
# the ground-truth valley positions along with the image. That makes ROI
# repeatability measurable rather than assumed, which matters here: our Phase
# I critique of Paper 1 was precisely that it never quantified how repeatable
# its ROI is, and Paper 2 inherits a preprocessing stage it admits can be
# wrong by about ten degrees. We should not repeat the omission we criticised.
#
# The hand is built from capsules (segments with thickness) combined with a
# min() of their signed distance functions, which is a union. No external
# geometry library, and the boundary is smooth enough to trace.

# Hand layout in palm-width units: palm centred at the origin, +y upwards.
# (base_x, base_y, tip_x, tip_y, radius) per digit, index -> little.
FINGERS = (
    (-0.300, 0.46, -0.400, 1.26, 0.085),   # index
    (-0.100, 0.50, -0.120, 1.40, 0.088),   # middle
    (0.100, 0.49, 0.145, 1.31, 0.085),     # ring
    (0.290, 0.44, 0.405, 1.06, 0.078),     # little
)
THUMB = (-0.430, -0.02, -0.880, 0.46, 0.105)
PALM_HALF_W, PALM_TOP, PALM_BOTTOM = 0.44, 0.47, -0.52
WRIST = (0.0, -0.45, 0.0, -1.35, 0.30)


def hand_valleys():
    """
    APPROXIMATE finger-valley positions in palm-width units.

    A valley sits between two adjacent digits, near the height where their
    capsules merge; this returns the midpoint of the two finger bases, nudged
    outwards, which is within a few pixels of the true boundary minimum but
    is not exact. It is deliberately only used as a loose sanity bound on the
    detector (does it find a valley roughly where one is?), never as the
    measure of quality.

    The measure that matters is ROI REPEATABILITY — render one identity under
    different rotations, translations and scales, extract the ROI from each,
    and ask how similar the results are. That needs no ground truth at all,
    and it is the number Paper 1 never reports and Paper 2 inherits as an
    unquantified error of "about ten degrees". See test_preprocess.py.

    Returned in contour order: thumb-index, index-middle, middle-ring,
    ring-little. The second and fourth are the pair the ROI is built from,
    which is the standard construction and more stable than Paper 1's
    fingertip-based axis, because fingertips move a long way when fingers
    splay and valleys barely move at all.
    """
    pts = []
    digits = [THUMB] + list(FINGERS)
    for (a, b) in zip(digits[:-1], digits[1:]):
        ax, ay = a[0], a[1]
        bx, by = b[0], b[1]
        pts.append(((ax + bx) / 2.0, (ay + by) / 2.0 + 0.06))
    return pts


def _capsule_sdf(xx, yy, x0, y0, x1, y1, r):
    """Signed distance to a thick segment; negative inside."""
    px, py = xx - x0, yy - y0
    dx, dy = x1 - x0, y1 - y0
    L2 = dx * dx + dy * dy
    t = np.clip((px * dx + py * dy) / L2, 0.0, 1.0)
    return np.sqrt((px - t * dx) ** 2 + (py - t * dy) ** 2) - r


def render_hand(identity_seed, size=320, rot_deg=0.0, dx=0.0, dy=0.0,
                scale=1.0, gain=1.0, bias=0.0, noise=NOISE_SD,
                background=0.07, seed=0, fingers=None):
    """
    Render a whole hand on a dark background.

    Returns (image, truth) where truth carries the valley pixel coordinates,
    the palm centre and the transform actually applied, so a test can compare
    what preprocessing recovers against what was drawn.

    `fingers` overrides the four finger capsules; passing an empty tuple
    draws a mitten — palm, thumb and wrist with no fingers — which has no
    finger valleys at all. That is the demo's deliberately unreadable input:
    preprocessing must decline it rather than guess an ROI.
    """
    if fingers is None:
        fingers = FINGERS
    rng = np.random.default_rng(seed)
    n = int(size)
    # Image coordinates: +y DOWN, as in an array. Palm-width units, with the
    # palm about a third of the frame so fingers and wrist both fit.
    span = 3.2 / scale
    t = ((np.arange(n) + 0.5) / n - 0.5) * span
    xx, yy = np.meshgrid(t, t)
    xx = xx - dx / n * span
    yy = yy - dy / n * span

    th = np.deg2rad(rot_deg)
    ct, st = np.cos(th), np.sin(th)
    hx = ct * xx + st * yy
    hy = -st * xx + ct * yy
    hy = -hy                      # flip so +y is up in hand coordinates

    parts = [_capsule_sdf(hx, hy, *THUMB), _capsule_sdf(hx, hy, *WRIST)]
    for f in fingers:
        parts.append(_capsule_sdf(hx, hy, *f))
    # palm as a thick vertical capsule, wide enough to span the finger bases
    parts.append(_capsule_sdf(hx, hy, 0.0, PALM_BOTTOM + 0.1,
                              0.0, PALM_TOP - 0.1, PALM_HALF_W))
    sdf = parts[0]
    for p in parts[1:]:
        sdf = np.minimum(sdf, p)

    # Soft edge, a couple of pixels wide, so the boundary is not aliased.
    edge = 1.6 * span / n
    inside = np.clip(0.5 - sdf / edge, 0.0, 1.0)

    # Palm texture: the same stroke model as the ROI renderer, in hand
    # coordinates, so the lines move with the hand exactly as real ones do.
    # Confined to the palm proper by a soft window, since lines running up
    # the fingers and down the wrist would be an artefact of the fixture.
    strokes = palm_params(identity_seed)
    palm_only = np.clip(0.5 - _capsule_sdf(hx, hy, 0.0, PALM_BOTTOM + 0.12,
                                           0.0, PALM_TOP - 0.12,
                                           PALM_HALF_W - 0.02) / 0.10,
                        0.0, 1.0)
    skin = np.full((n, n), 0.78, dtype=np.float64)
    for s in strokes:
        # stroke coordinates are ROI fractions; map onto the palm region
        px = (hx / 0.95 + 0.5) - s["x0"]
        py = (-hy / 0.95 + 0.5) - s["y0"]
        ca, sa = np.cos(s["angle"]), np.sin(s["angle"])
        along = ca * px + sa * py
        perp = -sa * px + ca * py - s["curve"] * (ca * px + sa * py) ** 2
        skin -= s["depth"] * np.exp(-(perp ** 2) / (2.0 * s["width"] ** 2)) \
            * np.exp(-((along / s["extent"]) ** 4)) * palm_only

    img = background + (skin - background) * inside
    img = gain * img + bias
    if noise > 0:
        img = img + rng.normal(0.0, noise, img.shape)
    img = np.clip(img, 0.0, 1.0)

    # Ground truth, mapped back into pixel coordinates.
    def to_px(x, y):
        yy_ = -y
        rx = ct * x - st * yy_
        ry = st * x + ct * yy_
        col = (rx / span + 0.5) * n - 0.5 + dx
        row = (ry / span + 0.5) * n - 0.5 + dy
        return float(col), float(row)

    truth = {
        "valleys": [to_px(x, y) for (x, y) in hand_valleys()],
        "valley_names": ["thumb-index", "index-middle",
                         "middle-ring", "ring-little"],
        "palm_centre": to_px(0.0, 0.0),
        "rot_deg": float(rot_deg), "dx": float(dx), "dy": float(dy),
        "scale": float(scale), "span": float(span),
    }
    return img, truth


def hand_sample(identity_seed, sample_index, size=320,
                rot_deg=12.0, shift_px=14.0, scale_jitter=0.10):
    """One captured hand image of one identity, with session variation."""
    seed = int(identity_seed) * 100019 + int(sample_index)
    if sample_index == 0:
        return render_hand(identity_seed, size=size, seed=seed)
    rng = np.random.default_rng(seed)
    return render_hand(
        identity_seed, size=size,
        rot_deg=float(rng.uniform(-rot_deg, rot_deg)),
        dx=float(rng.uniform(-shift_px, shift_px)),
        dy=float(rng.uniform(-shift_px, shift_px)),
        scale=float(rng.uniform(1.0 - scale_jitter, 1.0 + scale_jitter)),
        gain=float(rng.uniform(*GAIN_RANGE)),
        bias=float(rng.uniform(*BIAS_RANGE)),
        seed=seed)
