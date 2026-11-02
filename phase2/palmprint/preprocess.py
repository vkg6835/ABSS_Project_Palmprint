"""
preprocess.py — from a captured hand image to a repeatable 128x128 ROI.

This is the answer to "how do you obtain the data" on the input side, and it
is the stage both of our papers are weakest on. Paper 1 builds its ROI from
hand landmarks, which is the right idea, but never reports how repeatable the
result is. Paper 2 does not build one at all: it borrows Zhang et al.'s
pipeline and then admits the ROI it receives can be rotated by about ten
degrees, and spends seven stored templates per enrolment absorbing that error.
Since our Phase I critique made a point of that omission, we measure our own
(see test_preprocess.py) rather than repeat it.

The pipeline, step by step
--------------------------
1. Threshold the hand away from the background. Otsu's method is used rather
   than Paper 1's fixed value of 70: a hard-coded grey level ties the system
   to one scanner under one lamp, which was our criticism of that paper, so
   adopting it here would be indefensible. Otsu picks the threshold from each
   image's own histogram at no cost in complexity.

2. Keep the largest connected component and fill its holes, which removes
   background speckle and any dark blemish inside the hand.

3. Trace the outline. Moore boundary tracing with 8-connectivity, giving an
   ordered loop of boundary pixels.

4. Find the palm centre as the deepest interior point — the centre of the
   largest circle that fits inside the hand. The centroid is not used: it is
   dragged around by however much wrist and forearm happen to be in frame,
   which varies between captures and would make everything downstream vary
   with it.

5. Walk the outline and measure the distance to that centre. Fingertips are
   the local maxima of that signal and the finger valleys are the minima
   between them. Contour points lying on the image border are excluded first,
   because a wrist cut off by the frame edge otherwise reads as a sixth
   fingertip.

6. Build the coordinate frame from the index-middle and ring-little valleys,
   and cut a square window from it. Valleys are used rather than Paper 1's
   fingertip axis because fingertips swing a long way when the fingers splay
   and valleys barely move — which matters a great deal for a touchless
   capture, where nothing constrains the pose.

7. Resample that window to a fixed 128x128 by bilinear interpolation, so the
   ROI is scale-normalised: a larger hand, or one held closer to the camera,
   yields the same patch of skin at the same resolution.

Everything is numpy and Pillow. The connected-component labelling, hole
filling, distance transform, contour tracing and resampling are all
implemented here rather than imported, because scipy and OpenCV are not
available on the demo machine.
"""
import numpy as np

# Neighbour offsets in clockwise order, starting due north.
_N8 = ((-1, 0), (-1, 1), (0, 1), (1, 1),
       (1, 0), (1, -1), (0, -1), (-1, -1))


class PreprocessError(Exception):
    """Raised when an image cannot be reduced to an ROI.

    Worth having as a distinct exception: the fraction of images that fail
    outright is itself a result. Neither paper reports a failure rate, and a
    verification system that silently produces a garbage ROI is worse than
    one that declines.
    """


# ------------------------------------------------------------- thresholding
def otsu_threshold(img, bins=256):
    """
    The grey level maximising between-class variance.

    Replaces Paper 1's hard-coded 70. Their own justification is that "the
    environment is controlled", which is exactly the assumption that breaks
    the moment the device or the lighting changes.
    """
    img = np.asarray(img, dtype=np.float64)
    hist, edges = np.histogram(img.ravel(), bins=bins, range=(0.0, 1.0))
    hist = hist.astype(np.float64)
    total = hist.sum()
    if total == 0:
        raise PreprocessError("empty image")
    p = hist / total
    centres = (edges[:-1] + edges[1:]) / 2.0
    w0 = np.cumsum(p)
    w1 = 1.0 - w0
    m0 = np.cumsum(p * centres)
    mt = m0[-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        mu0 = m0 / w0
        mu1 = (mt - m0) / w1
        between = w0 * w1 * (mu0 - mu1) ** 2
    between = np.nan_to_num(between, nan=-1.0, posinf=-1.0, neginf=-1.0)
    return float(centres[int(np.argmax(between))])


# ------------------------------------------------- connected components
def _erode8(mask):
    """One step of binary erosion; outside the image counts as background."""
    h, w = mask.shape
    p = np.pad(mask, 1, mode="constant", constant_values=False)
    out = mask.copy()
    for dy, dx in _N8:
        out &= p[1 + dy:1 + dy + h, 1 + dx:1 + dx + w]
    return out


def label_components(mask):
    """
    Label 8-connected components. Returns (labels, count), labels 1..count.

    Run-length based with union-find: each row is reduced to runs of set
    pixels, runs are joined to any touching run on the previous row, and the
    labels are resolved at the end. That keeps the Python loop at one
    iteration per RUN rather than one per pixel, which is what makes this
    usable without scipy.

    Both rows' runs are sorted, so the runs of the previous row that can
    touch a given run form a contiguous block, found by a pointer that only
    moves forward. An earlier version compared every run against every run on
    the row above, which is quadratic in the number of runs: harmless on a
    clean synthetic hand with two or three runs per row, and seconds per
    image on a noisy photograph where thresholding leaves hundreds. The
    result is identical; only the cost changed.
    """
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    parent = [0]

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    def runs_of(row):
        d = np.diff(np.concatenate(([0], row.view(np.int8), [0])))
        starts = np.flatnonzero(d == 1)
        ends = np.flatnonzero(d == -1)        # exclusive
        return list(zip(starts.tolist(), ends.tolist()))

    labels = np.zeros((h, w), dtype=np.int32)
    prev = []                     # previous row: (start, end_exclusive, label)
    for y in range(h):
        cur = []
        n_prev = len(prev)
        j0 = 0
        for (s, e) in runs_of(mask[y]):
            # 8-connectivity: runs (ps, pe) and (s, e) touch, diagonally
            # included, exactly when ps <= e and s <= pe.
            while j0 < n_prev and prev[j0][1] < s:
                j0 += 1
            lab = 0
            j = j0
            while j < n_prev and prev[j][0] <= e:
                pl = prev[j][2]
                if lab == 0:
                    lab = pl
                else:
                    union(lab, pl)
                j += 1
            if lab == 0:
                parent.append(len(parent))
                lab = len(parent) - 1
            labels[y, s:e] = lab
            cur.append((s, e, lab))
        prev = cur

    if len(parent) == 1:
        return labels, 0
    roots = np.array([find(i) for i in range(len(parent))], dtype=np.int32)
    uniq, inv = np.unique(roots[1:], return_inverse=True)
    remap = np.zeros(len(parent), dtype=np.int32)
    remap[1:] = inv.ravel().astype(np.int32) + 1
    return remap[labels], int(len(uniq))


def largest_component(mask):
    """The biggest 8-connected blob, as a boolean mask."""
    labels, n = label_components(mask)
    if n == 0:
        raise PreprocessError("nothing above the threshold")
    counts = np.bincount(labels.ravel())
    counts[0] = 0
    return labels == int(np.argmax(counts))


def fill_holes(mask):
    """
    Fill enclosed background regions.

    Any background component that does not touch the image border is interior
    and gets filled: a dark mark on the palm must not become a hole in the
    silhouette, or the outline trace will follow it instead of the hand.
    Done with one lookup table rather than one pass per hole, so a mask
    riddled with small holes costs the same as a clean one.
    """
    mask = np.asarray(mask, dtype=bool)
    labels, n = label_components(~mask)
    if n == 0:
        return mask.copy()
    border = np.concatenate([labels[0], labels[-1],
                             labels[:, 0], labels[:, -1]])
    interior = np.ones(n + 1, dtype=bool)
    interior[0] = False
    interior[np.unique(border)] = False
    return mask | interior[labels]


def segment_hand(img, threshold=None, min_area_frac=0.02, polarity="auto"):
    """
    Steps 1 and 2: binary hand silhouette.

    polarity  "bright"  the hand is lighter than the background — a lit hand
                        against a dark backdrop, as in our synthetic data;
              "dark"    the reverse — a hand photographed on white paper;
              "auto"    try both and keep the one that is shaped like a hand.

    Auto relies on a property of the SCENE rather than of the hand: a
    background is a frame around its subject, so its largest component runs
    along most of the image border, whereas a hand meets the border only
    where the wrist leaves the frame. Of the two candidate silhouettes, the
    one touching less of the border is the hand. That matters for the demo,
    which accepts arbitrary photographs, and for any database whose backdrop
    turns out to be lighter than skin.

    Returns (mask, threshold, polarity_used).
    """
    img = np.asarray(img, dtype=np.float64)
    t = otsu_threshold(img) if threshold is None else float(threshold)
    options = {"bright": img > t, "dark": img <= t}
    if polarity == "auto":
        order = ("bright", "dark")
    elif polarity in options:
        order = (polarity,)
    else:
        raise ValueError("polarity must be 'auto', 'bright' or 'dark'")

    best = None
    for name in order:
        try:
            comp = largest_component(options[name])
        except PreprocessError:
            continue
        if comp.mean() < min_area_frac:
            continue
        border = np.concatenate([comp[0], comp[-1],
                                 comp[1:-1, 0], comp[1:-1, -1]])
        touch = float(border.mean())
        if best is None or touch < best[0]:
            best = (touch, name, comp)
    if best is None:
        raise PreprocessError("no region of the image is large enough to "
                              "be a hand")
    _, name, comp = best
    return fill_holes(comp), t, name


# --------------------------------------------------------- shape analysis
def depth_map(mask, max_iter=4096):
    """
    Distance from each interior pixel to the nearest background pixel, by
    repeated erosion. Chebyshev rather than Euclidean, which is close enough
    to locate the deepest point and costs a handful of array operations per
    unit of distance.
    """
    mask = np.asarray(mask, dtype=bool)
    d = mask.astype(np.int32)
    cur = mask
    k = 0
    while cur.any() and k < max_iter:
        cur = _erode8(cur)
        k += 1
        d[cur] = k + 1
    return d


def palm_centre(mask):
    """Step 4: the deepest interior point — the inscribed-circle centre."""
    d = depth_map(mask)
    flat = int(np.argmax(d))
    return (int(flat // mask.shape[1]), int(flat % mask.shape[1])), int(d.max())


def trace_contour(mask, max_steps=None):
    """
    Step 3: Moore boundary tracing, 8-connected, clockwise.

    Returns an (n, 2) array of (row, col) in traversal order.
    """
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    idx = np.argwhere(mask)
    if idx.size == 0:
        raise PreprocessError("empty mask")
    start = (int(idx[0][0]), int(idx[0][1]))     # topmost, then leftmost
    if max_steps is None:
        max_steps = 8 * (h + w) * 4

    contour = [start]
    cur = start
    back = 6                                     # we arrived from the west
    for _ in range(max_steps):
        nxt = None
        for k in range(1, 9):
            d = (back + k) % 8
            ny = cur[0] + _N8[d][0]
            nx = cur[1] + _N8[d][1]
            if 0 <= ny < h and 0 <= nx < w and mask[ny, nx]:
                nxt, took = (ny, nx), d
                break
        if nxt is None:
            break                                # isolated pixel
        back = (took + 4) % 8
        cur = nxt
        if cur == start:
            break
        contour.append(cur)
    if len(contour) < 16:
        raise PreprocessError("outline too short (%d points)" % len(contour))
    return np.array(contour, dtype=np.int32)


def _smooth_circular(v, win):
    """Moving average on a closed loop."""
    win = max(3, int(win) | 1)
    k = np.ones(win) / win
    pad = win // 2
    ext = np.concatenate([v[-pad:], v, v[:pad]])
    return np.convolve(ext, k, mode="valid")


def _local_maxima_circular(v, min_sep):
    """
    Indices of local maxima on a CLOSED loop, thinned by separation and
    taken strongest first.

    The outline of a hand is a loop, and treating it as one removes a whole
    class of bug. The alternative — cutting it open at the wrist and scanning
    the arc — requires knowing where the wrist is, which is only easy when
    the wrist leaves the frame. When the whole hand fits inside the image the
    loop never breaks, any cut lands in the middle of the hand, and excluding
    the ends of the arc silently discards a real fingertip.
    """
    n = len(v)
    cand = [i for i in range(n)
            if v[i] >= v[(i - 1) % n] and v[i] >= v[(i + 1) % n]]
    cand.sort(key=lambda i: -v[i])
    keep = []
    for i in cand:
        if all(min(abs(i - j), n - abs(i - j)) >= min_sep for j in keep):
            keep.append(i)
    return sorted(keep)


def find_keypoints(mask, contour=None, centre=None,
                   smooth_frac=0.012, sep_frac=0.03):
    """
    Step 5: find every protrusion of the outline and every concavity between
    two neighbouring protrusions.

    Deliberately does NOT try to decide which protrusion is which digit.
    Counting the five fingertips and working outwards looks natural and is
    brittle, because the wrist is a protrusion too: when it leaves the frame
    its cut edge is the farthest point of the entire outline, and when it
    fits inside the frame its rounded end is a genuine local maximum. Either
    way it competes with the thumb — which really is shorter than the
    fingers — for a place in the top five, so the identity of every tip
    shifts with however much forearm happened to be in shot.

    Instead, each neighbouring pair of protrusions defines a candidate
    concavity, described by how deep it is (prominence) and how far apart its
    flanking protrusions are (arc). Choosing among those is left to
    select_finger_valleys, which needs no notion of which finger is which.
    """
    if contour is None:
        contour = trace_contour(mask)
    if centre is None:
        centre, _ = palm_centre(mask)

    cy, cx = centre
    n = len(contour)
    r = np.hypot(contour[:, 0] - cy, contour[:, 1] - cx).astype(np.float64)
    rs = _smooth_circular(r, max(3, int(smooth_frac * n)))

    peaks = _local_maxima_circular(rs, max(4, int(sep_frac * n)))
    if len(peaks) < 4:
        raise PreprocessError("the outline has only %d bumps, so the "
                              "fingers are not visible" % len(peaks))

    cands = []
    for k, a in enumerate(peaks):
        b = peaks[(k + 1) % len(peaks)]
        arc = (b - a) % n
        if arc < 2:
            continue
        seg = np.arange(a, a + arc + 1) % n
        j = int(seg[int(np.argmin(rs[seg]))])
        cands.append({"index": j,
                      "arc": int(arc),
                      "prominence": float(min(rs[a], rs[b]) - rs[j]),
                      "point": tuple(int(v) for v in contour[j])})

    return {
        "tips": [tuple(int(v) for v in contour[i]) for i in peaks],
        "candidates": cands,
        "profile": rs,
        "contour_length": n,
        "centre": (int(cy), int(cx)),
        "max_radius": float(rs.max()),
    }


def select_finger_valleys(kp, min_prominence_frac=0.08, max_arc_frac=0.15):
    """
    Choose the two valleys the ROI is built from, without identifying digits.

    The four fingers are adjacent, similar in length and close together, so
    the three concavities between them are the NARROWEST well-formed
    concavities on the outline: their flanking protrusions are a short arc
    apart. Every competing candidate is wide — the thumb sits far from the
    index finger, and the wrist is far from everything — so ranking the
    candidates by arc length and keeping three isolates the inter-finger
    valleys on any hand pose, with no need to work out which tip is the thumb
    and no dependence on how much wrist is in frame.

    Of those three, the two that are FARTHEST APART in the image are the
    index-middle and ring-little valleys, and the third is the middle-ring
    valley between them. Selecting by distance rather than by position along
    the outline avoids having to reason about where the loop happens to
    start, and the pair spanning the full width of the four fingers is the
    standard and most stable baseline for a palm ROI.

    A candidate must also be reasonably deep (`min_prominence_frac` of the
    outline's maximum radius) so that a ripple in a noisy contour cannot pose
    as a valley.
    """
    cands = [c for c in kp["candidates"]
             if c["prominence"] >= min_prominence_frac * kp["max_radius"]
             and c["arc"] <= max_arc_frac * kp["contour_length"]]
    if len(cands) < 3:
        raise PreprocessError(
            "only %d of the %d gaps in the outline look like the valleys "
            "between fingers" % (len(cands), len(kp["candidates"])))

    narrow = sorted(cands, key=lambda c: c["arc"])[:3]
    pts = [c["point"] for c in narrow]
    best, pair = -1.0, (0, 1)
    for i in range(3):
        for j in range(i + 1, 3):
            d = float(np.hypot(pts[i][0] - pts[j][0], pts[i][1] - pts[j][1]))
            if d > best:
                best, pair = d, (i, j)
    mid = ({0, 1, 2} - set(pair)).pop()
    return pts[pair[0]], pts[pair[1]], pts[mid]


# ------------------------------------------------------------- the ROI
def roi_frame(v_a, v_b, centre, side_factor=1.35, offset_factor=0.95):
    """
    Step 6: the coordinate frame, from the two outer finger valleys.

    The line between the valleys is the ROI's horizontal axis; the window is
    pushed along the inward normal, towards the palm centre, by
    `offset_factor` of that line's length, and its side is `side_factor` of
    the same length. Expressing both in units of the valley separation is
    what makes the ROI scale-invariant: a hand held closer to the camera
    yields the same patch of skin at the same resolution.

    The defaults put the window between roughly 0.3 and 1.6 valley-
    separations below the valley line, which on a hand is the central palm —
    clear of the finger bases above and of the wrist crease below. That
    choice is geometric, and deliberately so. Sweeping the two factors
    against our synthetic fixture shows the genuine-impostor gap rising
    monotonically as the window grows (0.245 at 1.25/0.70 up to 0.328 at
    1.50/1.10), but that is the fixture rewarding sheer palm area, since it
    paints texture across the whole palm. Picking the maximum would be tuning
    on synthetic data and would push the window onto the wrist crease of a
    real hand. To be re-validated on real images, not re-tuned on these.

    Handedness is pinned down here rather than upstream. The two valleys
    arrive in outline order, and which one comes first depends on the
    direction the outline happened to be traced — so taking them as given
    would flip the axis, and with the normal always turned towards the palm,
    a flipped axis means an ROI rotated by 180 degrees. Two images of one
    palm could then produce opposite crops and score as an impostor pair.
    The fix is to demand a fixed orientation: the normal must be the
    anticlockwise quarter-turn of the axis AND point towards the palm
    centre, which uniquely determines the order of the two valleys.

    Returns (origin, ex, ey, side) with origin the top-left corner of the
    window in (row, col) and ex, ey its unit axes.
    """
    a = np.array(v_a, dtype=np.float64)
    b = np.array(v_b, dtype=np.float64)
    c = np.array(centre, dtype=np.float64)

    d = b - a
    L = float(np.hypot(d[0], d[1]))
    if L < 1e-6:
        raise PreprocessError("the two valleys coincide")

    ex = d / L
    nrm = np.array([-ex[1], ex[0]])          # quarter turn, fixed sense
    mid = (a + b) / 2.0
    if np.dot(c - mid, nrm) < 0:
        # the fixed-sense normal points away from the palm, so the valleys
        # arrived in the other order: swap them rather than flip the normal
        ex = -ex
        nrm = np.array([-ex[1], ex[0]])

    side = side_factor * L
    roi_centre = mid + nrm * (offset_factor * L)
    origin = roi_centre - ex * (side / 2.0) - nrm * (side / 2.0)
    return origin, ex, nrm, side


def sample_patch(img, origin, ex, ey, side, size=128):
    """
    Step 7: bilinear resample of the window onto a size x size grid.

    Written out rather than imported: it is four weighted lookups per output
    pixel, it keeps the dependency list at numpy, and doing it explicitly
    makes the geometry auditable in the presentation.
    """
    img = np.asarray(img, dtype=np.float64)
    h, w = img.shape
    t = (np.arange(size) + 0.5) / size * side
    gx, gy = np.meshgrid(t, t)
    rr = origin[0] + ey[0] * gy + ex[0] * gx
    cc = origin[1] + ey[1] * gy + ex[1] * gx

    r0 = np.clip(np.floor(rr).astype(np.int64), 0, h - 1)
    c0 = np.clip(np.floor(cc).astype(np.int64), 0, w - 1)
    r1 = np.clip(r0 + 1, 0, h - 1)
    c1 = np.clip(c0 + 1, 0, w - 1)
    fr = np.clip(rr - r0, 0.0, 1.0)
    fc = np.clip(cc - c0, 0.0, 1.0)

    return ((1 - fr) * (1 - fc) * img[r0, c0]
            + (1 - fr) * fc * img[r0, c1]
            + fr * (1 - fc) * img[r1, c0]
            + fr * fc * img[r1, c1])


def extract_roi(img, size=128, side_factor=1.35, offset_factor=0.95,
                threshold=None, diagnostics=False, polarity="auto"):
    """
    The whole pipeline: hand image in, ROI out.

    With diagnostics=True also returns every intermediate stage, which is
    what the demo walks through on screen and what the presentation's figures
    are drawn from.
    """
    img = np.asarray(img, dtype=np.float64)
    if img.ndim != 2:
        raise PreprocessError("expected a single-channel image")

    mask, t, used = segment_hand(img, threshold, polarity=polarity)
    contour = trace_contour(mask)
    centre, depth = palm_centre(mask)
    kp = find_keypoints(mask, contour, centre)
    v_a, v_b, v_mid = select_finger_valleys(kp)
    origin, ex, ey, side = roi_frame(v_a, v_b, centre,
                                     side_factor, offset_factor)
    roi = sample_patch(img, origin, ex, ey, side, size)

    if not diagnostics:
        return roi
    corners = [origin,
               origin + ex * side,
               origin + ex * side + ey * side,
               origin + ey * side]
    info = {
        "threshold": t, "polarity": used, "mask": mask, "contour": contour,
        "centre": centre, "palm_depth": depth,
        "tips": kp["tips"],
        "valleys": [c["point"] for c in kp["candidates"]],
        "profile": kp["profile"],
        "roi_valleys": (v_a, v_b), "middle_valley": v_mid,
        "origin": origin, "ex": ex, "ey": ey, "side": side,
        "corners": [(float(c[0]), float(c[1])) for c in corners],
        "angle_deg": float(np.degrees(np.arctan2(ex[0], ex[1]))),
    }
    return roi, info
