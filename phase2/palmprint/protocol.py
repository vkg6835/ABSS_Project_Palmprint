"""
protocol.py — datasets, the enrolment/probe protocol, and experiment runners.

This is where "how do you acquire the data" and "how do you do verification
with an input" become executable. Keeping the protocol in one module matters
for honesty as much as for tidiness: the enrolment/probe split is declared
once, up front, and every reported number comes out of the same function, so
there is no opportunity to quietly tune a threshold on the data it is later
measured against.

Protocol
--------
The first `n_enrol` samples of each identity are enrolled; every other image
of that identity is a probe. Each probe is compared against every enrolled
identity: the comparison with its own identity is a GENUINE score, the
comparisons with all other identities are IMPOSTOR scores. With N identities
and P probes that gives P genuine and P*(N-1) impostor comparisons.

This is Paper 2's protocol, and with n_enrol=1 it is also the realistic one:
at a real enrolment desk you get one scan, not thirty over three weeks. It
is deliberately harsher than Paper 1's, which enrols ten samples per person
and fits a threshold per user by cross-validation.

Memory note: images are encoded to templates one at a time and then released.
Only the templates are retained — a 32x32 code is about a kilobyte — so a
database of several thousand images never needs to be resident.
"""
import os
import time

import numpy as np
from PIL import Image

from . import han, metrics, rloc

IMAGE_EXT = (".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".pgm")


# ------------------------------------------------------------------- loading
def load_gray(path, size=None):
    """Read an image as a float array in [0, 1]; optionally resize square."""
    im = Image.open(path).convert("L")
    if size is not None:
        im = im.resize((int(size), int(size)), Image.Resampling.LANCZOS)
    return np.asarray(im, dtype=np.float64) / 255.0


def _visible(name):
    """Skip hidden entries and helper folders such as `_extra` or `.cache`."""
    return not (name.startswith(".") or name.startswith("_"))


def index_images(root, layout="subdirs", per_palm=10):
    """
    Walk a database directory and return [(identity, index, path), ...].

    layout="subdirs"  every immediate subdirectory is one identity — the
                      common case, and the layout our synthetic demo data
                      uses.
    layout="flat"     all files in one directory, identity taken from the
                      filename up to the first underscore or hyphen — e.g.
                      "012_03.bmp" belongs to identity "012".
    layout="tongji"   the Tongji release: two folders of flat-numbered
                      images, no per-person folders, identity derived from
                      the number. See palmprint/tongji.py, which also
                      checks the numbering against the pixels.

    Names beginning with "." or "_" are skipped, so a dataset folder can carry
    helper material (the demo keeps its deliberately unreadable example in
    `_extra/`) without it being mistaken for a person.

    Which layout the IITD release actually uses is NOT assumed here: it is
    checked against the database's own README on arrival, and the presets
    exist so that confirmation is a one-word change rather than a rewrite.
    """
    root = os.path.abspath(root)
    items = []
    if layout == "tongji":
        from . import tongji
        return tongji.index(root, per_palm=per_palm)
    if layout == "subdirs":
        for name in sorted(os.listdir(root)):
            sub = os.path.join(root, name)
            if not os.path.isdir(sub) or not _visible(name):
                continue
            files = sorted(f for f in os.listdir(sub)
                           if f.lower().endswith(IMAGE_EXT) and _visible(f))
            for i, f in enumerate(files):
                items.append((name, i, os.path.join(sub, f)))
    elif layout == "flat":
        files = sorted(f for f in os.listdir(root)
                       if f.lower().endswith(IMAGE_EXT) and _visible(f))
        seen = {}
        for f in files:
            stem = os.path.splitext(f)[0]
            ident = stem.replace("-", "_").split("_")[0]
            idx = seen.get(ident, 0)
            seen[ident] = idx + 1
            items.append((ident, idx, os.path.join(root, f)))
    else:
        raise ValueError("layout must be 'subdirs', 'flat' or 'tongji'")
    return items


def _as_image(x, size=None):
    """Accept either an array already in memory or a path to load."""
    if isinstance(x, np.ndarray):
        return x
    return load_gray(x, size)


def split_enrol_probe(items, n_enrol=1):
    """Partition [(identity, index, src), ...] into enrol and probe lists."""
    by_id = {}
    for ident, idx, src in items:
        by_id.setdefault(ident, []).append((idx, src))
    enrol, probe = [], []
    for ident in sorted(by_id):
        samples = sorted(by_id[ident])
        for k, (idx, src) in enumerate(samples):
            (enrol if k < n_enrol else probe).append((ident, idx, src))
    return enrol, probe


def protocol_summary(items, n_enrol=1):
    enrol, probe = split_enrol_probe(items, n_enrol)
    n_id = len({i for i, _, _ in items})
    return {"identities": n_id, "images": len(items),
            "enrolled": len(enrol), "probes": len(probe),
            "genuine_comparisons": len(probe),
            "impostor_comparisons": len(probe) * (n_id - 1)}


# --------------------------------------------------------------- experiments
def _accumulate(probe_scores, identities, probe_identity, gen, imp):
    """Split one probe's per-identity score vector into genuine/impostor."""
    j = identities.index(probe_identity)
    gen.append(probe_scores[j])
    imp.extend(probe_scores[:j])
    imp.extend(probe_scores[j + 1:])


def run_rloc(items, cfg=None, n_enrol=1, size=None, label=None,
             progress=None):
    """
    Full RLOC verification experiment -> metrics.ScoreSet.

    `progress` may be a callable taking (done, total) for a status line.
    """
    cfg = cfg or rloc.RlocConfig()
    label = label or ("RLOC: " + cfg.describe())
    enrol, probe = split_enrol_probe(items, n_enrol)

    gallery = rloc.Gallery(cfg)
    for ident, _, src in enrol:
        gallery.enrol(ident, _as_image(src, size))

    gen, imp = [], []
    t0 = time.time()
    for k, (ident, _, src) in enumerate(probe):
        code = rloc.encode(_as_image(src, size), cfg)
        _accumulate(gallery.scores(code), gallery.identities, ident, gen, imp)
        if progress is not None and (k + 1) % 25 == 0:
            progress(k + 1, len(probe))
    elapsed = time.time() - t0

    ss = metrics.ScoreSet(gen, imp, label)
    ss.elapsed = elapsed
    ss.config = cfg
    return ss


def run_han(items, mode="sobel", cells=8, strips=False, n_enrol=1,
            size=None, label=None, progress=None):
    """Full Han et al. baseline experiment -> metrics.ScoreSet."""
    gallery = han.HanGallery(mode=mode, cells=cells, strips=strips)
    label = label or ("Han: " + gallery.describe())
    enrol, probe = split_enrol_probe(items, n_enrol)

    for ident, _, src in enrol:
        gallery.enrol(ident, _as_image(src, size))

    gen, imp = [], []
    t0 = time.time()
    for k, (ident, _, src) in enumerate(probe):
        vec = gallery.extract(_as_image(src, size))
        _accumulate(gallery.scores(vec), gallery.identities, ident, gen, imp)
        if progress is not None and (k + 1) % 25 == 0:
            progress(k + 1, len(probe))
    elapsed = time.time() - t0

    ss = metrics.ScoreSet(gen, imp, label)
    ss.elapsed = elapsed
    return ss


# ------------------------------------------------- the missing ablation
def ablation_configs():
    """
    Paper 2's Table 4, as runnable configurations.

    Our Phase I finding was that this table, read carefully, does not support
    the paper's framing. Experiment 1 — MFRAT with plain pixel-to-pixel
    matching and no enlarged training set — reports 0.40% EER against
    Competitive Code's 0.47% under the same conditions. So the filter swap the
    title is about is worth 0.07 of the 0.31-point headline gain, roughly
    fifteen per cent; the rest comes from the enlarged training set and the
    cross-shaped neighbourhood, neither of which has anything to do with
    MFRAT, and neither of which was ever applied to the Gabor baseline.

    Running these four on our own database turns that reading from an
    argument into a measurement.
    """
    return [
        ("Exp 1  MFRAT only",
         rloc.RlocConfig(rotations=(0.0,), neighbourhood="pixel")),
        ("Exp 2  + enlarged training set",
         rloc.RlocConfig(neighbourhood="pixel")),
        ("Exp 3  + cross neighbourhood",
         rloc.RlocConfig(neighbourhood="cross")),
        ("Exp 4  3x3 square instead of cross",
         rloc.RlocConfig(neighbourhood="square")),
    ]


def alignment_configs():
    """
    Is the "enlarged training set" a property of RLOC, or a patch for a badly
    aligned ROI?

    Paper 2 stores seven rotated templates per enrolment because the ROI it
    inherits can be out by about ten degrees. Our own ROI extraction recovers
    hand rotation to well under a degree (see test_preprocess.py), which
    makes the question answerable: run the matcher with seven templates and
    with one, crossed with the pixel and cross neighbourhoods, on identically
    aligned data.

    On synthetic hands the single template is slightly BETTER than seven
    (d' 9.41 against 8.96 with the cross rule) at a seventh of the storage
    and half the matching time. That is not a paradox, and it is the same
    mechanism that makes the paper's own 3x3 square worse than its cross:
    every extra template is another chance for an IMPOSTOR to match, so
    widening tolerance costs discrimination, and it only pays when there is
    genuine misalignment to absorb.

    If this survives on the real database, the reading is that the enlarged
    training set measures the quality of the preprocessing rather than the
    merit of the orientation code — which tightens the Phase I argument that
    little of Paper 2's headline gain is attributable to MFRAT itself.

    Treat it as a hypothesis with a mechanism until it is re-run on IITD:
    synthetic hands give our ROI an easier task than real ones will.
    """
    return [
        ("7 templates, cross", rloc.RlocConfig()),
        ("1 template, cross", rloc.RlocConfig(rotations=(0.0,))),
        ("7 templates, pixel", rloc.RlocConfig(neighbourhood="pixel")),
        ("1 template, pixel", rloc.RlocConfig(rotations=(0.0,),
                                              neighbourhood="pixel")),
    ]
