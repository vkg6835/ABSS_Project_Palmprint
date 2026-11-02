"""
tongji.py — reading the Tongji palmprint database.

The release is 12,000 images of 600 palms captured in two sessions, and it
arrives as two flat folders:

    session1/00001.bmp ... session1/06000.bmp
    session2/00001.bmp ... session2/06000.bmp

There are no per-person folders, so the only thing tying an image to a palm
is its position in the numbering. Everything in this module therefore treats
that numbering as a CLAIM TO BE CHECKED rather than a fact:

  * index() reads the folder, confirms the numbering is complete and
    contiguous, and refuses to run on anything that does not fit;
  * pairing_check() tests the claim against the pixels — the images of one
    palm from the two sessions should agree with each other more than with
    images of other palms. If that does not hold, the numbering means
    something else (or the archive is not what it says), and every result
    built on it would be worthless while still looking plausible.

The ROI archive holds 128 x 128 8-bit greyscale BMPs, which is the input
both papers assume, so nothing here does its own segmentation.
"""
import os
import re

import numpy as np

from . import preprocess as pp, protocol, rloc

# 600 palms x 10 captures = 6000 files per session.
IMAGES_PER_PALM_DEFAULT = 10
SESSIONS_DEFAULT = ("session1", "session2")
_IMAGE_RE = re.compile(r"^(\d+)\.(bmp|png|jpg|jpeg|tif|tiff)$", re.IGNORECASE)


class TongjiError(Exception):
    """The folder is not a Tongji release, or not the one we expect."""


def identity(palm):
    """Stable name for palm number `palm` (1-based), the same in both
    sessions, so the matcher knows two captures are the same person."""
    return "P%03d" % int(palm)


def session_of(path):
    """'session1' or 'session2', from the path."""
    parts = re.split(r"[\\/]", os.path.abspath(path))
    for part in reversed(parts[:-1]):
        if part.lower().startswith("session"):
            return part.lower()
    return ""


def _scan_session(folder):
    """{number: filename} for one session folder, or raise with the reason."""
    if not os.path.isdir(folder):
        raise TongjiError("no folder named %s in %s"
                          % (os.path.basename(folder), os.path.dirname(folder)))
    found = {}
    for name in os.listdir(folder):
        m = _IMAGE_RE.match(name)
        if m:
            found[int(m.group(1))] = name
    if not found:
        raise TongjiError("no numbered images in %s" % folder)
    return found


def index(root, per_palm=IMAGES_PER_PALM_DEFAULT,
          sessions=SESSIONS_DEFAULT, strict=True):
    """
    [(identity, rank, path), ...] for every image in the release.

    `rank` runs 0..(per_palm * len(sessions) - 1) within a palm, with each
    session's captures contiguous and session 1 first, so that
    protocol.split_enrol_probe(n_enrol=1) enrols one capture from session 1
    and leaves everything else as a probe.

    With strict=True, raises TongjiError unless every session holds exactly
    the same, contiguous run of numbers, starting at 1 and divisible by
    per_palm. A truncated download or a different release stops here rather
    than producing results from a silently mis-grouped dataset.
    """
    root = os.path.abspath(root)
    per_session = {}
    for sess in sessions:
        per_session[sess] = _scan_session(os.path.join(root, sess))

    if strict:
        sizes = {s: len(f) for s, f in per_session.items()}
        if len(set(sizes.values())) != 1:
            raise TongjiError("sessions hold different numbers of images: %s"
                              % sizes)
        # Gaps before divisibility: an incomplete extraction must be reported
        # as what it is, not as "not a whole number of palms", which is the
        # confusing message the first version produced for a missing image.
        for sess, files in per_session.items():
            missing = [k for k in range(1, len(files) + 1) if k not in files]
            if missing:
                raise TongjiError(
                    "%s is missing %d of the %d numbered images (first "
                    "missing: %05d) — the download or extraction is "
                    "incomplete" % (sess, len(missing), len(files), missing[0]))
        n = next(iter(sizes.values()))
        if n % per_palm:
            raise TongjiError(
                "%d images per session is not a whole number of palms at "
                "%d images per palm (%d left over) — the numbering must "
                "differ from the expected layout" % (n, per_palm, n % per_palm))

    items = []
    numbers = sorted(per_session[sessions[0]])
    for number in numbers:
        palm = (number - 1) // per_palm + 1
        within = (number - 1) % per_palm
        for s_i, sess in enumerate(sessions):
            name = per_session[sess].get(number)
            if name is None:
                continue
            rank = s_i * per_palm + within
            items.append((identity(palm), rank,
                          os.path.join(root, sess, name)))
    items.sort(key=lambda t: (t[0], t[1]))
    return items


def describe(root, per_palm=IMAGES_PER_PALM_DEFAULT,
             sessions=SESSIONS_DEFAULT):
    """A dictionary of what is actually on disk, for printing and for the
    record of how a result was produced."""
    out = {"root": os.path.abspath(root), "sessions": {}, "per_palm": per_palm}
    sizes = {}
    for sess in sessions:
        try:
            files = _scan_session(os.path.join(root, sess))
        except TongjiError as e:
            out["sessions"][sess] = {"error": str(e)}
            continue
        nums = sorted(files)
        sizes[sess] = len(nums)
        out["sessions"][sess] = {
            "images": len(nums), "first": nums[0], "last": nums[-1],
            "gaps": sum(1 for k in range(nums[0], nums[-1] + 1)
                        if k not in files),
        }
    if sizes and len(set(sizes.values())) == 1:
        n = next(iter(sizes.values()))
        out["images"] = n * len(sizes)
        out["palms"] = n // per_palm if n % per_palm == 0 else None
    return out


# ------------------------------------------------------- the numbering test
def pairing_check(items, cfg=None, n_palms=40, pairs_per_palm=2, seed=0):
    """
    Do the same numbers in the two sessions really show the same palm?

    For `n_palms` palms, compare a capture from session 1 with a capture of
    the SAME number in session 2 (the claim), and with captures of other
    palms (the control), using the orientation-code agreement — no rotated
    templates, no cross neighbourhood, so this is a raw property of the
    images rather than of the matcher.

    A real palm photographed twice must score higher than two different
    palms. If it does not, the numbering does not mean what we think, and
    the caller should stop.

    Returns a dict of the two distributions and a verdict.
    """
    cfg = cfg or rloc.RlocConfig(rotations=(0.0,), neighbourhood="pixel")
    rng = np.random.default_rng(seed)

    by_identity = {}
    for ident, rank, path in items:
        by_identity.setdefault(ident, []).append((rank, path))
    ids = sorted(by_identity)
    if len(ids) < 3:
        raise TongjiError("need at least three palms to check the pairing")

    chosen = ids if len(ids) <= n_palms else sorted(
        rng.choice(ids, size=n_palms, replace=False).tolist())

    def first_in(path, session):
        return session_of(path) == session

    cache = {}

    def code(path):
        if path not in cache:
            roi = protocol.load_gray(path, 128)
            cache[path] = rloc.encode(roi, cfg)
        return cache[path]

    same, other = [], []
    for ident in chosen:
        s1 = [p for r, p in by_identity[ident] if first_in(p, "session1")]
        s2 = [p for r, p in by_identity[ident] if first_in(p, "session2")]
        if not s1 or not s2:
            continue
        for _ in range(pairs_per_palm):
            a = s1[int(rng.integers(len(s1)))]
            b = s2[int(rng.integers(len(s2)))]
            same.append(float(np.mean(code(a) == code(b))))
            # a control pair: this palm from session 1 against another palm
            other_ident = chosen[int(rng.integers(len(chosen)))]
            if other_ident == ident:
                continue
            o2 = [p for r, p in by_identity[other_ident] if first_in(p, "session2")]
            if not o2:
                continue
            c = o2[int(rng.integers(len(o2)))]
            other.append(float(np.mean(code(a) == code(c))))

    same = np.asarray(same, dtype=np.float64)
    other = np.asarray(other, dtype=np.float64)
    if same.size < 5 or other.size < 5:
        raise TongjiError("not enough images to check the pairing")
    m_same, m_other = float(same.mean()), float(other.mean())
    return {
        "pairs": (int(same.size), int(other.size)),
        "same_palm": m_same,
        "different_palm": m_other,
        "gap": m_same - m_other,
        "ok": bool(m_same > m_other),
    }


def write_manifest(root, name="Tongji palmprint database",
                   per_palm=IMAGES_PER_PALM_DEFAULT):
    """
    Write `manifest.json` into the dataset folder so the demo and the
    experiments treat it as pre-cropped palm ROIs.
    """
    import json
    path = os.path.join(root, "manifest.json")
    manifest = {
        "name": name,
        "kind": "roi",
        "layout": "tongji",
        "per_palm": per_palm,
        "note": ("Tongji release: 600 palms, 10 captures per palm per "
                 "session, two sessions. Pre-cropped 128 x 128 palm ROIs, "
                 "so no segmentation of ours is applied."),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return path
