"""
experiments.py — the Phase II evaluation, as a library.

run_experiments.py is the command line around it. Everything that produces a
number is here; everything that draws or formats one is there (and in
plots.py), so a change of wording never forces a recomputation.

The protocol
------------
Each palm is enrolled from ONE image: its first session-1 capture. Every other
image of that palm is a probe, labelled by group:

    cross   captured in a different session from the enrolment (Tongji:
            session 2, weeks later). Every headline number uses these.
    same    captured in the enrolment's own session, minutes apart. Kept and
            reported separately, because mixing them in flatters every
            method — on Tongji it turns RLOC's 0.11% EER into 0.06%.
    all     datasets without sessions (the synthetic set) have only this.

Every probe is compared with every enrolled palm: one genuine comparison and
N-1 impostor comparisons each.

What is computed
----------------
Features, once per image, in resumable chunks:
    MFRAT codes for every probe; MFRAT codes of every enrolment image at the
    seven enrolment rotations; the same two for the Gabor code; and Paper 1's
    Sobel and morphology feature vectors.
Score matrices, one per configuration (probes x palms).
Metrics per configuration and group, and a paired bootstrap over palms.

Resuming
--------
Work is saved as it is done, under a key that changes whenever the images or
the feature/matching code change. A `Clock` with a time budget stops cleanly
between units and the next run continues from there — needed in our own
sandbox, harmless on a laptop where one run simply finishes.
"""
import hashlib
import inspect
import json
import os
import time

import numpy as np

from . import gabor, han, metrics, mfrat, protocol, rloc, tongji

ROTATIONS = (0.0, 3.0, -3.0, 6.0, -6.0, 9.0, -9.0)
# Candidate Gabor frequencies (radians per pixel; wavelength = 2 pi / omega).
# The first grid stopped at 0.30, and 0.30 won — the edge of the grid, so the
# best value could have been beyond it. It was extended downwards rather
# than leave the Gabor arm handicapped by where we happened to stop looking.
GABOR_OMEGAS = (0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80)
SCORE_CHUNK = 2000              # probes per saved block of scores
GABOR_TUNE_PER_PALM = 2          # session-1 probes per palm used for tuning
FARS = (1e-2, 1e-3, 1e-4)
BOOT_SEED = 461                  # the course code: fixed, not chosen
HIST_BINS = 4096


class Deadline(Exception):
    """The time budget ran out between two units of work. Run again."""


class Clock(object):
    def __init__(self, max_seconds=None):
        self.t0 = time.time()
        self.max_seconds = max_seconds

    def elapsed(self):
        return time.time() - self.t0

    def check(self):
        if self.max_seconds and self.elapsed() > self.max_seconds:
            raise Deadline()


# =============================================================== protocol
class Split(object):
    """Who is enrolled from what, and which probes belong to which group."""

    def __init__(self, identities, enrol_paths, probe_paths, probe_owner,
                 probe_group, root):
        self.identities = list(identities)
        self.enrol_paths = list(enrol_paths)
        self.probe_paths = list(probe_paths)
        self.probe_owner = np.asarray(probe_owner, dtype=np.int64)
        self.probe_group = np.asarray(probe_group)
        self.root = root

    @property
    def groups(self):
        out = [g for g in ("cross", "same", "all")
               if np.any(self.probe_group == g)]
        return out

    def counts(self, group):
        rows = self.probe_group == group
        n = len(self.identities)
        return {"probes": int(rows.sum()), "genuine": int(rows.sum()),
                "impostor": int(rows.sum()) * (n - 1)}

    def rel(self, path):
        return os.path.relpath(path, self.root).replace("\\", "/")


def make_split(root, layout="subdirs", per_palm=10, max_palms=None):
    items = protocol.index_images(root, layout, per_palm=per_palm)
    if not items:
        raise ValueError("no images found in %s" % root)
    by_id = {}
    for ident, idx, path in items:
        by_id.setdefault(ident, []).append((idx, path))
    ids = sorted(by_id)
    if max_palms:
        ids = ids[:int(max_palms)]
    sessions = layout == "tongji"

    enrol, probes, owner, group = [], [], [], []
    for j, ident in enumerate(ids):
        caps = sorted(by_id[ident])
        enrol_path = caps[0][1]
        enrol.append(enrol_path)
        es = tongji.session_of(enrol_path) if sessions else ""
        for _, path in caps[1:]:
            probes.append(path)
            owner.append(j)
            if sessions:
                group.append("same" if tongji.session_of(path) == es
                             else "cross")
            else:
                group.append("all")
    return Split(ids, enrol, probes, owner, group, os.path.abspath(root))


# ================================================================== store
class Store(object):
    """Named arrays on disk, written atomically, under one cache key."""

    def __init__(self, folder):
        self.folder = folder
        os.makedirs(folder, exist_ok=True)

    def _path(self, name):
        return os.path.join(self.folder, name + ".npz")

    def has(self, name):
        return os.path.isfile(self._path(name))

    def load(self, name):
        with np.load(self._path(name), allow_pickle=False) as z:
            return {k: z[k] for k in z.files}

    def save(self, name, **arrays):
        tmp = self._path(name) + ".tmp"
        with open(tmp, "wb") as f:
            np.savez_compressed(f, **arrays)
        os.replace(tmp, self._path(name))


def cache_key(split, extra=""):
    """Images (path, size, time) + the code that turns them into numbers."""
    h = hashlib.sha256()
    for mod in (mfrat, rloc, gabor, han, tongji):
        with open(mod.__file__, "rb") as f:
            h.update(f.read())
    for fn in (_probe_chunk, _enrol_chunk, _gabor_chunk, rloc_scores,
               han_scores, make_split):
        h.update(inspect.getsource(fn).encode("utf-8"))
    h.update(repr((ROTATIONS, extra, len(split.identities))).encode("utf-8"))
    paths = split.enrol_paths + split.probe_paths
    # Checking 12,000 files one after another is instant on a local disk but
    # takes a minute on a network or virtual drive, where each check waits on
    # the round trip. The checks are independent, so they run in parallel.
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=16) as pool:
        stats = list(pool.map(os.stat, paths))
    for p, st in zip(paths, stats):
        h.update(("%s|%d|%d\n" % (split.rel(p), st.st_size, st.st_mtime_ns)
                  ).encode("utf-8"))
    return h.hexdigest()[:16]


# =============================================================== features
def _probe_chunk(paths):
    """MFRAT code + Paper 1's Sobel and morphology vectors, one read each."""
    cfg = rloc.RlocConfig()
    codes, sob, mor = [], [], []
    for p in paths:
        roi = protocol.load_gray(p, 128)
        codes.append(rloc.encode(roi, cfg))
        sob.append(han.features(roi, "sobel"))
        mor.append(han.features(roi, "morph"))
    return {"mfrat": np.stack(codes).astype(np.uint8),
            "sobel": np.stack(sob), "morph": np.stack(mor)}


def _enrol_chunk(paths):
    """MFRAT at the seven enrolment rotations + Paper 1's two vectors."""
    cfg = rloc.RlocConfig(rotations=ROTATIONS)
    codes, sob, mor = [], [], []
    for p in paths:
        roi = protocol.load_gray(p, 128)
        codes.append(rloc.enrol_codes(roi, cfg))
        sob.append(han.features(roi, "sobel"))
        mor.append(han.features(roi, "morph"))
    return {"mfrat": np.stack(codes).astype(np.uint8),
            "sobel": np.stack(sob), "morph": np.stack(mor)}


def _gabor_chunk(paths, omega, rotations):
    """Gabor codes at the given rotations: (n, len(rotations), 32, 32)."""
    out = []
    for p in paths:
        roi = protocol.load_gray(p, 128)
        out.append(np.stack([gabor.encode(rloc.rotate_roi(roi, a), omega)
                             for a in rotations]))
    return {"codes": np.stack(out).astype(np.uint8)}


def chunked(store, name, paths, fn, clock, log, chunk=400):
    """
    Run fn over `paths` in chunks, saving each, and return the concatenation.
    Chunks already on disk are skipped, so an interrupted run resumes.
    """
    n = len(paths)
    parts = []
    for k, c0 in enumerate(range(0, n, chunk)):
        part = "%s.part%04d" % (name, k)
        if not store.has(part):
            clock.check()
            t0 = time.time()
            store.save(part, **fn(paths[c0:c0 + chunk]))
            log("  %s: images %d-%d of %d (%.0f s)"
                % (name, c0 + 1, min(n, c0 + chunk), n, time.time() - t0))
        parts.append(store.load(part))
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def probe_features(split, store, clock, log):
    return chunked(store, "probe_features", split.probe_paths, _probe_chunk,
                   clock, log)


def enrol_features(split, store, clock, log):
    return chunked(store, "enrol_features", split.enrol_paths, _enrol_chunk,
                   clock, log, chunk=150)


def gabor_features(split, store, clock, log, omega):
    tag = "gabor%03d" % int(round(omega * 100))
    probes = chunked(store, tag + "_probes", split.probe_paths,
                     lambda ps: _gabor_chunk(ps, omega, (0.0,)), clock, log)
    enrol = chunked(store, tag + "_enrol", split.enrol_paths,
                    lambda ps: _gabor_chunk(ps, omega, ROTATIONS), clock, log,
                    chunk=150)
    return probes["codes"][:, 0], enrol["codes"]


# ================================================================ scoring
def rloc_scores(probe_codes, enrol_codes, n_templates, cfg):
    """
    (P, N) scores of every probe against every palm: equations (7)-(9),
    the configured neighbourhood and translation search, best of the first
    `n_templates` enrolment rotations. Bit-identical to Gallery.scores.
    """
    n = enrol_codes.shape[0]
    tmpl = enrol_codes[:, :n_templates]
    flat = tmpl.reshape((-1,) + tmpl.shape[2:])
    owner = np.repeat(np.arange(n), n_templates)
    return rloc.scores_batch(probe_codes, flat, owner, n, cfg
                             ).astype(np.float32)


def han_scores(probe_vecs, enrol_vecs):
    """(P, N) Pearson correlations — Paper 1's template matching with one
    enrolment sample per palm (HanGallery.scores, all probes at once)."""
    pz = han._zscore(probe_vecs)
    ez = han._zscore(enrol_vecs)
    return (pz @ ez.T / pz.shape[1]).astype(np.float32)


# The configurations. Each RLOC-family entry: (key, label, encoder,
# neighbourhood, templates). Exp 1-4 are Paper 2's Table 4; the Gabor arms
# mirror Exp 1-3 with the filter swapped and nothing else changed.
RLOC_FAMILY = [
    ("exp1", "Exp 1  MFRAT, pixel-to-pixel, 1 template", "mfrat", "pixel", 1),
    ("exp2", "Exp 2  + enlarged training set (7 templates)", "mfrat",
     "pixel", 7),
    ("rloc", "Exp 3  + cross neighbourhood  = RLOC", "mfrat", "cross", 7),
    ("exp4", "Exp 4  3x3 square instead of cross", "mfrat", "square", 7),
    ("align1", "RLOC with 1 template instead of 7", "mfrat", "cross", 1),
    ("gab1", "Gabor, pixel-to-pixel, 1 template", "gabor", "pixel", 1),
    ("gab2", "Gabor + enlarged training set", "gabor", "pixel", 7),
    ("gab3", "Gabor + enlarged set + cross", "gabor", "cross", 7),
    ("oneway", "RLOC, one direction only (no eq. 9 maximum)", "mfrat",
     "cross", 7),
    ("shift1", "RLOC, +/-1 shift search instead of +/-2", "mfrat", "cross", 7),
    ("first", "RLOC as we first built it (+/-1 shift, one direction)",
     "mfrat", "cross", 7),
]
# Matcher settings that differ from Paper 2's, per configuration. Every other
# configuration matches as the paper does: +/-2 code pixels of translation
# and equation (9)'s maximum over both directions (rloc.RlocConfig).
CFG_EXTRA = {
    "oneway": {"symmetric": False},
    "shift1": {"shift_range": 1},
    "first": {"shift_range": 1, "symmetric": False},
}
HAN_FAMILY = [
    ("han", "Han et al. 2003, Sobel features", "sobel"),
    ("han_morph", "Han et al. 2003, morphology features", "morph"),
]
LABELS = dict([(k, l) for k, l, *_ in RLOC_FAMILY] +
              [(k, l) for k, l, _ in HAN_FAMILY])


def score_name(key, omegas):
    enc = dict((k, e) for k, _, e, _, _ in RLOC_FAMILY).get(key)
    if enc != "gabor":
        return "scores_%s" % key
    om = omegas[key] if isinstance(omegas, dict) else omegas
    return "scores_%s_w%03d" % (key, int(round(om * 100)))


def metrics_unit(store, clock, key, omegas, S, split, W, main_group):
    """
    Every metric for one configuration, and its bootstrap samples — saved as
    one unit, so an interrupted run does not redo the ones already finished.
    The score histograms are not kept (millions of numbers); figures that
    need them recompute them for the two configurations they show.
    """
    name = "metrics3_%s_b%d" % (score_name(key, omegas)[len("scores_"):],
                                W.shape[0])
    if not store.has(name):
        clock.check()
        res = {}
        arrays = {}
        for g in split.groups:
            e = evaluate(S, split, g)
            for part in ("_det", "_farfrr"):
                for i, arr in enumerate(e.pop(part)):
                    arrays["%s%s_%d" % (g, part, i)] = np.asarray(arr)
            e.pop("_hist")
            res[g] = e
        samples = bootstrap_eers(S, split, main_group, W)
        res[main_group]["eer_ci95"] = ci(samples)
        meta = np.frombuffer(json.dumps(res, default=float).encode("utf-8"),
                             dtype=np.uint8)
        store.save(name, meta=meta, boot=samples, **arrays)
    z = store.load(name)
    res = json.loads(bytes(z["meta"]).decode("utf-8"))
    for g in res:
        res[g]["_det"] = (z["%s_det_0" % g], z["%s_det_1" % g])
        res[g]["_farfrr"] = tuple(z["%s_farfrr_%d" % (g, i)] for i in range(3))
        if "eer_ci95" in res[g]:
            res[g]["eer_ci95"] = tuple(res[g]["eer_ci95"])
    return res, z["boot"]


def score_all(split, store, clock, log, omegas, only=None):
    """Compute (or load) every score matrix. `omegas` gives each Gabor
    configuration its own wavelength. Returns {key: (P, N) array} and
    {key: seconds spent matching}."""
    out, secs = {}, {}
    need = [k for k in LABELS if (only is None or k in only)]
    pf = ef = None
    gab = {}
    for key, label, enc, nb, nt in RLOC_FAMILY:
        if key not in need:
            continue
        name = score_name(key, omegas)
        if not store.has(name):
            if enc == "mfrat":
                pf = pf or probe_features(split, store, clock, log)
                ef = ef or enrol_features(split, store, clock, log)
                pc, ec = pf["mfrat"], ef["mfrat"]
            else:
                om = omegas[key]
                if om not in gab:
                    gab[om] = gabor_features(split, store, clock, log, om)
                pc, ec = gab[om]
            clock.check()
            cfg = rloc.RlocConfig(neighbourhood=nb,
                                  rotations=ROTATIONS[:nt],
                                  **CFG_EXTRA.get(key, {}))
            # in blocks of probes, each saved, so an interrupted run resumes
            # mid-configuration (every row depends on its own probe only)
            parts, dt = [], 0.0
            for i, c0 in enumerate(range(0, len(pc), SCORE_CHUNK)):
                part = "%s.part%04d" % (name, i)
                if not store.has(part):
                    clock.check()
                    t0 = time.time()
                    store.save(part, scores=rloc_scores(
                        pc[c0:c0 + SCORE_CHUNK], ec, nt, cfg),
                        seconds=np.array(time.time() - t0))
                z = store.load(part)
                parts.append(z["scores"])
                dt += float(z["seconds"])
            S = np.concatenate(parts)
            store.save(name, scores=S, seconds=np.array(dt))
            for i in range(len(parts)):
                os.remove(store._path("%s.part%04d" % (name, i)))
            log("  scored %-46s %.0f s" % (label, dt))
        z = store.load(name)
        out[key], secs[key] = z["scores"], float(z["seconds"])
    for key, label, mode in HAN_FAMILY:
        if key not in need:
            continue
        name = "scores_%s" % key
        if not store.has(name):
            pf = pf or probe_features(split, store, clock, log)
            ef = ef or enrol_features(split, store, clock, log)
            t0 = time.time()
            S = han_scores(pf[mode], ef[mode])
            store.save(name, scores=S, seconds=np.array(time.time() - t0))
        z = store.load(name)
        out[key], secs[key] = z["scores"], float(z["seconds"])
    return out, secs


# ========================================================== Gabor tuning
GABOR_STAGES = [("gab1", "pixel", 1), ("gab2", "pixel", 7),
                ("gab3", "cross", 7)]


def tune_gabor(split, store, clock, log, omegas=GABOR_OMEGAS):
    """
    Choose each Gabor configuration's wavelength on session-1 data only.

    The wavelength decides how smooth the code is, and smoothness is itself
    a tolerance to misalignment — the same job the enlarged training set and
    the cross neighbourhood do in the metric. So the best wavelength for
    plain pixel-to-pixel matching need not be the best once those tricks are
    added, and tuning once, in the no-tricks setting, would hand the
    full-pipeline comparison a Gabor code chosen for a different system.
    (The first version did exactly that.) Each Gabor configuration therefore
    gets its own wavelength, chosen with its own matching rule.

    For each candidate, palms are enrolled from their first session-1 image
    and probed with GABOR_TUNE_PER_PALM other session-1 images each. Lowest
    EER wins; ties — common, because same-session matching is easy and an
    EER of a fraction of a percent is a handful of errors — go to the larger
    d', a continuous measure of how far apart the genuine and impostor
    scores sit. No session-2 image is ever looked at, so no choice can be
    fitted to the numbers we report. MFRAT's parameters are taken from its
    paper unchanged, so if anything the Gabor arm is favoured.
    """
    rows = np.flatnonzero(np.isin(split.probe_group, ("same", "all")))
    picked = []
    seen = {}
    for r in rows:
        o = split.probe_owner[r]
        if seen.get(o, 0) < GABOR_TUNE_PER_PALM:
            picked.append(r)
            seen[o] = seen.get(o, 0) + 1
    picked = np.asarray(picked, dtype=np.int64)
    probe_paths = [split.probe_paths[r] for r in picked]
    owner = split.probe_owner[picked]
    table = []
    for om in omegas:
        name = "gabor_tune3_w%03d" % int(round(om * 100))
        if not store.has(name):
            clock.check()
            t0 = time.time()
            enrol = _gabor_chunk(split.enrol_paths, om, ROTATIONS)["codes"]
            p = _gabor_chunk(probe_paths, om, (0.0,))["codes"][:, 0]
            vals = {}
            for key, nb, nt in GABOR_STAGES:
                cfg = rloc.RlocConfig(neighbourhood=nb,
                                      rotations=ROTATIONS[:nt])
                S = rloc_scores(p, enrol, nt, cfg)
                gen = S[np.arange(len(owner)), owner]
                mask = np.ones(S.shape, dtype=bool)
                mask[np.arange(len(owner)), owner] = False
                ss = metrics.ScoreSet(gen, S[mask])
                vals["eer_" + key] = np.array(ss.eer()[0])
                vals["dp_" + key] = np.array(ss.dprime())
            store.save(name, probes=np.array(len(owner)), **vals)
            log("  Gabor tuning, omega %.2f: session-1 EER %s (%.0f s)"
                % (om, " / ".join("%.3f%%" % (100 * float(vals["eer_" + k]))
                                  for k, _, _ in GABOR_STAGES),
                   time.time() - t0))
        z = store.load(name)
        table.append((om, {k: (float(z["eer_" + k]), float(z["dp_" + k]))
                           for k, _, _ in GABOR_STAGES}))
    best = {}
    for k, _, _ in GABOR_STAGES:
        best[k] = min(table, key=lambda t: (t[1][k][0], -t[1][k][1]))[0]
    return best, table, int(len(owner))


# ================================================================ metrics
def scoreset(S, split, group):
    rows = np.flatnonzero(split.probe_group == group)
    own = split.probe_owner[rows]
    sub = S[rows]
    gen = sub[np.arange(rows.size), own]
    mask = np.ones(sub.shape, dtype=bool)
    mask[np.arange(rows.size), own] = False
    return metrics.ScoreSet(gen, sub[mask]), sub, own


def evaluate(S, split, group):
    """Every number one configuration earns on one group of probes."""
    ss, sub, own = scoreset(S, split, group)
    eer, t_eer = ss.eer()
    out = {"genuine": ss.n_genuine, "impostor": ss.n_impostor,
           "eer": eer, "threshold": t_eer, "dprime": ss.dprime()}
    for far in FARS:
        frr, t = ss.frr_at_far(far)
        out["frr_at_far_%g" % far] = frr
        out["threshold_at_far_%g" % far] = t
    out["confusion_eer"] = ss.confusion(t_eer)
    out["confusion_far_0.001"] = ss.confusion(out["threshold_at_far_0.001"])
    # closed-set identification: is the best-scoring palm the right one?
    order = np.argsort(-sub, axis=1, kind="stable")
    rank = np.argmax(order == own[:, None], axis=1) + 1
    out["rank1"] = float(np.mean(rank == 1))
    out["rank5"] = float(np.mean(rank <= 5))
    # where the false rejects at the EER threshold come from: a few hard
    # palms, or everyone a little? (It decides how wide the interval is.)
    # in float64, as the ScoreSet does: numpy 2 compares a float32 array with
    # a Python float in float32, which rounds the threshold — and on Tongji
    # one Han genuine score sits 6e-9 from it, so this table and the
    # confusion matrix disagreed by one (460 against 461)
    gen_raw = sub[np.arange(sub.shape[0]), own].astype(np.float64)
    fr = gen_raw < t_eer
    per_palm = np.bincount(own[fr], minlength=len(split.identities))
    worst = np.argsort(-per_palm, kind="stable")[:5]
    out["false_rejects"] = int(fr.sum())
    out["palms_with_false_reject"] = int(np.sum(per_palm > 0))
    out["probes_per_palm"] = float(sub.shape[0]) / len(split.identities)
    out["worst_palms"] = [[split.identities[i], int(per_palm[i])]
                          for i in worst if per_palm[i] > 0]
    far, frr, _ = ss.det_rates(n=400)
    out["_det"] = (far, frr)
    out["_hist"] = (ss.genuine, ss.impostor)
    lo = float(np.percentile(ss.impostor, 0.1))
    hi = float(np.percentile(ss.genuine, 99.9))
    t = np.linspace(min(lo, t_eer), max(hi, t_eer), 400)
    out["_farfrr"] = (t, ss.far(t), ss.frr(t))
    return out


# ============================================================== bootstrap
def boot_weights(n_palms, reps, seed=BOOT_SEED):
    """How many times each palm is drawn in each replicate. Shared by every
    configuration, so differences between configurations are PAIRED."""
    rng = np.random.default_rng(seed)
    return rng.multinomial(n_palms, np.full(n_palms, 1.0 / n_palms),
                           size=reps).astype(np.float64)


def palm_histograms(S, split, group, edges):
    """Per-palm genuine and impostor score histograms on common bins."""
    rows = np.flatnonzero(split.probe_group == group)
    own = split.probe_owner[rows]
    n = len(split.identities)
    nb = len(edges) - 1
    sub = S[rows]
    b = np.clip(np.searchsorted(edges, sub, side="right") - 1, 0, nb - 1)
    gen_b = b[np.arange(rows.size), own]
    G = np.bincount(own * nb + gen_b, minlength=n * nb).reshape(n, nb)
    allc = np.bincount((np.repeat(own, sub.shape[1]) * nb + b.ravel()),
                       minlength=n * nb).reshape(n, nb)
    return G.astype(np.float64), (allc - G).astype(np.float64)


def binned_eer(Gw, Iw):
    """EER of each row of bootstrapped histograms (reps, bins)."""
    tg = Gw.sum(axis=1, keepdims=True)
    ti = Iw.sum(axis=1, keepdims=True)
    frr = np.cumsum(Gw, axis=1) / tg            # scores below each upper edge
    far = 1.0 - np.cumsum(Iw, axis=1) / ti      # scores at/above it
    d = far - frr
    k = np.argmax(d <= 0, axis=1)
    k = np.maximum(k, 1)
    r = np.arange(Gw.shape[0])
    d0, d1 = d[r, k - 1], d[r, k]
    w = np.where(d0 - d1 > 0, d0 / np.where(d0 - d1 > 0, d0 - d1, 1), 0.0)
    e0 = (far[r, k - 1] + frr[r, k - 1]) / 2.0
    e1 = (far[r, k] + frr[r, k]) / 2.0
    return e0 + w * (e1 - e0)


def bootstrap_eers(S, split, group, W):
    lo, hi = float(np.nanmin(S)), float(np.nanmax(S))
    pad = (hi - lo) * 1e-6 + 1e-9
    edges = np.linspace(lo - pad, hi + pad, HIST_BINS + 1)
    G, I = palm_histograms(S, split, group, edges)
    return binned_eer(W @ G, W @ I)


def ci(samples, level=0.95):
    a = (1.0 - level) / 2.0
    return (float(np.quantile(samples, a)), float(np.quantile(samples, 1 - a)))


# ================================================================== costs
def costs(split, omega=0.5, n=60):
    """Per-ROI extraction and per-comparison matching time on this machine,
    and template sizes. Measured, so they depend on the computer. `omega`
    is the Gabor wavelength actually used — the kernel, and so the cost,
    grows with the wavelength."""
    paths = split.probe_paths[:n]
    rois = [protocol.load_gray(p, 128) for p in paths]
    cfg = rloc.RlocConfig()

    def per_roi(fn):
        fn(rois[0])
        t0 = time.perf_counter()
        for r in rois:
            fn(r)
        return 1000.0 * (time.perf_counter() - t0) / len(rois)

    out = {
        "mfrat_ms": per_roi(lambda r: rloc.encode(r, cfg)),
        "gabor_ms": per_roi(lambda r: gabor.encode(r, omega)),
        "gabor_omega": omega,
        "sobel_ms": per_roi(lambda r: han.features(r, "sobel")),
        "morph_ms": per_roi(lambda r: han.features(r, "morph")),
        "enrol7_ms": per_roi(lambda r: rloc.enrol_codes(r, cfg)),
    }
    a = rloc.encode(rois[0], cfg)
    t7 = rloc.enrol_codes(rois[1], cfg)
    k = 200
    t0 = time.perf_counter()
    for _ in range(k):
        rloc.match_codes(a, t7, cfg).max()
    out["match_one_palm_ms"] = 1000.0 * (time.perf_counter() - t0) / k
    v1, v2 = han.features(rois[0]), han.features(rois[1])
    t0 = time.perf_counter()
    for _ in range(k):
        han.pearson(v1, v2)
    out["han_match_ms"] = 1000.0 * (time.perf_counter() - t0) / k
    bits = 32 * 32 * np.log2(6)
    out["rloc_template_bytes"] = int(np.ceil(bits / 8.0)) * len(ROTATIONS)
    out["rloc_template_bytes_one"] = int(np.ceil(bits / 8.0))
    out["han_template_bytes"] = 64 * 4
    return out


def write_json(path, obj):
    def clean(o):
        if isinstance(o, dict):
            return {k: clean(v) for k, v in o.items() if not k.startswith("_")}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        return o
    with open(path, "w", encoding="utf-8") as f:
        json.dump(clean(obj), f, indent=2)
