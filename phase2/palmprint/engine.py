"""
engine.py — everything the live demo does, with no GUI code in it.

"Whether your demo works or not" is one of the Phase II grading criteria, so
everything that decides anything lives here, where it can be tested without a
screen (test_demo.py), and demo.py is only a window around it. The same split
means the numbers on the demo's screen come from exactly the code the
experiments use.

What the engine does
--------------------
prepare()   Index a dataset folder, preprocess every image, enrol the first
            readable image of each person, and score every other image
            against everyone enrolled — the protocol of protocol.py — which
            gives the genuine and impostor distributions and an operating
            threshold for each matcher. The result is cached on disk, keyed
            on the images AND on the source code that processed them, so the
            second start-up is instant and an edited module can never be
            served stale numbers.
process()   One input image through every stage, keeping everything the
            screen needs to show — or the reason it was declined.
verify()    One-to-one: does this probe match the claimed identity?
identify()  One-to-many: whose palm is this?
enrol()     Add a new person live, from a single image.

Dataset folders
---------------
One subfolder per person (protocol.index_images), plus an optional
manifest.json:

    {"name": "Synthetic hands", "kind": "hand", "layout": "subdirs",
     "note": "...", "examples": {"unreadable": "_extra/no_fingers.png"}}

kind "hand" means whole-hand images, on which our own ROI extraction runs;
kind "roi" means pre-cropped palm images, which skip it. Folders whose names
start with "_" or "." are ignored as people.
"""
import hashlib
import json
import os
import time
import traceback

import numpy as np
from PIL import Image, ImageOps

from . import han, metrics, preprocess as pp, protocol, rloc

ENGINE_VERSION = "1"
_PKG = os.path.dirname(os.path.abspath(__file__))
# The modules whose code produces the cached arrays; any edit to one of them
# invalidates every cache built with the old code. (The engine's own
# processing functions are hashed separately — see _fingerprint.)
_SOURCES = ("preprocess.py", "mfrat.py", "rloc.py", "han.py", "protocol.py",
            "tongji.py")

MATCHERS = ("rloc", "han")
MATCHER_NAMES = {"rloc": "RLOC (Jia, Huang & Zhang 2008)",
                 "han": "Han et al. (2003)"}
# Used only if a dataset is too small to estimate a threshold from, and
# reported as such — never silently.
FALLBACK_THRESHOLDS = {"rloc": 0.70, "han": 0.60}


class EngineError(Exception):
    """A problem with the dataset or the request, explained in plain words."""


def _ms(t0):
    return 1000.0 * (time.perf_counter() - t0)


def read_manifest(root):
    path = os.path.join(root, "manifest.json")
    if not os.path.isfile(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_image(path, max_side=None):
    """
    Read any common image file as greyscale floats in [0, 1].

    Phone photographs record their orientation in EXIF rather than in the
    pixels, so that is applied first; and anything larger than `max_side`
    is reduced, because a 12-megapixel photograph carries no more palm-line
    information at our ROI size than a 640-pixel one, and takes thirty times
    longer to segment.
    """
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im)
        im = im.convert("L")
        if max_side and max(im.size) > max_side:
            s = float(max_side) / max(im.size)
            im = im.resize((max(1, int(round(im.size[0] * s))),
                            max(1, int(round(im.size[1] * s)))),
                           Image.Resampling.LANCZOS)
        return np.asarray(im, dtype=np.float64) / 255.0


class Trace(object):
    """Everything computed for one input image, stage by stage."""

    def __init__(self, name):
        self.name = name
        self.source = None          # absolute path, if the input was a file
        self.image = None           # greyscale input, float in [0, 1]
        self.ok = False
        self.stage_failed = None    # "load" or "roi" when declined
        self.reason = ""
        self.error_detail = ""      # traceback, if an unexpected error
        self.info = None            # preprocessing diagnostics
        self.roi = None             # 128 x 128 float
        self.code = None            # 32 x 32 orientation code — the template
        self.contrast = None        # 32 x 32 directional contrast
        self.sobel = None           # Han equation (1) magnitude map
        self.han_vec = None         # Han grid-mean feature vector
        self.ms = {}                # stage -> milliseconds

    @property
    def rotation_deg(self):
        """How far the hand is turned from upright, as the ROI frame saw it.

        The frame's axis points along the valley line, which for an upright
        hand is 180 degrees in image coordinates; the difference is the
        hand's rotation, wrapped to (-180, 180].
        """
        if not self.info:
            return None
        a = self.info["angle_deg"] - 180.0
        return (a + 180.0) % 360.0 - 180.0


class Engine(object):
    def __init__(self, root, rloc_cfg=None, han_cells=8, max_side=640,
                 roi_size=128, cache_dir=None, use_cache=True):
        self.root = os.path.abspath(root)
        if not os.path.isdir(self.root):
            raise EngineError("dataset folder not found: %s" % self.root)
        self.manifest = read_manifest(self.root)
        self.kind = self.manifest.get("kind", "hand")
        if self.kind not in ("hand", "roi"):
            raise EngineError("manifest kind must be 'hand' or 'roi'")
        self.layout = self.manifest.get("layout", "subdirs")
        self.per_palm = int(self.manifest.get("per_palm", 10))
        self.name = self.manifest.get("name", os.path.basename(self.root))
        self.note = self.manifest.get("note", "")
        self.cfg = rloc_cfg or rloc.RlocConfig()
        self.han_cells = int(han_cells)
        self.max_side = max_side
        self.roi_size = int(roi_size)
        self.cache_dir = cache_dir or os.path.join(os.path.dirname(_PKG),
                                                   ".cache")
        self.use_cache = bool(use_cache)
        self.ready = False
        self.cache_status = ""
        self.prepare_seconds = 0.0
        self.subset = []            # identities kept by prepare(max_palms=)
        self._reset()

    def _reset(self):
        self.rloc_gallery = rloc.Gallery(self.cfg)
        self.han_gallery = han.HanGallery("sobel", self.han_cells)
        self.rloc_templates = {}    # identity -> (n, h, w) uint8
        self.han_templates = {}     # identity -> (D,) first enrolled vector
        self.enrol_source = {}      # identity -> path of the enrolled image
        self.fte = []               # identities that could not be enrolled
        self.probes = []            # dicts: identity, path, ok, reason
        self.score_matrix = {}      # matcher -> (P, N) array
        self.scoresets = {}
        self.thresholds = {}
        self.eers = {}
        self.threshold_source = {}
        self.live_enrolled = []
        self._examples = {}
        self._truth = {}            # path key -> identity, from the index
        self.protocol = "all probes"
        self.protocol_detail = ""
        self.same_session = {}      # matcher -> ScoreSet, when sessions exist

    @property
    def identities(self):
        return list(self.rloc_gallery.identities)

    # ======================================================= who is who
    @staticmethod
    def _path_key(path):
        return os.path.normcase(os.path.normpath(os.path.abspath(path)))

    def display_name(self, path):
        """
        How an image is named on screen: its path inside the dataset, with
        forward slashes, so `session1/00001.bmp` and `session2/00001.bmp`
        stay distinguishable; just the file name for anything outside it.
        """
        try:
            rel = os.path.relpath(os.path.abspath(path), self.root)
        except ValueError:                  # another drive, on Windows
            return os.path.basename(path)
        if rel.startswith(".."):
            return os.path.basename(path)
        return rel.replace("\\", "/")

    def identity_of(self, path):
        """
        Whose palm an image is, according to the dataset's own index — or
        None for any image the index does not cover, such as a photograph
        brought along for live enrolment.

        This is the ONLY source of truth the demo uses to say whether a
        decision was right. An earlier version read the first folder under
        the dataset instead, which is the person on the synthetic set but
        `session1` on Tongji, so correct accepts were labelled FALSE ACCEPT.
        """
        if not path:
            return None
        return self._truth.get(self._path_key(path))

    def enrolled_from(self, path):
        """
        The identity whose template was built from exactly this image file,
        or None. Verifying a palm with its own enrolment image scores 1.000
        by construction (the unrotated template IS its code), which says
        nothing about the system — so the demo says so when it happens.
        """
        if not path:
            return None
        k = self._path_key(path)
        for ident, src in self.enrol_source.items():
            if src and self._path_key(src) == k:
                return ident
        return None

    def session_of(self, path):
        """The capture session of an indexed image, '' if the dataset has
        no sessions."""
        if self.layout != "tongji" or not path:
            return ""
        from . import tongji
        return tongji.session_of(path)

    # ================================================================ inputs
    def process(self, source, name=None):
        """
        One image through every stage. Never raises for a bad input: an
        unreadable file, a pose the preprocessing cannot read, or even an
        unexpected error comes back as a declined Trace with the reason
        attached, because a live demonstration must decline gracefully
        rather than crash.
        """
        if name is None:
            name = (self.display_name(source) if isinstance(source, str)
                    else "image")
        tr = Trace(name)

        t = time.perf_counter()
        try:
            if isinstance(source, np.ndarray):
                img = np.asarray(source, dtype=np.float64)
                if img.ndim == 3:
                    img = img.mean(axis=2)
                if img.max() > 1.0:
                    img = img / 255.0
            else:
                tr.source = os.path.abspath(source)
                img = load_image(source, self.max_side if self.kind == "hand"
                                 else None)
        except Exception as e:
            tr.stage_failed = "load"
            tr.reason = "could not read the image (%s)" % e.__class__.__name__
            tr.error_detail = traceback.format_exc()
            return tr
        tr.ms["load"] = _ms(t)
        tr.image = img

        t = time.perf_counter()
        if self.kind == "hand":
            try:
                roi, info = pp.extract_roi(img, self.roi_size,
                                           diagnostics=True)
            except pp.PreprocessError as e:
                tr.stage_failed, tr.reason = "roi", str(e)
                tr.ms["roi"] = _ms(t)
                return tr
            except Exception as e:
                tr.stage_failed = "roi"
                tr.reason = "preprocessing failed unexpectedly (%s: %s)" % (
                    e.__class__.__name__, e)
                tr.error_detail = traceback.format_exc()
                tr.ms["roi"] = _ms(t)
                return tr
            tr.info = info
        else:
            # A database of pre-cropped ROIs: the image IS the palm region.
            # An outside photograph only qualifies if it is also a square
            # crop of the palm centre; squashing a whole-hand photograph
            # into 128 x 128 would produce a code, a score and a confident
            # verdict that all mean nothing.
            h, w = img.shape
            if max(h, w) > 1.25 * min(h, w):
                tr.stage_failed = "roi"
                tr.reason = ("this database uses square crops of the palm "
                             "centre; this image is %d × %d — crop the "
                             "palm first" % (w, h))
                tr.ms["roi"] = _ms(t)
                return tr
            roi = img
            if roi.shape != (self.roi_size, self.roi_size):
                im = Image.fromarray(img.astype(np.float32))
                im = im.resize((self.roi_size, self.roi_size),
                               Image.Resampling.LANCZOS)
                roi = np.clip(np.asarray(im, dtype=np.float64), 0.0, 1.0)
        tr.roi = roi
        tr.ms["roi"] = _ms(t)

        t = time.perf_counter()
        tr.code, tr.contrast = rloc.encode_with_contrast(roi, self.cfg)
        tr.ms["encode"] = _ms(t)

        t = time.perf_counter()
        tr.sobel = han.sobel_max(roi)
        tr.han_vec = han.grid_mean(tr.sobel, self.han_cells).ravel()
        tr.ms["han"] = _ms(t)

        tr.ok = True
        return tr

    # =========================================================== preparation
    def prepare(self, progress=None, rebuild=False, max_palms=None):
        """
        Build — or load from cache — the galleries and the evaluation.

        `progress(fraction, message)` is called as work proceeds; the demo
        runs this on a worker thread and shows the messages, so the window
        stays responsive during a first start-up.

        `max_palms` keeps only the first N people, in identity order. The
        demo wants a few dozen palms it can open in seconds; the experiments
        want all of them and have minutes to spare. The limit is part of the
        cache key, so a 40-palm cache can never be mistaken for the full
        evaluation.
        """
        report = progress or (lambda frac, msg: None)
        t0 = time.perf_counter()
        items = protocol.index_images(self.root, self.layout,
                                      per_palm=self.per_palm)
        if not items:
            raise EngineError("no images found in %s" % self.root)
        if max_palms:
            keep = sorted({i for i, _, _ in items})[:int(max_palms)]
            items = [t for t in items if t[0] in set(keep)]
            self.subset = keep
        note = ("first %d palms" % len(self.subset)) if self.subset else ""
        key = self._fingerprint(items, extra=note)

        data = None
        if self.use_cache and not rebuild:
            report(0.0, "loading cached templates")
            data = self._load_cache(key)
        if data is not None:
            self.cache_status = "loaded from cache"
        else:
            data = self._compute(items, report)
            self.cache_status = "computed"
            if self.use_cache:
                self._save_cache(key, data)

        self._install(data)
        self.prepare_seconds = time.perf_counter() - t0
        self.ready = True
        report(1.0, "ready")
        return self

    def _galleries(self, ids, rloc_enrol, han_enrol):
        rg = rloc.Gallery(self.cfg)
        hg = han.HanGallery("sobel", self.han_cells)
        for i, ident in enumerate(ids):
            rg.enrol_codes(ident, rloc_enrol[i])
            hg.enrol_vector(ident, han_enrol[i])
        return rg, hg

    def _compute(self, items, report):
        by_id = {}
        for ident, idx, path in items:
            by_id.setdefault(ident, []).append((idx, path))

        total, done = len(items), 0
        enrol, fte, probes = [], [], []
        for ident in sorted(by_id):
            enrolled = False
            for idx, path in sorted(by_id[ident]):
                tr = self.process(path)
                done += 1
                report(0.80 * done / total,
                       "reading and encoding image %d of %d" % (done, total))
                rel = os.path.relpath(path, self.root).replace("\\", "/")
                if tr.ok and not enrolled:
                    # first readable image of each person is enrolled
                    enrol.append((ident, rel,
                                  rloc.enrol_codes(tr.roi, self.cfg),
                                  tr.han_vec))
                    enrolled = True
                else:
                    probes.append((ident, rel, tr.ok, tr.reason,
                                   tr.code, tr.han_vec))
            if not enrolled:
                fte.append(ident)
        if not enrol:
            raise EngineError("no image in the dataset could be enrolled")

        ids = [e[0] for e in enrol]
        rloc_enrol = np.stack([e[2] for e in enrol]).astype(np.uint8)
        han_enrol = np.stack([e[3] for e in enrol]).astype(np.float64)
        n_probe, n_id = len(probes), len(ids)
        code_shape = rloc_enrol.shape[2:]
        dim = han_enrol.shape[1]

        probe_ok = np.array([p[2] for p in probes], dtype=bool)
        probe_codes = np.zeros((n_probe,) + code_shape, dtype=np.uint8)
        probe_han = np.zeros((n_probe, dim), dtype=np.float64)
        for k, p in enumerate(probes):
            if p[2]:
                probe_codes[k] = p[4]
                probe_han[k] = p[5]

        rg, hg = self._galleries(ids, rloc_enrol, han_enrol)
        rs = np.full((n_probe, n_id), np.nan, dtype=np.float32)
        hs = np.full((n_probe, n_id), np.nan, dtype=np.float32)
        good = np.flatnonzero(probe_ok)

        # RLOC: every probe against every template in one batched pass —
        # bit-for-bit what scoring them one by one gives, a minute instead
        # of an hour on the full Tongji release (see rloc.scores_batch).
        def rloc_progress(done, steps):
            report(0.80 + 0.17 * done / steps,
                   "matching RLOC codes: pass %d of %d" % (done, steps))
        if good.size:
            rs[good] = rg.scores_batch(probe_codes[good],
                                       progress=rloc_progress)
        for n, k in enumerate(good):
            hs[k] = hg.scores(probe_han[k])
            if (n + 1) % 200 == 0 or n + 1 == good.size:
                report(0.97 + 0.03 * (n + 1) / good.size,
                       "matching Han features: probe %d of %d"
                       % (n + 1, good.size))

        meta = {"identities": ids,
                "enrol_paths": [e[1] for e in enrol],
                "fte": fte,
                "probes": [{"identity": p[0], "path": p[1], "ok": bool(p[2]),
                            "reason": p[3] or ""} for p in probes]}
        return {"meta": meta, "rloc_enrol": rloc_enrol, "han_enrol": han_enrol,
                "probe_codes": probe_codes, "probe_han": probe_han,
                "probe_ok": probe_ok, "rloc_scores": rs, "han_scores": hs}

    def _from_rel(self, rel):
        """A cached dataset-relative path back to a real one. Either
        separator is accepted, so a cache written on one system reads on
        another."""
        parts = [p for p in rel.replace("\\", "/").split("/") if p]
        return os.path.join(self.root, *parts)

    def _install(self, data):
        self._reset()
        meta = data["meta"]
        ids = list(meta["identities"])
        self.rloc_gallery, self.han_gallery = self._galleries(
            ids, data["rloc_enrol"], data["han_enrol"])
        for i, ident in enumerate(ids):
            self.rloc_templates[ident] = np.asarray(data["rloc_enrol"][i])
            self.han_templates[ident] = np.asarray(data["han_enrol"][i])
            self.enrol_source[ident] = self._from_rel(meta["enrol_paths"][i])
        self.fte = list(meta["fte"])
        self.probes = [{"identity": p["identity"],
                        "path": self._from_rel(p["path"]),
                        "ok": bool(p["ok"]), "reason": p["reason"]}
                       for p in meta["probes"]]
        self.score_matrix = {
            "rloc": np.asarray(data["rloc_scores"], dtype=np.float64),
            "han": np.asarray(data["han_scores"], dtype=np.float64)}

        col = {ident: j for j, ident in enumerate(ids)}
        own = np.array([col.get(p["identity"], -1) for p in self.probes],
                       dtype=np.int64)
        ok = np.array([p["ok"] for p in self.probes], dtype=bool)
        self._own, self._ok = own, ok

        # Who every indexed image belongs to — the demo's only ground truth.
        for i, ident in enumerate(ids):
            self._truth[self._path_key(self.enrol_source[ident])] = ident
        for p in self.probes:
            self._truth[self._path_key(p["path"])] = p["identity"]

        # Which probes the evaluation is built from. Where the dataset has
        # capture sessions, only probes from a DIFFERENT session than the
        # person's enrolment count: a second capture taken a minute after the
        # first is far easier to match than one taken weeks later, and on
        # Tongji mixing the two flatters RLOC's EER from 0.11% to 0.06% and
        # Han's from 7.7% to 6.0%. The same-session numbers are kept, but
        # separately and labelled.
        enrol_sess = {ident: self.session_of(self.enrol_source[ident])
                      for ident in ids}
        probe_sess = [self.session_of(p["path"]) for p in self.probes]
        cross = np.array([bool(s) and s != enrol_sess.get(p["identity"], "")
                          for s, p in zip(probe_sess, self.probes)],
                         dtype=bool)
        if cross.any():
            self._eval = ok & cross
            same_rows = ok & ~cross
            first = sorted({s for s in enrol_sess.values() if s})
            self.protocol = "cross-session"
            self.protocol_detail = ("enrolled from %s, tested only with "
                                    "images from another session"
                                    % " / ".join(first))
        else:
            self._eval = ok
            same_rows = None
            self.protocol = "all probes"
            self.protocol_detail = ""

        def scoreset(S, rows, label):
            gk = np.flatnonzero(rows & (own >= 0))
            gen = S[gk, own[gk]]
            imp_mask = np.zeros(S.shape, dtype=bool)
            imp_mask[rows] = True
            imp_mask[gk, own[gk]] = False
            imp = S[imp_mask]
            if gen.size and imp.size:
                return metrics.ScoreSet(gen, imp, label)
            return None

        for m in MATCHERS:
            S = self.score_matrix[m]
            ss = scoreset(S, self._eval, MATCHER_NAMES[m])
            if ss is not None:
                eer, thr = ss.eer()
                self.scoresets[m] = ss
                self.eers[m] = eer
                self.thresholds[m] = thr
                self.threshold_source[m] = (
                    "equal-error point of the %s evaluation" % self.protocol
                    if self.protocol != "all probes"
                    else "equal-error point of this dataset")
            else:
                self.thresholds[m] = FALLBACK_THRESHOLDS[m]
                self.threshold_source[m] = ("fallback value - too few images "
                                            "to estimate one")
            if same_rows is not None and same_rows.any():
                s2 = scoreset(S, same_rows, MATCHER_NAMES[m] + " same session")
                if s2 is not None:
                    self.same_session[m] = s2
        self._pick_examples()

    def _pick_examples(self):
        """
        One-click demonstration cases, chosen from the evaluation itself:
        a TYPICAL genuine attempt and a typical impostor attempt (the
        medians — not cherry-picked best cases), plus the hardest impostor,
        the one that came closest to being accepted, which is the honest
        case to show when explaining what the threshold is for.

        Drawn only from the probes the evaluation uses — so on a dataset with
        sessions, the genuine example is a capture from a later session than
        the enrolment, the harder and more honest case.
        """
        ex = {}
        if not self.probes:
            self._examples = ex
            return
        S = self.score_matrix["rloc"]
        own, ok = self._own, self._eval
        ids = self.identities

        gk = np.flatnonzero(ok & (own >= 0))
        if gk.size:
            gs = S[gk, own[gk]]
            k = int(gk[np.argsort(gs, kind="stable")[gs.size // 2]])
            ex["genuine"] = {"path": self.probes[k]["path"],
                             "claim": self.probes[k]["identity"],
                             "score": float(S[k, own[k]])}

        imp_mask = np.zeros(S.shape, dtype=bool)
        imp_mask[ok] = True
        for k in gk:
            imp_mask[k, own[k]] = False
        where = np.argwhere(imp_mask)
        if where.size:
            vals = S[imp_mask]
            order = np.argsort(vals, kind="stable")
            for label, pos in (("impostor", order[vals.size // 2]),
                               ("hardest", order[-1])):
                k, j = (int(v) for v in where[pos])
                ex[label] = {"path": self.probes[k]["path"],
                             "claim": ids[j], "score": float(S[k, j]),
                             "true_identity": self.probes[k]["identity"]}

        unread = (self.manifest.get("examples") or {}).get("unreadable")
        if unread:
            p = os.path.join(self.root, unread)
            if os.path.isfile(p):
                ex["unreadable"] = {"path": p, "claim": None}
        self._examples = ex

    def examples(self):
        return dict(self._examples)

    # ============================================================= decisions
    def _rloc_detail(self, trace, identity):
        return rloc.explain_match(trace.code, self.rloc_templates[identity],
                                  self.cfg)

    def _han_detail(self, trace, identity):
        c = self.han_cells
        return {"probe_grid": trace.han_vec.reshape(c, c),
                "template_grid": self.han_templates[identity].reshape(c, c)}

    def verify(self, trace, claimed, matcher="rloc"):
        """One-to-one: is this the person claimed?"""
        if matcher not in MATCHERS:
            raise ValueError("matcher must be one of %s" % (MATCHERS,))
        thr = self.thresholds.get(matcher, FALLBACK_THRESHOLDS[matcher])
        out = {"task": "verify", "matcher": matcher, "claimed": claimed,
               "threshold": thr}
        if not trace.ok:
            out.update(decision="declined", accept=False, score=None,
                       reason=trace.reason, ms=0.0)
            return out
        if claimed not in self.rloc_templates:
            raise EngineError("%s is not enrolled" % claimed)

        t = time.perf_counter()
        if matcher == "rloc":
            detail = self._rloc_detail(trace, claimed)
            score = detail["score"]
        else:
            # the same mean Pearson correlation the evaluation used
            score = self.han_gallery.verify(trace.han_vec, claimed)
            detail = self._han_detail(trace, claimed)
        accept = bool(score >= thr)
        out.update(score=float(score), accept=accept, detail=detail,
                   decision="accept" if accept else "reject", ms=_ms(t))
        return out

    def identify(self, trace, matcher="rloc", top=5):
        """One-to-many: whose palm is this, if anyone's?"""
        if matcher not in MATCHERS:
            raise ValueError("matcher must be one of %s" % (MATCHERS,))
        thr = self.thresholds.get(matcher, FALLBACK_THRESHOLDS[matcher])
        out = {"task": "identify", "matcher": matcher, "threshold": thr}
        if not trace.ok:
            out.update(decision="declined", accept=False, score=None,
                       reason=trace.reason, ranked=[], identity=None, ms=0.0)
            return out

        t = time.perf_counter()
        if matcher == "rloc":
            s = self.rloc_gallery.scores(trace.code)
            ids = self.rloc_gallery.identities
        else:
            s = self.han_gallery.scores(trace.han_vec)
            ids = self.han_gallery.identities
        order = np.argsort(-s, kind="stable")
        ranked = [(ids[k], float(s[k])) for k in order[:top]]
        best_id, best = ranked[0]
        accept = bool(best >= thr)
        detail = (self._rloc_detail(trace, best_id) if matcher == "rloc"
                  else self._han_detail(trace, best_id))
        out.update(ranked=ranked, score=best, accept=accept, detail=detail,
                   identity=best_id if accept else None,
                   candidate=best_id,
                   decision="accept" if accept else "reject", ms=_ms(t))
        return out

    def enrol(self, name, source):
        """
        Enrol a new person live, from one image. Returns the Trace: if it
        was declined, nothing is enrolled and trace.reason says why.
        Enrolling an existing name adds templates to that person.
        """
        name = (name or "").strip()
        if not name:
            raise EngineError("a name is needed to enrol someone")
        tr = self.process(source)
        if not tr.ok:
            return tr
        codes = rloc.enrol_codes(tr.roi, self.cfg)
        self.rloc_gallery.enrol_codes(name, codes)
        self.han_gallery.enrol_vector(name, tr.han_vec)
        if name in self.rloc_templates:
            self.rloc_templates[name] = np.concatenate(
                [self.rloc_templates[name], codes])
        else:
            self.rloc_templates[name] = codes
            self.han_templates[name] = tr.han_vec
            self.enrol_source[name] = tr.source
        if name not in self.live_enrolled:
            self.live_enrolled.append(name)
        return tr

    # ============================================================= reporting
    def summary(self):
        d = {"name": self.name, "kind": self.kind, "note": self.note,
             "enrolled": len(self.identities),
             "probes": len(self.probes),
             "declined": sum(1 for p in self.probes if not p["ok"]),
             "fte": len(self.fte), "cache": self.cache_status,
             "subset": len(self.subset),
             "seconds": self.prepare_seconds}
        for m in MATCHERS:
            ss = self.scoresets.get(m)
            d[m] = {"threshold": self.thresholds.get(m),
                    "threshold_source": self.threshold_source.get(m, ""),
                    "eer": self.eers.get(m),
                    "genuine": ss.n_genuine if ss else 0,
                    "impostor": ss.n_impostor if ss else 0}
            s2 = self.same_session.get(m)
            if s2 is not None:
                d[m]["same_session_eer"] = s2.eer()[0]
        d["protocol"] = self.protocol
        d["protocol_detail"] = self.protocol_detail
        return d

    # ================================================================= cache
    def _fingerprint(self, items, extra=""):
        """
        The cache key: the images, the settings, and the source code of
        everything that PRODUCES the cached arrays — the feature and matching
        modules in full, plus the engine's own image-processing and scoring
        functions. The engine's display logic (thresholds, examples, labels)
        is deliberately not part of it: that is recomputed from the cached
        scores on every start, so changing it costs nothing, whereas a full
        Tongji rebuild costs minutes.
        """
        import inspect
        h = hashlib.sha256()
        h.update(("engine %s\n" % ENGINE_VERSION).encode("utf-8"))
        h.update(json.dumps({"kind": self.kind, "layout": self.layout,
                             "roi": self.roi_size, "max_side": self.max_side,
                             "rloc": self.cfg.describe(),
                             "rotations": list(self.cfg.rotations),
                             "han_cells": self.han_cells,
                             "per_palm": self.per_palm, "subset": extra},
                            sort_keys=True).encode("utf-8"))
        for fn in _SOURCES:
            p = os.path.join(_PKG, fn)
            if os.path.isfile(p):
                with open(p, "rb") as f:
                    h.update(f.read())
        for fn in (load_image, Engine.process, Engine._compute,
                   Engine._galleries):
            h.update(inspect.getsource(fn).encode("utf-8"))
        for ident, idx, path in items:
            st = os.stat(path)
            rel = os.path.relpath(path, self.root).replace("\\", "/")
            h.update(("%s|%s|%d|%d\n" % (ident, rel, st.st_size,
                                         st.st_mtime_ns)).encode("utf-8"))
        return h.hexdigest()[:16]

    def _cache_path(self, key):
        safe = "".join(c if (c.isalnum() or c in "-_") else "_"
                       for c in self.name)[:40]
        return os.path.join(self.cache_dir, "%s-%s.npz" % (safe, key))

    def _load_cache(self, key):
        path = self._cache_path(key)
        if not os.path.isfile(path):
            return None
        try:
            with np.load(path, allow_pickle=False) as z:
                data = {k: z[k] for k in z.files}
            data["meta"] = json.loads(bytes(data["meta"]).decode("utf-8"))
            return data
        except Exception:
            return None                 # a damaged cache is simply rebuilt

    def _save_cache(self, key, data):
        path = self._cache_path(key)
        tmp = path + ".tmp"
        try:
            os.makedirs(self.cache_dir, exist_ok=True)
            arrays = {k: v for k, v in data.items() if k != "meta"}
            arrays["meta"] = np.frombuffer(
                json.dumps(data["meta"]).encode("utf-8"), dtype=np.uint8)
            with open(tmp, "wb") as f:
                np.savez_compressed(f, **arrays)
            os.replace(tmp, path)
        except OSError as e:
            self.cache_status += " (not saved: %s)" % e.__class__.__name__
