"""
metrics.py — verification performance: FAR, FRR, EER, ROC, confusion matrix.

Requirement 5 of the Phase II brief is "the performance of your experiment,
i.e. the FAR and FRR or confusion matrix". This module is the only place
those are computed, so every number in the presentation traces to one call.

Convention throughout: scores are SIMILARITIES and the decision is

    accept if score >= threshold

so FAR falls and FRR rises as the threshold rises. Both matchers in this
package follow that convention (RLOC returns a matched-pixel fraction, Han
returns a mean Pearson correlation), which avoids the sign confusion that
comes from mixing distances and similarities.

Definitions used:
  FAR(t)  fraction of IMPOSTOR comparisons scoring >= t   (wrongly accepted)
  FRR(t)  fraction of GENUINE  comparisons scoring <  t   (wrongly rejected)
  GAR(t)  1 - FRR(t)
  EER     the rate where FAR(t) = FRR(t), by linear interpolation between the
          two bracketing thresholds
  d'      (mean_gen - mean_imp) / sqrt((var_gen + var_imp) / 2), a
          threshold-free measure of how far apart the two distributions are

Pure numpy — no matplotlib import, so this module stays importable on a
machine that only has numpy and Pillow (which is the demo machine).
"""
import numpy as np


class ScoreSet(object):
    """
    The genuine and impostor score distributions of one experiment.

    Everything is derived from these two arrays by order statistics, using
    searchsorted rather than broadcasting, so a million impostor comparisons
    costs a sort instead of a million-by-million array.
    """

    def __init__(self, genuine, impostor, label=""):
        self.genuine = np.sort(np.asarray(genuine, dtype=np.float64).ravel())
        self.impostor = np.sort(np.asarray(impostor, dtype=np.float64).ravel())
        self.label = label
        if self.genuine.size == 0 or self.impostor.size == 0:
            raise ValueError("need at least one genuine and one impostor score")

    # ------------------------------------------------------------ basics
    @property
    def n_genuine(self):
        return int(self.genuine.size)

    @property
    def n_impostor(self):
        return int(self.impostor.size)

    def far(self, t):
        """Fraction of impostor scores >= t."""
        t = np.asarray(t, dtype=np.float64)
        idx = np.searchsorted(self.impostor, t, side="left")
        return (self.impostor.size - idx) / float(self.impostor.size)

    def frr(self, t):
        """Fraction of genuine scores < t."""
        t = np.asarray(t, dtype=np.float64)
        idx = np.searchsorted(self.genuine, t, side="left")
        return idx / float(self.genuine.size)

    def gar(self, t):
        return 1.0 - self.frr(t)

    def thresholds(self):
        """Every score that could serve as a decision boundary."""
        return np.unique(np.concatenate([self.genuine, self.impostor]))

    # --------------------------------------------------------------- EER
    def eer(self):
        """
        Returns (eer, threshold).

        FAR is non-increasing and FRR non-decreasing in t, so FAR - FRR
        changes sign exactly once. Both rates are step functions, constant on
        every interval (t[k-1], t[k]] between consecutive observed scores.

        Two cases:
          - FAR - FRR is exactly zero over some interval — always the case
            when the genuine and impostor scores do not overlap at all, the
            interval then being (highest impostor, lowest genuine]. The EER
            threshold is placed at the MIDPOINT of that interval. Taking an
            end instead, as a naive search does, puts the operating threshold
            exactly on the lowest genuine score ever observed, so a fresh
            genuine attempt scoring a hair below it is rejected — a poor
            threshold to stand a live demonstration on.
          - otherwise the crossing is interpolated linearly between the two
            bracketing thresholds, and the EER is read off the interpolated
            FAR and FRR curves at that crossing — the standard estimate. (An
            earlier version averaged the two rates at the upper bracket
            instead, which overstates the EER by up to half a step.)
        """
        t = self.thresholds()
        far, frr = self.far(t), self.frr(t)
        d = far - frr

        zero = np.flatnonzero(d == 0)
        if zero.size:
            i0, i1 = int(zero[0]), int(zero[-1])
            lo = t[i0 - 1] if i0 > 0 else t[i0]
            t_star = (lo + t[i1]) / 2.0
            return float((far[i1] + frr[i1]) / 2.0), float(t_star)

        if np.all(d > 0):
            return float(far[-1] + frr[-1]) / 2.0, float(t[-1])
        if np.all(d < 0):
            return float(far[0] + frr[0]) / 2.0, float(t[0])
        i = int(np.argmax(d < 0))          # first threshold where FRR wins
        lo, hi = i - 1, i
        w = d[lo] / (d[lo] - d[hi])        # d[lo] > 0 > d[hi]: w in (0, 1)
        t_star = t[lo] + w * (t[hi] - t[lo])
        far_x = far[lo] + w * (far[hi] - far[lo])
        frr_x = frr[lo] + w * (frr[hi] - frr[lo])
        return float((far_x + frr_x) / 2.0), float(t_star)

    def threshold_at_far(self, target_far):
        """The lowest threshold whose FAR does not exceed `target_far`."""
        t = self.thresholds()
        far = self.far(t)
        ok = np.nonzero(far <= target_far)[0]
        if ok.size == 0:
            return float(t[-1] + 1e-9)
        return float(t[ok[0]])

    def frr_at_far(self, target_far):
        """
        FRR at a fixed operating FAR — the operating point Paper 2 reports in
        its Table 5, and the one that actually matters for a door: how many
        legitimate users get turned away at an acceptable intrusion rate.
        """
        t = self.threshold_at_far(target_far)
        return float(self.frr(t)), t

    def dprime(self):
        g, i = self.genuine, self.impostor
        num = g.mean() - i.mean()
        den = np.sqrt((g.var() + i.var()) / 2.0)
        return float(num / den) if den > 0 else float("inf")

    # ----------------------------------------------------------- curves
    def roc(self, n=512):
        """
        (far, gar, thresholds) for plotting, with the thresholds chosen so
        that BOTH tails are drawn in detail: half of them sit at impostor
        quantiles for FARs spaced evenly on a log scale down to a single
        impostor, half at genuine quantiles for FRRs spaced likewise down to a
        single genuine score.

        (An earlier version took thresholds evenly by rank among all observed
        scores. With millions of impostor scores and thousands of genuine
        ones, nearly every point then fell in the impostor bulk, and the
        low-FAR region — the one a DET plot exists to show — got a handful.)
        """
        half = max(2, n // 2)
        ni, ng = self.impostor.size, self.genuine.size
        fars = np.logspace(np.log10(1.0 / ni), 0.0, half)
        frrs = np.logspace(np.log10(1.0 / ng), 0.0, half)
        # FAR(t) = fraction of impostors >= t: the threshold for a target FAR
        # is the impostor score with that fraction above it
        ti = self.impostor[np.clip(ni - np.ceil(fars * ni).astype(int), 0,
                                   ni - 1)]
        tg = self.genuine[np.clip(np.floor(frrs * ng).astype(int), 0,
                                  ng - 1)]
        t = np.unique(np.concatenate([ti, tg, [self.genuine[0],
                                               self.impostor[-1]]]))
        return self.far(t), self.gar(t), t

    def det_rates(self, n=512):
        """(far, frr) over the threshold range, for a DET-style plot."""
        far, gar, t = self.roc(n)
        return far, 1.0 - gar, t

    # ------------------------------------------------- confusion matrix
    def confusion(self, threshold):
        """
        The 2x2 table at one operating point.

        Rows are the truth (genuine / impostor), columns the decision
        (accept / reject).
        """
        tp = int(np.sum(self.genuine >= threshold))
        fn = self.n_genuine - tp
        fp = int(np.sum(self.impostor >= threshold))
        tn = self.n_impostor - fp
        return {"threshold": float(threshold),
                "genuine_accepted": tp, "genuine_rejected": fn,
                "impostor_accepted": fp, "impostor_rejected": tn,
                "far": fp / float(self.n_impostor),
                "frr": fn / float(self.n_genuine),
                "gar": tp / float(self.n_genuine),
                "accuracy": (tp + tn) / float(self.n_genuine
                                              + self.n_impostor)}

    # ----------------------------------------------------------- report
    def summary(self, operating_far=1e-3):
        eer, t_eer = self.eer()
        frr_op, t_op = self.frr_at_far(operating_far)
        return {"label": self.label,
                "n_genuine": self.n_genuine,
                "n_impostor": self.n_impostor,
                "eer": eer, "eer_threshold": t_eer,
                "dprime": self.dprime(),
                "operating_far": operating_far,
                "frr_at_operating_far": frr_op,
                "operating_threshold": t_op,
                "genuine_mean": float(self.genuine.mean()),
                "genuine_std": float(self.genuine.std()),
                "impostor_mean": float(self.impostor.mean()),
                "impostor_std": float(self.impostor.std())}

    def report(self, operating_far=1e-3):
        s = self.summary(operating_far)
        c = self.confusion(s["eer_threshold"])
        lines = [
            "%s" % (self.label or "scores"),
            "  comparisons      %d genuine, %d impostor"
            % (s["n_genuine"], s["n_impostor"]),
            "  genuine  score   %.4f +/- %.4f"
            % (s["genuine_mean"], s["genuine_std"]),
            "  impostor score   %.4f +/- %.4f"
            % (s["impostor_mean"], s["impostor_std"]),
            "  separation d'    %.2f" % s["dprime"],
            "  EER              %.3f %%  at threshold %.4f"
            % (100.0 * s["eer"], s["eer_threshold"]),
            "  FRR @ FAR=%.3g%%  %.3f %%  at threshold %.4f"
            % (100.0 * operating_far, 100.0 * s["frr_at_operating_far"],
               s["operating_threshold"]),
            "  at the EER point: %d/%d genuine accepted, "
            "%d/%d impostors rejected"
            % (c["genuine_accepted"], self.n_genuine,
               c["impostor_rejected"], self.n_impostor),
        ]
        return "\n".join(lines)


def confusion_table(cm):
    """Render a confusion dict as a small fixed-width table."""
    return "\n".join([
        "                   accepted    rejected",
        "  genuine        %10d  %10d" % (cm["genuine_accepted"],
                                         cm["genuine_rejected"]),
        "  impostor       %10d  %10d" % (cm["impostor_accepted"],
                                         cm["impostor_rejected"]),
        "  FAR %.4f %%   FRR %.4f %%   accuracy %.3f %%"
        % (100 * cm["far"], 100 * cm["frr"], 100 * cm["accuracy"]),
    ])


def compare(scoresets, operating_far=1e-3):
    """Side-by-side EER / d' table for several experiments."""
    w = max(len(s.label) for s in scoresets)
    head = ("%-*s  %9s  %7s  %12s" % (w, "experiment", "EER %", "d'",
                                      "FRR@FAR %"))
    rows = [head, "-" * len(head)]
    for s in scoresets:
        d = s.summary(operating_far)
        rows.append("%-*s  %9.3f  %7.2f  %12.3f"
                    % (w, s.label, 100 * d["eer"], d["dprime"],
                       100 * d["frr_at_operating_far"]))
    return "\n".join(rows)
