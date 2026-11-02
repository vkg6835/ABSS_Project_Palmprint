"""
gabor.py — a Gabor orientation code, for the experiment Paper 2 never ran.

Paper 2's title claim is that MFRAT, a sum along straight lines, is a better
line detector than the Gabor filters of Competitive Code. But its headline
gain over Competitive Code (EER 0.47% -> 0.16%) arrives together with two
changes that have nothing to do with the filter: seven rotated enrolment
templates, and the cross-shaped "pixel-to-area" matching rule. Its own
Table 4 gives the only like-for-like filter comparison, 0.40% against 0.47%,
and our Phase I critique argued that the filter therefore accounts for only
a small part of the gain.

This module lets that be measured instead of argued. It produces a code with
exactly the shape and meaning of mfrat.orientation_code — the index of the
winning direction at each point of a 32 x 32 lattice on a 128 x 128 ROI, by
winner-take-all over six directions — but from Gabor responses. Everything
downstream (rotated templates, neighbourhood matching, the batched scorer)
is shared, so swapping the filter is the ONLY difference between the two
arms of the comparison.

The filter
----------
The real part of the "neurophysiology-based" Gabor function that Competitive
Code uses (as we understand Kong & Zhang 2004, following Lee 1996):

    psi(x, y) = w / (sqrt(2 pi) k) * exp(-(w^2 / 8k^2)(4x'^2 + y'^2))
                * (cos(w x') - exp(-k^2 / 2))
    x' =  x cos(theta) + y sin(theta)
    y' = -x sin(theta) + y cos(theta)
    k  = sqrt(2 ln 2) (2^d + 1) / (2^d - 1)        d = bandwidth in octaves

Six orientations theta_j = j pi / 6. Each kernel is made exactly zero-mean
after truncation, so — like MFRAT with its equal pixel counts — the code is
exactly invariant to a change of brightness and contrast (f -> a f + b,
a > 0): every response is multiplied by the same a, and the argmin cannot
move.

The wavelength (w) is the one free parameter. Rather than quote a value we
cannot check, run_experiments.py chooses it on session-1 data only and
reports every number on session-2 data, so the choice cannot be tuned to the
test. That gives the Gabor arm, if anything, an advantage: MFRAT's
parameters are taken from its paper unchanged.
"""
import numpy as np

N_DIR = 6
BANDWIDTH = 1.5          # octaves
RADIUS_SIGMAS = 2.5      # kernel half-width, in units of the long sigma

_BANK_CACHE = {}


def kappa(bandwidth=BANDWIDTH):
    b = 2.0 ** bandwidth
    return np.sqrt(2.0 * np.log(2.0)) * (b + 1.0) / (b - 1.0)


def kernel(omega, theta, bandwidth=BANDWIDTH, radius=None):
    """The real, zero-mean Gabor kernel at one orientation."""
    k = kappa(bandwidth)
    sigma_long = 2.0 * k / omega                 # along the line (y')
    if radius is None:
        radius = int(np.ceil(RADIUS_SIGMAS * sigma_long))
    t = np.arange(-radius, radius + 1, dtype=np.float64)
    x, y = np.meshgrid(t, t)                     # x: columns, y: rows
    xp = x * np.cos(theta) + y * np.sin(theta)
    yp = -x * np.sin(theta) + y * np.cos(theta)
    env = np.exp(-(omega ** 2) / (8.0 * k ** 2) * (4.0 * xp ** 2 + yp ** 2))
    g = omega / (np.sqrt(2.0 * np.pi) * k) * env * (
        np.cos(omega * xp) - np.exp(-k ** 2 / 2.0))
    return g - g.mean()                          # exactly DC-free


def bank(omega, size, n_dir=N_DIR, bandwidth=BANDWIDTH):
    """
    The filter bank's FFTs for images of `size` x `size`, cached. Returns
    (kernels_fft, radius, padded_size).
    """
    key = (round(float(omega), 6), int(size), int(n_dir), float(bandwidth))
    if key in _BANK_CACHE:
        return _BANK_CACHE[key]
    kers = [kernel(omega, np.pi * j / n_dir, bandwidth) for j in range(n_dir)]
    r = kers[0].shape[0] // 2
    n = int(size) + 2 * r
    ffts = []
    for kk in kers:
        pad = np.zeros((n, n), dtype=np.float64)
        pad[:kk.shape[0], :kk.shape[1]] = kk
        # move the kernel centre to the origin, so the product of FFTs is a
        # centred (circular) correlation; the kernel is point-symmetric, so
        # correlation and convolution coincide
        pad = np.roll(pad, (-r, -r), axis=(0, 1))
        ffts.append(np.fft.rfft2(pad))
    out = (np.stack(ffts), r, n)
    _BANK_CACHE[key] = out
    return out


def responses(img, omega, n_dir=N_DIR, bandwidth=BANDWIDTH):
    """(n_dir, H, W) filter responses, edges handled by replicating the
    border (as MFRAT does), so no wrap-around reaches the ROI."""
    img = np.asarray(img, dtype=np.float64)
    h, w = img.shape
    if h != w:
        raise ValueError("expected a square ROI")
    ffts, r, n = bank(omega, h, n_dir, bandwidth)
    padded = np.pad(img, r, mode="edge")
    f = np.fft.rfft2(padded)
    resp = np.fft.irfft2(ffts * f[None], s=(n, n))
    return resp[:, r:r + h, r:r + w]


def orientation_code(img, omega, step=4, n_dir=N_DIR, bandwidth=BANDWIDTH):
    """
    The Gabor competitive code on the same lattice as mfrat.orientation_code:
    argmin over directions (palm lines are dark), sampled every `step` pixels
    starting at step // 2. Returns (code uint8, contrast) like MFRAT's.
    """
    resp = responses(img, omega, n_dir, bandwidth)
    off = step // 2
    sub = resp[:, off::step, off::step]
    return (np.argmin(sub, axis=0).astype(np.uint8), np.ptp(sub, axis=0))


def encode(img, omega, step=4):
    return orientation_code(img, omega, step)[0]
