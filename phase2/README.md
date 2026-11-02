# Phase II — palmprint verification

AI461 Advanced Biometric Systems and Security, SVNIT Surat.

Primary system: **RLOC** — Jia, Huang & Zhang, *Palmprint verification based on
robust line orientation code*, Pattern Recognition 41 (2008) 1504–1513.
Comparison baseline: **Han et al.**, *Personal authentication using palm-print
features*, Pattern Recognition 36 (2003) 371–381.

Both techniques are implemented so the comparison rests on our own
measurements under one protocol, rather than on numbers quoted from two
different databases.

## Running it

Requires only `numpy` and `Pillow`, both already installed, plus `tkinter`
for the demo window, which ships with Python on Windows. No scipy, no OpenCV,
no network access.

```
cd phase2
python selftest.py          # operator, matcher, protocol, metrics
python test_preprocess.py   # ROI extraction and its repeatability
python test_demo.py         # the demo's engine and display, without a window
python test_tongji.py       # the Tongji loader (--data PATH for a real copy)
python test_experiments.py  # the experiment pipeline, on a small synthetic set
python run_experiments.py   # every result table and figure (resumable)
python demo.py              # the live demo (or double-click run_demo.bat)
python demo.py --check      # rehearse the demo (or double-click check_demo.bat)
```

Each test file takes between a few seconds and a minute and ends with
`all checks passed`.

## The live demo

`run_demo.bat`, or `python demo.py`. The first start writes the synthetic
demo data to `demo_data/synthetic` and preprocesses it, which takes about
half a minute; after that, start-up takes a second or two because the
templates are cached in `.cache/`. The cache is keyed on the images and on
the source code, so editing either rebuilds it automatically.

Load a probe, then step through what the system does to it. Each step is one
key, so it can be driven from a presentation clicker:

| step | panel | what it shows |
|---|---|---|
| 1 | Input | the captured image, untouched |
| 2 | Region of interest | outline, finger valleys and ROI window drawn on the input; the 128×128 crop |
| 3 | Orientation code | the 32×32 code as a field of strokes (Han: the Sobel magnitude map) |
| 4 | Matching | which cells matched the claimed person's template (Han: the pooled 8×8 grids) |
| 5 | Decision | ACCEPT or REJECT, and where this score falls among every comparison in the dataset |

Keys: **Space / → / Page Down** next step · **← / Page Up** back ·
**A** every step · **R** back to step 1 · **G** genuine attempt · **I** impostor ·
**C** closest impostor · **U** unreadable image · **O** open any image ·
**E** enrol a new person · **T** verify ⇄ identify · **M** RLOC ⇄ Han ·
**H** hide the controls · **F11** full screen.

Things worth knowing before presenting with it:

- **The quick examples are typical, not cherry-picked.** "Genuine" and
  "Impostor" are the median attempts of the evaluation. "Closest impostor" is
  the one impostor that came nearest to being accepted — the honest case for
  explaining what the threshold is for. On real data it may be accepted, and
  if so the screen says FALSE ACCEPT.
- **When the true identity is known, the verdict says whether it was right** —
  including FALSE ACCEPT and FALSE REJECT in capitals when it was not.
- **The threshold is the dataset's equal-error point**, from the same
  evaluation the histogram shows. Where the two score distributions do not
  overlap at all, it sits midway between them rather than on the lowest
  genuine score ever seen, so a fresh genuine attempt is not rejected by a
  hair.
- **Live enrolment lasts until the window closes.** It is deliberately not
  saved, so every run starts from the same state. `demo_data/synthetic/_visitors`
  holds two people who are not in the database: enrol visitor A from one
  capture, verify the other, then identify visitor B — who is matched to
  nobody.
- **An unreadable image is declined, never guessed** — try U.
- **The synthetic data is labelled as such** in the window header. No
  performance claim is made on it.

**Rehearse on the laptop that will be used in class:** `check_demo.bat`. It
drives every scenario by itself, saves a screenshot of the real window for
each to `_screens/`, and writes `_screens/check_report.txt`. Do not touch the
mouse or keyboard while it runs (about a minute).

If anything goes wrong during the demo, the window stays open, says what
happened, and writes the details to `demo_error.log`.

## Datasets

Both papers assume the same input: a square, scale- and rotation-normalised
crop of the palm, not a photograph of a hand. A database that ships those
crops lets the comparison measure the two *feature extractors*, with no
segmentation of ours in the way.

**Tongji** (in use while IITD is pending). 600 palms, 10 captures per palm
per session, two sessions. Extract `ROI.rar` to `data/tongji/roi/`, keeping
the `session1/` and `session2/` folders, then:

```
python test_tongji.py --data data/tongji/roi
```

That checks the folder against the release's own shape, verifies the images
load as usable greyscale, **tests the numbering against the pixels** (see
below), and — only if every check passes — writes `manifest.json` so the demo
and the experiments read it. The demo then runs with `run_demo_tongji.bat`
(all 600 palms). The first start on new code encodes all 12,000 images and
then scores every probe against every palm — several minutes, about five;
every later start loads the saved templates in seconds.

*Use all 600 palms, not `--palms 40`.* An EER measured on a few dozen people
is mostly noise: across random sets of 40 Tongji palms the cross-session RLOC
EER ranges from 0% to 3.5%, and the first 40 happen to be the hardest block
in the database (they include P014 and P025, the two worst palms of all 600),
giving 3.7% against 0.11% for the whole release. `--palms` is for quick
checks, not for numbers anyone quotes.

*The evaluation is cross-session.* Each palm is enrolled from its first
session-1 capture and tested only with its session-2 captures — weeks later,
not a minute later. Mixing in the nine remaining session-1 captures flatters
both methods (RLOC 0.06% instead of 0.11%, Han 6.0% instead of 7.7%), so the
demo's threshold, histogram and examples, and the status bar's EER, all come
from the cross-session comparisons: 6,000 genuine and 3,594,000 impostor.
The same-session figures are kept and labelled separately.

*Bringing your own palm.* The ROIs in Tongji were cut out by the capture
device, so every one is framed the same way; RLOC was designed for that and
searches only ±8 pixels of shift (±2 code pixels) and ±9° of rotation. A
hand-held photo of your own palm must be framed just as consistently: a
square crop of the palm centre, taken from a fixed position (phone on a
stand, palm flat on a marked spot, same distance and lighting each time).
A non-square photo is declined outright, because squashing a whole-hand
picture into 128 × 128 would produce a confident verdict that means nothing.
Three photos of one hand, each enrolled in turn and matched against the other
two, are all accepted (0.815–0.867), with the winning shift between 1 and 2
code pixels — close to the edge of what the search covers, so a looser
capture discipline would start costing acceptances.

*Why the numbering is checked.* The release is two flat folders of numbered
images with no per-person folders, so the only thing tying an image to a palm
is `00001.bmp` in one session being the same person as `00001.bmp` in the
other. Every result rests on that, and a wrong assumption would produce
plausible-looking numbers from nonsense. `tongji.pairing_check()` therefore
compares captures of the same number across sessions with captures of
different numbers, using raw orientation-code agreement; if the first is not
higher, the loader says so and nothing should be run.

*What this cannot measure.* With pre-cropped ROIs, our own ROI extraction
(and the ±0.54° result from Phase I) is not exercised. That experiment needs
the `Original Images.rar` 800×600 photographs, run through
`preprocess.extract_roi`, and is a separate comparison.

## What is here

| file | what it is |
|---|---|
| `palmprint/preprocess.py` | hand segmentation, outline, finger valleys, the ROI frame |
| `palmprint/mfrat.py` | the MFRAT operator and the orientation code — Paper 2's feature extractor |
| `palmprint/rloc.py` | enrolment with rotated templates, pixel-to-area matching, the gallery |
| `palmprint/han.py` | the 2003 baseline: Sobel, morphology, grid pooling, Pearson matching |
| `palmprint/metrics.py` | FAR, FRR, EER, ROC, d′, confusion matrix |
| `palmprint/protocol.py` | dataset indexing, the enrol/probe split, experiment runners |
| `palmprint/synthetic.py` | parametric synthetic palms and hands — a test fixture, not a result |
| `palmprint/tongji.py` | reading the Tongji release: numbering into palms, the pairing check, the manifest |
| `palmprint/engine.py` | everything the demo decides: preprocessing, galleries, thresholds, cache |
| `palmprint/render.py` | every panel the demo draws, with Pillow |
| `palmprint/stage.py` | the audience's view, composed as one image |
| `demo.py` | the window around engine and stage |
| `make_demo_data.py` | writes the synthetic demo dataset |
| `run_demo.bat`, `check_demo.bat` | double-click launchers for the demo and its rehearsal |
| `run_demo_tongji.bat` | the demo on the Tongji database, all 600 palms |
| `selftest.py`, `test_preprocess.py`, `test_demo.py`, `test_tongji.py`, `test_experiments.py` | the checks |
| `palmprint/gabor.py` | a Gabor orientation code, for the MFRAT-against-Gabor experiment |
| `palmprint/experiments.py` | the evaluation: protocol, features, scores, metrics, bootstrap |
| `palmprint/plots.py` | the result figures, in the demo's style |
| `run_experiments.py` | every result table and figure, from the images |

## The results

`python run_experiments.py` writes `results/results.md` (every table),
`results/results.json` (every number) and `results/figures/` (DET, FAR/FRR,
score distributions, EERs with intervals). On all 600 Tongji palms it is a
job of tens of minutes on our laptop — most of it matching at the published
±2 shift search, then the Gabor wavelength search. It is meant to be left
running: work is saved as it goes in `results/cache/` (about 150 MB), so an
interrupted run resumes where it stopped rather than starting again.

Headline, cross-session (6,000 genuine, 3,594,000 impostor comparisons):
RLOC **0.11%** EER (95% interval 0.04–0.16%) against Han et al. 2003 at
**7.67%** (6.84–8.66%); at FAR 0.1%, RLOC turns away 0.12% of genuine users
and Han 44%. Seven genuine attempts out of 6,000 are rejected, and they come
from five palms. Closed-set identification is right for 99.95% of probes.

How the numbers are protected from fooling us: every EER carries a 95%
interval from a bootstrap over palms, and every comparison a paired interval
for the difference; the Gabor arm's one free parameter is tuned per
configuration on session-1 data only, never on the session-2 data reported;
and the experiment runner and the demo engine compute RLOC's scores by
separate code paths, which agree bit for bit — including between a Windows
laptop and a Linux machine.

**Why the demo's picture is drawn with Pillow, not built from Tk widgets.**
"Whether your demo works or not" is graded. Composing the whole audience view
as one image means the exact picture the class will see can be rendered and
checked on any machine without opening a window — `test_demo.py` does this at
seven window sizes, and checks nothing overlaps — and the window code that
remains is small enough to read in one sitting.

## Design notes worth knowing

**Matching is batched, and provably the same matcher.** Scoring a probe
against 4,200 templates one at a time takes ~1.5 s at the published
settings, which would make a full Tongji evaluation take several hours.
Because a probe cell holds exactly one direction, equations (7)–(9) can be
rewritten as a dot product between the probe's one-hot code and a 0/1 mask of
the directions present in each template neighbourhood — so every probe
against every template is one matrix product per translation, which numpy
hands to BLAS. Equation (9)'s second direction is the same product with the
roles swapped, so it costs one more product, not a second matcher. The counts are small
integers, exact in float32, and the score is count / area in float64 as
before: `selftest.py` checks the two give **bit-for-bit identical** scores for
the pixel, cross and square rules, with one or seven templates, at the
published settings and at our first version's, and they were also compared on
real Tongji codes against a 600-palm gallery. A full evaluation now takes
about three minutes of matching.

**Scores are similarities everywhere.** Both matchers return "higher is more
alike", and the decision is always `accept if score >= threshold`. Paper 1
states its rule as a distance (1 − R); mixing the two conventions is an easy
way to invert a result without noticing.

**The protocol is declared once**, in `protocol.py`: the first sample of each
identity is enrolled, every other image is a probe, and each probe is matched
against every enrolled identity. With one enrolled sample this is Paper 2's
protocol and also the realistic one — at an enrolment desk you get one scan,
not thirty over three weeks. It is deliberately harsher than Paper 1's, which
enrols ten samples per person and fits a threshold per user.

**Synthetic numbers are not results.** `synthetic.py` exists so that the weeks
spent waiting for database approval are not dead time. Synthetic identities
are far more distinctive than real palms, so any EER measured on them is
meaningless as a claim and none will appear in the presentation.

**One result of our own.** Paper 2 requires every direction of the operator to
be given the same number of pixels, and justifies it on the grounds that the
six sums are then comparable. The sharper statement, which the paper does not
make, is that equal pixel counts are the *exact* condition for the code to be
provably invariant to any affine illumination change: under f → af + b each
directional sum maps to a·r_k + b·|L_k|, so when every |L_k| is equal the same
increasing function is applied to all six sums and the argmin cannot move. The
self-test confirms the code is unchanged bit-for-bit under gain and bias, and
that removing eight pixels from one direction breaks it immediately. So
requirement 2 is not housekeeping — it is what buys the illumination
robustness the method is sold on.

**A limitation to state rather than hide.** MFRAT assigns a direction to every
point, including points with no line through them, where the six sums are
nearly equal and the winner is essentially arbitrary. The directional
contrast is what tells you whether a code value is trustworthy, and Paper 2
discards it. We keep it (`rloc.encode_with_contrast`), which is also the basis
of the fusion experiment we argued for in Phase I.

**We measure our own ROI, because we faulted Paper 1 for not measuring its
own.** Over 80 synthetic hands at ±14° rotation, ±12 px translation and ±10%
scale, extraction succeeds on all of them and recovers the hand's rotation
with a standard deviation of 0.54° and a worst case under 2°. The window side
tracks hand size to 1.7%, so the crop is genuinely scale-normalised. For
comparison, Paper 2 states the ROI it inherits can be out by about ±10°.

**On synthetic hands, one template matched seven — on real palms it did
not.** Paper 2 stores seven rotated templates per enrolment to absorb a
misaligned ROI. With our synthetic ROI accurate to half a degree, one template
scored no worse than seven, by the mechanism that makes the paper's 3×3
square worse than its cross: every extra template is one more chance for an
impostor to match. We called that a hypothesis until real data arrived. On
Tongji it is refuted: seven templates beat one, 0.11% against 0.25% EER, in
99.9% of bootstrap replicates. Tongji's ROIs come from its own capture
software, whose rotation accuracy we cannot measure, so what the result says
is that those ROIs carry rotation the templates absorb — not that our ROI
would need them. The synthetic finding stays a statement about synthetic
data; the experiment is reported in `results/results.md`, section 5.

**The matcher is the paper's, and we measured what it costs to depart
from it.** Paper 2 scores a pair as the larger of s(probe, template) and
s(template, probe) — equation (9) — and searches ±2 code pixels of
translation. Our first version kept the probe's direction only and searched
±1, to keep a full evaluation inside an hour. On Tongji that was expensive:
0.69% EER instead of 0.11%, and the ±1 search was almost all of it. Both
settings stay available as `RlocConfig(symmetric=..., shift_range=...)`, and
`results/results.md` section 8 reports the four variants, because a matcher
that differs from the paper's is a result about our code, not about RLOC.

**Where we deliberately depart from Paper 1.** It thresholds the hand at a
fixed grey level of 70, justified on the grounds that "the environment is
controlled". We use Otsu's method instead. Hard-coding a grey level was one of
our Phase I criticisms, so adopting it here would be indefensible, and Otsu
costs nothing.
