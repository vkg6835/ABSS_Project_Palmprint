#!/usr/bin/env python3
"""
make_demo_data.py — write the synthetic demo dataset to disk.

    python make_demo_data.py                  # demo_data/synthetic, 15 people
    python make_demo_data.py --people 20 --samples 6

demo.py runs this automatically the first time if the folder is missing, so
there is normally no need to call it by hand.

The images are whole synthetic hands (palmprint/synthetic.py) stored as
ordinary PNG files, one folder per person, so that the demo reads them
through exactly the same code path it will use for the IITD images. They are
a stand-in until the real database arrives, labelled as synthetic everywhere
they appear, and no performance claim is made on them.

Sample 1 of each person is a clean, centred capture — it is the one that gets
enrolled. Samples 2 onwards carry fresh session variation: rotation up to 12
degrees, a shift of up to 14 pixels, about 10% scale change, and a different
brightness and contrast.

The folder also holds `_extra/no_fingers.png`, a hand with no fingers, which
has no finger valleys and so cannot yield an ROI. It is there so the demo can
show the system declining an input rather than guessing. And `_visitors/`
holds two captures each of two people who are NOT in the database, for the
live-enrolment part of the demo: enrol visitor A from one capture, verify the
other, then show that visitor B is identified as nobody.

Folders starting with "_" are never mistaken for people.
"""
import argparse
import json
import os
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from palmprint import synthetic  # noqa: E402

DEFAULT_OUT = os.path.join(HERE, "demo_data", "synthetic")


def _save(img, path):
    arr = (np.clip(img, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    Image.fromarray(arr).save(path)


def make(out=DEFAULT_OUT, people=15, samples=6, size=400, progress=None):
    """Write the dataset; returns the folder. `progress(done, total)`."""
    os.makedirs(out, exist_ok=True)
    total = people * samples + 5
    done = 0
    for p in range(1, people + 1):
        ident = "S%03d" % p
        folder = os.path.join(out, ident)
        os.makedirs(folder, exist_ok=True)
        for k in range(samples):
            img, _ = synthetic.hand_sample(p, k, size=size)
            _save(img, os.path.join(folder, "%s_%d.png" % (ident, k + 1)))
            done += 1
            if progress:
                progress(done, total)

    extra = os.path.join(out, "_extra")
    os.makedirs(extra, exist_ok=True)
    img, _ = synthetic.render_hand(900, size=size, rot_deg=6.0, seed=900,
                                   fingers=())
    _save(img, os.path.join(extra, "no_fingers.png"))

    # Two people who are NOT in the database, two captures each, for live
    # enrolment: enrol from capture 1, then verify capture 2 — and show that
    # an unenrolled visitor is not identified as anybody.
    visitors = os.path.join(out, "_visitors")
    os.makedirs(visitors, exist_ok=True)
    for tag, seed in (("A", 901), ("B", 902)):
        for k, sample_index in ((1, 0), (2, 3)):
            img, _ = synthetic.hand_sample(seed, sample_index, size=size)
            _save(img, os.path.join(visitors,
                                    "visitor_%s_%d.png" % (tag, k)))
    if progress:
        progress(total, total)

    manifest = {
        "name": "Synthetic hands",
        "kind": "hand",
        "layout": "subdirs",
        "synthetic": True,
        "note": ("Synthetic test fixture standing in for the IITD database. "
                 "Not real palms: no performance claim is made on it."),
        "examples": {"unreadable": "_extra/no_fingers.png"},
        "visitors": "_visitors",
    }
    with open(os.path.join(out, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--people", type=int, default=15)
    ap.add_argument("--samples", type=int, default=6)
    ap.add_argument("--size", type=int, default=400)
    a = ap.parse_args(argv)

    def show(done, total):
        print("\r  writing image %d of %d" % (done, total), end="", flush=True)

    make(a.out, a.people, a.samples, a.size, progress=show)
    print("\n  done: %s" % a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
