# Biometric: Palmprint Verification

Palmprint verification project for **AI461 Advanced Biometric Systems and Security** at SVNIT Surat. It implements Robust Line Orientation Code (RLOC) and the Han et al. palmprint baseline, with a visual demonstration and an experiment pipeline for comparing both methods under one evaluation protocol.

## Quick start

Install Python with Tkinter support, then run:

```sh
python -m pip install -r requirements.txt
cd phase2
python demo.py
```

On Windows, you can also launch the demo with `phase2/run_demo.bat`. The default demo generates synthetic palm images locally, so it can run without downloading a biometric database.

## Project contents

- `phase2/palmprint/`: ROI extraction, orientation encoding, matching, metrics, dataset loaders, and demo rendering.
- `phase2/demo.py`: interactive verification and identification demo.
- `phase2/run_experiments.py`: evaluation, ablation studies, and result generation.
- `phase2/results/`: saved experiment reports and figures.
- Root `.pptx` files: Phase I literature critique and Phase II presentation.
- Root `.docx` file: demo run sheet.

See [the Phase II README](phase2/README.md) for dataset preparation, demo controls, evaluation details, and implementation notes. Published project measurements are recorded in [the experiment report](phase2/results/results.md).

## Verification

From `phase2/`, run the existing checks:

```sh
python selftest.py
python test_preprocess.py
python test_demo.py
python test_tongji.py
python test_experiments.py
```

## Datasets and generated files

The Tongji biometric dataset, dataset archives, generated feature caches, synthetic demo data, and rehearsal screenshots are excluded from Git. Obtain the dataset separately and follow the preparation instructions in the Phase II README. The synthetic demo data and caches are generated automatically when needed.

## References

- Jia, Huang, and Zhang, *Palmprint verification based on robust line orientation code*, Pattern Recognition 41 (2008), 1504–1513.
- Han et al., *Personal authentication using palm-print features*, Pattern Recognition 36 (2003), 371–381.
