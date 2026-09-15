# Third-party material

The code in this repository is original work, licensed under the
[Apache License 2.0](LICENSE). This file records the third-party material that
the repository does **not** own, so that its provenance and ownership are
unambiguous.

Nothing here is redistributed competition data. See
[Competition data](#competition-data) below.

## Photographic figures

The documentation portal and README display five photographic images of
Herculaneum papyrus scrolls. **Copyright in these images remains with their
original publishers.** They are re-hosted here as optimized WebP derivatives
purely to illustrate the scrolls this project works on.

They are not covered by this repository's Apache-2.0 licence.

| File in this repo | Original | Source | Used on |
|---|---|---|---|
| `docs/repo-overview/assets/img/figures/ct-scroll.webp` | `ct-scroll.jpg` | [Vesuvius Challenge — Surface Detection, Kaggle](https://www.kaggle.com/competitions/vesuvius-challenge-surface-detection) | Portal home (hero) |
| `docs/repo-overview/assets/img/figures/scroll-surface-trace.webp` | `scroll-surface-trace.png` | [Vesuvius Challenge — Surface Detection, Kaggle](https://www.kaggle.com/competitions/vesuvius-challenge-surface-detection) | README, portal — evaluation |
| `docs/repo-overview/assets/img/figures/scroll-1536x1017.webp` | `scroll-1536x1017.png` | [AIhub — the Vesuvius Challenge is using AI to virtually unroll Pompeiis ancient scrolls](https://aihub.org/2024/09/12/the-vesuvius-challenge-is-using-ai-to-virtually-unroll-pompeiis-ancient-scrolls/) | README, portal — data flow |
| `docs/repo-overview/assets/img/figures/scroll.webp` | `scroll.webp` | [scrollprize.org — first scroll](https://scrollprize.org/firstscroll) | Portal — pipeline |
| `docs/repo-overview/assets/img/figures/scroll-cross-sections.webp` | `scroll-cross-sections.webp` | [scrollprize.org — first scroll](https://scrollprize.org/firstscroll) | Portal home |

What is shown:

- **ct-scroll** — a Herculaneum scroll imaged by X-ray CT; the raw data behind the task.
- **scroll-surface-trace** — the recto surface traced through a slice of the CT volume, the object
  the composite score measures.
- **scroll-1536x1017** — a carbonised scroll virtually unrolled from CT scans, revealing the hidden
  writing surface.
- **scroll** — the sealed, carbonised scroll as scanned.
- **scroll-cross-sections** — from object to text: the sealed roll, cross-sections through the scan
  revealing the spiralled sheet, and the unwrapped surface where the Greek writing emerges.

### Changes made

Each image was resized (never upscaled) and re-encoded to WebP at quality 82, and some were
converted from RGBA to RGB. Pixel content is otherwise unaltered. The full-resolution originals are
**not** distributed with this repository.

### Attribution and removal

Every use is credited in an adjacent `figcaption` or `<sub>` line, linking back to the source. The
images are used for identification and illustration of the subject matter; the terms of the
originating site govern their reuse.

If you own the rights to one of these images and would like it removed or credited differently,
please [open an issue](https://github.com/mddunlap924/vesuvius/issues) and it will be actioned
promptly.

## External software

This project depends on third-party packages under their own licences. Notable ones:

| Package | Role | Licence |
|---|---|---|
| [PyTorch](https://pytorch.org/) | Core training and inference | BSD-3-Clause |
| [MONAI](https://monai.io/) | Medical-imaging networks, losses, transforms | Apache-2.0 |
| [nnU-Net v2](https://github.com/MIC-DKFZ/nnUNet) | Self-configuring baseline pipeline | Apache-2.0 |
| [surface-distance](https://github.com/google-deepmind/surface-distance) | Surface Dice metric | Apache-2.0 |
| [Hydra](https://hydra.cc/) | Experiment configuration | MIT |
| [Weights & Biases](https://wandb.ai/) | Experiment tracking | MIT |

`uv.lock` pins the full transitive set, each under its own licence.

### `topometrics` — an undeclared dependency

`src/approach/unetbasic/metrics/kaggle_metrics.py` imports
`topometrics.leaderboard.compute_leaderboard_score`. This package supplies the official competition
metric and is **supplied separately by the competition organisers**; it is not published on PyPI and
is therefore deliberately absent from `pyproject.toml` and `uv.lock`.

It is used as an imported dependency only — no `topometrics` source is vendored into this
repository. Obtain it from the competition and install it into your environment before running the
evaluation path.

## Competition data

No competition data is redistributed here. The repository contains only source code,
configuration, tests and documentation.

The CT volumes, ground-truth labels and derivatives produced by this project are excluded via
`.gitignore` (`*.tif`, `*.tiff`, `*.npy`, `*.csv`, `*.parquet`, `/data/`) and are governed by the
[Vesuvius Challenge competition rules](https://www.kaggle.com/competitions/vesuvius-challenge-surface-detection/rules).
Model checkpoints are likewise excluded and are not distributed.

## Trademarks

"Vesuvius Challenge", "Kaggle", "MONAI", "PyTorch", "nnU-Net" and other names are trademarks of
their respective owners. Use of these names here is descriptive only and does not imply
endorsement or affiliation.
