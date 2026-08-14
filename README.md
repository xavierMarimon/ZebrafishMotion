# ZebrafishMotion

Code, data, trained networks and hardware designs for

> **Characterizing latent motion features in scoliotic zebrafish: from nonlinear dynamics to machine-learning classification**
> X. Marimon\*, E. Saman-Sakkal\*, M. Cerrolaza, J. Bosch-Bayard, A. Portela
> *Nonlinear Dynamics*, Topical Collection "Biosystem Dynamics"
> \* joint first authors and corresponding authors

Contact: xavier.marimon@upc.edu, eduardo.saman@ucdconnect.ie

---

## What is here

| Folder | Contents |
| --- | --- |
| `Analysis-code/` | Python used for every number and figure in the paper: nonlinear time-series library, feature pipeline, surrogate testing, classification and figure scripts |
| `DLC-trained-networks/` | `ZebrafishNet` (single animal) and `MultiZebrafishNet` (several animals), DeepLabCut projects with configuration files and trained weights |
| `Inferred-trajectories/` | Pose estimates for all recordings, as DeepLabCut `.h5` files |
| `CAD-files/`, `CAD-images-and-video/`, `Laser-cut-files/` | 3D models and cutting files for the optomechanical platform |
| `Camera-setup-scripts/` | Camera configuration and synchronised acquisition |

## The two networks

Only the snapshot behind the published results is tracked for each network;
the intermediate training checkpoints are not.

`ZebrafishNet` is the single-animal network used for the results in the paper.
It tracks three landmarks — the head, on the dorsal midline of the cranium;
`Tail_B`, at the caudal peduncle; and `Tail_E`, at the distal margin of the
caudal fin — with a ResNet-50 backbone trained for 25 000 iterations on 200
manually labelled frames (training error 3.19 px, test error 3.50 px). The
snapshot used throughout is `snapshot-25000`.

`MultiZebrafishNet` is the multi-animal version, trained on the same landmark
set for simultaneous tracking of several fish in one arena. It is **not** used
for the results reported in the paper, which concern individually recorded
animals, and is released so that the platform can be reused for social assays.
Its snapshot is `snapshot-10000`.

## Reproducing the analysis

```bash
python -m pip install numpy scipy pandas matplotlib h5py scikit-learn
cd Analysis-code
python runner.py          # feature extraction with IAAFT surrogates (slow)
python classify.py        # leave-one-animal-out cross-validation, exact permutation tests
python figures.py         # all figures of the paper
```

`Analysis-code/features_final.csv` is the extracted feature matrix
(110 trials × 33 features, plus tracking-quality columns), so the statistics
and figures can be reproduced without re-running the surrogate pipeline.

Key entry points:

- `nld.py`, `nld2.py` — embedding, correlation dimension, Lyapunov exponent, recurrence quantification, ordinal and entropy measures, IAAFT surrogates
- `pipeline.py` — per-trial feature extraction
- `validate_nld.py` — validation of every estimator against systems with known values (Lorenz, logistic map, coloured noise)
- `classify.py` — LDA, leave-one-animal-out cross-validation, exact permutation tests

## How to cite

If you use this code, the trained networks, the pose estimates or the
hardware designs, please cite the article:

> Marimon, X.\*, Saman-Sakkal, E.\*, Cerrolaza, M., Bosch-Bayard, J. &
> Portela, A. Characterizing latent motion features in scoliotic zebrafish:
> from nonlinear dynamics to machine-learning classification.
> *Nonlinear Dynamics* (2026). Topical Collection "Biosystem Dynamics".
> \* joint first authors

```bibtex
@article{Marimon2026ZebrafishMotion,
  author  = {Marimon, Xavier and Saman-Sakkal, Eduardo and
             Cerrolaza, Miguel and Bosch-Bayard, Jorge and Portela, Alejandro},
  title   = {Characterizing latent motion features in scoliotic zebrafish:
             from nonlinear dynamics to machine-learning classification},
  journal = {Nonlinear Dynamics},
  year    = {2026},
  note    = {Topical Collection ``Biosystem Dynamics''}
}
```

`CITATION.cff` carries the same metadata in machine-readable form, so GitHub
shows a **Cite this repository** button in the sidebar and reference managers
can import it directly. Volume, pages and DOI will be added there once the
article is published.

## Licences

- Hardware designs (`CAD-files/`, `Laser-cut-files/`) — **CERN-OHL-S v2**
- Software (`Analysis-code/`) — **MIT**
- Data and trained weights — **CC-BY-4.0**

## Note on large files

**Recordings.** The video recordings are not in this repository. The
full-resolution set is about 21 GB and 134 of its files exceed GitHub's
100 MB hard limit. Both the full-resolution and the downsampled recordings
are available from the corresponding authors on reasonable request.

**Intermediate training checkpoints.** Only `snapshot-25000`
(`ZebrafishNet`) and `snapshot-10000` (`MultiZebrafishNet`) are tracked,
since those are the weights that produced the published results; the earlier
checkpoints are excluded in `.gitignore` but remain in the working copy.

**Pose estimates.** `Inferred-trajectories/` carries the DeepLabCut `.h5`
files and the assembly pickles. The rendered label overlay videos and the
`.csv` export of the same coordinates are omitted, since both are derived
from the `.h5` files.
