# nbody-net

Neural network-based prediction of gravitational N-body system evolution.
This repository contains the full pipeline for dataset generation, model training,
and evaluation of five deep learning architectures applied to the gravitational
three-body problem.

> **Associated thesis:** *Neural Network-Based Prediction of N-Body System Evolution*
> Daniela Cojocaru — West University of Timișoara, Faculty of Computer Science, 2026
> Supervisor: Prof. Dr. Daniela Zaharie

---

## Overview

| Architecture | Paradigm | Key idea |
|---|---|---|
| **MLP** | Baseline | Flat state vector → next state |
| **LSTM** | Temporal | Sequence of past states → next state |
| **GNN** | Graph-based | Message-passing over body interaction graph |
| **GNODE** | Physics-informed | GNN predicts accelerations → RK2 integrator propagates state |
| **PI-GNN** | Physics-informed | GNN + physics loss penalizing Newton's law deviations |

All models are trained on a custom RK4-generated dataset of 500 three-body simulations
(1,000,000 time steps) and evaluated on trajectory accuracy, conservation of physical
invariants (energy, linear momentum, angular momentum), zero-shot generalization to
larger systems, and inference runtime.

---

## Project Structure

```
nbody-net/
│
├── notebooks/
│   ├── 1_dataset_generation.ipynb   # Generate RK4 simulation data
│   ├── 2_training.ipynb             # Train one model at a time
│   └── 3_testing.ipynb              # Quantitative + qualitative evaluation
│
├── .gitignore
├── requirements.txt
└── README.md
```

---

## External Resources

The dataset and trained model weights are too large for GitHub and are hosted on
Google Drive. Download them before running the training or testing notebooks.

| Resource | Link |
|---|---|
| Dataset (CSV) | [Download from Google Drive](https://drive.google.com/drive/folders/1i_ZpxB4KGCVGEUoNZtXybUf2Xs8AQxWw?usp=sharing) |
| Trained Models (.keras) | [Download from Google Drive](https://drive.google.com/drive/folders/1WVOT6nRl6XlD28Q0u-kjLCjCNraf0yBb?usp=sharing) |

After downloading, place the files in your Google Drive under:

```
MyDrive/Colab Notebooks/
├── rk4_3body_20260604_1522.csv
├── model_MLP_nbody.keras
├── model_LSTM_nbody.keras
├── model_GNN_nbody.keras
├── model_GNODE_nbody.keras
└── model_PIGNN_nbody.keras
```

---

## Running the Notebooks

All notebooks are designed for **Google Colaboratory** with GPU acceleration.
No local installation is required. Each notebook mounts your Google Drive
automatically on the first run.

> **Recommended runtime:** GPU (T4 or higher)
> Go to: *Runtime → Change runtime type → GPU*

---

### Notebook 1 — Dataset Generation

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1BSJbUbGRPK6sx9UyO5f4w2lSt4XODc32?usp=sharing)

Generates a custom N-body simulation dataset using one of four numerical integrators
(Kepler, Symplectic Euler, Velocity Verlet, RK4).

**To reproduce the thesis dataset exactly:**

| Parameter | Value |
|---|---|
| Number of bodies (`N`) | `3` |
| Steps per simulation | `2000` |
| Number of simulations | `500` |
| Integration method | `3` (RK4) |

The notebook prompts you interactively for these values. Press Enter to accept
the defaults shown above.

At the end of generation, you will be prompted to save the dataset:
- **Google Drive** — saved automatically to `MyDrive/Colab Notebooks/`
- **Local download** — triggers a browser download of the CSV file

**Output file name format:** `rk4_3body_YYYYMMDD_HHMM.csv`

---

### Notebook 2 — Training

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1j9c7fnW9P500_3S-nGjCeI_G6Z07qKDH?usp=sharing)

Trains one model at a time on the generated dataset.

**Before running:**
1. Upload your dataset CSV to Google Drive
2. Update `CSV_PATH` in the configuration cell to match your file path:
```python
CSV_PATH = '/content/drive/MyDrive/Colab Notebooks/rk4_3body_YYYYMMDD_HHMM.csv'
```

**To train a model:**

Select the model when prompted:
```
1 = MLP
2 = LSTM
3 = GNN
4 = GNODE
5 = PI-GNN
```

**Training configuration used in the thesis:**

| Parameter | Value |
|---|---|
| Epochs | `100` |
| Batch size | `512` |
| Learning rate | `5 × 10⁻⁴` |
| Optimizer | Adam |
| Train/Test split | 80% / 20% (by simulation ID) |
| Random seed | `42` |

Each model is saved automatically to Google Drive and a local download is triggered
at the end of training.

> **Note:** Train models in order (MLP → LSTM → GNN → GNODE → PI-GNN) or
> select only the ones you need. Each model run is independent.

---

### Notebook 3 — Testing & Evaluation

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/17J1aqiZlOprIZGBMHgFigYxehY8CpJqT?usp=sharing)

Evaluates trained models across two workflows.

**Before running:**
1. Upload trained `.keras` model files to Google Drive
2. Verify the paths in `MODEL_PATHS` match your Drive structure

#### Workflow 1 — Quantitative Evaluation

Reproduces the numerical results from Tables 4.1-4.6 of the thesis.
Runs one model across `N_RUNS = 10` independent simulations and reports:

- Trajectory MAE and MSE across short / medium / long-term horizons
- Energy, linear momentum, and angular momentum conservation errors (MAE ± std, Max Error)
- Mean inference runtime ± std

**Configuration cell:**
```python
OPTION_MODEL     = 4    # 1=MLP, 2=LSTM, 3=GNN, 4=GNODE, 5=PI-GNN
OPTION_NUMERICAL = 3    # ground truth: 3=RK4
N_RUNS           = 10
steps            = 2000
```

> To reproduce the same results, run Workflow 1 for each model separately
> with the configuration above.

#### Workflow 2 — Qualitative Visualization

Produces trajectory plots for one model.
Reproduces Figures 4.1–4.2 of the thesis.

**Configuration cell:**
```python
OPTION_MODEL     = 4    # model to visualize
OPTION_NUMERICAL = 3    # ground truth integrator
steps            = 2000
```

For **zero-shot generalization plots** (Figure 4.2), change the number of bodies:
```python
N = 4    # or 5, or 10
```

> Note: zero-shot evaluation is only meaningful for GNN-based models
> (GNN, GNODE, PI-GNN). MLP and LSTM have fixed input size and cannot
> generalize to N ≠ 3.

---

## Physical Units

| Quantity | Unit | Value |
|----------|------|-------|
| Mass | M☉ (solar mass) | 10³⁰ kg |
| Length | AU (astronomical unit) | ≈ 1.5 × 10¹¹ m |
| Time | TU | ≈ 7,082,478 s ≈ 82 days |
| G | — | 1.0 (normalized) |

Softening parameter: `ε = 0.05 AU`
Time step: `dt = 0.001 TU` ≈ 2 hours

---

## Requirements

All dependencies are pre-installed in Google Colab. For reference:

```
tensorflow >= 2.x
numpy
pandas
matplotlib
scikit-learn
```

---

## Known Issues

- **Model path errors:** if the `.keras` files are not found, verify that the
  filenames in `MODEL_PATHS` exactly match the files saved in your Drive,
  including capitalization (e.g. `model_PIGNN_nbody.keras` not
  `model_PI-GNN_nbody.keras`).

- **GNN instability on N > 3:** the standard GNN diverges on systems larger
  than three bodies. This is an expected result documented in the thesis
  (Section 4.2) and not a bug.

---

## License

This project was developed as part of a Bachelor's thesis at the West University
of Timișoara, Faculty of Computer Science, 2026.

© 2026 Daniela Cojocaru. All rights reserved.