[README.md](https://github.com/user-attachments/files/27079312/README.md)

# ⚡ EV Flexibility Project

> **AI-powered smart charging optimization for electric vehicles** — using Machine Learning to maximize grid flexibility and prioritize critical charging sessions under constrained capacity.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [Pipeline Architecture](#pipeline-architecture)
- [Dataset Support](#dataset-support)
- [Key Components](#key-components)
- [Results](#results)
- [Getting Started](#getting-started)
- [Project Structure](#project-structure)
- [Dependencies](#dependencies)

---

## Overview

This project implements a full ML pipeline to optimize EV charging session management on public charging infrastructure. By predicting **stay duration** and **charging urgency**, the system intelligently prioritizes vehicles when grid capacity is limited — demonstrating measurable improvements over naive FIFO (First-In, First-Out) scheduling.

The project was developed and benchmarked across three real-world datasets: **EPFL**, **JPL**, and **Caltech**.

---

## Problem Statement

Public EV charging stations often face **capacity constraints** — not every vehicle can charge at full power simultaneously. A naive FIFO approach serves vehicles in arrival order, which can leave truly urgent users (those who *need* to charge now) unserved while flexible vehicles (those that could wait) occupy limited capacity.

**Goal:** Design an AI system that, under constrained capacity (e.g., only 5% of total theoretical power available), maximizes the satisfaction rate of **genuinely urgent** charging sessions.

---

## Pipeline Architecture

```
Raw CSV Data
     │
     ▼
┌─────────────────────────────┐
│  Universal Dataset Adapter  │  ← Column mapping, unit conversion,
│  (DATASET_CONFIGS)          │    date parsing, per-source config
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│  Physics-Based Labeling     │  ← RF = stay_duration / min_charge_time
│  is_flexible = RF ≥ 1+θ    │    Threshold θ configurable per dataset
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│  Feature Engineering        │  ← arrival_hour, day_of_week,
│                             │    min_charge_time, pred_ratio
└─────────────┬───────────────┘
              │
     ┌────────┴────────┐
     ▼                 ▼
┌──────────┐     ┌──────────────┐
│ Model 1  │     │   Model 2    │
│ Stay     │────▶│   Urgency    │  ← pred_ratio feeds urgency classifier
│ Duration │     │   Classifier │
│ (RF Reg) │     │  (RF Class.) │
└──────────┘     └──────┬───────┘
                        │
                        ▼
          ┌─────────────────────────┐
          │   Charging Simulator    │
          │  FIFO  vs  ML-Enhanced  │  ← capacity_fraction = 5%
          └─────────────┬───────────┘
                        │
                        ▼
              Satisfaction Score (%)
              for truly urgent sessions
```

---

## Dataset Support

The pipeline uses a **universal adapter** (`DATASET_CONFIGS`) that normalizes heterogeneous data sources into a common schema. Adding a new dataset only requires defining its configuration entry.

| Dataset        | Source   | Notes                                      |
|----------------|----------|--------------------------------------------|
| `epfl_original`| EPFL     | Primary benchmark, European charging site  |
| `jpl_data`     | JPL      | US dataset, single-class edge case handled |
| `caltech_data` | Caltech  | US dataset, corrupted dates managed        |

All datasets are mapped to a standardized schema:

| Standard Column  | Description                          |
|------------------|--------------------------------------|
| `power_max`      | Maximum charging power (W)           |
| `energy_asked`   | Energy requested (Wh)                |
| `stay_duration`  | Total parking duration (minutes)     |
| `arrival_time`   | Session start timestamp              |
| `is_flexible`    | Binary label — 1 if flexible, 0 if urgent |

---

## Key Components

### `load_and_standardize(dataset_key)` — Universal Loader
Reads raw CSV, applies column mapping, unit conversions, and computes the physics-based `is_flexible` label using the **Constraint Factor (RF)**:

```
RF = stay_duration / min_charge_time
is_flexible = 1  if  RF ≥ 1 + threshold
```

### `StayDurationPredictor` — Model 1
Random Forest Regressor predicting how long a vehicle will remain plugged in, using:
- `arrival_hour`, `day_of_week`, `energy_asked`, `power_max`

### `UrgencyPredictor` — Model 2
Random Forest Classifier predicting charging urgency (probability that a session is non-flexible), using:
- `stay_duration`, `energy_asked`, `pred_ratio` (ratio of min charge time to predicted stay)

Includes **single-class safety mode** for datasets where all sessions share the same label.

### `ChargingSimulator` — Benchmarking Engine
Simulates power distribution under capacity constraints.

| Mode           | Strategy                                         |
|----------------|--------------------------------------------------|
| `fifo`         | Serve vehicles in random arrival order (baseline)|
| `ml_enhanced`  | Sort by urgency score before serving             |

Default `capacity_fraction = 0.05` simulates a crisis scenario (5% of total grid capacity).

### Robustness Test — Noise Injection
Tests system resilience by injecting Gaussian noise into `stay_duration` at levels from 0% to 50% of standard deviation, evaluating performance degradation:

| Status    | Condition              |
|-----------|------------------------|
| ✅ STABLE  | Loss < 5%              |
| ⚠️ DÉGRADÉ | Loss between 5–20%     |
| ❌ CRITIQUE | Loss > 20%            |

---

## Results

The benchmark compares **FIFO vs ML-Enhanced** satisfaction of urgent sessions at 5% grid capacity:

| Dataset        | Sessions  | FIFO Score | ML Score | Improvement |
|----------------|-----------|------------|----------|-------------|
| `epfl_original`| 1,878     | 11.1%      | **25.0%**| **+13.9 pts** ✅ |
| `jpl_data`     | 10,999    | 5.0%       | 5.0%     | 0.0 pts ➖  |
| `caltech_data` | 259,375   | 5.0%       | **10.5%**| **+5.5 pts** ✅ |

> Metric: % of **truly urgent** sessions (is_flexible = 0) successfully served under 5% grid capacity constraint.

### Interpretation

- **EPFL** — strongest result: the ML model more than **doubles** the satisfaction of urgent users (11.1% → 25.0%). The dataset has a well-balanced flexibility distribution, ideal for learning urgency patterns.
- **JPL** — no improvement: the dataset is likely near-entirely composed of flexible sessions (single-class edge case). The urgency classifier activates its **single-class safety mode** and cannot differentiate between users, falling back to FIFO-equivalent behavior.
- **Caltech** — solid improvement of **+5.5 pts** on 259,375 sessions. The ML gain is more modest than EPFL due to the larger scale and more uniform distribution of sessions, but the result is statistically meaningful.

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/your-username/EV_Flexibility_Project.git
```

### 2. Set up Google Drive

Create the following folder structure in your Google Drive:

```
MyDrive/
└── EV_Flexibility_Project/
    └── data/
        ├── EPFL_dataset.csv
        ├── JPL_dataset.csv
        └── Caltech_dataset.csv
```

### 3. Open in Google Colab

Upload and open the notebook in [Google Colab](https://colab.research.google.com/), then run the cells in order:

| Cell | Purpose                        |
|------|--------------------------------|
| 0    | Mount Google Drive             |
| 1    | Import libraries               |
| 2    | Load dataset configurations    |
| 3    | Universal data loader          |
| 3.5  | Viability audit per dataset    |
| 4    | Feature engineering            |
| 5    | Train Stay Duration model      |
| 6    | Train Urgency Classifier       |
| 7    | Charging Simulator             |
| 8    | Full benchmark (FIFO vs ML)    |
| 9    | Robustness test (noise)        |
| 10   | Final visualization            |

---

## Project Structure

```
EV_Flexibility_Project/
├── notebook.ipynb          # Main Colab notebook (all 10 cells)
├── README.md               # This file
└── data/                   # (on Google Drive, not committed)
    ├── EPFL_dataset.csv
    ├── JPL_dataset.csv
    └── Caltech_dataset.csv
```

---

## Dependencies

All standard libraries available in Google Colab — no additional installation required.

| Library        | Usage                                      |
|----------------|--------------------------------------------|
| `pandas`       | DataFrame manipulation                     |
| `numpy`        | Numerical operations                       |
| `scikit-learn` | Random Forest models, metrics, data split  |
| `matplotlib`   | Base plotting                              |
| `seaborn`      | Statistical visualization                  |

---

## Authors

Developed as part of the **Energy & Sustainable Cities** engineering curriculum at ESILV Paris (École Supérieure d'Ingénieurs Léonard de Vinci).

---

## License

This project is for academic and research purposes.
