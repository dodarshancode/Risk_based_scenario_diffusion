# Risk-Guided Diffusion for Critical Traffic Scenario Generation

This repository implements a compact, reproducible pipeline for **traffic scenario generation with diffusion models**, **criticality metrics**, and **closed-loop evaluation**. It is designed as a portfolio project to demonstrate modern scenario generation methods for autonomous driving testing.

The project trains a diffusion model to generate traffic participant trajectories and uses **risk-guided sampling** (classifier-free guidance) to bias generation toward safety-critical cases, then runs a closed-loop ego controller to automatically surface critical test cases.

## Key Features

- **Diffusion-based scenario generation** over multi-step trajectories (DDPM-style sampling).
- **Risk conditioning + classifier-free guidance** to steer scenario generation toward critical behavior.
- **Criticality metrics**: minimum gap and TTC-derived risk scoring to label and rank scenarios.
- **Closed-loop evaluation**: ego controller reacts to generated lead behavior and logs collisions/near-misses.
- **Testing campaign outputs**: JSON stats, NPZ scenario files, and visualizations for reporting.

## Quickstart

### Install
```bash
pip install torch numpy matplotlib tqdm
```

### Run training + evaluation
```bash
python risk_diffusion_scenario_generation.py
```

The script will:

Generate a synthetic car-following dataset (normal + critical scenarios).

Train a risk-conditioned diffusion model.

Run an evaluation campaign across multiple guidance scales.

Save metrics, critical scenario artifacts, and plots.

### Outputs
After a successful run, artifacts are written under:

text
experiments/
├── best_model.pt
├── checkpoint_epoch_*.pt
└── evaluation/
    ├── evaluation_stats.json
    ├── critical_scenarios/
    │   ├── scenario_000.npz
    │   └── ...
    └── visualizations/
        ├── guidance_analysis.png
        └── top_critical_scenarios.png

## Method Overview
1) Scenario representation
A scenario is a pair of time-indexed trajectories (ego + lead). The diffusion model generates the lead trajectory over a fixed horizon.

2) Risk conditioning + guidance
Each training sample is labeled with a binary risk_label (normal vs critical). At inference, classifier-free guidance increases the probability of generating critical outcomes by scaling the conditional score.

3) Criticality metrics
Two primary metrics are computed during evaluation:

Minimum inter-vehicle distance across the horizon.

Time-to-Collision (TTC) under a standard closing-speed approximation.

A combined risk score is used to rank scenarios, enabling automatic extraction of the most critical test cases from a large campaign.

4) Closed-loop evaluation
A lightweight ego controller (IDM-like car-following) runs in closed loop against the generated lead trajectory. The system logs:

collisions,

harsh braking events,

minimum gap and minimum TTC.
