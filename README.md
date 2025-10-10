# Forgotten Forge - MAKING OF VERMICULAR

[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-blue.svg)](https://creativecommons.org/licenses/by-nc/4.0/)
[![License: Elastic License 2.0](https://img.shields.io/badge/Commercial%20License-ELv2-orange)](LICENSE-COMMERCIAL.txt)

---
# VERMICULAR – Reproducible Scripts for the Paper

Empirical optimization of quantum algorithms on NISQ hardware via systematic testing, targeted dynamical decoupling (DD) placement, and automated strategy discovery.

This repository contains all scripts, configs, and example artifacts to reproduce the paper’s core results: 2‑qubit Grover success rates, multi‑stage performance, empirical σ_c measurements, and ablation studies.

## Contents

- VERMICULAR core implementation
- Strategy scanning and auto-optimization tools (Auto-Optimizer v2.0, Quantum Solver Pro, Live Scanner)
- Platform adaptations (IQM, Rigetti via AWS Braket)
- Analysis scripts and example JSON results
- Run manifests, seeds, and recommended transpiler settings

## Repository Structure

- `vermicular.py` — VERMICULAR core (2‑qubit Grover with DD at pre‑oracle and post‑diffusion)
- `live_circuit_o.py` — Systematic strategy scanner (brute force over strategies × depth)
- `auto_opti2.py` — Auto-Optimizer v2.0 (BalancedQuantumOptimizer: multi-objective with functionality validation)
- `qsp.py` — Quantum Solver Pro (early alpha; comprehensive toolbox). Cite with tag + commit.
- `iqm_corrected_execution.py` — IQM specifics (index mapping, adapter)
- `process_iqm_complete.py` — Analysis and simplified tomography pipelines
- `db_vs.py` — VERMICULAR vs. Standard Grover mini-benchmark

## Requirements

- Python 3.10+
- Recommended: `virtualenv` or `conda`
- Core packages:
    - `amazon-braket-sdk`, `boto3`
    - `numpy`, `scipy`, `matplotlib`
    - `pandas`, `tqdm` (optional for analysis)

Install (example):

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -U pip
pip install -r env/requirements.txt
```

Minimal install (if no `requirements.txt`):

```bash
pip install amazon-braket-sdk boto3 numpy scipy matplotlib pandas tqdm
```

Notes:

- Real QPU runs require an AWS account with Braket access and credentials configured.
- Without Braket access, local runs use Braket’s `LocalSimulator` (“braket_dm”) for smoke tests and parameter pre-selection.

## Quick Start

Local simulation (smoke tests):

```bash
python [vermicular.py] --demo
python db_[vs.py](http://vs.py) --sim-only
```

Strategy scanning (simulated):

```bash
python live_circuit_[o.py] \
  --algo grover \
  --depths 1 2 4 \
  --strategies baseline dd_xx dd_xy4 dd_cpmg gate_compression virtual_z echo_spin \
  --shots 1000
```

Auto-Optimizer v2.0 (balanced):

```bash
python auto_[opti2.py] \
  --algo grover \
  --perf-weight 0.5 \
  --res-weight 0.5 \
  --max-iters 5 \
  --target-sigma 0.15 \
  --min-perf 0.6
```

QPU execution (simplified, example):

```bash
python [vermicular.py] --backend iqm --shots 256 --save-manifest manifests/run_iqm_YYYYMMDD.json
python [vermicular.py] --backend rigetti --shots 256 --save-manifest manifests/run_rigetti_YYYYMMDD.json
```

Backends are resolved via AWS Braket device IDs under the hood. Configure your AWS profile and region accordingly.

## Reproducing Core Results

1) 2‑qubit Grover success rates

Standard vs. VERMICULAR vs. random/uniform DD

Scripts: `db_[vs.py]`, [`vermicular.py`]

```bash
python db_[vs.py] --backend iqm --shots 256 --out results/iqm_grover_2q.json
python db_[vs.py] --backend rigetti --shots 256 --out results/rigetti_grover_2q.json
```

Expected: ~5× improvement over baseline (see paper tables for reference values).

2) Multi‑stage search

Script: [`vermicular.py`] (multi‑stage mode) or a dedicated `multi_stage_*.py` script

```bash
python [vermicular.py] --backend iqm --multistage 3 --shots 256 --out results/iqm_multistage.json
```

3) Empirical σ_c measurement

Scripts: `auto_[opti2.py]` (uses `measure_sigma_c_accurate`), `process_iqm_[complete.py]`

Pipeline: discrete noise levels → measurement → information functional I(ε) → gradient → argmax

```bash
python auto_[opti2.py](http://opti2.py) --sigma-only \
  --noise-grid 0,0.05,0.10,0.15,0.20,0.25,0.30 \
  --shots 256 \
  --out results/sigma_product.json
```

4) Ablation studies (pre‑oracle only, post‑diffusion only, both, double)

Scripts: `db_[vs.py]`, [`vermicular.py`]

```bash
python db_[vs.py](http://vs.py) --backend iqm --dd pre   --shots 256
python db_[vs.py](http://vs.py) --backend iqm --dd post  --shots 256
python db_[vs.py](http://vs.py) --backend iqm --dd both  --shots 256
python db_[vs.py](http://vs.py) --backend iqm --dd both --double --shots 256
```

## Seeds, Transpiler Settings, Barriers

- Seeds (optional, recommended for reproducibility):
    - Simulator seed: `12345`
    - Orchestration seed: `4242`

```bash
python db_[vs.py] --backend iqm --shots 256 --seed-sim 12345 --seed-orch 4242
```

- Compiler/transpiler:
    - AWS Braket defaults (`optimization_level=1`)
    - Protect DD sequences from cancellation with barriers:
        
        ```python
        circuit.x(i); circuit.barrier(i); circuit.x(i)
        ```
        
- Manifests/logs:
    - Each core run can write a JSON manifest with seeds, depth, (post‑transpile) gate counts if available, backend IDs, timestamps:
        
        ```bash
        --save-manifest manifests/run_<backend>_<date>.json
        ```
        

## Platform Notes

- IQM
    - 1‑based indexing (adapter in `iqm_corrected_[execution.py]`)
- Rigetti
    - Higher crosstalk in our tests → stronger DD advised
- Local simulator
    - `LocalSimulator("braket_dm")` is used for quick tests and parameter sweeps

## Key Scripts (Short Guide)

- [`vermicular.py`](http://vermicular.py)
    
    Generates VERMICULAR circuits, places DD at pre‑oracle and post‑diffusion, protects DD with barriers.
    
- `live_circuit_[o.py]`
    
    Scans strategies across several depths and collects success rates.
    
- `auto_[opti2.py](http://opti2.py)`
    
    BalancedQuantumOptimizer: preserves functionality (`is_functional` checks), measures σ_c over a noise grid, optimizes a composite score (performance, σ_c, efficiency).
    
- [`qsp.py`] (alpha)
    
    Quantum Solver Pro: extended toolbox (DD variants, echo, HEB, brute force strategy sets). Early alpha!
    
- `process_iqm_[complete.py]`
    
    Analysis utilities and simplified tomography (basis-level extraction).
    
- `db_[vs.py]
    
    Standard Grover vs. VERMICULAR (simulator/QPU), JSON exports for tables/plots.
    

## Data Outputs

- JSON files in `results/` typically include:
    - Config: shots, noise grid, seeds
    - Success rates, timing
    - Optional σ_c estimates
- Example: `results/grover_benchmark_results_YYYYMMDD_HHMMSS.json`

## Known Limitations

- σ_c is empirically defined; robustness can be cross-checked with alternative coherence measures.
- Full transpiler logs and idle window durations may not always be available; include them in manifests when possible.
- QPU runs are calibration-dependent; small deviations from the paper are expected.

## License

Copyright (c) 2025 ForgottenForge.xyz

with structure preservation and functionality validation!

Dual Licensed under (see LICENSE):
- Creative Commons Attribution 4.0 International (CC BY 4.0)
- Elastic License 2.0 (ELv2)

Commercial licensing available. Contact: nfo@forgottenforge.xyz


## Contact

- Email: [theqa@posteo.com](mailto:theqa@posteo.com)
- Web: https://www.theqa.space

## Troubleshooting

- Braket not installed:
    
    `pip install amazon-braket-sdk` and configure AWS credentials.
    
- DD gets “optimized away”:
    
    Ensure barriers are inserted; verify post‑transpile gate counts and depth.
    
- Diverging success rates:
    
    Check calibration window, reduce shots for pilot runs, set simulator seeds, and compare manifests.
    
