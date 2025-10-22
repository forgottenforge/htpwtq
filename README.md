# Forgotten Forge - Parameter Optimization and Noise Resilience for QAOA on NISQ Devices

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![AWS](https://img.shields.io/badge/AWS-Cloud-orange?logo=amazonaws)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0.en.html)
[![License: Commercial](https://img.shields.io/badge/License-Commercial-orange)](LICENSE-COMMERCIAL.txt)
![Status: Early Alpha](https://img.shields.io/badge/Status-Early%20Alpha-red?style=for-the-badge)
![Status: Early Alpha](https://img.shields.io/badge/status-early--alpha-red?style=flat-square)
![Status: Early Alpha](https://img.shields.io/badge/status-early--alpha-red)



---
Welcome to the open-source companion repository for the paper  
**"Parameter Optimization and Noise Resilience Analysis for QAOA on NISQ Devices"**  
by *Forgotten Forge*.


---

## 🌐 Overview

This repository accompanies the research on **quantum algorithm optimization** and **noise resilience analysis** across real NISQ hardware platforms (IQM Garnet, Rigetti Ankaa-3, IonQ Forte-1).  
It provides the complete framework, experimental protocols, and analysis tools used in the study.

The work explores:
- Empirical **parameter optimization** for the Quantum Approximate Optimization Algorithm (QAOA)
- Strategic **dynamical decoupling (DD)** placement for Grover’s algorithm
- Introduction of the **critical noise threshold (σc)** as a quantitative resilience metric
- **Cross-platform validation** and reproducibility across superconducting and ion-trap hardware

---

## 🧠 What You'll Find Here

- Full experimental framework, data workflows, and analysis scripts  
- Reproducible setups for both **simulator** and **hardware** execution  
- The **SigmaCSuite** open-source toolkit for measuring noise resilience  
- Reproducibility details including statistical methods and parameter tuning protocols  
- [Sigma C Rust Live Demo](https://play.rust-lang.org/?version=stable&mode=debug&edition=2024&gist=d8c088f6eaeb5c421674154e3ea40653)

## License
🤝 Collaboration and Industry Partnerships

We warmly welcome collaboration with quantum technology companies and research institutions interested in extending or integrating the framework into their workflows.
Our goal is to bridge academic research and practical quantum development by jointly improving algorithmic optimization and validation tools for NISQ-era hardware.

If your team is exploring hardware-specific algorithm tuning, noise characterization, or hybrid quantum-classical optimization, we’d be glad to discuss joint experiments or integration projects.

Copyright (c) 2025 ForgottenForge.xyz

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published
by the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

For commercial licensing without AGPL-3.0 obligations, contact:
[nfo@forgottenforge.xyz]

SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial


## Contact

- Email: nfo@forgottenforge.xyz
- Web: https://www.forgottenforge.xyz

## Troubleshooting

- Braket not installed:
    
    `pip install amazon-braket-sdk` and configure AWS credentials.
    
- DD gets “optimized away”:
    
    Ensure barriers are inserted; verify post‑transpile gate counts and depth.
    
- Diverging success rates:
    
    Check calibration window, reduce shots for pilot runs, set simulator seeds, and compare manifests.
    
