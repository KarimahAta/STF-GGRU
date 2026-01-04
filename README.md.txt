# STF-GGRU: Integrated Spatial-Temporal Feature Alignment with Graph Convolutional and Gated Recurrent Network for Traffic Flow Prediction

This repository provides the official implementation of the STF-GGRU model proposed in the paper:

“Integrated Spatial-Temporal Feature Alignment with Graph Convolutional and Gated Recurrent Network for Traffic Flow Prediction”

The proposed framework is designed to accurately predict traffic flow by jointly modeling temporal dynamics, spatial dependencies, and adaptive feature importance.

STF-GGRU integrates Graph Convolutional Networks (GCN) and Gated Recurrent Units (GRU) with a novel Integrated Spatial-Temporal Feature Alignment (ISTFA) module, which combines Dynamic K-Nearest Neighbor (D-KNN) and Centered Kernel Alignment (CKA) to dynamically capture feature-based and spatial correlations.

---

## Repository Structure
STF-GGRU/
├── src/
├── data/
├── configs/
├── experiments/
├── requirements.txt
├── LICENSE
└── README.md


---

## Datasets

This work uses the PeMSD4 and PeMSD8 datasets provided by the California Department of Transportation Performance Measurement System (PeMS).

Data source:
http://pems.dot.ca.gov/

The datasets contain real-world traffic measurements collected at 5-minute intervals, including traffic flow, traffic occupancy, and traffic speed.

Due to data size and licensing constraints, the datasets are not included in this repository. Detailed information about data access and preprocessing is provided in `data/README.md`.

---

## Model Overview

The STF-GGRU architecture consists of three main components:

- A temporal module based on Gated Recurrent Units (GRU) for modeling sequential traffic evolution.
- A spatial module based on Graph Convolutional Networks (GCN) for capturing inter-sensor dependencies.
- An Integrated Spatial-Temporal Feature Alignment (ISTFA) module that adaptively aligns spatial and feature-based relationships using D-KNN and CKA.

The framework supports ablation studies, allowing selective removal of the temporal module, spatial module, ISTFA, D-KNN, or CKA.

---

## Configuration

All experimental settings and hyperparameters are defined using YAML configuration files located in the `configs/` directory.

Example configuration files include:
- `configs/pemsd4.yaml`
- `configs/pemsd8.yaml`
- `configs/sensitivity.yaml`

Ablation experiment settings are provided in the `experiments/ablation/` directory.

---

## Running the Code

### Requirements
- Python 3.8 or later
- PyTorch
- NumPy
- PyYAML

Install dependencies using:
```bash
pip install -r requirements.txt
All hyperparameters are explicitly defined in configuration files.
Preprocessing, training, and evaluation scripts are fully provided.
Random seeds are fixed to ensure consistent and reproducible results.

License

This project is licensed under the MIT License.
See the LICENSE file for details.

Code Availability Statement

The code supporting the findings of this study is publicly available in this GitHub repository.
Due to data size and licensing restrictions, the PeMS datasets are not redistributed.
All scripts required for preprocessing, training, and evaluation are provided to enable full reproducibility.

Contact

For questions related to this implementation, please contact the corresponding author.

