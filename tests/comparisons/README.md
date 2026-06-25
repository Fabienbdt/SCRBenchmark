# Algorithm Comparison Tests

This folder contains comparison tests between SCRBenchmark implementations and the original author implementations.

## Overview

| Algorithm | Python | Original Framework | SCRBenchmark Framework | Status |
|-----------|--------|--------------------|------------------------|--------|
| scDeepCluster | 3.9* | PyTorch 1.8 | PyTorch | IDENTICAL |
| scCDCG | **3.7** | PyTorch 1.12 | PyTorch | IDENTICAL |
| scMAE | **3.10** | PyTorch Lightning | PyTorch | IDENTICAL (methodology) |
| scNAME | >=3.8 | TensorFlow >=2.2 | PyTorch | IDENTICAL (methodology) |

IMPORTANT: Python versions in bold are explicitly required by the original authors.

## Folder Structure

```
tests/comparisons/
|-- README.md                    # This file
|-- __init__.py                  # Python module
|-- conftest.py                  # Shared pytest fixtures
|-- run_all_comparisons.py       # Main script
|-- compare_scdeepcluster.py     # scDeepCluster tests
|-- compare_sccdcg.py            # scCDCG tests
|-- compare_scmae.py             # scMAE tests
|-- compare_scname.py            # scNAME tests
|-- envs/                        # Virtual environments
|   |-- setup_environments.sh    # Setup script
|   |-- requirements_*.txt       # Per-algorithm dependencies
|-- data/                        # Test data
`-- results/                     # Test outputs
```

## Usage

### Run all tests

```bash
cd tests/comparisons
python run_all_comparisons.py
```

### Options

```bash
# Reports only (no pytest)
python run_all_comparisons.py --reports-only

# Pytest only (no reports)
python run_all_comparisons.py --tests-only
```

### Test a specific algorithm

```bash
# Generate report and run tests
python compare_scdeepcluster.py

# With pytest
pytest compare_scdeepcluster.py -v
```

## Conda Environment Setup

Original algorithms require specific Python versions (see table above).
The script uses conda to create environments with the correct versions.

### Prerequisites

- Anaconda or Miniconda installed

### Show required versions

```bash
cd tests/comparisons/envs
./setup_environments.sh summary
```

### Create environments

```bash
cd tests/comparisons/envs

# Create all environments
./setup_environments.sh all

# Or a specific environment
./setup_environments.sh scname
```

### Activate an environment

```bash
conda activate env_scname
```

## Comparison Details

### scDeepCluster
- Framework: PyTorch -> PyTorch
- Matching components:
  - ZINB loss
  - Encoder/decoder architecture
  - Soft clustering (Student's t-distribution)
  - Target distribution
  - Training loop (pretrain + clustering)

### scCDCG
- Framework: PyTorch -> PyTorch
- Matching components:
  - Autoencoder (AE_NN)
  - Full network with GCN
  - Sinkhorn normalization
  - Cluster assignment
  - Losses: MSE, orthogonality, covariance, KL
- Fix: Modulo bug fixed (epoch // 10 -> epoch % 10)

### scMAE
- Framework: PyTorch Lightning -> PyTorch standalone
- Matching components:
  - MLP architecture
  - Bernoulli masking
  - Scaling factor 1/(1-rate)
  - Reconstruction loss

### scNAME
- Framework: TensorFlow 1.x -> PyTorch
- Matching components:
  - mask_generator (Bernoulli)
  - pretext_generator (column shuffle)
  - Encoder/decoder architecture with 4 heads
  - ZINB loss, mask BCE, neighbor loss, kmeans loss
  - Phases: pretrain + finetune

## Validation Metrics

The tests use multiple metrics to validate equivalence:

1. Direct output comparison: For PyTorch-to-PyTorch implementations
2. ARI/NMI on clustering: Validate final clustering results
3. Embedding correlation: Compare latent representations
4. Component unit tests: Validate each algorithm block

## Important Notes

1. Reproducibility: Use `random_state=42` for reproducible results
2. Synthetic data: Tests use synthetic data with clear clusters
3. Thresholds: ARI > 0.2 is the minimum threshold for clustering tests
4. Cross-framework: TF->PyTorch ports are validated methodologically, not numerically
