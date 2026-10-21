# SEPAR

SEPAR (Spatial gene Expression PAttern Recognition) is a computational method designed for analyzing spatial transcriptomics data. It provides an unsupervised framework for interpretable spatial pattern recognition and pattern-specific gene identification.

## Features

* Unsupervised spatial pattern recognition
* Pattern-specific gene identification
* Gene expression pattern refinement
* Improved spatially variable gene detection
* Spatial domain clustering
* Multi-omics Integration

## Installation

### Prerequisites

Before installing SEPAR, ensure you have Python 3.7 or later installed. SEPAR requires the following package dependencies:

```python
pandas>=2.0.3
numpy>=1.23.5
scanpy>=1.9.6
anndata>=0.8.0
matplotlib>=3.6.1
scipy>=1.10.0
scikit-learn>=1.2.0
tqdm>=4.64.1
```

### Setting Up

1. Clone the repository:
```bash
git clone https://github.com/zerovain/SEPAR.git
cd SEPAR
```

2. Create and activate a conda environment (recommended):
```bash
conda create -n separ python=3.8
conda activate separ
```

3. Install dependencies:
```bash
conda install pandas numpy scipy matplotlib scikit-learn tqdm
conda install -c conda-forge scanpy anndata
```

## Quick Start

```python
import scanpy as sc
from SEPAR_model import SEPAR

# Load data
adata = sc.read_h5ad('your_data.h5ad')

# Initialize and run SEPAR
separ = SEPAR(adata, n_cluster=8)
separ.preprocess()
separ.compute_graph()
separ.compute_weight()
separ.separ_algorithm(r=30, alpha=1.0, beta=0.1, gamma=0.1)
separ.clustering()
```

## Documentation

For detailed usage instructions and tutorials, please visit our [documentation](https://separ.readthedocs.io/).

## Example Datasets

SEPAR has been tested on various spatial transcriptomics datasets, including:
- 10x Visium datasets
- Stereo-seq datasets
- osmFISH datasets
- MISAR-seq datasets

Example notebooks demonstrating the analysis of these datasets can be found in the `example/` directory.

## Parameters

Key parameters in SEPAR algorithm:
- `n_cluster`: Number of spatial domains to identify
- `r`: Number of patterns to identify
- `alpha`: Weight for graph regularization
- `beta`: Weight for sparsity penalty
- `gamma`: Weight for pattern orthogonality

## Output

SEPAR generates several key outputs:
1. Spatial patterns (`separ.Wpn`)
2. Gene loadings (`separ.Hpn`)
3. Clustering results (`separ.labelres`)
4. Pattern-specific genes (via `identify_pattern_specific_genes()`)

## Contributing

We welcome contributions! Please feel free to submit a Pull Request.

## Citation

If you use SEPAR in your research, please cite:
```bibtex
[Citation information will be added upon publication]
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

For questions and issues:
- Open an issue in the GitHub repository
- Check existing [documentation](https://separ.readthedocs.io/)
- Contact the maintainers

## Acknowledgments

We thank all contributors and users who have helped improve SEPAR.
