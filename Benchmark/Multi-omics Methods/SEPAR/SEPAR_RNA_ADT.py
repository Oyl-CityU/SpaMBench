"""SEPAR: 00 human lymph node RNA + ADT example.
Run in the method environment: python SEPAR_main.py
"""

import os
import sys

# Dependencies.
from unittest.mock import patch
import random
from scipy import sparse
import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from SEPAR_model import SEPAR
import SEPAR_model as separ_module

# 保存结果
output_dir = 'results/00_Human_lymph_node_slice1'
os.makedirs(os.path.join(output_dir, 'latent'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'labels'), exist_ok=True)

random.seed(2024)
np.random.seed(2024)
original_nmf = separ_module.non_negative_factorization
original_kmeans = separ_module.KMeans

# load data

file_fold = '../../datasets/00_Human_lymph_node_slice1/'
data1 = sc.read_h5ad(file_fold + "adata_RNA.h5ad")
data2 = sc.read_h5ad(file_fold + "adata_ADT.h5ad")
data1.var_names_make_unique()
data2.var_names_make_unique()
data2 = data2[data1.obs_names].copy()
# data1.X = data1.X.toarray()

adata_rna = data1
adata_adt = data2

n_clusters = 10
print("===========n_clusters:{}=========".format(n_clusters))

# Normalize each modality
sc.pp.normalize_total(adata_adt, target_sum=1e4)
sc.pp.log1p(adata_adt)
sc.pp.normalize_total(adata_rna, target_sum=1e4)
sc.pp.log1p(adata_rna)

# Combine modalities
x_combined = sparse.hstack([sparse.csr_matrix(adata_adt.X),
                           sparse.csr_matrix(adata_rna.X)], format="csr")
var_combined = pd.concat([
    adata_adt.var.assign(batch='ADT'),
    adata_rna.var.assign(batch='RNA')
])

# Create combined AnnData
adata_combined = ad.AnnData(
    X=x_combined,
    obs=adata_adt.obs,
    var=var_combined
)
adata_combined.obsm['spatial'] = adata_rna.obsm['spatial']

# Initialize SEPAR
# n_cluster = len(np.unique(Y_atac))
n_cluster = n_clusters
separ = SEPAR(adata_combined, n_cluster=n_cluster)

# Preprocess without additional normalization
separ.preprocess(min_cells=50, normalize=False)
separ.compute_graph(radius_rate=1.3)

# Feature selection
separ.select_morani(nslt=5000)

# Check modality distribution
batch_counts = separ.adata.var['batch'].value_counts()
print("Selected features per modality:", batch_counts)

for run_id in range(10):
    print(f"\n=================== Run {run_id+1}/{10} =================")

    # model train
    # Compute weights and run SEPAR
    separ.compute_weight(n_cluster=n_cluster)
    with patch.object(separ_module, 'non_negative_factorization',
                      lambda *a, **kw: original_nmf(*a, **dict(kw, random_state=None))):
        separ.separ_algorithm(
            r=30,              # Number of patterns
            alpha=0.5,         # Graph regularization
            beta=0.01,         # Sparsity penalty (previously l1)
            gamma=0.5,         # Pattern orthogonality (previously lam)
            mean=False
        )

    # Perform clustering
    with patch.object(separ_module, 'KMeans',
                      lambda *a, **kw: original_kmeans(*a, **dict(kw, random_state=None))):
        separ.clustering(n_cluster=n_clusters, N1=15, N2=1)

    # Save latent features
    latent_features = separ.Wpn
    np.save(os.path.join(output_dir, 'latent', f'latent_{run_id}.npy'), latent_features)

    # Save this method's clustering labels.
    pre_label = separ.labelres
    np.save(os.path.join(output_dir, 'labels', f'labels_{run_id}.npy'), pre_label)

print(f"\nAll experiments completed. Results saved to: {output_dir}")
