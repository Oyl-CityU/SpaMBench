"""PRESENT: 05 MISAR-E15 RNA + ATAC example.
Run in the method environment: python PRESENT_RNA_ATAC.py
"""

import os
import sys

# Dependencies.
import numpy as np
import pandas as pd
import scanpy as sc
import torch
from PRESENT import PRESENT_function

# 保存结果
output_dir = 'results/05_MISAR_E15'
os.makedirs(os.path.join(output_dir, 'latent'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'labels'), exist_ok=True)

# Inputs and original single-sample model settings.
device = 'cuda' if torch.cuda.is_available() else 'cpu'
device_id = 0
n_clusters = 12
file_fold = '../../datasets/05_MISAR_E15/'
adata_rna = sc.read_h5ad(file_fold + 'adata_RNA.h5ad')
adata_atac = sc.read_h5ad(file_fold + 'adata_ATAC.h5ad')
adata_rna.var_names_make_unique()
adata_atac.var_names_make_unique()
adata_atac = adata_atac[adata_rna.obs_names].copy()
adata_rna.X = adata_rna.X.astype(np.float32)
adata_atac.X = adata_atac.X.astype(np.float32)

for run_id in range(10):
    print(f'Run {run_id + 1}/10', flush=True)
    adata = PRESENT_function(
        spatial_key='spatial', adata_rna=adata_rna.copy(), adata_atac=adata_atac.copy(),
        gene_min_cells=1, peak_min_cells_fraction=0.03, num_hvg=3000,
        nclusters=n_clusters, d_lat=50, k_neighbors=6,
        intra_neighbors=6, inter_neighbors=6, epochs=100,
        lr=0.001, batch_size=320, device=device, device_id=device_id)
    adata = adata[adata_rna.obs_names].copy()
    np.save(os.path.join(output_dir, 'latent', f'latent_{run_id}.npy'), adata.obsm['embeddings'])
    np.save(os.path.join(output_dir, 'labels', f'labels_{run_id}.npy'), adata.obs['LeidenClusters'].to_numpy())
    adata.write_h5ad(os.path.join(output_dir, 'adata_output.h5ad'))
