"""PRESENT: 00 human lymph node RNA + ADT example.
Run in the method environment: python PRESENT_main.py
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
output_dir = 'results/00_Human_lymph_node_slice1'
os.makedirs(os.path.join(output_dir, 'latent'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'labels'), exist_ok=True)

# Inputs and original single-sample model settings.
device = 'cuda' if torch.cuda.is_available() else 'cpu'
device_id = 0
n_clusters = 10
file_fold = '../../datasets/00_Human_lymph_node_slice1/'
adata_rna = sc.read_h5ad(file_fold + 'adata_RNA.h5ad')
adata_adt = sc.read_h5ad(file_fold + 'adata_ADT.h5ad')
adata_rna.var_names_make_unique()
adata_adt.var_names_make_unique()
adata_adt = adata_adt[adata_rna.obs_names].copy()
adata_rna.X = adata_rna.X.astype(np.float32)
adata_adt.X = adata_adt.X.astype(np.float32)

for run_id in range(10):
    print(f'Run {run_id + 1}/10', flush=True)
    adata = PRESENT_function(
        spatial_key='spatial', adata_rna=adata_rna.copy(), adata_adt=adata_adt.copy(),
        gene_min_cells=1, protein_min_cells=1, num_hvg=3000,
        nclusters=n_clusters, d_lat=50, k_neighbors=6,
        intra_neighbors=6, inter_neighbors=6, epochs=100,
        lr=0.001, batch_size=320, device=device, device_id=device_id)
    adata = adata[adata_rna.obs_names].copy()
    np.save(os.path.join(output_dir, 'latent', f'latent_{run_id}.npy'), adata.obsm['embeddings'])
    np.save(os.path.join(output_dir, 'labels', f'labels_{run_id}.npy'), adata.obs['LeidenClusters'].to_numpy())
    adata.write_h5ad(os.path.join(output_dir, 'adata_output.h5ad'))
