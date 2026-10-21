"""SEDR example for the 00 human lymph node RNA dataset."""

import os
import time

import numpy as np
import pandas as pd
import scanpy as sc
import torch
from sklearn.decomposition import PCA
import SEDR


random_seed = 2023
SEDR.fix_seed(random_seed)
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

file_fold = '../../datasets/00_Human_lymph_node_slice1/'
output_dir = 'results/00_Human_lymph_node_slice1'

adata = sc.read_h5ad(file_fold + 'adata_RNA.h5ad')
adata.var_names_make_unique()

n_clusters = len(np.unique(adata.obs['final_annot']))
print('Number of clusters: {}'.format(n_clusters))

# Preprocessing from the SEDR tutorial.
sc.pp.filter_genes(adata, min_counts=10)
sc.pp.normalize_total(adata, target_sum=1e6)
sc.pp.highly_variable_genes(adata, flavor='seurat_v3', n_top_genes=2000)
adata = adata[:, adata.var['highly_variable']].copy()
sc.pp.scale(adata)

# sklearn PCA is used because Scanpy PCA is not stable in the original example.
adata.obsm['X_pca'] = PCA(n_components=200, random_state=42).fit_transform(adata.X)
graph_dict = SEDR.graph_construction(adata, 12)

os.makedirs(os.path.join(output_dir, 'latent'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'labels'), exist_ok=True)
runtimes = []

for run_id in range(10):
    print('=================== Run {}/10 ================='.format(run_id + 1))

    start_time = time.time()
    sedr_net = SEDR.Sedr(adata.obsm['X_pca'], graph_dict,
                         mode='clustering', device=device)
    sedr_net.train_with_dec(N=1)
    sedr_features, _, _, _ = sedr_net.process()
    adata.obsm['SEDR'] = sedr_features
    runtimes.append(time.time() - start_time)

    SEDR.mclust_R(adata, n_clusters, use_rep='SEDR', key_added='SEDR')

    np.save(os.path.join(output_dir, 'latent', 'latent_{}.npy'.format(run_id)),
            adata.obsm['SEDR'])
    np.save(os.path.join(output_dir, 'labels', 'labels_{}.npy'.format(run_id)),
            adata.obs['SEDR'].to_numpy())
    adata.write_h5ad(os.path.join(output_dir, '{}_{}.h5ad'.format(
        '00_Human_lymph_node_slice1', run_id)))

pd.DataFrame({'run': range(10), 'runtime_seconds': runtimes}).to_csv(
    os.path.join(output_dir, 'log_summary.csv'), index=False)
print('All runs completed. Results saved to {}'.format(output_dir))
