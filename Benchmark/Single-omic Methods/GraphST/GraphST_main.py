"""GraphST example for the 00 human lymph node RNA dataset."""

import os
import time

import numpy as np
import pandas as pd
import scanpy as sc
import torch
from GraphST import GraphST
from GraphST.utils import clustering


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
file_fold = '../../datasets/00_Human_lymph_node_slice1/'
output_dir = 'results/00_Human_lymph_node_slice1'

adata = sc.read_h5ad(file_fold + 'adata_RNA.h5ad')
adata.var_names_make_unique()

n_clusters = len(np.unique(adata.obs['final_annot']))
print('Number of clusters: {}'.format(n_clusters))

os.makedirs(os.path.join(output_dir, 'latent'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'labels'), exist_ok=True)

radius = 50
tool = 'mclust'  # mclust, leiden, or louvain
runtimes = []

for run_id in range(10):
    print('=================== Run {}/10 ================='.format(run_id + 1))

    start_time = time.time()
    model = GraphST.GraphST(adata, device=device)
    adata = model.train()
    runtimes.append(time.time() - start_time)

    if tool == 'mclust':
        clustering(adata, n_clusters, radius=radius, method=tool, refinement=True)
    elif tool in ['leiden', 'louvain']:
        clustering(adata, n_clusters, radius=radius, method=tool,
                   start=0.1, end=2.0, increment=0.01, refinement=False)

    np.save(os.path.join(output_dir, 'latent', 'latent_{}.npy'.format(run_id)),
            adata.obsm['emb'])
    np.save(os.path.join(output_dir, 'labels', 'labels_{}.npy'.format(run_id)),
            adata.obs['mclust'].to_numpy())
    adata.write_h5ad(os.path.join(output_dir, '{}_{}.h5ad'.format(
        '00_Human_lymph_node_slice1', run_id)))

pd.DataFrame({'run': range(10), 'runtime_seconds': runtimes}).to_csv(
    os.path.join(output_dir, 'log_summary.csv'), index=False)
print('All runs completed. Results saved to {}'.format(output_dir))
