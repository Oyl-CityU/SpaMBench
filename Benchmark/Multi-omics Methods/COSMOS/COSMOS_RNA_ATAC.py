"""COSMOS: 05 MISAR-E15 RNA + ATAC example.
Run in the method environment: python COSMOS_RNA_ATAC.py
"""

import os
import sys

# Dependencies.
from unittest.mock import patch
import random
import numpy as np
import pandas as pd
import scanpy as sc
import torch
from COSMOS import cosmos
from os.path import join
import warnings

# 保存结果
output_dir = 'results/05_MISAR_E15'
os.makedirs(os.path.join(output_dir, 'latent'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'labels'), exist_ok=True)

warnings.filterwarnings('ignore')
random_seed = 20
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)

# load data
file_fold = '../../datasets/05_MISAR_E15/'
adata1 = sc.read_h5ad(file_fold + 'adata_RNA.h5ad')
adata2 = sc.read_h5ad(file_fold + 'adata_ATAC.h5ad')
adata1.var_names_make_unique()
adata2.var_names_make_unique()
adata2 = adata2[adata1.obs_names].copy()

n_clusters = 12
print("===========n_clusters:{}=========".format(n_clusters))

sc.pp.normalize_per_cell(adata1)
sc.pp.log1p(adata1)
sc.pp.log1p(adata2)
adata1.obs['x_pos'] = adata1.obsm['spatial'][:, 0]
adata1.obs['y_pos'] = adata1.obsm['spatial'][:, 1]
adata2.obs['x_pos'] = adata2.obsm['spatial'][:, 0]
adata2.obs['y_pos'] = adata2.obsm['spatial'][:, 1]

def screen_res(df_embedding, n_cluster=5, res_s=0.1, res_e=1.0, step=0.1):
    opt_res = res_s
    opt_clusters = None
    for res in np.arange(res_s, res_e, step):
        embedding_adata = sc.AnnData(df_embedding)
        sc.pp.neighbors(embedding_adata, n_neighbors=50, use_rep='X')
        sc.tl.leiden(embedding_adata, resolution=float(res))
        clusters = list(embedding_adata.obs["leiden"])
        cluster_num = len(np.unique(clusters))
        print('res = ' + str(round(res, 2)) + ', Cluster# = ' + str(cluster_num))
        if cluster_num == n_cluster:
            opt_res = res
            opt_clusters = clusters
    print('optimal res = ' + str(round(opt_res, 2)))
    return opt_res, opt_clusters

for run_id in range(10):
    print(f"\n=================== Run {run_id+1}/{10} =================")

    # COSMOS training
    cosmos_comb = cosmos.Cosmos(adata1=adata1.copy(), adata2=adata2.copy())
    cosmos_comb.preprocessing_data(n_neighbors=10)
    with patch.object(torch, 'manual_seed', lambda _: None), \
            patch.object(random, 'seed', lambda _: None), \
            patch.object(np.random, 'seed', lambda _: None):
        cosmos_comb.train(spatial_regularization_strength=0.05, z_dim=50,
                          lr=1e-3, wnn_epoch=500, total_epoch=1000, max_patience_bef=10, max_patience_aft=30, min_stop=200,
                          random_seed=random_seed, gpu=0, regularization_acceleration=True, edge_subset_sz=1000000)
    weights = cosmos_comb.weights
    df_embedding = pd.DataFrame(cosmos_comb.embedding)

    # Screening resolution that matches the given number of clusters
    opt_res_cosmos, opt_clusters_cosmos = screen_res(
        df_embedding, res_s=0.5, res_e=0.9, step=0.02, n_cluster=n_clusters)

    # Save latent features
    latent_features = df_embedding.values
    np.save(os.path.join(output_dir, 'latent', f'latent_{run_id}.npy'), latent_features)
    pre_label = np.array(opt_clusters_cosmos) if opt_clusters_cosmos is not None else None
    if pre_label is not None:
        np.save(os.path.join(output_dir, 'labels', f'labels_{run_id}.npy'), pre_label)

    # Record

def screen_res(df_embedding, n_cluster=5, res_s=0.1, res_e=1.0, step=0.1):
    opt_res = res_s
    opt_clusters = None
    for res in np.arange(res_s, res_e, step):
        embedding_adata = sc.AnnData(df_embedding)
        sc.pp.neighbors(embedding_adata, n_neighbors=50, use_rep='X')
        sc.tl.leiden(embedding_adata, resolution=float(res))
        clusters = list(embedding_adata.obs["leiden"])
        cluster_num = len(np.unique(clusters))
        print('res = ' + str(round(res, 2)) + ', Cluster# = ' + str(cluster_num))
        if cluster_num == n_cluster:
            opt_res = res
            opt_clusters = clusters
    print('optimal res = ' + str(round(opt_res, 2)))
    return opt_res, opt_clusters

print(f"\nAll experiments completed. Results saved to: {output_dir}")
