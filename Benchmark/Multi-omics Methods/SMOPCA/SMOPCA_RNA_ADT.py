"""SMOPCA: 00 human lymph node RNA + ADT example.
Run in the method environment: python SMOPCA_main.py
"""

import os
import sys

# Dependencies.
import logging
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import scanpy as sc
from SMOPCA.model import SMOPCA
from SMOPCA.utils import clr_normalize_each_cell
import warnings

# 保存结果
output_dir = 'results/00_Human_lymph_node_slice1'
os.makedirs(os.path.join(output_dir, 'latent'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'labels'), exist_ok=True)

for handler in logging.root.handlers[:]:  # avoid DEBUG level information in jupyter notebook
    logging.root.removeHandler(handler)
logging.basicConfig(level=logging.INFO)  # use DEBUG for verbose information
warnings.filterwarnings('ignore')

file_fold = '../../datasets/00_Human_lymph_node_slice1/'
adata1 = sc.read_h5ad(file_fold + 'adata_RNA.h5ad')
adata2 = sc.read_h5ad(file_fold + 'adata_ADT.h5ad')
adata1.var_names_make_unique()
adata2.var_names_make_unique()
adata2 = adata2[adata1.obs_names].copy()

n_clusters = 10
print("===========n_clusters:{}=========".format(n_clusters))

top_3000_svgs_path = "00_top3000_genes_pvalues.csv"
top_3000_svgs = pd.read_csv(top_3000_svgs_path, index_col=0)
top_genes = top_3000_svgs.index.tolist()
adata1 = adata1[:, top_genes]

# Same preprocess_adata defaults; avoid calling .toarray() on scaled dense X.
sc.pp.filter_genes(adata1, min_cells=25)
sc.pp.highly_variable_genes(adata1, flavor="seurat_v3", n_top_genes=2000)
sc.pp.normalize_total(adata1, target_sum=1e4)
sc.pp.log1p(adata1)
sc.pp.scale(adata1)
adata1 = adata1[:, adata1.var['highly_variable']].copy()
adata2 = clr_normalize_each_cell(adata2)
sc.pp.scale(adata2)
X1 = adata1.X.toarray() if hasattr(adata1.X, 'toarray') else np.asarray(adata1.X)
X2 = adata2.X.toarray() if hasattr(adata2.X, 'toarray') else np.asarray(adata2.X)
pos = np.asarray(adata1.obsm['spatial'])

for run_id in range(10):
    print(f"\n=================== Run {run_id+1}/{10} =================")

    smopca = SMOPCA(Y_list=[X1.T, X2.T], Z_dim=20, pos=pos, intercept=False)
    smopca.estimateParams(sigma_init_list=(1, 1), tol_sigma=2e-5,
                          sigma_xtol_list=(1e-6, 1e-6), gamma_init=5, estimate_gamma=True, gamma_tol=1)
    z = smopca.calculatePosterior()

    y_pred = KMeans(n_clusters=n_clusters, n_init=100).fit_predict(z)

    # Save latent features
    latent_features = z
    np.save(os.path.join(output_dir, 'latent', f'latent_{run_id}.npy'), latent_features)

    # Save clustering results
    pre_label = y_pred
    np.save(os.path.join(output_dir, 'labels', f'labels_{run_id}.npy'), pre_label)

print(f"\nAll experiments completed. Results saved to: {output_dir}")
