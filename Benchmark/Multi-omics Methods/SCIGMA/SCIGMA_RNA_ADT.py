"""SCIGMA: 00 human lymph node RNA + ADT example.
Run in the method environment: python SCIGMA_main.py
"""

import os
import sys

# Dependencies.
import numpy as np
import pandas as pd
import scanpy as sc
import torch
from os.path import join
from rpy2 import robjects
from rpy2.robjects import default_converter, numpy2ri
from rpy2.robjects.conversion import localconverter
from scigma_core.data import clr_normalize_each_cell, build_network, feature_graph
from scigma_core.utils import seed
import scigma_core.SCIGMA as scigma_module

# 保存结果
output_dir = 'results/00_Human_lymph_node_slice1'
os.makedirs(os.path.join(output_dir, 'latent'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'labels'), exist_ok=True)

robjects.r.library('mclust')
# Same mclust EEE; modern conversion avoids deprecated numpy2ri.activate().
mclust = robjects.r('''function(x, k) {
    fit <- mclust::Mclust(x, G=k, modelNames="EEE")
    if (is.null(fit$classification)) stop("mclust returned no labels")
    fit$classification
}''')

# Load paired RNA/ADT counts.
n_clusters = 10

file_fold = '../../datasets/00_Human_lymph_node_slice1/'
adata1 = sc.read_h5ad(file_fold + 'adata_RNA.h5ad')
adata2 = sc.read_h5ad(file_fold + 'adata_ADT.h5ad')
adata1.var_names_make_unique()
adata2.var_names_make_unique()
adata2 = adata2[adata1.obs_names].copy()

cell_ids = adata1.obs_names.copy()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# RNA+ADT: SPOTS preprocessing and model defaults, as in our basic benchmark.
# 1000 is utils.seed's default
random_seed = 1000
seed(random_seed)

sc.pp.filter_genes(adata1, min_cells=10)
sc.pp.highly_variable_genes(adata1, flavor='seurat_v3', n_top_genes=3000)
sc.pp.normalize_total(adata1, target_sum=1e6)
sc.pp.log1p(adata1)
adata1.layers['logcounts'] = adata1.X.copy()

sc.pp.scale(adata1)
high = adata1[:, adata1.var['highly_variable']].copy()
sc.pp.pca(high, n_comps=adata2.n_vars - 1)
adata1.obsm['feat'] = high.obsm['X_pca'].copy()
adata2 = clr_normalize_each_cell(adata2)
adata2.layers['logcounts'] = adata2.X.copy()
sc.pp.scale(adata2)
sc.pp.pca(adata2, n_comps=adata2.n_vars - 1)
adata2.obsm['feat'] = adata2.obsm['X_pca'].copy()

for a in [adata1, adata2]:
    a.uns['adj_spatial'] = build_network(a.obsm['spatial'], n_neighbors=8)
g1, g2 = feature_graph(adata1, adata2, k=8)
adata1.obsm['adj_feature'], adata2.obsm['adj_feature'] = g1, g2
assert adata1.obs_names.equals(cell_ids) and adata2.obs_names.equals(cell_ids)
data = {'adata_omics1': adata1, 'adata_omics2': adata2}

for run_id in range(10):
    print(f'\n=================== Run {run_id+1}/10 =================')

    original_seed = scigma_module.seed
    try:
        scigma_module.seed = lambda _: None
        # ForestFire cannot sample more cells than exist in the dataset.
        model = scigma_module.SCIGMA(data, seed_num=random_seed, device=device,
                                     batch_size=min(6000, len(cell_ids)))
    finally:
        scigma_module.seed = original_seed
    output = model.train(1000)
    latent_features = output['emb_latent_combined']

    adata1.obsm['SCIGMA'] = latent_features
    np.save(join(output_dir, 'latent', f'latent_{run_id}.npy'), latent_features)

    cluster_features = latent_features  # Tutorial mclust does not use PCA.
    robjects.r['set.seed'](1000)
    with localconverter(default_converter + numpy2ri.converter):
        pre_label = np.asarray(mclust(cluster_features, n_clusters), dtype=int)
    np.save(join(output_dir, 'labels', f'labels_{run_id}.npy'), pre_label)

    del model

print(f'\nAll experiments completed. Results saved to: {output_dir}')
