"""SpaMV: 05 MISAR-E15 RNA + ATAC example.
Run in the method environment: python SpaMV_RNA_ATAC.py
"""

import os
import sys

# Dependencies.
import numpy as np
import pandas as pd
import scanpy as sc
import torch
from SpaMV.preprocess_keep_cells import preprocess_dc_keep_cells
from SpaMV.utils import clustering
from os.path import join
from rpy2 import robjects
from rpy2.robjects import default_converter, numpy2ri
from rpy2.robjects.conversion import localconverter
import SpaMV.spamv as spamv_module
import SpaMV.utils as cluster_module


def mclust_R(adata, num_cluster, modelNames='EEE', used_obsm='emb', random_seed=2020):
    robjects.r.library('mclust')
    robjects.r['set.seed'](random_seed)
    classify = robjects.r(
        'function(x, k, model) {mclust::Mclust(x, G=k, modelNames=model)$classification}')
    with localconverter(default_converter + numpy2ri.converter):
        labels = np.asarray(classify(adata.obsm[used_obsm], num_cluster, modelNames), dtype=int)
    adata.obs['mclust'] = pd.Categorical(labels)
    adata.obs['clusters_mclust'] = adata.obs['mclust'].copy()
    return adata

cluster_module.mclust_R = mclust_R

# 保存结果
output_dir = 'results/05_MISAR_E15'
os.makedirs(os.path.join(output_dir, 'latent'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'labels'), exist_ok=True)

# Load paired RNA/ATAC counts.
n_clusters = 12

file_fold = '../../datasets/05_MISAR_E15/'
adata1 = sc.read_h5ad(file_fold + 'adata_RNA.h5ad')
adata2 = sc.read_h5ad(file_fold + 'adata_ATAC.h5ad')
adata1.var_names_make_unique()
adata2.var_names_make_unique()
adata2 = adata2[adata1.obs_names].copy()

cell_ids = adata1.obs_names.copy()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Official constructor seed is 0;
random_seed = 0
spamv_module.set_seed(random_seed)
# Domain clustering tutorial, RNA+ADT (scale=True)
datasets = preprocess_dc_keep_cells([adata1.copy(), adata2.copy()],
                                    ['Transcriptome', 'Epigenome'], scale=True)

for run_id in range(10):
    print(f'\n=================== Run {run_id+1}/10 =================')

    original_seed = spamv_module.set_seed
    spamv_module.set_seed = lambda _: None
    try:
        model = spamv_module.SpaMV(datasets, interpretable=False, device=device)
    finally:
        spamv_module.set_seed = original_seed
    model.train()
    latent_features = model.get_embedding()

    adata1.obsm['SpaMV'] = latent_features
    np.save(join(output_dir, 'latent', f'latent_{run_id}.npy'), latent_features)

    datasets[0].obsm['SpaMV'] = model.get_embedding()
    clustering(datasets[0], n_clusters=n_clusters, key='SpaMV')
    pre_label = datasets[0].obs['SpaMV']

    np.save(join(output_dir, 'labels', f'labels_{run_id}.npy'), pre_label)

    del model

print(f'\nAll experiments completed. Results saved to: {output_dir}')
