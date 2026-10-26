"""SpaMode: 00 human lymph node RNA + ADT example.
Run in the method environment: python SpaMode_main.py
"""

import os
import sys

# Dependencies.
import numpy as np
import pandas as pd
import scanpy as sc
import torch
from SpaMode.Train import Train_SpaMode
from SpaMode.preprocess import construct_neighbor_graph_SpaMode
from SpaMode.preprocess import fix_seed, pca, clr_normalize_each_cell
from SpaMode.utils import clustering
from rpy2 import robjects
from rpy2.robjects import default_converter, numpy2ri
from rpy2.robjects.conversion import localconverter
import SpaMode.utils as cluster_module


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
output_dir = 'results/00_Human_lymph_node_slice1'
os.makedirs(os.path.join(output_dir, 'latent'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'labels'), exist_ok=True)

# 00 human lymph node configuration.
cfg = {
    'training_epoch': 1000,
    'learning_rate': 0.0001,
    'weight_decay': 0,
    'weight_recon': [5.0, 5.0],
    'weight_bi': 10,
    'weight_adv': 0.1,
    'weight_kl': 1e-06,
    'weight_moe': 100.0,
    'optimizer': 'AdamW',
    'earlystop': True,
    'early_stop_patience': 20,
    'dataset_name': '00_Human_lymph_node_slice1',
    'n_clusters': 10,
}

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
random_seed = 2024

dataset_name = cfg['dataset_name']
n_clusters = cfg['n_clusters']

# Set the seed once per dataset, outside the ten training runs.
fix_seed(random_seed)
print(f'\n=========== {dataset_name}, n_clusters: {n_clusters} ===========')

# Load and align paired modalities. Never remove cells.
file_fold = '../../datasets/00_Human_lymph_node_slice1/'
adata1 = sc.read_h5ad(file_fold + 'adata_RNA.h5ad')
second_modality = 'ADT'
adata2 = sc.read_h5ad(file_fold + 'adata_ADT.h5ad')
adata1.var_names_make_unique()
adata2.var_names_make_unique()
assert adata1.obs_names.is_unique and adata2.obs_names.is_unique
assert set(adata1.obs_names) == set(adata2.obs_names), 'Modality cell IDs differ'
adata2 = adata2[adata1.obs_names].copy()
original_ids = adata1.obs_names.copy()

# RNA preprocessing.
sc.pp.filter_genes(adata1, min_cells=10)
sc.pp.highly_variable_genes(adata1, flavor='seurat_v3', n_top_genes=3000)
sc.pp.normalize_total(adata1, target_sum=1e4)
sc.pp.log1p(adata1)
sc.pp.scale(adata1)
adata1_high = adata1[:, adata1.var['highly_variable']].copy()
adata1.obsm['feat'] = pca(adata1_high, n_comps=min(30, adata1_high.n_obs, adata1_high.n_vars))

# Protein preprocessing.
adata2 = clr_normalize_each_cell(adata2)
sc.pp.scale(adata2)
n_comps = min(30, adata2.n_obs, adata2.n_vars)
adata2.obsm['feat'] = pca(adata2, n_comps=n_comps)
assert adata1.obs_names.equals(original_ids) and adata2.obs_names.equals(original_ids)

datatype = cfg.get('datatype', '10x')
data = construct_neighbor_graph_SpaMode(
    [adata1, adata2], datatype=datatype, Spatial_neighbors=3, Feature_neighbors=20)

for run_id in range(10):
    print(f'\n=================== Run {run_id+1}/10 =================')

    # Dataset-specific parameters are preserved from the existing config.
    model = Train_SpaMode(data, datatype=datatype, device=device, Arg=cfg.copy())
    output = model.train()
    adata = adata1.copy()
    adata.obsm['SpaMode'] = output['SpaMode'].copy()

    rng_state = np.random.get_state()
    try:
        clustering(adata, key='SpaMode', add_key='SpaMode', n_clusters=n_clusters,
                   method='mclust', use_pca=True)
    finally:
        np.random.set_state(rng_state)

    latent_features = adata.obsm['SpaMode']
    pre_label = adata.obs['SpaMode'].to_numpy()
    np.save(os.path.join(output_dir, 'latent', f'latent_{run_id}.npy'), latent_features)
    np.save(os.path.join(output_dir, 'labels', f'labels_{run_id}.npy'), pre_label)

    del model, output, adata

print(f'\nAll experiments completed. Results saved to: {output_dir}')
