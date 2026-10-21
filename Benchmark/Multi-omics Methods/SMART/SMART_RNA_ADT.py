"""SMART: 00 human lymph node RNA + ADT example.
Run in the method environment: python SMART_main.py
"""

import os
import sys

# Dependencies.
import numpy as np
import pandas as pd
import scanpy as sc
import torch
from muon import prot as pt
from os.path import join
from rpy2 import robjects
from rpy2.robjects import default_converter, numpy2ri
from rpy2.robjects.conversion import localconverter
from smart.MNN import Mutual_Nearest_Neighbors
from smart.build_graph import Cal_Spatial_Net
from smart.train import train_SMART
from smart.utils import clustering
from smart.utils import set_seed, pca
import smart.utils as cluster_module

# R compatibility only: preserve the method's PCA and mclust EEE settings.

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

n_clusters = 10

file_fold = '../../datasets/00_Human_lymph_node_slice1/'
adata1 = sc.read_h5ad(file_fold + 'adata_RNA.h5ad')
adata2 = sc.read_h5ad(file_fold + 'adata_ADT.h5ad')
adata1.var_names_make_unique()
adata2.var_names_make_unique()
adata2 = adata2[adata1.obs_names].copy()

cell_ids = adata1.obs_names.copy()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Tutorial 2 (human lymph node): seed once before all repeats.
random_seed = 2024
set_seed(random_seed)

sc.pp.filter_genes(adata1, min_cells=10)
sc.pp.highly_variable_genes(adata1, flavor='seurat_v3', n_top_genes=3000)
sc.pp.normalize_total(adata1, target_sum=1e4)
sc.pp.log1p(adata1)
sc.pp.scale(adata1)
adata1.obsm['feat'] = pca(adata1[:, adata1.var['highly_variable']], n_comps=30)

adata2.X = adata2.X.astype(float)
pt.pp.clr(adata2)
sc.pp.scale(adata2)
adata2.obsm['feat'] = pca(adata2, n_comps=30)
Cal_Spatial_Net(adata1, model='KNN', n_neighbors=6)
Cal_Spatial_Net(adata2, model='KNN', n_neighbors=6)
adatas = [adata1, adata2]
x = [torch.FloatTensor(a.obsm['feat']).to(device) for a in adatas]
edges = [torch.LongTensor(a.uns['edgeList']).to(device) for a in adatas]

for run_id in range(10):
    print(f'\n=================== Run {run_id+1}/10 =================')

    numpy_state = np.random.get_state()
    triplets = [Mutual_Nearest_Neighbors(a, key='feat', n_nearest_neighbors=3,
                farthest_ratio=0.6) for a in adatas]
    np.random.set_state(numpy_state)
    model = train_SMART(features=x, edges=edges, triplet_samples_list=triplets,
                        weights=[1, 1, 1, 1], emb_dim=64, n_epochs=300,
                        lr=5e-3, weight_decay=1e-6, device=device,
                        window_size=10, slope=1e-4)
    with torch.no_grad():
        latent_features = model(x, edges)[0].cpu().numpy()

    adata1.obsm['SMART'] = latent_features

    tool = 'mclust'  # mclust, leiden, and louvain
    clustering(adata1, key='SMART', add_key='SMART',
               n_clusters=n_clusters, method=tool, use_pca=True)
    # ari = adjusted_rand_score(adata1.obs['final_annot'], adata1.obs['SMART'])
    pre_label = adata1.obs['SMART']

    np.save(join(output_dir, 'latent', f'latent_{run_id}.npy'), latent_features)
    np.save(join(output_dir, 'labels', f'labels_{run_id}.npy'), pre_label)

    del model

print(f'\nAll experiments completed. Results saved to: {output_dir}')
