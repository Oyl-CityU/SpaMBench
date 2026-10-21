"""PRAGA: 00 human lymph node RNA + ADT example.
Run in the method environment: python PRAGA_main.py
"""

import os
import sys

for variable in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ.setdefault(variable, '1')

# Dependencies.
from types import SimpleNamespace
import numpy as np
import pandas as pd
import scanpy as sc
import torch
from PRAGA.Train_model import Train
from PRAGA.preprocess import fix_seed, clr_normalize_each_cell, pca, construct_neighbor_graph
from PRAGA.utils import clustering
from rpy2 import robjects
from rpy2.robjects import default_converter, numpy2ri
from rpy2.robjects.conversion import localconverter
import PRAGA.utils as cluster_module

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

# Configuration: 00 human lymph node, RNA + ADT.
args = SimpleNamespace(data_type='10x', n_clusters=10, init_k=10, KNN_k=20,
                       alpha=0.9, cl_weight=1, RNA_weight=5, ADT_weight=5, tau=2, seed=2024,
                       output_dir=output_dir)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
fix_seed(args.seed)
file_fold = '../../datasets/00_Human_lymph_node_slice1/'
adata_omics1 = sc.read_h5ad(file_fold + 'adata_RNA.h5ad')
adata_omics2 = sc.read_h5ad(file_fold + 'adata_ADT.h5ad')
adata_omics1.var_names_make_unique()
adata_omics2.var_names_make_unique()
adata_omics2 = adata_omics2[adata_omics1.obs_names].copy()
adata_omics2 = adata_omics2[adata_omics1.obs_names].copy()

# Preprocessing.
# RNA
sc.pp.filter_genes(adata_omics1, min_cells=10)
sc.pp.highly_variable_genes(adata_omics1, flavor="seurat_v3", n_top_genes=3000)
sc.pp.normalize_total(adata_omics1, target_sum=1e4)
sc.pp.log1p(adata_omics1)
sc.pp.scale(adata_omics1)
adata_omics1_high = adata_omics1[:, adata_omics1.var['highly_variable']]
adata_omics1.obsm['feat'] = pca(adata_omics1_high, n_comps=adata_omics2.n_vars - 1)
# Protein
adata_omics2 = clr_normalize_each_cell(adata_omics2)
sc.pp.scale(adata_omics2)
adata_omics2.obsm['feat'] = pca(adata_omics2, n_comps=adata_omics2.n_vars - 1)
data = construct_neighbor_graph(adata_omics1, adata_omics2, datatype=args.data_type, Arg=args)

# Independent training runs.
for run_id in range(10):
    print(f"\n=================== Run {run_id+1}/{10} =================")
    model = Train(data, datatype=args.data_type, device=device, Arg=args)
    # train model
    output = model.train()
    adata = adata_omics1.copy()
    adata.obsm['emb_latent_omics1'] = output['emb_latent_omics1'].copy()
    adata.obsm['emb_latent_omics2'] = output['emb_latent_omics2'].copy()
    adata.obsm['PRAGA'] = output['PRAGA'].copy()

    # Clustering
    tool = 'mclust'  # mclust, leiden, and louvain
    clustering(adata, key='PRAGA', add_key='PRAGA',
               n_clusters=args.n_clusters, method=tool, use_pca=True)

    label = adata.obs['PRAGA']
    laten = adata.obsm['PRAGA']

    # Save latent features
    latent_features = laten
    np.save(os.path.join(args.output_dir, 'latent', f'latent_{run_id}.npy'), latent_features)

    # Save this method's clustering labels.
    pre_label = label
    np.save(os.path.join(args.output_dir, 'labels', f'labels_{run_id}.npy'), pre_label)

print(f"\nAll experiments completed. Results saved to: {args.output_dir}")
