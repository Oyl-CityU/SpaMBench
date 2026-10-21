"""SpatialGlue: 05 MISAR-E15 RNA + ATAC example.
Run in the method environment: python SpatialGlue_RNA_ATAC.py
"""

import os
import sys

# Dependencies.
import numpy as np
import pandas as pd
import scanpy as sc
import torch
from SpatialGlue.SpatialGlue_pyG import Train_SpatialGlue
from SpatialGlue.preprocess import clr_normalize_each_cell, pca, lsi
from SpatialGlue.preprocess import construct_neighbor_graph
from SpatialGlue.preprocess import fix_seed
from SpatialGlue.utils import clustering
from rpy2 import robjects
from rpy2.robjects import default_converter, numpy2ri
from rpy2.robjects.conversion import localconverter
import SpatialGlue.utils as cluster_module

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
output_dir = 'results/05_MISAR_E15'
os.makedirs(os.path.join(output_dir, 'latent'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'labels'), exist_ok=True)

# Environment configuration. SpatialGlue pacakge can be implemented with either CPU or GPU. GPU acceleration is highly recommend for imporoved efficiency.
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# the location of R, which is required for the 'mclust' algorithm. Please replace the path below with local R installation path

# read data
file_fold = '../../datasets/05_MISAR_E15/'
adata_omics1 = sc.read_h5ad(file_fold + 'adata_RNA.h5ad')
adata_omics2 = sc.read_h5ad(file_fold + 'adata_ATAC.h5ad')
adata_omics1.var_names_make_unique()
adata_omics2.var_names_make_unique()
adata_omics2 = adata_omics2[adata_omics1.obs_names].copy()

# Specify data type
data_type = 'Spatial-epigenome-transcriptome'
n_clusters = 12
print("=========n_clusters:{}=============".format(n_clusters))
# Fix random seed
random_seed = 2022
fix_seed(random_seed)

# RNA
sc.pp.filter_genes(adata_omics1, min_cells=10)
sc.pp.highly_variable_genes(adata_omics1, flavor="seurat_v3", n_top_genes=3000)
sc.pp.normalize_total(adata_omics1, target_sum=1e4)
sc.pp.log1p(adata_omics1)
sc.pp.scale(adata_omics1)

adata_omics1_high = adata_omics1[:, adata_omics1.var['highly_variable']]
adata_omics1.obsm['feat'] = pca(adata_omics1_high, n_comps=50)

# ATAC: HVG selection followed by the package TF-IDF/LSI transform.
sc.pp.highly_variable_genes(adata_omics2, flavor='seurat_v3', n_top_genes=3000)
lsi(adata_omics2, use_highly_variable=True, n_components=51)
adata_omics2.obsm['feat'] = adata_omics2.obsm['X_lsi'].copy()

data = construct_neighbor_graph(adata_omics1, adata_omics2, datatype=data_type)

for run_id in range(10):
    print(f"\n=== Run {run_id+1}/{10} ===")

    # define model
    model = Train_SpatialGlue(data, datatype=data_type, device=device)
    # train model
    output = model.train()
    adata = adata_omics1.copy()
    adata.obsm['emb_latent_omics1'] = output['emb_latent_omics1'].copy()
    adata.obsm['emb_latent_omics2'] = output['emb_latent_omics2'].copy()
    adata.obsm['SpatialGlue'] = output['SpatialGlue'].copy()
    adata.obsm['alpha'] = output['alpha']
    adata.obsm['alpha_omics1'] = output['alpha_omics1']
    adata.obsm['alpha_omics2'] = output['alpha_omics2']

    # we set 'mclust' as clustering tool by default. Users can also select 'leiden' and 'louvain'
    tool = 'mclust'  # mclust, leiden, and louvain
    clustering(adata, key='SpatialGlue', add_key='SpatialGlue',
               n_clusters=n_clusters, method=tool, use_pca=True)

    # Save latent features
    latent_features = adata.obsm['SpatialGlue']
    np.save(os.path.join(output_dir, 'latent', f'latent_{run_id}.npy'), latent_features)
    # Save this method's clustering labels.
    pre_label = adata.obs['mclust']
    np.save(os.path.join(output_dir, 'labels', f'labels_{run_id}.npy'), pre_label)

print(f"\nAll experiments completed. Results saved to: {output_dir}")
