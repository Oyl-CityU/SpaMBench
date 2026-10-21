"""STAGATE example for the 00 human lymph node RNA dataset."""

import os
import time

import numpy as np
import pandas as pd
import scanpy as sc
import STAGATE
import rpy2.robjects as robjects
from rpy2.robjects import default_converter, numpy2ri
from rpy2.robjects.conversion import localconverter


def mclust_R(adata, num_cluster, modelNames='EEE', used_obsm='STAGATE',
             random_seed=2020):
    """Run the Mclust clustering used in the STAGATE example."""
    robjects.r.library('mclust')
    robjects.r['set.seed'](random_seed)
    rmclust = robjects.r['Mclust']
    with localconverter(default_converter + numpy2ri.converter):
        result = rmclust(adata.obsm[used_obsm], num_cluster, modelNames)
    labels = np.asarray(result[-2], dtype=int)
    adata.obs['mclust'] = pd.Categorical(labels)
    return adata


dataset_name = '00_Human_lymph_node_slice1'
file_fold = '../../datasets/00_Human_lymph_node_slice1/'
output_dir = 'results/00_Human_lymph_node_slice1'

adata = sc.read_h5ad(file_fold + 'adata_RNA.h5ad')
adata.var_names_make_unique()
print(adata)

# Normalization and spatial graph settings from the STAGATE example.
sc.pp.highly_variable_genes(adata, flavor='seurat_v3', n_top_genes=3000)
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
STAGATE.Cal_Spatial_Net(adata, rad_cutoff=110)
STAGATE.Stats_Spatial_Net(adata)

n_clusters = len(np.unique(adata.obs['final_annot']))
print('Number of clusters: {}'.format(n_clusters))

os.makedirs(os.path.join(output_dir, 'latent'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'labels'), exist_ok=True)
runtimes = []

for run_id in range(10):
    print('=================== Run {}/10 ================='.format(run_id + 1))

    start_time = time.time()
    adata = STAGATE.train_STAGATE(adata, alpha=0)
    sc.pp.neighbors(adata, use_rep='STAGATE')
    sc.tl.umap(adata)
    runtimes.append(time.time() - start_time)
    adata = mclust_R(adata, n_clusters, used_obsm='STAGATE')

    np.save(os.path.join(output_dir, 'latent', 'latent_{}.npy'.format(run_id)),
            adata.obsm['STAGATE'])
    np.save(os.path.join(output_dir, 'labels', 'labels_{}.npy'.format(run_id)),
            adata.obs['mclust'].to_numpy())
    adata.write_h5ad(os.path.join(output_dir, '{}_{}.h5ad'.format(
        dataset_name, run_id)))

pd.DataFrame({'run': range(10), 'runtime_seconds': runtimes}).to_csv(
    os.path.join(output_dir, 'log_summary.csv'), index=False)
print('All runs completed. Results saved to {}'.format(output_dir))
