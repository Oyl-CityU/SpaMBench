"""SpaMosaic: 00 human lymph node RNA + ADT example.
Run in the method environment: python SpaMosaic_main.py
"""

import os
import sys

# Dependencies.
import numpy as np
import pandas as pd
import scanpy as sc
from os.path import join
from rpy2 import robjects
from rpy2.robjects import default_converter, numpy2ri
from rpy2.robjects.conversion import localconverter
from spamosaic.framework import SpaMosaic
from spamosaic.preprocessing import RNA_preprocess, ADT_preprocess
import spamosaic.framework as framework
import spamosaic.utils as cluster_module
import spamosaic.utils as utls


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

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'  # for CuBLAS operation and you have CUDA >= 10.2

file_fold = '../../datasets/00_Human_lymph_node_slice1/'
ad1_rna = sc.read_h5ad(file_fold + 'adata_RNA.h5ad')
ad1_adt = sc.read_h5ad(file_fold + 'adata_ADT.h5ad')
ad1_rna.var_names_make_unique()
ad1_adt.var_names_make_unique()
ad1_adt = ad1_adt[ad1_rna.obs_names].copy()

n_clusters = 10
# n_clusters = 10
print("=========n_clusters:{}=============".format(n_clusters))
print(ad1_rna)
print(ad1_adt)

ad1_rna.obs['src'] = '00'
ad1_adt.obs['src'] = '00'
framework.set_seeds(1234)
input_dict = {
    'rna': [ad1_rna],
    'adt': [ad1_adt]
}

input_key = 'dimred_bc'
RNA_preprocess(input_dict['rna'], batch_corr=True, favor='scanpy',
               n_hvg=5000, batch_key='src', key=input_key)
ADT_preprocess(input_dict['adt'], batch_corr=True, batch_key='src', key=input_key)

for run_id in range(10):
    print(f"\n=================== Run {run_id+1}/{10} =================")

    original_seed = framework.set_seeds
    framework.set_seeds = lambda _: None
    model = SpaMosaic(
        modBatch_dict=input_dict, input_key=input_key,
        batch_key='src', intra_knn=2, inter_knn=2, w_g=0.8,
        seed=1234,
        device='cuda:0'
    )

    framework.set_seeds = original_seed
    model.train(net='wlgcn', lr=0.01, T=0.01, n_epochs=100)

    ad_embs = model.infer_emb(input_dict, emb_key='emb', final_latent_key='merged_emb')
    ad_mosaic = sc.concat(ad_embs)
    ad_mosaic = utls.get_umap(ad_mosaic, use_reps=['merged_emb'])

    utls.clustering(ad_mosaic, n_cluster=n_clusters,
                    used_obsm='merged_emb', algo='mclust', key='mclust')
    utls.split_adata_ob(ad_embs, ad_mosaic, 'obs', 'mclust')

    # 确保输出目录存在

    # Save latent features
    latent_features = ad_mosaic.obsm['merged_emb']
    np.save(os.path.join(output_dir, 'latent', f'latent_{run_id}.npy'), latent_features)

    # Save this method's clustering labels.
    pre_label = ad_mosaic.obs['mclust']
    np.save(os.path.join(output_dir, 'labels', f'labels_{run_id}.npy'), pre_label)

print(f"\nAll experiments completed. Results saved to: {output_dir}")
