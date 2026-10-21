"""SpatialCOC: 05 MISAR-E15 RNA + ATAC example.
Run in the method environment: python SpatialCOC_RNA_ATAC.py
"""

import os
import sys

# Dependencies.
from scipy.sparse import csr_matrix
import numpy as np
import pandas as pd
import scanpy as sc
import torch
from Model.INR import INRModel
from Model.model import DCCAE
from Model.preprocess import fix_seed, preprocessing
from Model.utils import mclust_R
from rpy2 import robjects
from rpy2.robjects import default_converter, numpy2ri
from rpy2.robjects.conversion import localconverter
import Model.utils as cluster_module
import warnings

# R compatibility only: preserve the method's PCA and mclust EEE settings.

def mclust_R(adata, num_cluster, modelNames='EEE', used_obsm='emb', random_seed=2024):
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

warnings.filterwarnings('ignore')

fix_seed(2024)

file_fold = '../../datasets/05_MISAR_E15/'
adata_modality_1 = sc.read_h5ad(file_fold + "adata_RNA.h5ad")
adata_modality_2 = sc.read_h5ad(file_fold + "adata_ATAC.h5ad")
adata_modality_1.var_names_make_unique()
adata_modality_2.var_names_make_unique()
adata_modality_2 = adata_modality_2[adata_modality_1.obs_names].copy()

n_clusters = 12
print("===========n_clusters:{}=========".format(n_clusters))

sc.pp.filter_genes(adata_modality_1, min_cells=10)
sc.pp.highly_variable_genes(adata_modality_1, flavor='seurat_v3', n_top_genes=3000, subset=True)
adata_modality_1, adata_modality_2 = preprocessing(
    adata_modality_1, adata_modality_2, 'Spatial-epigenome-transcriptome')
# INR module training

def Spatially_Continuous_Reconstruction(adata, epoch_num):
    coords = adata.obsm['spatial'].astype(np.float32)
    node_feats = adata.X.astype(np.float32)

    coords = torch.from_numpy(coords).float()
    if isinstance(node_feats, csr_matrix):
        node_feats = node_feats.toarray()
    node_feats = torch.from_numpy(node_feats).float()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    node_feats = node_feats.to(device)
    coords = coords.to(device)

    print(f'out_dim: {node_feats.shape[1]}')

    model = INRModel(
        X=node_feats,
        spatial_coord=coords,
        device=device,
        learning_rate=1e-4,
        reg_par=0,
        epoch_num=epoch_num,
        print_train_log_info=True
    )

    reconstructed_X = model.fit()
    adata.uns['INR'] = reconstructed_X

# Spatial reconstruction (shared preprocessing before ten fusion runs).
Spatially_Continuous_Reconstruction(adata_modality_1, 1500)
# modality 2
Spatially_Continuous_Reconstruction(adata_modality_2, 200)
adata_modality_1.X = adata_modality_1.uns['INR']
sc.pp.pca(adata_modality_1)
adata_modality_2.obsm['X_pca'] = np.asarray(adata_modality_2.uns['INR'])

for run_id in range(10):
    print(f"\n=================== Run {run_id+1}/{10} =================")
    # Train
    n_output = 30
    features1 = adata_modality_1.obsm['X_pca'].shape[1]  # Feature sizes
    features2 = adata_modality_2.obsm['X_pca'].shape[1]
    layers1 = [256, 256, n_output]  # nodes in each hidden layer and the output size
    layers2 = [256, 256, n_output]

    X = np.ascontiguousarray(adata_modality_1.obsm['X_pca'])
    Y = np.ascontiguousarray(adata_modality_2.obsm['X_pca'])

    use_rep = ['DCCAE_X', "DCCAE_Y", "DCCAE"]
    n_DCCAE = 10
    epochs = 60
    model = DCCAE(input_size1=features1, input_size2=features2, n_components=n_DCCAE,
                  layer_sizes1=layers1, layer_sizes2=layers2, epoch_num=epochs)
    model.fit([X, Y])
    Xs_transformed = model.transform([X, Y])
    adata_modality_1.obsm["DCCAE_X"] = Xs_transformed[0]
    adata_modality_2.obsm["DCCAE_Y"] = Xs_transformed[1]
    adata_modality_1.obsm["DCCAE"] = np.concatenate(
        (adata_modality_1.obsm["DCCAE_X"], adata_modality_2.obsm["DCCAE_Y"]), axis=1)

    use_rep = ['DCCAE_X', 'DCCAE_Y', 'DCCAE']
    mclust_R(adata_modality_1, used_obsm=use_rep[2], num_cluster=n_clusters)
    print(adata_modality_1)

    # Save latent features
    latent_features = adata_modality_1.obsm["DCCAE"]
    np.save(os.path.join(output_dir, 'latent', f'latent_{run_id}.npy'), latent_features)
    # Save clustering results
    pre_label = adata_modality_1.obs["clusters_mclust"]
    np.save(os.path.join(output_dir, 'labels', f'labels_{run_id}.npy'), pre_label)

print(f"\nAll experiments completed. Results saved to: {output_dir}")
