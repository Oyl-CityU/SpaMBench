"""CANDIES: 05 MISAR-E15 RNA + ATAC example.
Run in the method environment: python CANDIES_RNA_ATAC.py
"""

import os
import sys

sys.path.insert(0, 'CANDIES')

# Dependencies.
import numpy as np
import pandas as pd
import scanpy as sc
import torch
from codes.AutoEncoder import train_atac
from codes.DiTs import seed_everything
from codes.ZINB_encoder import encoder_ZINB
from codes.get_graph import construct_neighbor_graph, adjacent_matrix_preprocessing
from codes.integration import train_and_infer
from codes.preprocess1 import pca, lsi
from codes.train_diff import run_diff

# 保存结果
output_dir = 'results/05_MISAR_E15'
os.makedirs(os.path.join(output_dir, 'latent'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'labels'), exist_ok=True)

# read data

file_fold = '../../datasets/05_MISAR_E15/'
adata_omics1 = sc.read_h5ad(file_fold + 'adata_RNA.h5ad')
adata_omics2 = sc.read_h5ad(file_fold + 'adata_ATAC.h5ad')
adata_omics1.var_names_make_unique()
adata_omics2.var_names_make_unique()
adata_omics2 = adata_omics2[adata_omics1.obs_names].copy()

n_clusters = 12
print("===========n_clusters:{}=========".format(n_clusters))

def run_leiden(adata1, n_cluster, use_rep="embeddings", key_added="Nleiden", range_min=0, range_max=3, max_steps=30, tolerance=0):
    adata = adata1.copy()
    sc.pp.neighbors(adata, use_rep=use_rep)
    this_step = 0
    this_min = float(range_min)
    this_max = float(range_max)
    while this_step < max_steps:
        this_resolution = this_min + ((this_max - this_min) / 2)
        sc.tl.leiden(adata, resolution=this_resolution)
        this_clusters = adata.obs['leiden'].nunique()

        if this_clusters > n_cluster + tolerance:
            this_max = this_resolution
        elif this_clusters < n_cluster - tolerance:
            this_min = this_resolution
        else:
            print("Succeed to find %d clusters at resolution %.3f" % (n_cluster, this_resolution))
            adata1.obs[key_added] = adata.obs["leiden"]

            return adata1

        this_step += 1

    print('Cannot find the number of clusters')
    adata1.obs[key_added] = adata.obs["leiden"]
    return adata1

# Data Processing
# RNA
sc.pp.filter_genes(adata_omics1, min_cells=10)
sc.pp.highly_variable_genes(adata_omics1, flavor="seurat_v3", n_top_genes=3000)
sc.pp.normalize_total(adata_omics1, target_sum=1e4)
sc.pp.log1p(adata_omics1)
sc.pp.scale(adata_omics1)

adata_omics1_high = adata_omics1[:, adata_omics1.var['highly_variable']]
adata_omics1.obsm['feat'] = pca(adata_omics1_high, n_comps=50)

# ATAC: follow the original RNA-ATAC pipeline with TF-IDF/LSI.
if 'X_lsi' not in adata_omics2.obsm:
    sc.pp.highly_variable_genes(adata_omics2, flavor="seurat_v3", n_top_genes=3000)
    lsi(adata_omics2, use_highly_variable=False, n_components=51)
adata_omics2.obsm['feat'] = adata_omics2.obsm['X_lsi'].copy()
adata_omics1_high = adata_omics1[:, adata_omics1.var['highly_variable']]
adata_omics1.obsm['feat'] = pca(
    adata_omics1_high, n_comps=adata_omics2.obsm['X_lsi'].shape[1])

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

adata_omics1, adata_omics2 = construct_neighbor_graph(adata_omics1, adata_omics2)

adj = adjacent_matrix_preprocessing(adata_omics1, adata_omics2)
adj_spatial_omics1 = adj['adj_spatial_omics1'].to(device)
adj_spatial_omics2 = adj['adj_spatial_omics2'].to(device)

ae_model = encoder_ZINB(
    adata=adata_omics1,
    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
    epochs=300,
    dim_output=64
)

adata_omics1.obsm['emb_ZINB'], adj_mat = ae_model.train()

train_atac(adata_omics2, adj_spatial_omics2, epochs=600)

# Denoise the RNA representation using the ATAC representation as condition.
emb_latent_omics1 = adata_omics1.obsm['emb_ZINB']
emb_latent_omics2 = adata_omics2.obsm['emb_latent_omics2']

df_omics1 = pd.DataFrame(emb_latent_omics1, index=adata_omics1.obs_names)
df_omics2 = pd.DataFrame(emb_latent_omics2, index=adata_omics2.obs_names)
aligned_emb_latent_omics1 = df_omics1.to_numpy()
aligned_emb_latent_omics2 = df_omics2.reindex(df_omics1.index).to_numpy()

class ConditionalDiffusionDataset():
    def __init__(self, adata_omics1, adata_omics2):
        self.adata_omics1 = adata_omics1
        self.adata_omics2 = adata_omics2

        self.st_sample = torch.tensor(self.adata_omics1, dtype=torch.float32)
        self.con_sample = torch.tensor(self.adata_omics2, dtype=torch.float32)
        self.con_data = torch.tensor(self.adata_omics2, dtype=torch.float32)

    def __len__(self):
        return len(self.adata_omics1)

    def __getitem__(self, idx):
        return self.st_sample[idx], self.con_sample[idx], self.con_data

# denoise the first modality, condition on the second modality.
dataset = ConditionalDiffusionDataset(aligned_emb_latent_omics1, aligned_emb_latent_omics2)

com_mtx = run_diff(dataset, k=3, batch_size=512, hidden_size=256, learning_rate=1e-3, num_epoch=1000, diffusion_step=800, depth=6, head=16, pca_dim=50,
                   device=device, classes=6, patience=40, bias=0.5)
adata_omics1.obsm['denoise_emb'] = com_mtx

# Integration phase

adata1 = adata_omics1.copy()
adata2 = adata_omics2.copy()

adata2 = adata2[adata1.obs_names].copy()

data_type = 'Spatial-epigenome-transcriptome'

adata1.obsm['feat'] = adata1.obsm['denoise_emb']
adata2.obsm['feat'] = adata2.obsm['emb_latent_omics2']

adata1, adata2 = construct_neighbor_graph(adata1, adata2)

adj = adjacent_matrix_preprocessing(adata1, adata2)
adj_spatial_omics1 = adj['adj_spatial_omics1'].to(device)
adj_spatial_omics2 = adj['adj_spatial_omics2'].to(device)
adj_feature_omics1 = adj['adj_feature_omics1'].to(device)
adj_feature_omics2 = adj['adj_feature_omics2'].to(device)

features_omics1 = torch.FloatTensor(adata1.obsm['feat'].copy()).to(device)
features_omics2 = torch.FloatTensor(adata2.obsm['feat'].copy()).to(device)

# Set the integration experiment seed once after shared learned preprocessing.
seed = 2025
seed_everything(seed)

for run_id in range(10):
    print(f"\n=================== Run {run_id+1}/{10} =================")

    result = train_and_infer(
        features_omics1=features_omics1,
        features_omics2=features_omics2,
        adj_spatial_omics1=adj_spatial_omics1,
        adj_feature_omics1=adj_feature_omics1,
        adj_spatial_omics2=adj_spatial_omics2,
        adj_feature_omics2=adj_feature_omics2,
        device=device,
        epochs=200
    )

    adata = adata1.copy()
    adata.obsm['CANDIES'] = result['emb_latent_combined'].detach().cpu().numpy().copy()

    adata = run_leiden(adata, n_cluster=n_clusters, use_rep="CANDIES", key_added="CANDIES")

    # Save latent features
    latent_features = adata.obsm['CANDIES']
    np.save(os.path.join(output_dir, 'latent', f'latent_{run_id}.npy'), latent_features)

    # Save clustering results
    pre_label = adata.obs["CANDIES"]
    np.save(os.path.join(output_dir, 'labels', f'labels_{run_id}.npy'), pre_label)

print(f"\nAll experiments completed. Results saved to: {output_dir}")
