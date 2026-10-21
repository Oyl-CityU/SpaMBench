"""SpaMGCN: 00 human lymph node RNA + ADT example.
Run in the method environment: python SpaMGCN_main.py
"""

import os
import sys

sys.path.insert(0, 'MGCN-main')

# Dependencies.
import argparse
import random
from scipy.sparse.csc import csc_matrix
from scipy.sparse.csr import csr_matrix
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
import scanpy as sc
import scipy
import torch
import torch.nn.functional as F
from model.Creat_model import creat_model
from torch.backends import cudnn
from train.train3 import Noise_Cross_Entropy
from utils.misc import load_configs
from utils.preprocess import construct_graph_by_coordinate, transform_adjacent_matrix, preprocess_graph

# 保存结果
output_dir = 'results/00_Human_lymph_node_slice1'
os.makedirs(os.path.join(output_dir, 'latent'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'labels'), exist_ok=True)

def fix_seed(seed):
    #seed = 2023
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False

    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

def build_args():

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', type=str, default="acm")
    parser.add_argument('--seed', type=int, default=20)
    parser.add_argument('--cuda', type=bool, default=True)
    parser.add_argument('--n_input', type=int, default=None)
    parser.add_argument('--n_z', type=int, default=None)
    parser.add_argument('--freedom_degree', type=float, default=1.0)
    parser.add_argument('--epoch', type=int, default=None)
    parser.add_argument('--shuffle', type=bool, default=True)
    parser.add_argument('--sigma', type=float, default=None)
    parser.add_argument('--loss_n', type=float, default=None)
    parser.add_argument('--loss_w', type=float, default=None)
    parser.add_argument('--loss_s', type=float, default=None)
    parser.add_argument('--loss_a', type=float, default=None)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--acc', type=float, default=-1)
    parser.add_argument('--f1', type=float, default=-1)
    args = parser.parse_args([])
    return args

def pca(adata, use_reps=None, n_comps=10):
    """Dimension reduction with PCA algorithm"""

    pca = PCA(n_components=n_comps)
    if use_reps is not None:
        feat_pca = pca.fit_transform(adata.obsm[use_reps])
    else:
        if isinstance(adata.X, csc_matrix) or isinstance(adata.X, csr_matrix):
            feat_pca = pca.fit_transform(adata.X.toarray())
        else:
            feat_pca = pca.fit_transform(adata.X)

    return feat_pca

def clr_normalize_each_cell(adata, inplace=True):
    """Normalize count vector for each cell, i.e. for each row of .X"""

    def seurat_clr(x):
        # TODO: support sparseness
        s = np.sum(np.log1p(x[x > 0]))
        exp = np.exp(s / len(x))
        return np.log1p(x / exp)

    if not inplace:
        adata = adata.copy()

    # apply to dense or sparse matrix, along axis. returns dense matrix
    adata.X = np.apply_along_axis(
        seurat_clr, 1, (adata.X.A if scipy.sparse.issparse(adata.X) else np.array(adata.X))
    )
    return adata

fix_seed(2024)
args = build_args()
args = load_configs(args, "MGCN-main/config/configs.yml")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

file_fold = '../../datasets/00_Human_lymph_node_slice1/'
adata_omics1 = sc.read_h5ad(file_fold + 'adata_RNA.h5ad')
adata_omics2 = sc.read_h5ad(file_fold + 'adata_ADT.h5ad')

adata_omics1.var_names_make_unique()
adata_omics2.var_names_make_unique()

# RNA
sc.pp.filter_genes(adata_omics1, min_cells=10)
sc.pp.highly_variable_genes(adata_omics1, flavor="seurat_v3", n_top_genes=3000)
sc.pp.normalize_total(adata_omics1, target_sum=1e4)
sc.pp.log1p(adata_omics1)
sc.pp.scale(adata_omics1)

adata_omics1 = adata_omics1[:, adata_omics1.var['highly_variable']]
adata_omics1.obsm['feat'] = pca(adata_omics1, n_comps=adata_omics2.n_vars - 1)
# Protein
adata_omics2 = clr_normalize_each_cell(adata_omics2)
sc.pp.scale(adata_omics2)
adata_omics2.obsm['feat'] = pca(adata_omics2, n_comps=adata_omics2.n_vars - 1)

def create_adj(adata):
    cell_position_omics1 = adata.obsm['spatial']
    adj_omics1 = construct_graph_by_coordinate(cell_position_omics1, n_neighbors=3)
    adata.uns['adj_spatial'] = adj_omics1
    adj_spatial_omics1 = adata.uns['adj_spatial']
    adj_spatial_omics1 = transform_adjacent_matrix(adj_spatial_omics1)
    adj_spatial_omics1 = adj_spatial_omics1.toarray()
    adj_spatial_omics1 = adj_spatial_omics1 + adj_spatial_omics1.T
    adj_spatial_omics1 = np.where(adj_spatial_omics1 > 1, 1, adj_spatial_omics1)
    adj = preprocess_graph(adj_spatial_omics1)
    return adj

args.n_input = adata_omics2.n_vars - 1
args.n_input1 = adata_omics2.n_vars - 1
args.n_clusters = 10

print("===========n_clusters:{}=========".format(args.n_clusters))
random_seek = 2024
args.Type = '10x'

args.random_seed = random_seek

args.loss_n = 0.001
args.epoch = 300
args.tool = 'kmeans'

args.n_clusters1 = 10
args.n_clusters2 = 10
adj_train = create_adj(adata_omics1)
adj_train = adj_train.to(device)
features_omics1 = torch.FloatTensor(adata_omics1.obsm['feat'].copy()).to(device)
features_omics2 = torch.FloatTensor(adata_omics2.obsm['feat'].copy()).to(device)

for run_id in range(10):
    print(f"\n=================== Run {run_id+1}/{10} =================")

    model = creat_model('spamgcn', args).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    dense_adj = adj_train.to_dense()
    for epoch in range(args.epoch):
        xh, zh, ah, za, zg, xh2, zh2, ah2, za2, zg2, fea = model(
            features_omics1, features_omics2, adj_train)
        loss_ae = F.mse_loss(xh, features_omics1) + F.mse_loss(xh2, features_omics2)
        loss_w = F.mse_loss(zh, torch.spmm(adj_train, features_omics1)) + \
            F.mse_loss(zh2, torch.spmm(adj_train, features_omics2))
        loss_a = F.binary_cross_entropy_with_logits(fea @ fea.T, dense_adj)
        loss_s = F.mse_loss(zg, za) + F.mse_loss(zg2, za2)
        loss = loss_ae + args.loss_w * loss_w + args.loss_a * loss_a + args.loss_s * loss_s
        if epoch > args.epoch * 0.1:
            loss = loss + args.loss_n * Noise_Cross_Entropy(fea, dense_adj)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print(f'Epoch {epoch}: {loss.item():.6f}', flush=True)
    model.eval()
    with torch.no_grad():
        adata_omics1.obsm['spaMGCN'] = model(
            features_omics1, features_omics2, adj_train)[-1].cpu().numpy()

    adata_omics1.obs['pred'] = KMeans(n_clusters=10, n_init=10).fit_predict(
        adata_omics1.obsm['spaMGCN']).astype(str)

    # Save latent features
    latent_features = adata_omics1.obsm['spaMGCN']
    print(latent_features.shape)
    np.save(os.path.join(output_dir, 'latent', f'latent_{run_id}.npy'), latent_features)

    # Save clustering results
    pre_label = adata_omics1.obs['pred']
    print(pre_label.shape)
    np.save(os.path.join(output_dir, 'labels', f'labels_{run_id}.npy'), pre_label)

print(f"\nAll experiments completed. Results saved to: {output_dir}")
