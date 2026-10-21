"""SMMGCL: 00 human lymph node RNA + ADT example.
Run in the method environment: python SMMGCL_main.py
"""

import os
import sys

# Dependencies.
from types import SimpleNamespace
from typing import Optional
import argparse
from scipy.sparse import issparse
from scipy.sparse.csc import csc_matrix
from scipy.sparse.csr import csr_matrix
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors, kneighbors_graph
import anndata
import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy
import scipy.sparse as sp
import torch
from model import SMMGCL
from os.path import join
from train import Train
from utils import setup_seed
import matplotlib
import pdb
import sklearn
import warnings

# 保存结果
output_dir = 'results/00_Human_lymph_node_slice1'
os.makedirs(os.path.join(output_dir, 'latent'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'labels'), exist_ok=True)

matplotlib.use('Agg')
warnings.filterwarnings('ignore')

def construct_graph_by_feature(feature, k=20, mode="connectivity", metric="correlation",
                               include_self=False):
    adj_f = kneighbors_graph(feature, k, mode=mode, metric=metric,
                             include_self=include_self)

    return adj_f

def construct_graph_by_coordinate(cell_position, n_neighbors=3):
    adj_s = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(cell_position)
    adj_s = adj_s.kneighbors_graph(cell_position)
    return adj_s

def construct_adjacency_matrix(adj_wave, prunning_one, prunning_two, common_neighbors):
    if prunning_one:
        # Pruning strategy 1
        original_adj_wave = adj_wave.toarray()
        judges_matrix = original_adj_wave == original_adj_wave.T
        np_adj_wave = original_adj_wave * judges_matrix
        adj_wave = sp.csc_matrix(np_adj_wave)
    else:
        # transform the matrix to be symmetric (Instead of Pruning strategy 1)
        np_adj_wave = construct_symmetric_matrix(adj_wave.toarray())
        adj_wave = sp.csc_matrix(np_adj_wave)

    # obtain the adjacency matrix without self-connection
    adj = sp.csc_matrix(np_adj_wave)
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()

    if prunning_two:
        # Pruning strategy 2
        adj = adj.toarray()
        b = np.nonzero(adj)
        rows = b[0]
        cols = b[1]
        dic = {}
        for row, col in zip(rows, cols):
            if row in dic.keys():
                dic[row].append(col)
            else:
                dic[row] = []
                dic[row].append(col)
        for row, col in zip(rows, cols):
            if len(set(dic[row]) & set(dic[col])) < common_neighbors:
                adj[row][col] = 0
        adj = sp.csc_matrix(adj)
        adj.eliminate_zeros()
    adj_hat = construct_adjacency_hat(adj)
    return adj, adj_wave, adj_hat

def construct_adjacency_hat(adj):
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return adj_normalized

def construct_symmetric_matrix(original_matrix):
    result_matrix = np.zeros(original_matrix.shape, dtype=float)
    num = original_matrix.shape[0]
    for i in range(num):
        for j in range(num):
            if original_matrix[i][j] == 0:
                continue
            elif original_matrix[i][j] == 1:
                result_matrix[i][j] = 1
                result_matrix[j][i] = 1
            else:
                print("The value in the original matrix is illegal!")
                pdb.set_trace()
    assert (result_matrix == result_matrix.T).all() == True

    if ~(np.sum(result_matrix, axis=1) > 1).all():
        print("There existing a outlier!")
        pdb.set_trace()

    return result_matrix

def construct_sparse_float_tensor(np_matrix):
    sp_matrix = sp.csc_matrix(np_matrix)
    three_tuple = sparse_to_tuple(sp_matrix)
    sparse_tensor = torch.sparse.FloatTensor(torch.LongTensor(three_tuple[0].T),
                                             torch.FloatTensor(three_tuple[1]),
                                             torch.Size(three_tuple[2]))
    return sparse_tensor

def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape

def graph_construction(feature, spatial, view_no, k, a, prunning_one=True, prunning_two=True,
                       common_neighbors=2):
    adj_s = construct_graph_by_coordinate(spatial, n_neighbors=k)
    adj_f = construct_graph_by_feature(feature, k=k, mode="connectivity", metric="correlation", )
    adj_s = a * adj_s + (1 - a) * adj_f
    adj, adj_wave, adj_hat = construct_adjacency_matrix(
        adj_s, prunning_one, prunning_two, common_neighbors)

    graph_dict = {
        "adj_hat": adj_hat,
        "adj_wave": adj_wave,
    }

    if view_no == 0:
        RNA_graph_dict = graph_dict
        return RNA_graph_dict
    else:
        ADT_graph_dict = graph_dict
        return ADT_graph_dict

def clr_normalize_each_cell(adata, inplace=True):

    def seurat_clr(x):
        # TODO: support sparseness
        s = np.sum(np.log1p(x[x > 0]))
        exp = np.exp(s / len(x))
        return np.log1p(x / exp)

    if not inplace:
        adata = adata.copy()

    adata.X = np.apply_along_axis(
        seurat_clr, 1, (adata.X.toarray() if scipy.sparse.issparse(adata.X) else np.array(adata.X))
    )
    return adata

def normalize(adata, highly_genes=3000, min_genes=100, min_cells=100):
    sc.pp.filter_genes(adata, min_cells=10)
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.scale(adata)
    return adata

def load_datas(adata_omics1, adata_omics2, highly_genes=3000, k=15, a=0.5):

    # if dataset == 'Dataset1_Mouse_Spleen1':
    # else:

    adata_omics1.obsm['spatial'] = np.rot90(np.array(adata_omics1.obsm['spatial'])).T
    adata_omics1.obsm['spatial'][:, 0] = -1 * adata_omics1.obsm['spatial'][:, 0]
    adata_omics2.obsm['spatial'] = np.rot90(np.array(adata_omics2.obsm['spatial'])).T
    adata_omics2.obsm['spatial'][:, 0] = -1 * adata_omics2.obsm['spatial'][:, 0]

    def pca(adata, use_reps=None, n_comps=10):
        pca = PCA(n_components=n_comps)
        if use_reps is not None:
            feat_pca = pca.fit_transform(adata.obsm[use_reps])
        else:
            if isinstance(adata.X, csc_matrix) or isinstance(adata.X, csr_matrix):
                feat_pca = pca.fit_transform(adata.X.toarray())
            else:
                feat_pca = pca.fit_transform(adata.X)

        return feat_pca

    adata_omics1 = normalize(adata_omics1, highly_genes, 0, 10)
    adata_omics2 = clr_normalize_each_cell(adata_omics2)
    sc.pp.scale(adata_omics2)
    feat1 = pca(adata_omics1, n_comps=50)

    if issparse(feat1):
        feat = feat1.toarray()
    else:
        feat = feat1
    adata1 = ad.AnnData(pd.DataFrame(feat, dtype=np.float32))
    adata1.var_names_make_unique()
    adata1.obsm['spatial'] = adata_omics1.obsm['spatial']
    adata1.var_names_make_unique()

    if issparse(adata_omics2.X):
        feat = adata_omics2.X.toarray()
    else:
        feat = adata_omics2.X
    adata2 = ad.AnnData(
        pd.DataFrame(feat, index=adata_omics2.obs.index, columns=np.array(adata_omics2.var.index), dtype=np.float32))
    adata2.var_names_make_unique()
    adata2.obsm['spatial'] = adata_omics2.obsm['spatial']
    adata2.var_names_make_unique()

    RNA_graph_dict = graph_construction(adata1.X, adata1.obsm['spatial'], 0, k, a)
    ADT_graph_dict = graph_construction(adata2.X, adata2.obsm['spatial'], 1, k, a)

    return adata1, adata2, RNA_graph_dict, ADT_graph_dict

def construct_sparse_float_tensor(np_matrix):
    sp_matrix = sp.csc_matrix(np_matrix)
    three_tuple = sparse_to_tuple(sp_matrix)
    sparse_tensor = torch.sparse.FloatTensor(torch.LongTensor(three_tuple[0].T),
                                             torch.FloatTensor(three_tuple[1]),
                                             torch.Size(three_tuple[2]))
    return sparse_tensor

def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape

def loaddata(adata_omics1, adata_omics2, RNA_graph_dict, ADT_graph_dict):
    print("load dataset")
    # path = "../generate_data/" + dataset + "/"

    feature_list = []
    spatial_list = []
    adj_wave_list = []
    adj_hat_list = []
#################################################
    feature = torch.from_numpy(adata_omics1.X).float()
    feature_list.append(feature)
    spatial_list.append(adata_omics1.obsm['spatial'])
    # graph_dict = np.load(path + str(0) + "_graph_dict.npy", allow_pickle=True).tolist()
    graph_dict = RNA_graph_dict
    adj_hat = graph_dict["adj_hat"]
    adj_wave = graph_dict["adj_wave"]
    adj_hat = construct_sparse_float_tensor(adj_hat)
    adj_hat_list.append(adj_hat)
    adj_wave = construct_sparse_float_tensor(adj_wave)
    adj_wave_list.append(adj_wave)
########################################
    feature = torch.from_numpy(adata_omics2.X).float()
    feature_list.append(feature)
    spatial_list.append(adata_omics2.obsm['spatial'])
    # graph_dict = np.load(path + str(1) + "_graph_dict.npy", allow_pickle=True).tolist()
    graph_dict = ADT_graph_dict
    adj_hat = graph_dict["adj_hat"]
    adj_wave = graph_dict["adj_wave"]
    adj_hat = construct_sparse_float_tensor(adj_hat)
    adj_hat_list.append(adj_hat)
    adj_wave = construct_sparse_float_tensor(adj_wave)
    adj_wave_list.append(adj_wave)
    return adata_omics1, adata_omics2, feature_list, spatial_list, adj_wave_list, adj_hat_list,

args = SimpleNamespace(seed=100, weight_decay=1e-5, optimizer='RMSprop', lr=1e-5,
                       cuda=torch.cuda.is_available(), cuda_device='0', dataset='00_Human_lymph_node_slice1',
                       fusion_type='att', rg_weight=0.1, cl_weight=0.01, con_weight=0.01,
                       num_epochs=300, num_clusters=10, hidden_dims=[32])

file_fold = '../../datasets/00_Human_lymph_node_slice1/'
adata_omics1 = sc.read_h5ad(file_fold + 'adata_RNA.h5ad')
adata_omics2 = sc.read_h5ad(file_fold + 'adata_ADT.h5ad')
adata_omics1.var_names_make_unique()
adata_omics2.var_names_make_unique()
adata_omics2 = adata_omics2[adata_omics1.obs_names].copy()

args.num_clusters = 10
print("=========n_clusters:{}=============".format(args.num_clusters))

setup_seed(args.seed)
adata1, adata2, RNA_graph_dict, ADT_graph_dict = load_datas(
    adata_omics1, adata_omics2, highly_genes=3000, k=9, a=0.3)

adata1, adata2, feature_list, spatial_list, adj_wave_list, adj_hat_list = loaddata(
    adata1, adata2, RNA_graph_dict, ADT_graph_dict)

args.n = feature_list[0].shape[0]
args.num_views = len(feature_list)
args.view_dims = []
for j in range(args.num_views):
    args.view_dims.append(feature_list[j].shape[1])

for run_id in range(10):
    print(f"\n=================== Run {run_id+1}/{10} =================")

    model = SMMGCL(args.view_dims, args.hidden_dims, args.num_clusters, args.fusion_type)

    embedding = Train(model=model,
                      feature_list=feature_list.copy(),
                      adj_hat_list=adj_hat_list.copy(),
                      adj_wave_list=adj_wave_list.copy(), args=args)
    adata1.obsm['embedding'] = embedding
    adata2.obsm['embedding'] = embedding

    kmeans = KMeans(n_clusters=args.num_clusters, n_init=5)
    y_pred = kmeans.fit_predict(embedding)

    adata1.obs['y_pred'] = y_pred.astype(str)
    adata2.obs['y_pred'] = y_pred.astype(str)
    print("===finish===")

    # 保存结果

    # Save latent features
    latent_features = embedding
    np.save(os.path.join(output_dir, 'latent', f'latent_{run_id}.npy'), latent_features)
    pre_label = y_pred
    np.save(os.path.join(output_dir, 'labels', f'labels_{run_id}.npy'), pre_label)

print(f"\nAll experiments completed. Results saved to: {output_dir}")
