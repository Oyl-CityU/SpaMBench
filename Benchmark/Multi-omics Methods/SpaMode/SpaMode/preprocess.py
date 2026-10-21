import os
import scipy
import anndata
import sklearn
import torch
import random
import numpy as np
import scanpy as sc
import pandas as pd
from typing import Optional
import scipy.sparse as sp
from torch.backends import cudnn
from scipy.sparse import coo_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import kneighbors_graph


def normalize_coordinates(coords):
    if isinstance(coords, np.ndarray):
        coords = torch.from_numpy(coords).requires_grad_(False)

    min_coords = torch.min(coords, dim=0).values  # (x_min, y_min)
    max_coords = torch.max(coords, dim=0).values  # (x_max, y_max)

    normalized_coords = (coords - min_coords) / (max_coords - min_coords)

    return normalized_coords

def construct_neighbor_graph_SpaMode(adata_omics_list, datatype='SPOTS', Spatial_neighbors=3, Feature_neighbors=20):

    print('Spatial_neighbors =', Spatial_neighbors)
    print('Feature_neighbors =', Feature_neighbors)

    for adata in adata_omics_list:
        cell_position = adata.obsm['spatial']
        adj_spatial = construct_graph_by_coordinate(cell_position, n_neighbors=Spatial_neighbors)
        adata.uns['adj_spatial'] = adj_spatial
        adj_feat = kneighbors_graph(adata.obsm['feat'], Feature_neighbors, mode="distance", metric="correlation",
                                    include_self=False)
        adata.obsm['adj_feature'] = adj_feat

    return adata_omics_list

def construct_neighbor_graph(adata_omics1, adata_omics2, datatype='SPOTS', Spatial_neighbors=0, Feature_neighbors=20):
    """
    Construct neighbor graphs, including feature graph and spatial graph.
    Feature graph is based expression data while spatial graph is based on cell/spot spatial coordinates.

    Parameters
    ----------
    n_neighbors : int
        Number of neighbors.

    Returns
    -------
    data : dict
        AnnData objects with preprossed data for different omics.

    """

    # construct spatial neighbor graphs
    ################# spatial graph #################
    if Spatial_neighbors==0:
        if datatype in ['Stereo-CITE-seq', 'Spatial-epigenome-transcriptome']:
            n_neighbors = 6
        elif datatype == 'Sim_L':
            n_neighbors = 4
        else:
            n_neighbors = 3
    else:
        n_neighbors=Spatial_neighbors
    # omics1

    print('Spatial_neighbors =', n_neighbors)
    print('Feature_neighbors =', Feature_neighbors)

    if adata_omics1:
        cell_position_omics1 = adata_omics1.obsm['spatial']
        adj_omics1 = construct_graph_by_coordinate(cell_position_omics1, n_neighbors=n_neighbors)
        adata_omics1.uns['adj_spatial'] = adj_omics1

    # omics2
    if adata_omics2:
        cell_position_omics2 = adata_omics2.obsm['spatial']
        adj_omics2 = construct_graph_by_coordinate(cell_position_omics2, n_neighbors=n_neighbors)
        adata_omics2.uns['adj_spatial'] = adj_omics2

    ################# feature graph #################
    if datatype == 'Multi':
        feature_graph_omics1, feature_graph_omics2 = construct_graph_by_feature(adata_omics1, adata_omics2, k=60)
    else:
        feature_graph_omics1, feature_graph_omics2 = construct_graph_by_feature(adata_omics1, adata_omics2, k=Feature_neighbors)
    if adata_omics1:
        adata_omics1.obsm['adj_feature'] = feature_graph_omics1
    # omics2
    if adata_omics2:
        adata_omics2.obsm['adj_feature'] = feature_graph_omics2

    data = {'adata_omics1': adata_omics1, 'adata_omics2': adata_omics2}

    return data


def pca(adata, use_reps=None, n_comps=10):
    """Dimension reduction with PCA algorithm"""

    from sklearn.decomposition import PCA
    from scipy.sparse.csc import csc_matrix
    from scipy.sparse.csr import csr_matrix
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

    import numpy as np
    import scipy

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


def construct_graph_by_feature(adata_omics1, adata_omics2, k=20, mode="distance", metric="correlation",
                               include_self=False):
    """Constructing feature neighbor graph according to expresss profiles"""
    if adata_omics1:
        feature_graph_omics1 = kneighbors_graph(adata_omics1.obsm['feat'], k, mode=mode, metric=metric,
                                                include_self=include_self)
    else:
        feature_graph_omics1 = None
    if adata_omics2:
        feature_graph_omics2 = kneighbors_graph(adata_omics2.obsm['feat'], k, mode=mode, metric=metric,
                                                include_self=include_self)
    else:
        feature_graph_omics2 = None

    # print(adata_omics1.obsm['feat'].shape)
    # print(feature_graph_omics1.shape)
    # assert 0

    return feature_graph_omics1, feature_graph_omics2


def construct_graph_by_coordinate(cell_position, n_neighbors=3):
    """Constructing spatial neighbor graph according to spatial coordinates."""
    nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(cell_position)
    _, indices = nbrs.kneighbors(cell_position)
    # print(indices.shape)
    # assert 0
    x = indices[:, 0].repeat(n_neighbors)
    y = indices[:, 1:].flatten()
    adj = pd.DataFrame(columns=['x', 'y', 'value'])
    adj['x'] = x
    adj['y'] = y
    adj['value'] = np.ones(x.size)
    return adj


def transform_adjacent_matrix(adjacent):
    n_spot = adjacent['x'].max() + 1
    # print(max(adjacent['x']))  # 查看最大的列索引
    # print(max(adjacent['y']))  # 查看最大的列索引
    # print(n_spot)  # 查看矩阵的列维度
    # assert 0
    adj = coo_matrix((adjacent['value'], (adjacent['x'], adjacent['y'])), shape=(n_spot, n_spot))
    return adj


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""

    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


# ====== Graph preprocessing
def preprocess_graph(adj):
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return sparse_mx_to_torch_sparse_tensor(adj_normalized)


def _process_adj(adata_omics1, feat_key='adj_feature'):
    if adata_omics1:
        adj_spatial_omics1 = adata_omics1.uns['adj_spatial']
        adj_spatial_omics1 = transform_adjacent_matrix(adj_spatial_omics1)
        adj_spatial_omics1 = adj_spatial_omics1.toarray()  # To ensure that adjacent matrix is symmetric
        adj_spatial_omics1 = adj_spatial_omics1 + adj_spatial_omics1.T
        adj_spatial_omics1 = np.where(adj_spatial_omics1 > 1, 1, adj_spatial_omics1)
        adj_spatial_omics1 = preprocess_graph(
            adj_spatial_omics1)  # sparse adjacent matrix corresponding to spatial graph
        adj_feature_omics1 = torch.FloatTensor(adata_omics1.obsm[feat_key].copy().toarray())
        adj_feature_omics1 = adj_feature_omics1 + adj_feature_omics1.T
        adj_feature_omics1 = np.where(adj_feature_omics1 > 1, 1, adj_feature_omics1)
        adj_feature_omics1 = preprocess_graph(
            adj_feature_omics1)  # sparse adjacent matrix corresponding to feature graph
    else:
        adj_spatial_omics1 = None
        adj_feature_omics1 = None
    return adj_spatial_omics1, adj_feature_omics1


def adjacent_matrix_preprocessing(adata_omics1, adata_omics2, key='adj_feature'):
    """Converting dense adjacent matrix to sparse adjacent matrix"""

    ######################################## construct spatial graph ########################################
    adj_spatial_omics1, adj_feature_omics1 = _process_adj(adata_omics1, feat_key=key)
    adj_spatial_omics2, adj_feature_omics2 = _process_adj(adata_omics2, feat_key=key)

    adj = {'adj_spatial_omics1': adj_spatial_omics1,
           'adj_spatial_omics2': adj_spatial_omics2,
           'adj_feature_omics1': adj_feature_omics1,
           'adj_feature_omics2': adj_feature_omics2,
           }

    return adj


def lsi(
        adata: anndata.AnnData, n_components: int = 20,
        use_highly_variable: Optional[bool] = None, **kwargs
) -> None:
    r"""
    LSI analysis (following the Seurat v3 approach)
    """
    if use_highly_variable is None:
        use_highly_variable = "highly_variable" in adata.var
    adata_use = adata[:, adata.var["highly_variable"]] if use_highly_variable else adata
    X = tfidf(adata_use.X)
    # X = adata_use.X
    X_norm = sklearn.preprocessing.Normalizer(norm="l1").fit_transform(X)
    X_norm = np.log1p(X_norm * 1e4)
    X_lsi = sklearn.utils.extmath.randomized_svd(X_norm, n_components, **kwargs)[0]
    X_lsi -= X_lsi.mean(axis=1, keepdims=True)
    X_lsi /= X_lsi.std(axis=1, ddof=1, keepdims=True)
    # adata.obsm["X_lsi"] = X_lsi
    adata.obsm["X_lsi"] = X_lsi[:, 1:]


def tfidf(X):
    r"""
    TF-IDF normalization (following the Seurat v3 approach)
    """
    idf = X.shape[0] / X.sum(axis=0)
    if scipy.sparse.issparse(X):
        tf = X.multiply(1 / X.sum(axis=1))
        return tf.multiply(idf)
    else:
        tf = X / X.sum(axis=1, keepdims=True)
        return tf * idf


def fix_seed(seed):
    # seed = 2023
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
