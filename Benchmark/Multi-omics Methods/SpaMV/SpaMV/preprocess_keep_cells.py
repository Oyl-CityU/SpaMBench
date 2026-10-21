"""Local adaptation of SpaMV preprocess_dc: retain every input cell.
Original model/package files are untouched. Feature filters and HVG defaults remain.
"""
from typing import List, Optional
import anndata
from anndata import AnnData
import numpy as np
import scipy
import scipy.sparse
import sklearn.preprocessing
import sklearn.utils.extmath
import scanpy as sc

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
        seurat_clr, 1, (adata.X.toarray() if scipy.sparse.issparse(adata.X) else np.array(adata.X))
    )
    return adata

def lsi_keep_cells(adata, n_components=20, use_highly_variable=None,
                   random_state=0, key_added='X_lsi', **kwargs):
    # Official TF-IDF / L1 / log1p / randomized-SVD / row standardization.
    # Zero-count denominators are guarded instead of removing cells.
    if use_highly_variable is None:
        use_highly_variable = 'highly_variable' in adata.var
    X = adata[:, adata.var['highly_variable']].X if use_highly_variable else adata.X
    row = np.asarray(X.sum(axis=1)).ravel()
    col = np.asarray(X.sum(axis=0)).ravel()
    inv = np.divide(1., row, out=np.zeros_like(row, dtype=float), where=row!=0)
    idf = np.divide(float(X.shape[0]), col, out=np.zeros_like(col,dtype=float), where=col!=0)
    if scipy.sparse.issparse(X):
        tfidf = X.multiply(inv[:,None]).multiply(idf)
    else:
        tfidf = X * inv[:,None] * idf
    normalized = sklearn.preprocessing.Normalizer(norm='l1').fit_transform(tfidf)
    normalized = np.log1p(normalized * 1e4)
    latent = sklearn.utils.extmath.randomized_svd(normalized, n_components,
                 random_state=random_state, **kwargs)[0]
    latent -= latent.mean(axis=1, keepdims=True)
    std = latent.std(axis=1, ddof=1, keepdims=True)
    latent = np.divide(latent, std, out=np.zeros_like(latent), where=std!=0)
    latent[row==0] = 0  # No ATAC signal: retain the row with a finite zero representation.
    adata.obsm[key_added] = latent[:,1:]

def preprocess_dc_keep_cells(datasets: List[AnnData], omics_names: List[str], prune: bool = True, min_cells: int = 10,
                  min_genes: int = 200, hvg: bool = True, n_top_genes: int = 3000, normalization: bool = True,
                  target_sum: float = 1e4, log1p: bool = True, scale: bool = False):
    '''
    # preprocess step for domain clustering
    '''
    # prune low quality features and spots
    if prune:
        for i in range(len(datasets)):
            if any(substring.lower() in omics_names[i].lower() for substring in
                   ['Transcriptomics', 'Transcriptome', 'RNA', 'Gene']):
                datasets[i].var['mt'] = np.logical_or(datasets[i].var_names.str.startswith('MT-'),
                                                      datasets[i].var_names.str.startswith('mt-'))
                datasets[i].var['rb'] = datasets[i].var_names.str.startswith(('RP', 'Rp', 'rp'))
                sc.pp.calculate_qc_metrics(datasets[i], qc_vars=['mt'], inplace=True)
                # Cell removal disabled: retain even zero-count/100%-MT cells.
                mask_gene = np.logical_and(~datasets[i].var['mt'], ~datasets[i].var['rb'])
                datasets[i] = datasets[i][:, mask_gene].copy()
            sc.pp.filter_genes(datasets[i], min_cells=min_cells)
            # sc.pp.filter_cells(..., min_genes=min_genes) disabled: keep all cells.
    remained_spots = datasets[0].obs_names
    if not remained_spots.is_unique:
        raise ValueError('Duplicate RNA cell IDs')
    n_comps = min(50, datasets[0].n_vars - 1, len(remained_spots) - 1)
    for i in range(1, len(datasets)):
        if not datasets[i].obs_names.is_unique or set(remained_spots) != set(datasets[i].obs_names):
            raise ValueError('Modality cell IDs differ; cannot retain all paired cells')
        n_comps = min(n_comps, datasets[i].n_vars - 1)
    if n_comps < 1:
        raise ValueError('Insufficient features for PCA/LSI after gene filtering')
    for i in range(len(datasets)):
        datasets[i] = datasets[i][remained_spots, :].copy()
        if any(substring.lower() in omics_names[i].lower() for substring in
               ['Transcriptomics', 'Transcriptome', 'RNA', 'Gene']):
            if hvg:
                sc.pp.highly_variable_genes(datasets[i], flavor='seurat_v3', n_top_genes=n_top_genes, subset=False)
            if normalization:
                sc.pp.normalize_total(datasets[i], target_sum=target_sum)
            if hvg:
                datasets[i] = datasets[i][:, datasets[i].var.highly_variable]
            if log1p:
                sc.pp.log1p(datasets[i])
            if scale:
                sc.pp.scale(datasets[i])
            sc.pp.pca(datasets[i], n_comps=n_comps, key_added='embedding')
        elif any(substring.lower() in omics_names[i].lower() for substring in
                 ['Proteomics', 'Proteome', 'ADT', 'Protein']):
            if normalization:
                datasets[i] = clr_normalize_each_cell(datasets[i])
            if scale:
                sc.pp.scale(datasets[i])
            sc.pp.pca(datasets[i], n_comps=n_comps, key_added='embedding')
        elif any(substring.lower() in omics_names[i].lower() for substring in ['Epigenomics', 'Epigenome', 'peaks']):
            lsi_keep_cells(datasets[i], use_highly_variable=False, n_components=n_comps + 1, key_added='embedding')
        elif any(substring.lower() in omics_names[i].lower() for substring in
                 ['Metabolomics', 'Metabolome', 'Metabonomics', 'Metabonome']):
            if normalization:
                sc.pp.normalize_total(datasets[i], target_sum=target_sum)
            if log1p:
                sc.pp.log1p(datasets[i])
            if scale:
                sc.pp.scale(datasets[i])
            sc.pp.pca(datasets[i], n_comps=n_comps, key_added='embedding')
        elif any(substring.lower() in omics_names[i].lower() for substring in ['H3K27ac', 'H3K27me3']):
            if scale:
                sc.pp.scale(datasets[i])
            sc.pp.pca(datasets[i], n_comps=n_comps, key_added='embedding')
        else:
            raise Exception("The preprocess step for " + omics_names[
                i] + " has not been implemented. Please preprocess by your own approach.")
        datasets[i] = anndata.AnnData(datasets[i].obsm['embedding'], obs=datasets[i].obs, obsm=datasets[i].obsm,
                                      uns=datasets[i].uns)
    for adata in datasets:
        if not adata.obs_names.equals(remained_spots) or not np.isfinite(adata.X).all():
            raise ValueError('Cell alignment or finite-value check failed')
    return datasets
