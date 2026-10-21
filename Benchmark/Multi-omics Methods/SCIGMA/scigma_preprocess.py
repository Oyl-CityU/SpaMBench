"""Count-compatible, all-cell adaptation of SCIGMA SPOTS/Stereo/ATAC preprocessing."""
import numpy as np
import scanpy as sc
import scipy.sparse as sp
from sklearn.preprocessing import Normalizer
from sklearn.utils.extmath import randomized_svd


def lsi_counts(adata, n_components=50):
    # preprocessing() already chose 3000 peaks using seurat_v3 on counts.
    # Do not call the original lsi's seurat HVG on counts (expm1 overflow).
    X=adata[:,adata.var['highly_variable']].X
    row=np.asarray(X.sum(axis=1)).ravel(); col=np.asarray(X.sum(axis=0)).ravel()
    inv=np.divide(1.,row,out=np.zeros_like(row,dtype=float),where=row!=0)
    idf=np.divide(float(X.shape[0]),col,out=np.zeros_like(col,dtype=float),where=col!=0)
    tf=X.multiply(inv[:,None]).multiply(idf) if sp.issparse(X) else X*inv[:,None]*idf
    normalized=np.log1p(Normalizer(norm='l1').fit_transform(tf)*1e4)
    n=min(n_components,min(X.shape))
    if n<2:raise ValueError('Insufficient ATAC features for LSI')
    z=randomized_svd(normalized,n_components=n)[0]
    z-=z.mean(axis=1,keepdims=True)
    std=z.std(axis=1,ddof=1,keepdims=True)
    z=np.divide(z,std,out=np.zeros_like(z),where=std!=0)
    z[row==0]=0
    adata.obsm['X_lsi']=z[:,1:]


def preprocessing_keep_cells(rna,other,datatype='SPOTS',n_neighbors=8,feat_neighbors=8,rna_mode='default'):
    from data import clr_normalize_each_cell,build_network,feature_graph
    if datatype not in ['SPOTS','Stereo-CITE-seq','Spatial-ATAC-RNA-seq']:
        raise ValueError('Unsupported datatype: '+datatype)
    ids=rna.obs_names.copy()
    if not ids.is_unique or not other.obs_names.is_unique or set(ids)!=set(other.obs_names):
        raise ValueError('Both modalities must have identical unique cell IDs')
    other=other[ids].copy()
    # sc.pp.filter_cells(..., min_genes=80/200) deliberately disabled.
    sc.pp.filter_genes(rna,min_cells=10)
    atac=datatype=='Spatial-ATAC-RNA-seq'
    if atac:sc.pp.filter_genes(other,min_cells=10)
    if atac:rna.layers['counts']=rna.X.copy()
    if not atac and rna_mode=='default':
        sc.pp.highly_variable_genes(rna,flavor='seurat_v3',n_top_genes=3000)
    sc.pp.normalize_total(rna,target_sum=1e6)
    sc.pp.log1p(rna)
    if datatype=='SPOTS':rna.layers['logcounts']=rna.X.copy()
    sc.pp.scale(rna)
    if atac and rna_mode=='default':
        sc.pp.highly_variable_genes(rna,flavor='seurat_v3',n_top_genes=3000,layer='counts')
    high=rna[:,rna.var['highly_variable']].copy() if rna_mode=='default' else rna.copy()
    requested=50 if atac else other.n_vars-1
    n=min(requested,high.n_obs-1,high.n_vars-1)
    if n<1:raise ValueError('Insufficient RNA features for PCA')
    sc.pp.pca(high,n_comps=n,use_highly_variable=False)
    rna.obsm['feat']=high.obsm['X_pca'].copy()
    if atac:
        sc.pp.highly_variable_genes(other,flavor='seurat_v3',n_top_genes=3000)
        lsi_counts(other,50)  # Always recompute from current counts; no stale X_lsi reuse.
        other.obsm['feat']=other.obsm['X_lsi'].copy()
    else:
        other=clr_normalize_each_cell(other)
        if datatype=='SPOTS':other.layers['logcounts']=other.X.copy()
        sc.pp.scale(other)
        n=min(other.n_vars-1,other.n_obs-1)
        if n<1:raise ValueError('Insufficient ADT features for PCA')
        sc.pp.pca(other,n_comps=n,use_highly_variable=False)
        other.obsm['feat']=other.obsm['X_pca'].copy()
    for a in [rna,other]:
        if not a.obs_names.equals(ids) or not np.isfinite(a.obsm['feat']).all():
            raise ValueError('Cell alignment/nonfinite feature check failed')
        a.uns['adj_spatial']=build_network(a.obsm['spatial'],n_neighbors=n_neighbors)
    g1,g2=feature_graph(rna,other,k=feat_neighbors)
    rna.obsm['adj_feature']=g1;other.obsm['adj_feature']=g2
    rna.uns['benchmark_rna_pca_genes']=high.var_names.to_numpy(dtype=str)
    return {'adata_omics1':rna,'adata_omics2':other}
