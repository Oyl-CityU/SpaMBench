"""SpaMultiVAE: 05 MISAR-E15 RNA + ATAC example.
Run in the method environment: python SpaMultiVAE_RNA_ATAC.py
"""

import os
import sys

sys.path.insert(0, 'src/spaMultiVAE')

# Dependencies.
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse
import torch
from os.path import join
from preprocess import normalize, geneSelection
from spaMultiVAE import SPAMULTIVAE

def preprocess_spatial_atac(adata, min_spots_ratio=0.01, top_n_variable=30000):
    """
    Preprocess spatial ATAC-seq data:
    1. Binarize counts (set >0 to 1)
    2. Filter peaks present in >min_spots_ratio of spots
    3. Select top_n_variable most variable peaks
    
    Parameters:
        adata (AnnData): AnnData object with peaks x spots count matrix
        min_spots_ratio (float): Minimum fraction of spots where a peak must be present (default: 0.01)
        top_n_variable (int): Number of top variable peaks to keep (default: 30000)
    
    Returns:
        AnnData: Processed AnnData object
    """
    # 1. Binarize the matrix (set all counts >0 to 1)
    adata.X = (adata.X > 0).astype(np.int8)
    
    # 2. Filter peaks present in >min_spots_ratio of spots
    min_spots = min_spots_ratio * adata.n_obs
    peak_presence = np.array((adata.X > 0).sum(axis=0)).flatten()  # Sum across spots
    mask_peaks = (peak_presence > min_spots)
    adata = adata[:, mask_peaks].copy()
    print(f"After min_spots filtering: {adata.n_vars} peaks remaining")
    
    # 3. Select top variable peaks (based on presence/absence variance)
    # Note: For binary data, variance = p*(1-p), where p is presence rate
    presence_rate = np.array(adata.X.mean(axis=0)).flatten()
    peak_variance = presence_rate * (1 - presence_rate)
    
    # Get indices of top N most variable peaks
    top_var_indices = np.argsort(peak_variance)[-top_n_variable:]
    adata = adata[:, top_var_indices].copy()
    print(f"After variable peak selection: {adata.n_vars} peaks remaining")
    
    return adata


# 保存结果
output_dir = 'results/05_MISAR_E15'
os.makedirs(os.path.join(output_dir, 'latent'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'labels'), exist_ok=True)

'''
Parameter setting
'''

class Args(object):
    def __init__(self):
        self.data_file = '00_data'
        self.select_genes = 0
        self.select_proteins = 0
        self.batch_size = "auto"
        self.maxiter = 5000
        self.train_size = 0.95
        self.patience = 200
        self.lr = 5e-3
        self.weight_decay = 1e-6
        self.gene_noise = 0
        self.protein_noise = 0
        self.dropoutE = 0
        self.dropoutD = 0
        self.encoder_layers = [128, 64]
        self.GP_dim = 2
        self.Normal_dim = 18
        self.gene_decoder_layers = [128]
        self.protein_decoder_layers = [128]
        self.init_beta = 10
        self.min_beta = 4
        self.max_beta = 25
        self.KL_loss = 0.025
        self.num_samples = 1
        self.fix_inducing_points = True
        self.inducing_point_steps = 19
        self.fixed_gp_params = False
        self.loc_range = 20.
        self.kernel_scale = 20.
        self.model_file = "model.pt"
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

args = Args()

file_fold = '../../datasets/05_MISAR_E15/'
args.data_file = file_fold

os.makedirs(output_dir, exist_ok=True)
args.model_file = output_dir + "/model.pt"
print(args.model_file)

data1 = sc.read_h5ad(file_fold + "adata_RNA.h5ad")
data2 = sc.read_h5ad(file_fold + "adata_ATAC.h5ad")
data1.var_names_make_unique()
data2.var_names_make_unique()
data2 = data2[data1.obs_names].copy()
data2 = preprocess_spatial_atac(data2)

# 提取spatkX提取的3000高变基因
top_3000_svgs_path = "05_top3000_genes_pvalues.csv"
top_3000_svgs = pd.read_csv(top_3000_svgs_path, index_col=0)
top_genes = top_3000_svgs.index.tolist()
data1 = data1[:, top_genes]
print(data1)

n_clusters = 12
# n_clusters = 12
print("===========n_clusters:{}=========".format(n_clusters))

x1 = (data1.X.toarray() if scipy.sparse.issparse(
    data1.X) else np.asarray(data1.X)).astype('float64')
x2 = (data2.X.toarray() if scipy.sparse.issparse(
    data2.X) else np.asarray(data2.X)).astype("float64")
loc = data1.obsm["spatial"].astype("float64")

if args.batch_size == "auto":
    if x1.shape[0] <= 1024:
        args.batch_size = 128
    elif x1.shape[0] <= 2048:
        args.batch_size = 256
    else:
        args.batch_size = 512
else:
    args.batch_size = int(args.batch_size)

args.batch_size = 256
print(args.batch_size)

print(args)

if args.select_genes > 0:
    importantGenes = geneSelection(x1, n=args.select_genes, plot=False)
    x1 = x1[:, importantGenes]
    # np.savetxt("selected_genes.txt", importantGenes, delimiter=",", fmt="%i")

if args.select_proteins > 0:
    importantProteins = geneSelection(x2, n=args.select_proteins, plot=False)
    x2 = x2[:, importantProteins]
    # np.savetxt("selected_proteins.txt", importantProteins, delimiter=",", fmt="%i")

scaler = MinMaxScaler()
loc = scaler.fit_transform(loc) * args.loc_range

print(x1.shape)
print(x2.shape)
print(loc.shape)

eps = 1e-5
initial_inducing_points = np.mgrid[0:(1 + eps):(1. / args.inducing_point_steps),
                                   0:(1 + eps):(1. / args.inducing_point_steps)].reshape(2, -1).T * args.loc_range
print(initial_inducing_points.shape)

adata1 = sc.AnnData(x1, dtype="float64")
adata1 = normalize(adata1,
                   size_factors=False,
                   normalize_input=True,
                   logtrans_input=True)

adata2 = sc.AnnData(x2, dtype="float64")
adata2 = normalize(adata2,
                   size_factors=False,
                   normalize_input=True,
                   logtrans_input=True)

adata2_no_scale = sc.AnnData(x2, dtype="float64")
adata2_no_scale = normalize(adata2_no_scale,
                            size_factors=False,
                            normalize_input=False,
                            logtrans_input=True)

# Fit the second-modality background prior on the processed ATAC features.
gm = GaussianMixture(n_components=2, covariance_type="diag", n_init=20).fit(adata2_no_scale.X)
back_idx = np.argmin(gm.means_, axis=0)
protein_log_back_mean = np.log(
    np.expm1(gm.means_[back_idx, np.arange(adata2_no_scale.n_vars)]) + 1e-8)
protein_log_back_scale = np.sqrt(gm.covariances_[back_idx, np.arange(adata2_no_scale.n_vars)])
print("ATAC background prior shape", protein_log_back_mean.shape)

for run_id in range(10):
    print(f"\n=================== Run {run_id+1}/{10} =================")

    model = SPAMULTIVAE(gene_dim=adata1.n_vars, protein_dim=adata2.n_vars, GP_dim=args.GP_dim, Normal_dim=args.Normal_dim, dynamicVAE=False,
                        encoder_layers=args.encoder_layers, gene_decoder_layers=args.gene_decoder_layers, protein_decoder_layers=args.protein_decoder_layers,
                        gene_noise=args.gene_noise, protein_noise=args.protein_noise, encoder_dropout=args.dropoutE, decoder_dropout=args.dropoutD,
                        fixed_inducing_points=args.fix_inducing_points, initial_inducing_points=initial_inducing_points,
                        fixed_gp_params=args.fixed_gp_params, kernel_scale=args.kernel_scale, N_train=adata1.n_obs, KL_loss=args.KL_loss, init_beta=args.init_beta, min_beta=args.min_beta,
                        max_beta=args.max_beta, protein_back_mean=protein_log_back_mean, protein_back_scale=protein_log_back_scale, dtype=torch.float64,
                        device=args.device)

    if not os.path.isfile(args.model_file):
        model.train_model(pos=loc, gene_ncounts=adata1.X, gene_raw_counts=adata1.raw.X, gene_size_factors=adata1.obs.size_factors,
                          protein_ncounts=adata2.X, protein_raw_counts=adata2.raw.X,
                          lr=args.lr, weight_decay=args.weight_decay, batch_size=args.batch_size, num_samples=args.num_samples,
                          train_size=args.train_size, maxiter=args.maxiter, patience=args.patience, save_model=True, model_weights=args.model_file)
    else:
        model.load_model(args.model_file)

    final_latent = model.batching_latent_samples(
        X=loc, gene_Y=adata1.X, protein_Y=adata2.X, batch_size=args.batch_size)
    print(final_latent.shape)

    latent = final_latent
    y_pred = KMeans(n_clusters=n_clusters, n_init=100).fit_predict(latent[:, :])
    # Save latent features
    latent_features = final_latent
    np.save(os.path.join(output_dir, 'latent', f'latent_{run_id}.npy'), latent_features)
    # Save clustering results
    pre_label = y_pred
    np.save(os.path.join(output_dir, 'labels', f'labels_{run_id}.npy'), pre_label)

print(f"\nAll experiments completed. Results saved to: {output_dir}")
