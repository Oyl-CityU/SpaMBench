"""MISO: 05 MISAR-E15 RNA + ATAC example.
Run in the method environment: python MISO_RNA_ATAC.py
"""

import os
import sys

# Dependencies.
import random
import numpy as np
import pandas as pd
import scanpy as sc
import torch
from miso import Miso
from miso.utils import preprocess
from os.path import join

# 保存结果
output_dir = 'results/05_MISAR_E15'
os.makedirs(os.path.join(output_dir, 'latent'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'labels'), exist_ok=True)

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# Fix the random seed once before the ten runs.
random_seed = 100
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)

# Load data:
file_fold = '../../datasets/05_MISAR_E15/'
adata1 = sc.read_h5ad(file_fold + 'adata_RNA.h5ad')
adata2 = sc.read_h5ad(file_fold + 'adata_ATAC.h5ad')
adata1.var_names_make_unique()
adata2.var_names_make_unique()
adata2 = adata2[adata1.obs_names].copy()

n_clusters = 12
print('===========n_clusters:{}========='.format(n_clusters))

rna = preprocess(adata1.copy(), modality='rna')
protein = preprocess(adata2.copy(), modality='atac')
assert rna.shape[0] == protein.shape[0] == adata1.n_obs

for run_id in range(10):
    print(f'\n=================== Run {run_id+1}/{10} =================')

    model = Miso([rna, protein], ind_views='all', combs='all',
                 sparse=False, device=device)
    model.train()
    latent_features = model.emb.copy()

    rng_state = np.random.get_state()
    pre_label = model.cluster(n_clusters=n_clusters)
    np.random.set_state(rng_state)

    # Save the latent representation and labels for each run.
    np.save(join(output_dir, 'latent', f'latent_{run_id}.npy'), latent_features)
    np.save(join(output_dir, 'labels', f'labels_{run_id}.npy'), pre_label)

    del model

print(f'\nAll experiments completed. Results saved to: {output_dir}')
