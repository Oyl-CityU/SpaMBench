"""SpaMICS: 05 MISAR-E15 RNA + ATAC example.
Run in the method environment: python SpaMICS_RNA_ATAC.py
"""

import os
import sys

# Dependencies.
import numpy as np
import pandas as pd
import scanpy as sc
import torch
from model import SpaMICS
from processing import preprocess
from utils import construct_graph, refine_adj_spatial, post_proC
import opt
import tqdm

# 保存结果
output_dir = 'results/05_MISAR_E15'
os.makedirs(os.path.join(output_dir, 'latent'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'labels'), exist_ok=True)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
opt.args.device = str(device)
opt.args.n_cluster = 12
file_fold = '../../datasets/05_MISAR_E15/'
adata1 = sc.read_h5ad(file_fold + 'adata_RNA.h5ad')
adata2 = sc.read_h5ad(file_fold + 'adata_ATAC.h5ad')
adata1.var_names_make_unique()
adata2.var_names_make_unique()
adata2 = adata2[adata1.obs_names].copy()
adata2 = adata2[adata1.obs_names].copy()
# preprocess sets seed 2022 once; no reseeding in the training loop.
data = preprocess(adata1, adata2, datatype='RNA-ATAC')
x = data['adata_omics1'].obsm['feat']
y = data['adata_omics2'].obsm['feat']
af, nf = construct_graph(x, k=10)
bf, ng = construct_graph(y, k=10)
spatial, _ = construct_graph(adata1.obsm['spatial'], k=10)
X_omics1, X_omics2 = [torch.as_tensor(v, dtype=torch.float32, device=device) for v in (x, y)]
adj_feature_omics1, adj_feature_omics2, adj_spatial_omics1, adj_spatial_omics2 = [
    torch.as_tensor(v, dtype=torch.float32, device=device)
    for v in (nf, ng, refine_adj_spatial(af, spatial), refine_adj_spatial(bf, spatial))]
opt.args.n_omics1, opt.args.n_omics2 = x.shape[1], y.shape[1]
print("=" * 10 + " Pretraining has begun! " + "=" * 10)

for run_id in range(10):
    print(f"\n=================== Run {run_id+1}/{10} =================")

    model = SpaMICS(X_omics2.shape[0]).to(device)
    optimizer0 = torch.optim.Adam(filter(lambda p: p.requires_grad,
                                  model.parameters()), lr=opt.args.pretrain_lr)
    pbar = tqdm.tqdm(range(31), ncols=200)

    for epoch in pbar:
        loss_rec = model(X_omics1, X_omics2, adj_feature_omics1, adj_feature_omics2, adj_spatial_omics1,
                         adj_spatial_omics2, stage=0)

        pretrain_loss = loss_rec
        optimizer0.zero_grad()
        pretrain_loss.backward()
        optimizer0.step()

        pbar.set_postfix({'loss': '{0:1.4f}'.format(pretrain_loss)})

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad,
                                 model.parameters()), lr=opt.args.pretrain_lr)
    pbar = tqdm.tqdm(range(opt.args.pretrain_epoch + 1), ncols=200)
    for epoch in pbar:
        loss_rec = model(X_omics1, X_omics2, adj_feature_omics1, adj_feature_omics2, adj_spatial_omics1,
                         adj_spatial_omics2, stage=1)

        pretrain_loss = loss_rec
        optimizer.zero_grad()
        pretrain_loss.backward()
        optimizer.step()

        pbar.set_postfix({'loss': '{0:1.4f}'.format(pretrain_loss)})

    optimizer2 = torch.optim.Adam(filter(lambda p: p.requires_grad,
                                  model.parameters()), lr=opt.args.train_lr)
    pbar2 = tqdm.tqdm(range(opt.args.epoch + 1), ncols=200)
    for epoch in pbar2:

        loss_rec, loss_self, loss_reg, loss_dis, S = model(X_omics1, X_omics2, adj_feature_omics1, adj_feature_omics2,
                                                           adj_spatial_omics1, adj_spatial_omics2, stage=2)

        total_loss = loss_rec + opt.args.lambda_1 * loss_self + \
            opt.args.lambda_2 * loss_reg + opt.args.lambda_3 * loss_dis

        optimizer2.zero_grad()
        total_loss.backward()
        optimizer2.step()

        pbar2.set_postfix({'loss': '{0:1.4f}'.format(total_loss)})
    model.eval()
    with torch.no_grad():
        S = model(X_omics1, X_omics2, adj_feature_omics1, adj_feature_omics2,
                  adj_spatial_omics1, adj_spatial_omics2, stage=2)[4]
    S_cpu = S.detach().cpu().numpy()
    pred, _ = post_proC(S_cpu, opt.args.n_cluster)

    # Save latent features
    latent_features = S_cpu
    print(latent_features.shape)
    np.save(os.path.join(output_dir, 'latent', f'latent_{run_id}.npy'), latent_features)

    # Save clustering results
    pre_label = pred
    print(pre_label.shape)
    np.save(os.path.join(output_dir, 'labels', f'labels_{run_id}.npy'), pre_label)

print(f"\nAll experiments completed. Results saved to: {output_dir}")
