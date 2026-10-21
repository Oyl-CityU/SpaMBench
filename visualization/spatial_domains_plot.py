import os
import warnings

import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches

warnings.filterwarnings(
    'ignore',
    message="No data for colormapping provided via 'c'. Parameters 'cmap' will be ignored",
    category=UserWarning
)

# Dataset and plotting parameters.
dataset_name = '00_Human_lymph_node_slice1'
methods = [
    'spaMultiVAE', 'SpatialGlue', 'SpaMosaic', 'COSMOS', 'SMMGCL', 'PRESENT',
    'SpatialCOC', 'PRAGA', 'CANDIES', 'SMOPCA', 'SEPAR', 'SpaMGCN',
    'SpaMICS', 'MISO', 'SpaMode', 'SMART', 'SpaMV', 'SCIGMA'
]
s_size = 45
cmap = 'tab20'
color = 'final_annot'

path = 'results/{}_results'.format(dataset_name)
plot_path = '{}/plot'.format(path)
os.makedirs(plot_path, exist_ok=True)

# Read the combined AnnData and the ten-run ARI table.
adata = sc.read_h5ad('{}/{}_adata_RNA_result.h5ad'.format(path, dataset_name))
ARI_results = pd.read_csv('{}/{}_ARI_all_results.csv'.format(path, dataset_name), header=0)
mean_ARI = ARI_results.iloc[:10].apply(pd.to_numeric, errors='coerce').mean()
method_ARI = dict(zip(ARI_results.columns, mean_ARI))

plt.rcParams['font.size'] = 12
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']

# The ground-truth annotation is categorical; its category codes are 0..K-1.
if color in adata.obs:
    true_categories = list(adata.obs[color].astype('category').cat.categories)
    true_labels = adata.obs[color].astype('category').cat.codes

    fig, ax = plt.subplots(figsize=(4, 3.5))
    sc.pl.embedding(adata, basis='spatial', color=color, title='Ground Truth',
                    ax=ax, s=s_size, show=False, cmap=cmap)
    legend = ax.get_legend()
    gt_handles = []
    gt_labels = []
    if legend is not None:
        gt_handles = getattr(legend, 'legendHandles', None)
        if gt_handles is None:
            gt_handles = getattr(legend, 'legend_handles', [])
        gt_labels = [text.get_text() for text in legend.get_texts()]
        legend.remove()

    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title('Ground Truth')
    if gt_handles:
        ax.legend(gt_handles, gt_labels, loc='upper left', bbox_to_anchor=(0.99, 0.85),
                  frameon=False, fontsize=9, handletextpad=0.1,
                  labelspacing=0.4, borderpad=0.3, columnspacing=0.2)
    plt.savefig('{}/{}_ground_truth_domains.pdf'.format(plot_path, dataset_name),
                bbox_inches='tight', dpi=300)

    # Method labels were matched to these zero-based ground-truth category codes.
    domain_labels = [str(i) for i in range(len(true_categories))]
else:
    raise KeyError('Ground-truth column {!r} was not found in adata.obs'.format(color))


def draw_method_comparison(nrows, ncols, file_name, legend_bbox, legend_fontsize,
                           legend_labelspacing, layout_rect, w_pad, h_pad):
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.8, nrows * 3.25))
    ax_list = np.asarray(axes).reshape(-1)

    for i, method in enumerate(methods):
        ax = ax_list[i]
        if method not in adata.obs:
            ax.text(0.5, 0.5, 'Result unavailable', ha='center', va='center', transform=ax.transAxes)
            title = method
        else:
            # Keep categorical labels in numeric order (0, 1, ..., K-1).
            method_labels = adata.obs[method].astype(str)
            categories = sorted(pd.unique(method_labels), key=lambda value: int(value))
            adata.obs[method] = pd.Categorical(method_labels, categories=categories, ordered=True)
            score = method_ARI.get(method, np.nan)
            score_text = '{:.3f}'.format(score) if pd.notna(score) else 'NA'
            title = '{}\n(ARI: {})'.format(method, score_text)
            sc.pl.embedding(adata, basis='spatial', color=method, cmap=cmap,
                            title=title, ax=ax, s=s_size, show=False)

        legend = ax.get_legend()
        if legend is not None:
            legend.remove()
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color('black')
            spine.set_linewidth(0.8)
        ax.set_facecolor('white')

        rect = patches.Rectangle((0, 1.02), 1, 0.2, transform=ax.transAxes,
                                 color='#E6E7E8', clip_on=False)
        ax.add_patch(rect)
        ax.text(0.5, 1.12, title, transform=ax.transAxes,
                ha='center', va='center', fontsize=14)
        ax.set_title('')

    for ax in ax_list[len(methods):]:
        ax.axis('off')

    if gt_handles:
        fig.legend(gt_handles, domain_labels, loc='center right', bbox_to_anchor=legend_bbox,
                   frameon=False, fontsize=legend_fontsize, handletextpad=0.2,
                   labelspacing=legend_labelspacing, borderpad=0.3, columnspacing=0.2)
    plt.tight_layout(rect=layout_rect, w_pad=w_pad, h_pad=h_pad)
    plt.savefig('{}/{}'.format(plot_path, file_name), bbox_inches='tight', dpi=300)


# The same 18 methods are shown in both requested layouts.
# draw_method_comparison(
#     2, 9, '{}_results_domains_2x9.pdf'.format(dataset_name),
#     legend_bbox=(0.93, 0.5), legend_fontsize=14, legend_labelspacing=0.4,
#     layout_rect=[0, 0, 0.9, 1], w_pad=0.4, h_pad=0.7
# )
# draw_method_comparison(
#     3, 6, '{}_results_domains_3x6.pdf'.format(dataset_name),
#     legend_bbox=(0.93, 0.5), legend_fontsize=14, legend_labelspacing=0.4,
#     layout_rect=[0, 0, 0.88, 1], w_pad=0.35, h_pad=0.6
# )

# Display the ground-truth plot and both method-comparison figures.
# plt.show()
