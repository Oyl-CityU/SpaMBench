import os

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
import seaborn as sns

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['font.size'] = 16

metrics = ['ARI', 'NMI', 'HOM', 'COM']
titles = ['ARI', 'NMI', 'HOM', 'COM']
methods = [
    'spaMultiVAE', 'SpatialGlue', 'SpaMosaic', 'COSMOS', 'SMMGCL', 'PRESENT',
    'SpatialCOC', 'PRAGA', 'CANDIES', 'SMOPCA', 'SEPAR', 'SpaMGCN',
    'SpaMICS', 'MISO', 'SpaMode', 'SMART', 'SpaMV', 'SCIGMA'
]
save_path = 'results/00_boxplots'
os.makedirs(save_path, exist_ok=True)


def read_metric_data(dataset, metric):
    file_path = f'results/{dataset}_results/{dataset}_{metric}_all_results.csv'
    return pd.read_csv(file_path, header=0).reindex(columns=methods).apply(
        pd.to_numeric, errors='coerce'
    )


def plot_dataset_group(datasets, file_name):
    data_by_metric = {
        metric: pd.concat(
            [read_metric_data(dataset, metric) for dataset in datasets],
            ignore_index=True
        )
        for metric in metrics
    }

    fig, axes = plt.subplots(1, 4, figsize=(20, 8))
    for i, metric in enumerate(metrics):
        data = data_by_metric[metric]
        valid_methods = [method for method in methods if data[method].notna().any()]
        medians = data[valid_methods].median().sort_values(ascending=False)
        sns.boxplot(data=data[medians.index], orient='h', palette='tab20', ax=axes[i])
        axes[i].xaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
        axes[i].set_title(titles[i], fontsize=20)
        axes[i].set_xlabel('')
        axes[i].set_ylabel('')
        axes[i].tick_params(axis='both', labelsize=11)
        for spine in axes[i].spines.values():
            spine.set_linewidth(2)
            spine.set_color('black')

    plt.tight_layout()
    plt.savefig(f'{save_path}/{file_name}', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close(fig)


plot_dataset_group(
    ['00_Human_lymph_node_slice1', '01_Human_lymph_node_slice2'],
    '00_01_ARI_NMI_HOM_COM_boxplot.pdf'
)
plot_dataset_group(
    ['02_Human_tonsil_slice1', '03_Human_tonsil_slice2'],
    '02_03_ARI_NMI_HOM_COM_boxplot.pdf'
)
plot_dataset_group(
    ['05_MISAR_E15', '06_MISAR_E18'],
    '05_06_ARI_NMI_HOM_COM_boxplot.pdf'
)
plot_dataset_group(
    ['12_Slide_tags'],
    '12_ARI_NMI_HOM_COM_boxplot.pdf'
)
