import argparse
from pathlib import Path
import os
import re
import sys

import warnings
import numpy as np
import pandas as pd
import scanpy as sc
import squidpy as sq
from sklearn.metrics import (adjusted_rand_score, normalized_mutual_info_score,
                             homogeneity_score,
                             completeness_score, silhouette_score)
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import trustworthiness

try:
    import numpy._core  
except ImportError:
    import numpy.core
    sys.modules.setdefault('numpy._core', numpy.core)
    for module_name, module in list(sys.modules.items()):
        if module_name.startswith('numpy.core.'):
            sys.modules.setdefault('numpy._core' + module_name[len('numpy.core'):], module)

# This script lives in 01_spatial_multi_omics_methods/22_results_clear_up_update.
method_root = Path(__file__).resolve().parent.parent
data_root = method_root / '00_data'
output_root = method_root / '22_results_clear_up_update' / 'results'
output_root.mkdir(parents=True, exist_ok=True)

# Dataset order follows the original 00–13 benchmark, then the two added image datasets.
dataset_names = [
    '00_Human_lymph_node_slice1', '01_Human_lymph_node_slice2',
    '02_Human_tonsil_slice1', '03_Human_tonsil_slice2', '04_DBiT_seq',
    '05_MISAR_E15', '06_MISAR_E18', '07_Mouse_Breast_Tumor',
    '08_Human_Breast_Cancer', '09_Human_skin', '10_Mouse_Thymus1',
    '11_Mouse_postnatal_brain_Slice3', '12_Slide_tags',
    '13_Mouse_postnatal_brain_Slice1', '15_cosMX_human_breast',
    '16_Xenium_human_ccRCC_label'
]

# Relative paths are from 01_spatial_multi_omics_methods.
method_paths = {
    'spaMultiVAE': ['01-spaVAE/spaVAE'],
    'SpatialGlue': ['02-spatialGlue'],
    'SpaMosaic': ['03-SpaMosaic/SpaMosaic/00_pipline_code'],
    'COSMOS': ['04-COSMOS/COSMOS/00_pipline_code'],
    'SMMGCL': ['05_SMMGCL/SMMGCL/00_pipline_code'],
    'PRESENT': ['06_Presnt/PRESENT'],
    'SpatialCOC': ['08_SpaKnit/SpatialCOC-main/00_pipline_code'],
    'PRAGA': ['09_PRAGA/PRAGA'],
    'CANDIES': ['10_CANDIES/CANDIES-main/00_pipline_code'],
    'SMOPCA': ['11_SMOPCA/SMOPCA-main/00_pipline_code'],
    'SEPAR': ['12_SEPAR/SEPAR-main/00_pipline_cde'],
    'SpaMGCN': ['13_spaMGCN/spaMGCN-master/00_pipline_code'],
    'SpaMICS': ['14/SpaMICS-main'],
    'MISO': ['16_MISO/miso-main'],
    'SpaMode': ['18_SpaMode/SpaMode-master'],
    'SMART': ['19_SMART/SMART-main-main'],
    'SpaMV': ['20_SpaMV/SpaMV-main'],
    'SCIGMA': ['21_SCIGMA/SCIGMA-main']
}

# Ground-truth fields used by the existing benchmark; None means unsupervised metrics only.
gt_keys = {
    '00_Human_lymph_node_slice1': 'final_annot',
    '01_Human_lymph_node_slice2': 'final_annot',
    '02_Human_tonsil_slice1': 'final_annot',
    '03_Human_tonsil_slice2': 'final_annot',
    '04_DBiT_seq': None,
    '05_MISAR_E15': 'Combined_Clusters_annotation',
    '06_MISAR_E18': 'Combined_Clusters_annotation',
    '07_Mouse_Breast_Tumor': None,
    '08_Human_Breast_Cancer': None,
    '09_Human_skin': None,
    '10_Mouse_Thymus1': None,
    '11_Mouse_postnatal_brain_Slice3': None,
    '12_Slide_tags': 'cell_type',
    '13_Mouse_postnatal_brain_Slice1': None,
    '15_cosMX_human_breast': None,
    '16_Xenium_human_ccRCC_label': 'cell_type'
}

n_runs = 10
marker_top_n = 5
marker_n_perms = 100
umap_random_state = 2025
umap_n_neighbors = 15


def compute_chaos(labels, spatial):
    labels = np.asarray(labels)
    spatial = StandardScaler().fit_transform(np.asarray(spatial, dtype=float))
    total = 0.0
    for group in np.unique(labels):
        points = spatial[labels == group]
        if len(points) <= 2:
            continue
        nn = NearestNeighbors(n_neighbors=2).fit(points)
        distances, _ = nn.kneighbors(points)
        total += distances[:, 1].sum()
    return float(total / len(labels))


def compute_pas(labels, spatial, k=10):
    labels = np.asarray(labels)
    spatial = np.asarray(spatial, dtype=float)
    k = min(k, len(labels) - 1)
    if k < 1:
        return np.nan
    nn = NearestNeighbors(n_neighbors=k + 1).fit(spatial)
    _, indices = nn.kneighbors(spatial)
    neighbor_labels = labels[indices[:, 1:]]
    return float(np.mean(np.sum(neighbor_labels != labels[:, None], axis=1) > k / 2))


def compute_latent_asw(labels, latent):
    return float(silhouette_score(np.asarray(latent, dtype=float), labels, metric='euclidean'))


def compute_trustworthiness(latent):
    latent = np.asarray(latent, dtype=np.float32)
    n_obs = latent.shape[0]
    if n_obs < 4:
        return np.nan
    graph_neighbors = min(umap_n_neighbors, n_obs - 1)
    trust_neighbors = min(10, (n_obs - 1) // 2)
    tmp = sc.AnnData(X=latent)
    sc.pp.neighbors(tmp, n_neighbors=graph_neighbors, use_rep='X', random_state=umap_random_state)
    sc.tl.umap(tmp, random_state=umap_random_state)
    return float(trustworthiness(latent, tmp.obsm['X_umap'], n_neighbors=trust_neighbors))


def compute_marker_scores(adata, pred_key, top_n=marker_top_n, n_perms=marker_n_perms):
    marker_data = adata.copy()
    # Count matrices may be stored as uint64; per-cell normalization divides in place
    # and cannot cast the resulting float values back to an integer dtype.
    marker_data.X = marker_data.X.astype(np.float32)
    marker_data.obs[pred_key] = marker_data.obs[pred_key].astype('category')
    counts = marker_data.obs[pred_key].value_counts()
    keep = counts.index[counts > 3]
    marker_data = marker_data[marker_data.obs[pred_key].isin(keep)].copy()
    if marker_data.n_obs == 0 or marker_data.obs[pred_key].nunique() < 2:
        return np.nan, np.nan
    sc.pp.normalize_per_cell(marker_data)
    sc.pp.log1p(marker_data)
    sc.tl.rank_genes_groups(marker_data, groupby=pred_key)
    ranked = marker_data.uns['rank_genes_groups']['names']
    selected = []
    for group in ranked.dtype.names:
        selected.extend(list(ranked[group][:top_n]))
    selected = [gene for gene in pd.unique(selected) if gene in marker_data.var_names]
    if not selected:
        return np.nan, np.nan
    sq.gr.spatial_neighbors(marker_data)
    sq.gr.spatial_autocorr(marker_data, mode='moran', genes=selected, n_perms=n_perms,
                           n_jobs=1, show_progress_bar=False)
    sq.gr.spatial_autocorr(marker_data, mode='geary', genes=selected, n_perms=n_perms,
                           n_jobs=1, show_progress_bar=False)
    return float(np.median(marker_data.uns['moranI']['I'])), float(np.median(marker_data.uns['gearyC']['C']))


def evaluate_one_run(adata, gt_key, labels, latent, spatial_key='spatial'):
    """Return benchmark, latent representation, and marker spatial metrics."""
    labels = np.asarray(labels).reshape(-1)
    latent = np.asarray(latent)
    spatial = np.asarray(adata.obsm[spatial_key])
    result = {key: np.nan for key in [
        'ARI', 'NMI', 'HOM', 'COM', 'CHAOS', 'PAS',
        'Latent_ASW', 'Trustworthiness', 'MoranI', 'GearyC'
    ]}
    errors = []
    if gt_key is not None and gt_key in adata.obs:
        y_true = adata.obs[gt_key].astype(str).to_numpy()
        y_pred = labels.astype(str)
        for key, function in [
            ('ARI', adjusted_rand_score), ('NMI', normalized_mutual_info_score),
            ('HOM', homogeneity_score),
            ('COM', completeness_score)
        ]:
            try:
                result[key] = float(function(y_true, y_pred))
            except Exception as exc:
                errors.append(f'{key}: {type(exc).__name__}: {exc}')
    for key, function, args in [
        ('CHAOS', compute_chaos, (labels, spatial)),
        ('PAS', compute_pas, (labels, spatial)),
        ('Latent_ASW', compute_latent_asw, (labels, latent)),
        ('Trustworthiness', compute_trustworthiness, (latent,))
    ]:
        try:
            result[key] = function(*args)
        except Exception as exc:
            errors.append(f'{key}: {type(exc).__name__}: {exc}')
    try:
        marker_i, geary_c = compute_marker_scores(adata, '_prediction')
        result['MoranI'], result['GearyC'] = marker_i, geary_c
    except Exception as exc:
        errors.append(f'Marker autocorrelation: {type(exc).__name__}: {exc}')
    result['metric_errors'] = ' | '.join(errors)
    return result


def _run_id(path):
    match = re.search(r'(?:_|-)(\d+)$', Path(path).stem)
    return int(match.group(1)) if match else None


def _npy_for_run(result_dir, kind, run_id):
    candidates = []
    for folder in [result_dir / kind, result_dir]:
        if folder.exists():
            candidates.extend(folder.rglob('*.npy'))
    candidates = sorted(set(candidates), key=lambda x: (0 if x.parent.name == kind else 1, len(str(x))))
    for path in candidates:
        if _run_id(path) == run_id and kind.lower() in path.stem.lower():
            return path
    return None


def _has_complete_run_sidecars(result_dir):
    """Return True when each run has labels, latent, and its own cell IDs."""
    for run_id in range(n_runs):
        if _npy_for_run(result_dir, 'labels', run_id) is None:
            return False
        if _npy_for_run(result_dir, 'latent', run_id) is None:
            return False
        if not any((result_dir / name).is_file() for name in [
            f'obs_names_{run_id}.npy', f'cell_ids_{run_id}.npy'
        ]) and not any((result_dir / name).is_file() for name in ['obs_names.csv', 'cell_ids.csv']):
            return False
    return True


def _result_candidates(method, dataset):
    candidates = []
    for rel in method_paths[method]:
        root = method_root / rel
        for result_name in ['results', 'results_extra_image', 'results_extra_image_update', 'results_extra_image_update2']:
            if result_name == 'results':
                candidates.append(root / result_name / dataset)
            else:
                candidates.append(root / result_name / dataset / 'default')
                candidates.append(root / result_name / dataset)
    return candidates


def _choose_result_dir(method, dataset):
    scored = []
    for order, path in enumerate(_result_candidates(method, dataset)):
        if not path.is_dir():
            continue
        pairs = sum(_npy_for_run(path, 'labels', i) is not None and _npy_for_run(path, 'latent', i) is not None
                    for i in range(n_runs))
        has_h5ad = (path / 'adata_output.h5ad').is_file()
        scored.append((pairs, int(has_h5ad), -order, path))
    return max(scored, default=(0, 0, 0, None))[3]


def _read_cell_ids(result_dir, run_id, output_adata=None):
    for path in [result_dir / f'obs_names_{run_id}.npy', result_dir / f'cell_ids_{run_id}.npy']:
        if path.is_file():
            return np.asarray(np.load(path, allow_pickle=True)).astype(str).reshape(-1)
    for path in [result_dir / 'obs_names.csv', result_dir / 'cell_ids.csv']:
        if path.is_file():
            frame = pd.read_csv(path)
            col = 'cell_id' if 'cell_id' in frame.columns else frame.columns[0]
            return frame[col].astype(str).to_numpy()
    if output_adata is not None:
        return np.asarray(output_adata.obs_names).astype(str)
    return None


def _key_for_run(keys, run_id, kind):
    preferred = ([f'pred_{run_id}', f'labels_{run_id}', f'labels_mclust_{run_id}', f'prediction_{run_id}']
                 if kind == 'labels' else [f'latent_{run_id}', f'X_latent_{run_id}', f'emb_latent_{run_id}'])
    for key in preferred:
        if key in keys:
            return key
    for key in keys:
        if _run_id(key) == run_id and kind.lower() in key.lower():
            return key
    return None


def _load_artifacts(result_dir, run_id, output_adata, method):
    label_path = _npy_for_run(result_dir, 'labels', run_id)
    latent_path = _npy_for_run(result_dir, 'latent', run_id)
    labels = None
    latent = None
    if output_adata is not None:
        # A per-run H5AD carries its own observation names, so prefer its
        # run-specific outputs over NPY side files that may be stale or lack
        # cell IDs (PRESENT has both formats in some result folders).
        key = _key_for_run(output_adata.obs.columns, run_id, 'labels')
        if key is not None:
            labels = output_adata.obs[key].to_numpy()
        key = _key_for_run(output_adata.obsm.keys(), run_id, 'latent')
        if key is not None:
            latent = np.asarray(output_adata.obsm[key])
        # Some pipelines keep only the final run in adata_output.h5ad instead of writing per-run NPYs.
        # Use those method-named fields only for the run recorded in uns['run_id'].
        stored_run = output_adata.uns.get('run_id')
        if stored_run is not None and int(np.asarray(stored_run).reshape(-1)[0]) == run_id:
            if labels is None and method in output_adata.obs:
                labels = output_adata.obs[method].to_numpy()
            if latent is None and method in output_adata.obsm:
                latent = np.asarray(output_adata.obsm[method])
    if labels is None and label_path:
        labels = np.load(label_path, allow_pickle=True).reshape(-1)
    if latent is None and latent_path:
        latent = np.load(latent_path, allow_pickle=True)
    ids = _read_cell_ids(result_dir, run_id, output_adata)
    return labels, latent, ids


def _align_run(adata, labels, latent, ids):
    labels = np.asarray(labels).reshape(-1)
    latent = np.asarray(latent)
    if latent.ndim != 2 or len(labels) != latent.shape[0]:
        raise ValueError(f'label/latent dimensions differ: labels={labels.shape}, latent={latent.shape}')
    base_ids = np.asarray(adata.obs_names).astype(str)
    if ids is None:
        if len(labels) != adata.n_obs:
            raise ValueError(f'no cell-id sidecar; result rows={len(labels)}, input cells={adata.n_obs}')
        return adata.copy(), labels, latent, 'aligned_by_input_order'
    ids = np.asarray(ids).astype(str).reshape(-1)
    if len(ids) != len(labels):
        raise ValueError(f'cell-id count {len(ids)} does not match output rows {len(labels)}')
    if len(set(ids)) != len(ids):
        raise ValueError('duplicate cell IDs in result')
    base_index = {cell: i for i, cell in enumerate(base_ids)}
    pairs = [(base_index[cell], j) for j, cell in enumerate(ids) if cell in base_index]
    if not pairs:
        raise ValueError('no matching cell IDs between output and input data')
    input_rows = np.asarray([x[0] for x in pairs], dtype=int)
    output_rows = np.asarray([x[1] for x in pairs], dtype=int)
    status = 'aligned_by_cell_id' if len(pairs) == adata.n_obs == len(ids) else 'aligned_common_cells'
    return adata[input_rows].copy(), labels[output_rows], latent[output_rows], status


def _metric_row(dataset, method, run_id, status, gt_key, **kwargs):
    row = dict(dataset=dataset, method=method, run_id=run_id, status=status,
               ground_truth_key=gt_key or '', n_cells_input=np.nan, n_cells_evaluated=np.nan,
               n_clusters=np.nan, metric_errors='')
    row.update({key: np.nan for key in [
        'ARI', 'NMI', 'HOM', 'COM', 'CHAOS', 'PAS',
        'Latent_ASW', 'Trustworthiness', 'MoranI', 'GearyC'
    ]})
    row.update(kwargs)
    return row


def evaluate_dataset(dataset, methods=None):
    data_path = data_root / dataset / 'adata_RNA.h5ad'
    if not data_path.is_file():
        raise FileNotFoundError(data_path)
    adata = sc.read_h5ad(data_path)
    adata.var_names_make_unique()
    if 'spatial' not in adata.obsm:
        raise KeyError(f"{dataset}: expected adata.obsm['spatial']")
    gt_key = gt_keys.get(dataset)
    if gt_key not in adata.obs:
        gt_key = None
    dataset_dir = output_root / f'{dataset}_results'
    dataset_dir.mkdir(parents=True, exist_ok=True)
    method_rows = []
    status_rows = []

    selected_methods = list(method_paths) if methods is None else methods
    for method in selected_methods:
        result_dir = _choose_result_dir(method, dataset)
        shared_output_adata = None
        shared_h5ad = result_dir / 'adata_output.h5ad' if result_dir is not None else None
        if (shared_h5ad is not None and shared_h5ad.is_file()
                and not _has_complete_run_sidecars(result_dir)):
            try:
                shared_output_adata = sc.read_h5ad(shared_h5ad)
            except Exception as exc:
                warnings.warn(f'{method}/{dataset}: unable to read {shared_h5ad.name}: {exc}')
        rows = []
        available = 0
        for run_id in range(n_runs):
            if result_dir is None:
                row = _metric_row(dataset, method, run_id, 'missing_output_dir', gt_key)
            else:
                try:
                    output_adata = shared_output_adata
                    run_h5ad = result_dir / f'adata_output_{run_id}.h5ad'
                    if run_h5ad.is_file():
                        try:
                            output_adata = sc.read_h5ad(run_h5ad)
                        except Exception as exc:
                            warnings.warn(f'{method}/{dataset}: unable to read {run_h5ad.name}: {exc}')
                    labels, latent, ids = _load_artifacts(result_dir, run_id, output_adata, method)
                    if labels is None or latent is None:
                        missing = 'missing_labels' if labels is None else 'missing_latent'
                        row = _metric_row(dataset, method, run_id, missing, gt_key,
                                          result_dir=str(result_dir))
                    else:
                        adata_run, labels, latent, align_status = _align_run(adata, labels, latent, ids)
                        if len(labels) != adata_run.n_obs or latent.shape[0] != adata_run.n_obs:
                            raise ValueError('aligned row count mismatch')
                        adata_run.obs['_prediction'] = pd.Categorical(labels.astype(str))
                        metrics = evaluate_one_run(adata_run, gt_key, labels, latent, spatial_key='spatial')
                        row = _metric_row(dataset, method, run_id, align_status, gt_key,
                                          result_dir=str(result_dir), n_cells_input=adata.n_obs,
                                          n_cells_evaluated=adata_run.n_obs,
                                          n_clusters=len(pd.unique(labels)), **metrics)
                        available += 1
                except Exception as exc:
                    row = _metric_row(dataset, method, run_id, 'evaluation_error', gt_key,
                                      result_dir=str(result_dir), metric_errors=f'{type(exc).__name__}: {exc}')
            rows.append(row)
        method_rows.extend(rows)
        status_rows.append(dict(dataset=dataset, method=method,
                                result_dir=str(result_dir) if result_dir else '',
                                evaluated_runs=available, expected_runs=n_runs,
                                status='complete' if available == n_runs else 'partial_or_missing'))
        print(f'{dataset} | {method}: {available}/{n_runs} runs', flush=True)

    run_metrics = pd.DataFrame(method_rows)
    metric_columns = ['ARI', 'NMI', 'HOM', 'COM', 'CHAOS', 'PAS',
                      'Latent_ASW', 'Trustworthiness', 'MoranI', 'GearyC']
    # Save one wide table per metric: 10 run rows by 18 method columns.
    for metric in metric_columns:
        wide = run_metrics.pivot(index='run_id', columns='method', values=metric)
        wide = wide.reindex(index=range(n_runs), columns=selected_methods)
        result_path = dataset_dir / f'{dataset}_{metric}_all_results.csv'
        if methods is not None and result_path.is_file():
            existing = pd.read_csv(result_path).reindex(index=range(n_runs))
            for method in selected_methods:
                existing[method] = wide[method]
            wide = existing.reindex(columns=list(method_paths))
        else:
            wide = wide.reindex(columns=list(method_paths))
        wide.columns.name = None
        wide.to_csv(result_path, index=False)
    return run_metrics


def run_all_datasets(datasets=None, methods=None):
    datasets = dataset_names if datasets is None else datasets
    for dataset in datasets:
        evaluate_dataset(dataset, methods=methods)
    print(f'Finished. Results: {output_root.resolve()}')


def main():
    global marker_top_n, marker_n_perms
    parser = argparse.ArgumentParser(description='Evaluate spatial multi-omics benchmark outputs.')
    parser.add_argument('--dataset', nargs='+', choices=dataset_names,
                        help='Dataset name(s). By default, evaluate all configured datasets.')
    parser.add_argument('--method', nargs='+', choices=list(method_paths),
                        help='Method name(s). By default, evaluate all configured methods.')
    parser.add_argument('--list-datasets', action='store_true',
                        help='Print available dataset names and exit.')
    parser.add_argument('--marker-top-n', type=int, default=marker_top_n,
                        help='Top marker genes per cluster for spatial autocorrelation (default: %(default)s).')
    parser.add_argument('--marker-perms', type=int, default=marker_n_perms,
                        help='Permutation count for Moran/Geary marker scores (default: %(default)s).')
    args = parser.parse_args()
    if args.list_datasets:
        print('\n'.join(dataset_names))
        return
    marker_top_n = args.marker_top_n
    marker_n_perms = args.marker_perms
    run_all_datasets(args.dataset, args.method)


if __name__ == '__main__':
    main()
