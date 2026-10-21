#!/usr/bin/env python3
"""Mask nonzero entries in one modality while leaving its paired modality intact."""

import argparse
import json
import shutil
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "00_data"
OUTPUT_ROOT = DATA_ROOT / "18_modality_specific_masking"
DATASETS = {
    "02_Human_tonsil_slice1": ("RNA", "ADT"),
    "12_Slide_tags": ("RNA", "ATAC"),
}
MASK_LEVELS = (25, 50, 75)
SEED = 2024


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overwrite", action="store_true", help="replace existing masking outputs")
    args = parser.parse_args()
    targets = [OUTPUT_ROOT / f"{dataset}_mask_{modality}_{percent}"
               for dataset, modalities in DATASETS.items()
               for modality in modalities for percent in MASK_LEVELS]
    if not args.overwrite and any(path.exists() and any(path.iterdir()) for path in targets):
        raise FileExistsError("Masking output exists; use --overwrite to replace it")

    summary = []
    for dataset, modalities in DATASETS.items():
        source = DATA_ROOT / dataset
        originals = {modality: ad.read_h5ad(source / f"adata_{modality}.h5ad")
                     for modality in modalities}
        rngs = {
            modality: np.random.default_rng(
                np.random.SeedSequence([SEED, int(dataset[:2]), index])
            )
            for index, modality in enumerate(modalities)
        }

        for masked_modality, original in originals.items():
            if original.raw is not None or original.layers:
                raise ValueError(f"{dataset}/{masked_modality}: resolve raw/layers before masking")

            matrix = sparse.csr_matrix(original.X, copy=True)
            matrix.sum_duplicates()
            matrix.eliminate_zeros()
            matrix.sort_indices()
            order = rngs[masked_modality].permutation(matrix.nnz)
            unchanged_modality = next(m for m in modalities if m != masked_modality)

            for percent in MASK_LEVELS:
                count = matrix.nnz * percent // 100
                masked = matrix.copy()
                masked.data[order[:count]] = 0
                masked.eliminate_zeros()

                output = OUTPUT_ROOT / f"{dataset}_mask_{masked_modality}_{percent}"
                output.mkdir(parents=True, exist_ok=True)
                result = original.copy()
                result.X = masked
                info = {
                    "dataset": dataset,
                    "masked_modality": masked_modality,
                    "unchanged_modality": unchanged_modality,
                    "mask_percent": percent,
                    "seed": SEED,
                    "n_cells": original.n_obs,
                    "n_features": original.n_vars,
                    "original_nonzero": matrix.nnz,
                    "masked_nonzero": count,
                    "sampling": "nested uniform sampling of nonzero X entries",
                }
                result.uns["modality_masking"] = info
                result.write_h5ad(output / f"adata_{masked_modality}.h5ad", compression="gzip")
                shutil.copy2(source / f"adata_{unchanged_modality}.h5ad",
                             output / f"adata_{unchanged_modality}.h5ad")
                np.save(output / "masked_csr_positions.npy", np.sort(order[:count]))
                (output / "masking_metadata.json").write_text(json.dumps(info, indent=2))
                summary.append({"output_directory": output.name, **info})
                pd.DataFrame(summary).to_csv(OUTPUT_ROOT / "masking_summary.csv", index=False)
                print(f"{output.name}: masked {count:,}/{matrix.nnz:,} nonzero entries")


if __name__ == "__main__":
    main()
