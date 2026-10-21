#!/usr/bin/env python3
"""Shuffle spatial coordinates for a selected fraction of cells; keep expression unchanged."""

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "00_data"
OUTPUT_ROOT = DATA_ROOT / "19_spatial_coordinate_perturbation"
DATASETS = {
    "02_Human_tonsil_slice1": ("RNA", "ADT"),
    "12_Slide_tags": ("RNA", "ATAC"),
}
PERTURBATION_LEVELS = (25, 50, 75, 100)
SEED = 2024


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overwrite", action="store_true", help="replace existing perturbation outputs")
    args = parser.parse_args()
    targets = [OUTPUT_ROOT / f"{dataset}_spatial_shuffle_{percent}"
               for dataset in DATASETS for percent in PERTURBATION_LEVELS]
    if not args.overwrite and any(path.exists() and any(path.iterdir()) for path in targets):
        raise FileExistsError("Perturbation output exists; use --overwrite to replace it")

    summary = []
    for dataset, modalities in DATASETS.items():
        source = DATA_ROOT / dataset
        originals = {modality: ad.read_h5ad(source / f"adata_{modality}.h5ad")
                     for modality in modalities}
        reference = originals[modalities[0]]
        coordinates = np.asarray(reference.obsm["spatial"], dtype=float)
        rng = np.random.default_rng(np.random.SeedSequence([SEED, int(dataset[:2])]))
        selection_order = rng.permutation(reference.n_obs)

        for percent in PERTURBATION_LEVELS:
            count = reference.n_obs * percent // 100
            selected = selection_order[:count]
            permutation = rng.permutation(count)
            while np.any(np.all(coordinates[selected[permutation]] == coordinates[selected], axis=1)):
                permutation = rng.permutation(count)

            donor = np.arange(reference.n_obs)
            donor[selected] = selected[permutation]
            perturbed = coordinates[donor].copy()
            changed = np.any(perturbed != coordinates, axis=1)
            output = OUTPUT_ROOT / f"{dataset}_spatial_shuffle_{percent}"
            output.mkdir(parents=True, exist_ok=True)

            info = {
                "dataset": dataset,
                "seed": SEED,
                "requested_percent": percent,
                "changed_cells": int(changed.sum()),
                "total_cells": reference.n_obs,
                "spatial_key": "spatial",
                "method": "partial coordinate permutation without fixed points",
                "paired_modalities_share_permutation": True,
            }
            for modality, original in originals.items():
                result = original.copy()
                result.obsm["spatial"] = perturbed.copy()
                for column, axis in (("X", 0), ("Y", 1)):
                    if column in original.obs and np.allclose(
                        pd.to_numeric(original.obs[column], errors="coerce"), coordinates[:, axis]
                    ):
                        result.obs[column] = perturbed[:, axis]
                result.uns["spatial_perturbation"] = info
                result.write_h5ad(output / f"adata_{modality}.h5ad", compression="gzip")

            mapping = pd.DataFrame({
                "cell_id": reference.obs_names,
                "coordinate_donor_cell_id": reference.obs_names[donor],
                "coordinate_donor_row": donor,
                "perturbed": changed,
            })
            for axis in range(coordinates.shape[1]):
                mapping[f"original_coordinate_{axis}"] = coordinates[:, axis]
                mapping[f"perturbed_coordinate_{axis}"] = perturbed[:, axis]
            mapping.to_csv(output / "coordinate_mapping.csv", index=False)
            (output / "perturbation_metadata.json").write_text(json.dumps(info, indent=2))
            summary.append({"output_directory": output.name, **info})
            pd.DataFrame(summary).to_csv(OUTPUT_ROOT / "perturbation_summary.csv", index=False)
            print(f"{output.name}: shuffled coordinates for {changed.sum():,}/{reference.n_obs:,} cells")


if __name__ == "__main__":
    main()
