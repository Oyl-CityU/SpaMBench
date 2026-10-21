# Benchmark datasets

This table summarizes the 16 datasets used in the manuscript. `Cell types` gives the number of annotated cell types only when curated labels are available. `Reference clusters` gives the target number of clusters used by the benchmark methods; for datasets without ground-truth labels, this is the benchmark-specified clustering target rather than a known biological reference.

| No. | Dataset | Technique | Species | Tissue | Cells | Cell types | Reference clusters |
|---:|---|---|---|---|---:|---:|---:|
| 1 | Human Lymph Node Slice1 | 10X Visium | Human | Lymph node | 3,484 | 10 | 10 |
| 2 | Human Lymph Node Slice2 | 10X Visium | Human | Lymph node | 3,359 | 10 | 10 |
| 3 | Human Tonsil Slice1 | 10X Visium | Human | Tonsil | 4,326 | 4 | 4 |
| 4 | Human Tonsil Slice2 | 10X Visium | Human | Tonsil | 4,518 | 4 | 4 |
| 5 | Mouse Embryonic Brain E15.5 | MISAR-seq | Mouse | Embryonic brain | 1,949 | 12 | 12 |
| 6 | Mouse Embryonic Brain E18.5 | MISAR-seq | Mouse | Embryonic brain | 2,129 | 14 | 14 |
| 7 | Human Melanoma | Slide-tags | Human | Melanoma | 2,535 | 10 | 10 |
| 8 | Mouse Breast Tumor | SPOTS | Mouse | Breast | 1,978 | — | 5 |
| 9 | Mouse Embryo | DBiT | Mouse | Embryo | 936 | — | 13 |
| 10 | Human Breast Cancer | 10X Visium | Human | Breast | 4,169 | — | 10 |
| 11 | Human Skin | Spatial CITE-seq | Human | Skin | 1,691 | — | 10 |
| 12 | Mouse Thymus | Stereo CITE-seq | Mouse | Thymus | 4,697 | — | 8 |
| 13 | Mouse Postnatal Brain Slice1 | Spatial ATAC-RNA-seq | Mouse | Postnatal brain | 2,372 | — | 8 |
| 14 | Mouse Postnatal Brain Slice2 | Spatial ATAC-RNA-seq | Mouse | Postnatal brain | 9,215 | — | 8 |
| 15 | Human Renal ccRCC | 10X Xenium | Human | Kidney | 359,085 | 18 | 18 |
| 16 | CosMx Human Breast Cancer | CosMx | Human | Breast | 152,451 | — | 17 |

`—` means curated cell-type labels are unavailable. `Reference clusters` does not imply that those unlabeled datasets have biological ground-truth annotations.

All datasets are stored in H5AD format and have been made publicly available on Zenodo: **[Zenodo record link to be added]**.

