
## =============================Pipline code================================================
# Data types：['10x', 'SPOTS', 'Stereo-CITE-seq']

# RNA + ADT 数据集
# python main.py --file_fold '/00_data/00_Human_lymph_node_slice1/' --data_type '10x' --n_clusters 10 --init_k 10 --KNN_k 20 --RNA_weight 1 --ADT_weight 1  --output_dir "00_pipline_results/00_Human_lymph_node_slice1"

# python main.py --file_fold '/00_data/01_Human_lymph_node_slice2/' --data_type '10x' --n_clusters 10 --init_k 10 --KNN_k 20 --RNA_weight 1 --ADT_weight 1  --output_dir "00_pipline_results/01_Human_lymph_node_slice2"

# python main.py --file_fold '/00_data/02_Human_tonsil_slice1/' --data_type '10x' --n_clusters 4 --init_k 4 --KNN_k 20 --RNA_weight 1 --ADT_weight 1  --output_dir "00_pipline_results/02_Human_tonsil_slice1"

# python main.py --file_fold '/00_data/03_Human_tonsil_slice2/' --data_type '10x' --n_clusters 4 --init_k 4 --KNN_k 20 --RNA_weight 1 --ADT_weight 1  --output_dir "00_pipline_results/03_Human_tonsil_slice2"

# python main.py --file_fold '/00_data/04_DBiT_seq/' --data_type '10x' --n_clusters 13 --init_k 13 --KNN_k 20 --RNA_weight 1 --ADT_weight 1  --output_dir "00_pipline_results/04_DBiT_seq"

# python main.py --file_fold '/00_data/07_Mouse_Breast_Tumor/' --data_type 'SPOTS' --n_clusters 5 --init_k 5 --KNN_k 20 --RNA_weight 1 --ADT_weight 1  --output_dir "00_pipline_results/07_Mouse_Breast_Tumor"

# 08 Error：
# python main.py --file_fold '/00_data/08_Human_Breast_Cancer/' --data_type '10x' --n_clusters 10 --init_k 10 --KNN_k 20 --RNA_weight 1 --ADT_weight 1  --output_dir "00_pipline_results/08_Human_Breast_Cancer"

# python main.py --file_fold '/00_data/09_Human_skin/' --data_type '10x' --n_clusters 10 --init_k 10 --KNN_k 20 --RNA_weight 1 --ADT_weight 1  --output_dir "00_pipline_results/09_Human_skin"

# python main.py --file_fold '/00_data/10_Mouse_Thymus1/' --data_type 'Stereo-CITE-seq' --n_clusters 8 --init_k 8 --KNN_k 20 --RNA_weight 1 --ADT_weight 1  --output_dir "00_pipline_results/10_Mouse_Thymus1"


# # RNA + ATAC dataset: All Error
# python main.py --file_fold '/00_data/05_MISAR_E15/' --data_type 'Spatial-epigenome-transcriptome' --n_clusters 12 --init_k 12 --KNN_k 20 --RNA_weight 1 --ADT_weight 1 --output_dir "00_pipline_results/05_MISAR_E15"
# python main.py --file_fold '/00_data/12_Slide_tags/' --data_type 'Spatial-epigenome-transcriptome' --n_clusters 10 --init_k 10 --KNN_k 20 --RNA_weight 1 --ADT_weight 1 --output_dir "00_pipline_results/12_Slide_tags"

