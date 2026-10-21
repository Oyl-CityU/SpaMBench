from utils import construct_graph, refine_adj_spatial, convert_to_tensor
from processing import preprocess
import scanpy as sc
import numpy as np
import opt
import pandas as pd


def load_data():
    """
        Load and preprocess data from specified datasets.

        Constructs feature graphs and spatial graphs, refines the spatial graphs,
        and returns PCA features, normalized graphs, and labels.

        Returns:
            tuple: Contains PCA features of each omics data, normalized feature graphs,
                   labels, and refined spatial graphs for both omics datasets.
    """

    if opt.args.name in ["00_Human_lymph_node_slice1", "01_Human_lymph_node_slice2"]:
        adata_omics1, adata_omics2, label = load_human_lymph_node(opt.args.name)
        datatype = 'RNA-ADT'
    elif opt.args.name in ["02_Human_tonsil_slice1", "03_Human_tonsil_slice2", "02_Human_tonsil_slice1_noise_01", 
                           "02_Human_tonsil_slice1_noise_02", "02_Human_tonsil_slice1_noise_03", "02_Human_tonsil_slice1_noise_04",
                            "02_Human_tonsil_slice1_mask_01", "02_Human_tonsil_slice1_mask_03", "02_Human_tonsil_slice1_mask_05", 
                            "02_Human_tonsil_slice1_mask_07", "02_Human_tonsil_slice1_mask_09"]:
        adata_omics1, adata_omics2, label = load_human_tonsil(opt.args.name)
        datatype = 'RNA-ADT'
    elif opt.args.name == "04_DBiT_seq":
        adata_omics1, adata_omics2, label = load_DBiT(opt.args.name)
        datatype = 'RNA-ADT'
    elif opt.args.name in ["05_MISAR_E15", "06_MISAR_E18"]:
        adata_omics1, adata_omics2, label = load_mouse_brain(opt.args.name)
        datatype = 'RNA-ATAC'
    elif opt.args.name in ['07_Mouse_Breast_Tumor']:
        adata_omics1, adata_omics2, label = load_mouse_breast(opt.args.name)
        datatype = 'RNA-ADT'
        opt.args.lambda_1, opt.args.lambda_2, opt.args.lambda_3 = 1, 0.01, 1
    elif opt.args.name in ['08_Human_Breast_Cancer', "09_Human_skin", "10_Mouse_Thymus1"]:
        adata_omics1, adata_omics2, label = load_human_breast_cancer(opt.args.name)
        datatype = 'RNA-ADT'
    elif opt.args.name in ["12_Slide_tags", "12_Slide_tags_noise_01", "12_Slide_tags_noise_02", "12_Slide_tags_noise_03", 
                "12_Slide_tags_mask_01", "12_Slide_tags_mask_03", "12_Slide_tags_mask_05", "12_Slide_tags_mask_07", "12_Slide_tags_mask_09",
                "12_Slide_tags_noise_04"]:
        adata_omics1, adata_omics2, label = load_human_melanoma(opt.args.name)
        datatype = 'RNA-ATAC'
        opt.args.lambda_1, opt.args.lambda_2, opt.args.lambda_3 = 1, 0.01, 1
    elif opt.args.name in ["11_Mouse_postnatal_brain_Slice3", '13_Mouse_postnatal_brain_Slice1']:
        adata_omics1, adata_omics2, label = load_atac(opt.args.name)
        datatype = 'RNA-ATAC'
        opt.args.lambda_1, opt.args.lambda_2, opt.args.lambda_3 = 1, 0.01, 1
    else:
        raise ValueError(f"Dataset {opt.args.name} not supported")
    
    
    # # --------------------已手动注释Pre函数中筛选3000 RNA高变的代码  -----ADT数据集：00   ATAC数据集：05------------------------------
    # HVG 
    # print("  --------1----------  ")
    # sc.pp.highly_variable_genes(adata_omics1, flavor="seurat_v3", n_top_genes=3000)
    # hvg_genes = adata_omics1.var_names[adata_omics1.var['highly_variable']].tolist()
    # adata_omics1 = adata_omics1[:, adata_omics1.var['highly_variable']].copy()

    # # SVG   
    # top_3000_svgs_path = "/data/chenfuqun/01_working/04_spatial_multi-omics_review/01_paper_demo/01-spaVAE/spaVAE/00_pp_datasets/SVGs_files/05_top3000_genes_pvalues.csv"
    # top_3000_svgs = pd.read_csv(top_3000_svgs_path, index_col=0)
    # top_genes = top_3000_svgs.index.tolist() 
    # adata_omics1 = adata_omics1[:, top_genes]

    # # None  

    # print(adata_omics1)
    # print(adata_omics2)
    # ----------------------------------------------------------------------------------  

    # Preprocessing data
    data = preprocess(adata_omics1, adata_omics2, datatype=datatype)
    X_pca_omics1, X_pca_omics2 = data['adata_omics1'].obsm['feat'].copy(), data['adata_omics2'].obsm['feat'].copy()

    # Feature Graph Construction
    adj_omics1, adj_norm_omics1 = construct_graph(X_pca_omics1, k=opt.args.k)
    adj_omics2, adj_norm_omics2 = construct_graph(X_pca_omics2, k=opt.args.k)

    # Spatial Graph Construction
    adj_spatial, adj_spatial_norm = construct_graph(adata_omics1.obsm['spatial'], k=opt.args.spatial_k)

    # # Refine the spatial graphs using the feature graphs
    adj_spatial_refine_omics1 = refine_adj_spatial(adj_omics1, adj_spatial)
    adj_spatial_refine_omics2 = refine_adj_spatial(adj_omics2, adj_spatial)

    # Convert to tensor and move to the correct device
    X_pca_omics1, X_pca_omics2 = convert_to_tensor([X_pca_omics1, X_pca_omics2])
    adj_norm_omics1, adj_norm_omics2, adj_spatial_refine_omics1, adj_spatial_refine_omics2 = convert_to_tensor(
        [adj_norm_omics1, adj_norm_omics2, adj_spatial_refine_omics1, adj_spatial_refine_omics2])

    if opt.args.show:
        print("-------------Details Of The Dataset------------")
        print("-----------------------------------------------")
        print("Dataset name         :", opt.args.name)
        print("Omics1 shape         :", X_pca_omics1.shape)
        print("Omics2 shape         :", X_pca_omics2.shape)
        if label is not None:
            print("Category num         :", max(label) - min(label) + 1)
            print("Category distribution:")
            for i in range(max(label + 1)):
                print("Label", i, end=":")
                print(len(label[np.where(label == i)]))
        print("-----------------------------------------------")

    return X_pca_omics1, X_pca_omics2, adj_norm_omics1, adj_norm_omics2, label, adj_spatial_refine_omics1, adj_spatial_refine_omics2


def load_human_lymph_node(name):
    """
        Load human lymph node data from .h5ad files.

        Args:
            name (str): Dataset name indicating which files to load.

        Returns:
            tuple: Contains RNA and ADT AnnData objects and their labels.
    """
    adata_rna = sc.read_h5ad(f'/data/chenfuqun/01_working/04_spatial_multi-omics_review/01_spatial_multi_omics_methods/00_data/{name}/adata_RNA.h5ad')
    adata_adt = sc.read_h5ad(f'/data/chenfuqun/01_working/04_spatial_multi-omics_review/01_spatial_multi_omics_methods/00_data/{name}/adata_ADT.h5ad')

    adata_rna.var_names_make_unique()
    adata_adt.var_names_make_unique()

    # label = np.load(f'./data/10X/{name}/label.npy')
    label = adata_adt.obs["final_annot"].cat.codes

    return adata_rna, adata_adt, label


def load_mouse_brain(name):
    """
    Load mouse brain data from .h5ad files.

    Args:
        name (str): Dataset name indicating which files to load.

    Returns:
        tuple: Contains RNA and ATAC AnnData objects and their labels (None for this dataset).
    """
    adata_rna = sc.read_h5ad(f'/data/chenfuqun/01_working/04_spatial_multi-omics_review/01_spatial_multi_omics_methods/00_data/{name}/adata_RNA.h5ad')
    adata_atac = sc.read_h5ad(f'/data/chenfuqun/01_working/04_spatial_multi-omics_review/01_spatial_multi_omics_methods/00_data/{name}/adata_ATAC.h5ad')
    adata_rna.var_names_make_unique()
    adata_atac.var_names_make_unique()

    label = adata_rna.obs['Combined_Clusters']
    # print(adata_rna, adata_atac, adata_rna.obs['Combined_Clusters'])
    return adata_rna, adata_atac, label


def load_human_tonsil(name):
    """
    Load human tonsil data from .h5ad files.

    Args:
        name (str): Dataset name indicating which files to load.

    Returns:
        tuple: Contains RNA and ADT AnnData objects and their labels (None for this dataset).
    """
    adata_rna = sc.read_h5ad(f'/data/chenfuqun/01_working/04_spatial_multi-omics_review/01_spatial_multi_omics_methods/00_data/14_add_noise_and_mask_dataset/{name}/adata_RNA.h5ad')
    adata_adt = sc.read_h5ad(f'/data/chenfuqun/01_working/04_spatial_multi-omics_review/01_spatial_multi_omics_methods/00_data/14_add_noise_and_mask_dataset/{name}/adata_ADT.h5ad')
    adata_rna.var_names_make_unique()
    adata_adt.var_names_make_unique()

    label = adata_rna.obs['final_annot']
    from sklearn.preprocessing import LabelEncoder
    label_encoder = LabelEncoder()
    labels_numeric = label_encoder.fit_transform(label)
    adata_rna.obs['lab_numeric'] = labels_numeric
    print(adata_rna, adata_adt, adata_rna.obs['final_annot'])

    return adata_rna, adata_adt, adata_rna.obs['lab_numeric']


def load_human_breast_cancer(name):
    """
    Load human breast cancer data from .h5ad files.

    Args:
        name (str): Dataset name indicating which files to load.

    Returns:
        tuple: Contains RNA and ADT AnnData objects and their labels (None for this dataset).
    """
    adata_rna = sc.read_h5ad(f'/data/chenfuqun/01_working/04_spatial_multi-omics_review/01_spatial_multi_omics_methods/00_data/{name}/adata_RNA.h5ad')
    adata_adt = sc.read_h5ad(f'/data/chenfuqun/01_working/04_spatial_multi-omics_review/01_spatial_multi_omics_methods/00_data/{name}/adata_ADT.h5ad')

    adata_rna.var_names_make_unique()
    adata_adt.var_names_make_unique()
    label = None
    return adata_rna, adata_adt, label


def load_human_melanoma(name):
    """
    Load human melanoma data from .h5ad files.

    Args:
        name (str): Dataset name indicating which files to load.

    Returns:
        tuple: Contains RNA and ATAC AnnData objects and their labels (None for this dataset).
    """
    adata_rna = sc.read_h5ad(f'/data/chenfuqun/01_working/04_spatial_multi-omics_review/01_spatial_multi_omics_methods/00_data/14_add_noise_and_mask_dataset/{name}/adata_RNA.h5ad')
    adata_atac = sc.read_h5ad(f'/data/chenfuqun/01_working/04_spatial_multi-omics_review/01_spatial_multi_omics_methods/00_data/14_add_noise_and_mask_dataset/{name}/adata_ATAC.h5ad')
    adata_rna.var_names_make_unique()
    adata_atac.var_names_make_unique()
    label = adata_rna.obs['cell_type'].astype('category').cat.codes
    return adata_rna, adata_atac, label


def load_atac(name):
    adata_rna = sc.read_h5ad(f'/data/chenfuqun/01_working/04_spatial_multi-omics_review/01_spatial_multi_omics_methods/00_data/{name}/adata_RNA.h5ad')
    adata_atac = sc.read_h5ad(f'/data/chenfuqun/01_working/04_spatial_multi-omics_review/01_spatial_multi_omics_methods/00_data/{name}/adata_ATAC.h5ad')
    adata_rna.var_names_make_unique()
    adata_atac.var_names_make_unique()
    label = None

    return adata_rna, adata_atac, label

def load_mouse_breast(name):
    adata_rna = sc.read_h5ad(f'/data/chenfuqun/01_working/04_spatial_multi-omics_review/01_spatial_multi_omics_methods/00_data/{name}/adata_RNA.h5ad')
    adata_atac = sc.read_h5ad(f'/data/chenfuqun/01_working/04_spatial_multi-omics_review/01_spatial_multi_omics_methods/00_data/{name}/adata_ADT.h5ad')
    adata_rna.var_names_make_unique()
    adata_atac.var_names_make_unique()

    label = np.load("/data/chenfuqun/01_working/04_spatial_multi-omics_review/01_spatial_multi_omics_methods/00_data/07_Mouse_Breast_Tumor/labels.npy")

    return adata_rna, adata_atac, label

def load_DBiT(name):
    adata_rna = sc.read_h5ad(f'/data/chenfuqun/01_working/04_spatial_multi-omics_review/01_spatial_multi_omics_methods/00_data/{name}/adata_RNA.h5ad')
    adata_atac = sc.read_h5ad(f'/data/chenfuqun/01_working/04_spatial_multi-omics_review/01_spatial_multi_omics_methods/00_data/{name}/adata_ADT.h5ad')
    adata_rna.var_names_make_unique()
    adata_atac.var_names_make_unique()

    label = adata_rna.obs['ground_truth'].astype('category').cat.codes

    return adata_rna, adata_atac, label

