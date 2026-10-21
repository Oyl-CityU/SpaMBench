import pandas as pd
import numpy as np
import scanpy as sc
import anndata as ad
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix as csr
from scipy.sparse import issparse
from scipy.linalg import block_diag
from sklearn.metrics import pairwise_distances
from sklearn.decomposition import non_negative_factorization
from sklearn.cluster import KMeans
from numpy.linalg import norm
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

class SEPARmult:
    def __init__(self, adata_list, section_ids, n_cluster):
        self.n_cluster = n_cluster
        self.adata_list = adata_list
        self.section_ids = section_ids
        self.exp_mat_list = []
        self.loc_list = []
        self.stgraph_list = []
        self.G1_list = []
        self.adj_list = []
        self.U = None
        self.V = None
        self.Wpn = None
        self.Hpn = None
        self.labelres = None
        self.adata_concat = None
        self.adj_concat = None
        self.exp_mat = None
        
        for adata in self.adata_list:
            if issparse(adata.X):
                self.exp_mat_list.append(adata.X.toarray().astype(np.float16))
            else:
                self.exp_mat_list.append(adata.X.astype(np.float16))

    def preprocess(self, min_cells=0, normalize=True, n_top_genes=3000):
        for adata in self.adata_list:
            adata.var_names_make_unique()
            sc.pp.filter_genes(adata, min_cells=min_cells)
            sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=n_top_genes)
            if normalize:
                sc.pp.normalize_total(adata, target_sum=1e4)
                sc.pp.log1p(adata)
            print(f'After filtering and preprocessing: {adata.shape}')

    def compute_graph(self, radius_rate=1.2):
        for i, adata in enumerate(self.adata_list):
            loc = np.array(adata.obsm['spatial'])
            pair_dist = pairwise_distances(loc)
            
            scale = loc.max(0) - loc.min(0)
            radius = np.sqrt(scale[0] * scale[1] / loc.shape[0])
            rad_cutoff1 = radius * radius_rate
            G1 = (pair_dist < rad_cutoff1).astype(float)

            adata.obsp["connectivities"] = csr(G1)
            Gedge = G1.sum(0)
            
            # Print the mean number of neighbors for each sample
            mean_neighbors = Gedge.mean()
            print(f'Mean number of neighbors for sample {self.section_ids[i]}: {mean_neighbors:.2f}')
            
            row_sum = G1.sum(1)[:, np.newaxis]
            stgraph = np.power(row_sum, -1) * G1
            self.stgraph_list.append(csr(stgraph))
            self.G1_list.append(G1)
            self.loc_list.append(loc)
            self.adj_list.append(csr(G1))


    def filter_shared_genes(self):  
        # Get intersection of var_names across all AnnData objects  
        shared_genes = set(self.adata_list[0].var_names)  
        for adata in self.adata_list[1:]:  
            shared_genes = shared_genes.intersection(set(adata.var_names))  
        
        # Filter each AnnData to keep only shared genes  
        shared_genes = list(shared_genes)  
        for i, adata in enumerate(self.adata_list):  
            self.adata_list[i] = adata[:, shared_genes].copy()        
        print(f"Number of shared genes: {len(shared_genes)}")

    def select_morani(self, nslt=1000):
        morani_list = []
        for adata in self.adata_list:
            morani = sc.metrics.morans_i(adata)
            morani_list.append(morani)

        # Select top genes based on Moran's I across all samples
        morani_combined = np.mean(morani_list, axis=0)
        m_order = np.flip(np.argsort(morani_combined))
        slt_m = m_order[:nslt]

        for i, adata in enumerate(self.adata_list):
            self.adata_list[i] = adata[:, slt_m]
            self.exp_mat_list[i] = self.adata_list[i].X.toarray().astype(np.float16)
        print('Moran\'s I selection finished')
    


    def concat_data(self, annotation = False):
        self.adata_concat = ad.concat(self.adata_list, label="slice_name", keys=self.section_ids)
        if annotation:
            self.adata_concat.obs['Ground Truth'] = self.adata_concat.obs['Ground Truth'].astype('category')
        self.adata_concat.obs["batch_name"] = self.adata_concat.obs["slice_name"].astype('category')

        # Concatenate adjacency matrices
        self.adj_concat = self.adj_list[0].todense()
        for batch_id in range(1, len(self.section_ids)):
            self.adj_concat = block_diag(self.adj_concat, self.adj_list[batch_id].todense())
        
        self.adj_concat = csr(self.adj_concat)
        self.adata_concat.uns['edgeList'] = np.nonzero(self.adj_concat)
        self.exp_mat = self.adata_concat.X.toarray().astype(np.float16)
        print('Data concatenation finished')

    def compute_weight(self, metric='cosine', n_cluster=8):
        exp_mat_mean_old = self.adj_concat.dot(self.adata_concat.X.toarray().astype(np.float16))
        exp_pair_dist = pairwise_distances(exp_mat_mean_old, metric=metric)
        d_star = np.percentile(exp_pair_dist.flatten(), 100 / n_cluster)
        sigma = np.sqrt(2) * d_star
        w_pd = np.exp(-exp_pair_dist**2 / sigma**2)

        wG = ((self.adj_concat > 0).astype(np.int16).multiply(w_pd)).toarray()
        wG = wG - np.eye(wG.shape[0])
        wD = wG.sum(axis=0)
        self.wG = wG
        self.wD = wD

        self.wGt = wG + np.eye(wG.shape[0])
        row_sum = self.wGt.sum(1)[:, np.newaxis]
        self.wGt = np.power(row_sum, -1) * self.wGt

    def separ_algorithm(self, r, alpha, beta, gamma, max_iter=100, mean=True):  
        """  
        Implementation of SEPAR algorithm using graph regularized NMF.  
        
        The optimization problem is:  
        min_{W≥0,H≥0} ||X-WH||_F^2 + α*tr(Ω W^T L W) + β||W||_1 + (γ/2)∑_{i<j}⟨h_i,h_j⟩^2  
        
        Parameters  
        ----------  
        r : int  
            Number of spatial patterns to identify  
        alpha : float  
            Weight parameter for graph regularization term (tr(Ω W^T L W))  
        beta : float  
            Weight parameter for sparsity penalty (||W||_1)  
        gamma : float  
            Weight parameter for pattern orthogonality term  
        max_iter : int, default=100  
            Maximum number of iterations  
        mean : bool, default=True  
            Whether to use mean-adjusted expression matrix  
        """  
        # Initialize expression matrix  
        X = self.adj_concat.dot(self.exp_mat) if mean else self.exp_mat  
        X = np.maximum(X, 0)  

        # Transpose and normalize  
        Xt = X.T  
        norms = norm(Xt, axis=0)  
        Xt = Xt / (norms + 1e-10)  
        self.Xt = Xt  

        # Initialize W and H using standard NMF  
        W, H, _ = non_negative_factorization(Xt, n_components=r, random_state=0)  
        self.U = W.copy()  # H in the paper (gene loadings)  
        self.V = H.T.copy()  # W in the paper (spatial patterns)  
        
        # Normalize patterns  
        ul = np.linalg.norm(self.U, axis=0)  
        self.U = self.U / ul  
        self.V = self.V * ul  
        self.V = self.V.T  

        # Get graph Laplacian components  
        W_mean = csr(self.wG)  # Graph weight matrix  
        d_mean = self.wD      # Diagonal degree matrix  

        self.err_list = []  

        # Iterative optimization  
        for i in tqdm(range(max_iter), desc="Processing iterations"):  
            # Calculate Pattern Significance Scores (PSS)  
            pss = self.sim_res(self.V.T, self.U.T, X)  
            
            # Update W (spatial patterns)  
            self.U[self.U == 0] = 1e-10  
            self.V = self.V * (self.U.T @ Xt + alpha * (W_mean.dot(self.V.T) * pss).T) / (  
                (self.U.T @ self.U) @ self.V + alpha * pss.reshape([r, 1]) * (self.V * d_mean) + beta)  
            self.V[self.V == 0] = 1e-10  

            # Update H (gene loadings)  
            self.U = self.U * (Xt @ self.V.T) / (  
                self.U @ (self.V @ self.V.T) + gamma * self.U @ (self.U.T @ self.U - np.eye(r)))  
            
            # Calculate reconstruction error  
            err = np.linalg.norm(Xt - self.U @ self.V, ord='fro')  
            self.err_list.append(err)  

            # Normalize patterns  
            ull = np.linalg.norm(self.U, axis=0)  
            self.U = self.U / (ull + 1e-10)  
            self.V = (self.V.T * ull).T  

        # Store final results  
        self.Wpn = self.V.T  # Spatial patterns  
        self.Hpn = self.U.T  # Gene loadings


    def clustering(self, n_cluster=8, N1=None, N2=3):  
        """  
        Perform spatial domain identification using filtered patterns.  
        
        Parameters  
        ----------  
        n_cluster : int  
            Number of spatial domains to identify  
        N1 : int, optional  
            Number of patterns to exclude based on l2-norm  
            If None, will be set to r - 5 - n_cluster, where r is the number of patterns  
        N2 : int, default=3  
            Number of additional patterns to exclude based on PSS  
        """  
        # Calculate Pattern Specificity Scores (PSS)  
        sim_all = self.sim_res(self.Wpn, self.Hpn, self.exp_mat)  
        
        # Calculate l2-norms of patterns  
        pattern_norms = np.linalg.norm(self.Wpn, axis=0)  
        
        # First filtering: Remove patterns with small l2-norms  
        if N1 is None:  
            N1 = self.Wpn.shape[1] - 5 - n_cluster  # r - 5 - n_cluster  
        norm_threshold = np.sort(pattern_norms)[-N1]  
        norm_mask = pattern_norms > norm_threshold  
        
        # Apply first filter to PSS scores  
        filtered_pss = sim_all.copy()  
        filtered_pss[~norm_mask] = 0  
        
        # Second filtering: Remove patterns with low PSS  
        pss_threshold = np.sort(filtered_pss)[-(self.Wpn.shape[1]-N1-N2)]  
        pss_mask = filtered_pss >= pss_threshold  
        
        # Get filtered pattern matrix  
        Ws = self.Wpn[:, pss_mask]  
        
        # Perform k-means clustering  
        kmeans = KMeans(n_clusters=n_cluster, random_state=0).fit(Ws / np.max(Ws, axis=0))  
        self.labelres = kmeans.labels_  
        
        # Store results in AnnData object  
        self.adata_concat.obs['clustering'] = self.labelres.astype(str) 
        return kmeans.labels_


    def recognize_svgs(self, err_tol=0.7):  
        """  
        Identify spatially variable genes and return their ranking.  
        
        Parameters:  
        -----------  
        err_tol : float  
            Error tolerance threshold for SVG identification  
            
        Returns:  
        --------  
        dict : Contains SVGs, all genes ranking, and error rates  
        """  
        Xn = self.Xt.T  
        whn = self.Wpn @ self.Hpn  
        resn = Xn - whn  
        err_rate_p = norm(resn, axis=0) / norm(Xn, axis=0)  
        
        # Sort genes by error rate  
        gene_ranking = np.argsort(err_rate_p)  
        
        # Identify SVGs  
        svg_mask = err_rate_p < err_tol  
        svgs = self.adata_concat.var_names[svg_mask]  
        
        return {  
            'svgs': svgs,  
            'gene_ranking': gene_ranking,  
            'error_rates': err_rate_p  
        }  

    def identify_pattern_specific_genes(self, n_patterns=30, threshold=0.3, normalize='l2'):  
        """  
        Identify and rank genes specific to each spatial pattern.  
        
        Parameters  
        ----------  
        n_patterns : int, default=30  
            Number of patterns to analyze  
        threshold : float, default=0.3  
            Threshold for determining pattern-specific genes  
        normalize : str, default='l2'  
            Normalization method ('l2' or 'max')  
            
        Returns  
        -------  
        list of list  
            For each pattern, returns a list of genes sorted by specificity  
        """  
        # Normalize patterns  
        if normalize == 'l2':  
            pattern_norms = np.linalg.norm(self.Wpn, axis=0)  
        elif normalize == 'max':  
            pattern_norms = np.max(np.abs(self.Wpn), axis=0)  
        
        # Normalize W and H matrices  
        Wpnn = self.Wpn / pattern_norms  
        Hpnn = (self.Hpn.T * pattern_norms).T  
        self.Hpnr = Hpnn.copy()  # Store raw normalized H  
        
        # Normalize H columns to sum to 1  
        Hpnn = Hpnn / Hpnn.sum(0)  
        
        # Store normalized matrices  
        self.Wpnn = Wpnn  
        self.Hpnn = Hpnn  
        
        # Initialize results  
        pattern_genes = []  
        genes_per_pattern = np.zeros([n_patterns,])  
        pattern_specific_mask = []  
        
        # Process each pattern  
        for i in range(n_patterns):  
            # Get gene loadings for current pattern  
            pattern_loadings = Hpnn[i, :]  
            
            # Identify specific genes using threshold  
            is_specific = pattern_loadings > threshold  
            genes_per_pattern[i] = np.sum(is_specific)  
            
            # If no genes pass threshold, use the highest loading  
            if genes_per_pattern[i] < 1:  
                is_specific = pattern_loadings > pattern_loadings[np.argsort(-pattern_loadings)[0]]  
            
            # Store binary mask  
            pattern_specific_mask.append(is_specific)  
            
            # Get gene names and their loadings  
            gene_names = self.adata_concat.var_names  
            specific_genes = gene_names[is_specific]  
            specific_loadings = pattern_loadings[is_specific]  
            
            # Sort genes by loading values  
            sorted_idx = np.argsort(-specific_loadings)  
            sorted_genes = specific_genes[sorted_idx]  
            
            pattern_genes.append(sorted_genes)  
        
        # Store results as attributes  
        self.pattern_specific_mask = pattern_specific_mask  # Boolean mask for each pattern  
        self.genes_per_pattern = genes_per_pattern  # Number of genes per pattern  
        
        return pattern_genes  

    def get_refined_expression(self):  
        """  
        Get refined gene expression matrix after denoising.  
        
        Returns  
        -------  
        AnnData  
            AnnData object containing refined expression data with original   
            gene names and spot indices  
        """  
        # Calculate refined expression matrix  
        refined_exp = self.Wpn @ self.Hpn  
        
        # Create new AnnData object with refined expression  
        adata_refined = self.adata_concat.copy()  
        adata_refined.X = refined_exp  
        
        return adata_refined  

    @staticmethod  
    def sim_res(Wpn1, Hpn1, X):  
        """  
        Calculate Pattern Significance Scores (PSS) for spatial patterns.  
        
        Parameters  
        ----------  
        Wpn1 : ndarray, (n_spots, n_patterns)  
            Spatial pattern matrix  
        Hpn1 : ndarray, (n_patterns, n_genes)  
            Gene loading matrix  
        X : ndarray, (n_spots, n_genes)  
            Gene expression matrix  
            
        Returns  
        -------  
        ndarray, (n_patterns,)  
            Pattern Significance Scores    
        """  
        # Calculate pattern normalization factors  
        col_sums = Wpn1.sum(axis=0)  
        non_zero_cols = col_sums != 0  
        
        # Initialize significance scores  
        sim_all1 = np.zeros(Wpn1.shape[1])  
        
        if non_zero_cols.any():  
            # Project and normalize expression matrix  
            Xi = (X.T @ Wpn1[:, non_zero_cols]) / col_sums[non_zero_cols]  
            Hi = Hpn1[non_zero_cols, :]  
            
            # Compute cosine similarity  
            xi_norms = np.sqrt(np.sum(Xi**2, axis=0))  
            hi_norms = np.sqrt(np.sum(Hi**2, axis=1))  
            numerator = np.sum(Xi.T * Hi, axis=1)  
            sim_all1[non_zero_cols] = numerator / (xi_norms * hi_norms)  
        
        return sim_all1
