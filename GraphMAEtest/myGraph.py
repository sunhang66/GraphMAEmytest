import pandas as pd
#import calculate_adj
import anndata as ad
import scanpy as sc
from operator import index
import re
from scipy.spatial.distance import pdist, squareform
import pandas as pd
import numpy as np
from scipy import sparse
import dgl
import torch
folder_name = "/home/sunhang/Embedding/CCST/dataset/DLPFC/"
samplea_list = {151507,
 151508,
 151509,
 151510,
 151669,
 151670,
 151671,
 151672,
 151673,
 151674,
 151675,
 151676}
sample_name = str(151673)
gene_exp_data_file = folder_name + sample_name + "_DLPFC_count.csv"
gene_loc_data_file = folder_name + sample_name + "_DLPFC_col_name.csv"
#save file
npz_file = folder_name + sample_name + "_DLPFC_distacne.npz"
gene_csv = pd.read_csv(gene_exp_data_file,index_col= 0 )
gene_loc_data_csv = pd.read_csv(gene_loc_data_file,index_col=0)
gene_loc_data_csv = gene_loc_data_csv.fillna("None")
row_name = "imagerow"
col_name = "imagecol"
cell_loc = gene_loc_data_csv[[row_name,col_name]].values
distance_np = pdist(cell_loc, metric = "euclidean")
distance_np_X =squareform(distance_np)
distance_loc_csv = pd.DataFrame(index=gene_loc_data_csv.index, columns=gene_loc_data_csv.index,data = distance_np_X)
threshold = 10
num_big = np.where((0< distance_np_X)&(distance_np_X < threshold))[0].shape[0]
adj_matrix = np.zeros(distance_np_X.shape)
non_zero_point = np.where((0 < distance_np_X) & (distance_np_X < threshold))
adj_matrix = np.zeros(distance_np_X.shape)
non_zero_point = np.where((0< distance_np_X)&(distance_np_X<threshold))
for i in range(num_big):
    x = non_zero_point[0][i]
    y = non_zero_point[1][i]
    adj_matrix[x][y] = 1 
adj_matrix  = np.float32(adj_matrix)
adj_matrix_crs = sparse.csr_matrix(adj_matrix)
graph = dgl.from_scipy(adj_matrix_crs,eweight_name='w')
min_cells = 5
pca_n_comps = 200
adata = ad.AnnData(gene_csv.values.T, obs=distance_loc_csv, var=pd.DataFrame(index = gene_csv.index), dtype='int32')
sc.pp.filter_genes(adata, min_cells=5)
adata_X = sc.pp.normalize_total(adata, target_sum=1, exclude_highly_expressed=True, inplace=False)['X']
adata_X = sc.pp.scale(adata_X)
adata_X = sc.pp.pca(adata_X, n_comps=pca_n_comps)
from sklearn.preprocessing import LabelEncoder
# Creating a instance of label Encoder.
le = LabelEncoder()
# Using .fit_transform function to fit label
# encoder and return encoded label
label = le.fit_transform(gene_loc_data_csv['layer_guess_reordered_short'])
gene_loc_data_csv["lay_num"] = label
graph.ndata["feat"] = torch.tensor(adata_X.copy())
num_features = graph.ndata["feat"].shape[1]
num_classes = len(set(gene_loc_data_csv.lay_num))