from time import time
from xml.sax.handler import feature_validation
from sklearn.metrics import (adjusted_rand_score, normalized_mutual_info_score, 
                             silhouette_score, calinski_harabasz_score,
                             davies_bouldin_score)
import logging
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from graphmae.utils import (
    
    build_args,
    create_optimizer,
    mask_edge,
    set_random_seed,
    TBLogger,
    get_current_lr,
    load_best_configs,

    
)
from scanpy import read_10x_h5
from collections import Counter
from graphmae.datasets.data_util import load_dataset
from graphmae.evaluation import node_classification_evaluation
from graphmae.models import build_model
from ogb.nodeproppred import DglNodePropPredDataset
from sklearn.cluster import KMeans
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
import warnings
warnings.filterwarnings("ignore")
columns  = ["dataset_name","graph_devise","threshold_num","node_num","edge_num",
            "feature_dim","feature_dim_num","mask_rate","encode_network","decode_network",
            "num_hidden","num_layers","activation","max_epoch","lr","first_result","second_result"]
result_pd = pd.DataFrame(columns  = columns)
class myDict(object):
    def __init__(self,mydict):
        self.mydict = mydict
        self.length = []
        self.keys = []
        for key,values in self.mydict.items():
            self.keys.append(key)
            self.length.append(len(values))
        self.nums = [1] * len(self.length)
        for i in range(len(self.length)):
            for j in range(i,len(self.length)):
                self.nums[i] *= self.length[j]
        self.para_dis = []
        print(self.length)
        print(self.nums)
                
    def getindex(self,index):
        result = []
        value = index
        for i in range(len(self.nums) - 1):
            result.append(value // self.nums[i+1])
            value = value - result[i] * self.nums[i+1]
        result.append(value) 
        result_dict = dict()
        for index,value in enumerate(result):
            result_dict[self.keys[index]] = self.mydict.get(self.keys[index])[value]
        return result_dict
    
    #para_dis = []
    def myiter(self):
        #para_dis = []
        for i in range(0,self.nums[0]):
            self.para_dis.append(self.getindex(i))
        return self.para_dis
def kMeans_use(embedding,cluster_number):
    kmeans = KMeans(n_clusters=cluster_number,
                init="k-means++",
                random_state=0)
    pred = kmeans.fit_predict(embedding)
    return pred
def pretrain(model, graph, feat, optimizer, max_epoch, device, scheduler, num_classes, lr_f, weight_decay_f, max_epoch_f, linear_prob, logger=None):
    logging.info("start training..")
    graph = graph.to(device)
    x = feat.to(device)

    epoch_iter = tqdm(range(max_epoch))

    for epoch in epoch_iter:
        model.train()

        loss, loss_dict = model(graph, x)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        epoch_iter.set_description(f"# Epoch {epoch}: train_loss: {loss.item():.4f}")
        if logger is not None:
            loss_dict["lr"] = get_current_lr(optimizer)
            logger.note(loss_dict, step=epoch)

        #if (epoch + 1) % 200 == 0:
            #node_classification_evaluation(model, graph, x, num_classes, lr_f, weight_decay_f, max_epoch_f, device, linear_prob, mute=True)

    # return best_model
    return model
import argparse
parser = argparse.ArgumentParser(description="GAT")
parser.add_argument("--seeds", type=int, nargs="+", default=[0])
parser.add_argument("--dataset", type=str, default="cora")
parser.add_argument("--device", type=int, default=-1)
parser.add_argument("--max_epoch", type=int, default=200,
                    help="number of training epochs")
parser.add_argument("--warmup_steps", type=int, default=-1)

parser.add_argument("--num_heads", type=int, default=4,
                    help="number of hidden attention heads")
parser.add_argument("--num_out_heads", type=int, default=1,
                    help="number of output attention heads")
parser.add_argument("--num_layers", type=int, default=2,
                    help="number of hidden layers")
parser.add_argument("--num_hidden", type=int, default=256,
                    help="number of hidden units")
parser.add_argument("--residual", action="store_true", default=False,
                    help="use residual connection")
parser.add_argument("--in_drop", type=float, default=.2,
                    help="input feature dropout")
parser.add_argument("--attn_drop", type=float, default=.1,
                    help="attention dropout")
parser.add_argument("--norm", type=str, default=None)
parser.add_argument("--lr", type=float, default=0.005,
                    help="learning rate")
parser.add_argument("--weight_decay", type=float, default=5e-4,
                    help="weight decay")
parser.add_argument("--negative_slope", type=float, default=0.2,
                    help="the negative slope of leaky relu for GAT")
parser.add_argument("--activation", type=str, default="prelu")
parser.add_argument("--mask_rate", type=float, default=0.5)
parser.add_argument("--drop_edge_rate", type=float, default=0.0)
parser.add_argument("--replace_rate", type=float, default=0.0)

parser.add_argument("--encoder", type=str, default="gat")
parser.add_argument("--decoder", type=str, default="gat")
parser.add_argument("--loss_fn", type=str, default="byol")
parser.add_argument("--alpha_l", type=float, default=2, help="`pow`inddex for `sce` loss")
parser.add_argument("--optimizer", type=str, default="adam")

parser.add_argument("--max_epoch_f", type=int, default=30)
parser.add_argument("--lr_f", type=float, default=0.001, help="learning rate for evaluation")
parser.add_argument("--weight_decay_f", type=float, default=0.0, help="weight decay for evaluation")
parser.add_argument("--linear_prob", action="store_true", default=False)

parser.add_argument("--load_model", action="store_true")
parser.add_argument("--save_model", action="store_true")
parser.add_argument("--use_cfg", action="store_true")
parser.add_argument("--logging", action="store_true")
parser.add_argument("--scheduler", action="store_true", default=False)
parser.add_argument("--concat_hidden", action="store_true", default=False)

# for graph classification
parser.add_argument("--pooling", type=str, default="mean")
parser.add_argument("--deg4feat", action="store_true", default=False, help="use node degree as input feature")
parser.add_argument("--batch_size", type=int, default=32)
columns  = ["dataset_name","graph_devise","threshold_num","node_num","edge_num",
            "feature_dim","feature_dim_num","mask_rate","encode_network","decode_network","norm",
            "num_hidden","num_layers","activation","max_epoch","lr","first_result","second_result"]
parameters = {
    "dataset_name":("151673",),
    "graph_devise":("CCST",),
    "threshold_num":(8,),
    "feature_dim":("PCA",),
    "feature_dim_num" : (600,),
    "mask_rate" : np.arange(0.3,0.6,0.1),
    "code_networt_and_norm" : (["gat","dotgat",None],["dotgat","gat",None]),
    "num_hidden" : (128,256,512),
    "num_layers" : (1,2,3),
    "activation" : ("relu","gelu","prelu","elu"),
    "max_epoch" : np.arange(500,3000,500),
    "lr" : (0.001,),
}
# parameters = {
#     "dataset_name":("151673",),
#     "graph_devise":("CCST",),
#     "threshold_num":(8,),
#     "feature_dim":("PCA",),
#     "feature_dim_num" : (600,),
#     "mask_rate" : (0.4,),
#     "code_networt_and_norm" : (["gat",None],),
#     "num_hidden" : (128,),
#     "num_layers" : (1,),
#     "activation" : ("relu",),
#     "max_epoch" : (500,),
#     "lr" : (0.001,),
# }
mydict = myDict(parameters)
parameters_list  = mydict.myiter()
result_file = "/home/sunhang/Embedding/Spatial_dataset/DLPFC"
time = 0
args = parser.parse_args([])
for choose_parameter in parameters_list:

    #args = parser.parse_args([])
    args.lr = choose_parameter["lr"]
    args.lr_f = 0.01
    args.num_hidden = choose_parameter["num_hidden"]
    args.num_heads = 4
    args.weight_decay = 2e-4
    args.weight_decay_f= 1e-4
    args.max_epoch= choose_parameter["max_epoch"]
    args.max_epoch_f= 300
    args.mask_rate= choose_parameter["mask_rate"]
    args.num_layers= choose_parameter["num_layers"]
    args.encoder= choose_parameter["code_networt_and_norm"][0]
    args.decoder= choose_parameter["code_networt_and_norm"][1]
    args.norm = choose_parameter["code_networt_and_norm"][2]
    args.activation= choose_parameter["activation"]
    args.in_drop= 0.2
    args.attn_drop= 0.1
    args.linear_prob= True
    args.loss_fn= "sce" 
    args.drop_edge_rate=0.0
    args.optimizer= "adam"
    args.replace_rate= 0.05 
    args.alpha_l= 3
    args.scheduler= True
    args.dataset = choose_parameter["dataset_name"]
    pca_n_comps = choose_parameter["feature_dim_num"]


    #默认参数传递
    device = args.device if args.device >= 0 else "cpu"
    seeds = args.seeds
    dataset_name = args.dataset
    max_epoch = args.max_epoch
    max_epoch_f = args.max_epoch_f
    num_hidden = args.num_hidden
    num_layers = args.num_layers
    encoder_type = args.encoder
    decoder_type = args.decoder
    replace_rate = args.replace_rate

    optim_type = args.optimizer 
    loss_fn = args.loss_fn

    lr = args.lr
    weight_decay = args.weight_decay
    lr_f = args.lr_f
    weight_decay_f = args.weight_decay_f
    linear_prob = args.linear_prob
    load_model = args.load_model
    save_model = args.save_model
    logs = args.logging
    use_scheduler = args.scheduler



    folder_name = "/home/sunhang/Embedding/Spatial_dataset/DLPFC"
    sample_name = dataset_name
    gene_loc_data_file = folder_name + "/" +sample_name+ "/" + sample_name + "_DLPFC_col_name.csv"
    adata_file = folder_name + "/" +sample_name+ "/" + sample_name + "_filtered_feature_bc_matrix.h5"
    gene_loc_data_csv = pd.read_csv(gene_loc_data_file,index_col=0)
    gene_loc_data_csv.index = gene_loc_data_csv.barcode
    gene_loc_data_csv = gene_loc_data_csv.fillna("None")
    le = LabelEncoder()
    label = le.fit_transform(gene_loc_data_csv['layer_guess_reordered_short'])
    gene_loc_data_csv["lay_num"] = label
    num_classes = len(set(gene_loc_data_csv.lay_num))
    if((gene_loc_data_csv['layer_guess_reordered_short'] == "None").any()):
        num_classes = len(set(gene_loc_data_csv.lay_num)) - 1

    # Create a group with location informatio
    row_name = "imagerow"
    col_name = "imagecol"
    cell_loc = gene_loc_data_csv[[row_name,col_name]].values
    distance_np = pdist(cell_loc, metric = "euclidean")
    distance_np_X =squareform(distance_np)
    distance_loc_csv = pd.DataFrame(index=gene_loc_data_csv.index, columns=gene_loc_data_csv.index,data = distance_np_X)
    threshold = 8
    num_big = np.where((0< distance_np_X)&(distance_np_X < threshold))[0].shape[0]
    #num_big = np.where((0< distance_np_X)&(distance_np_X < threshold))[0].shape[0]
    adj_matrix = np.zeros(distance_np_X.shape)
    non_zero_point = np.where((0 < distance_np_X) & (distance_np_X < threshold))
    adj_matrix = np.zeros(distance_np_X.shape)
    non_zero_point = np.where((0< distance_np_X)&(distance_np_X<threshold))
    for i in range(num_big):
        x = non_zero_point[0][i]
        y = non_zero_point[1][i]
        adj_matrix[x][y] = 1 
    adj_matrix = adj_matrix + np.eye(distance_np_X.shape[0])
    adj_matrix  = np.float32(adj_matrix)
    adj_matrix_crs = sparse.csr_matrix(adj_matrix)
    graph = dgl.from_scipy(adj_matrix_crs,eweight_name='w')


    adata = read_10x_h5(adata_file)
    adata.obs = pd.merge(adata.obs,gene_loc_data_csv,left_index=True,right_index=True)
    adata.var_names=[i.upper() for i in list(adata.var_names)]
    adata.var["genename"]=adata.var.index.astype("str")
    adata.var_names_make_unique

    sc.pp.filter_genes(adata, min_cells=5)
    adata_X = sc.pp.normalize_total(adata, target_sum=1, exclude_highly_expressed=True, inplace=False)['X']
    adata_X = sc.pp.scale(adata_X)
    adata_X = sc.pp.pca(adata_X, n_comps=pca_n_comps)
    graph.ndata["feat"] = torch.tensor(adata_X.copy())
    num_features = graph.ndata["feat"].shape[1]
    args.num_features = num_features
    acc_list = []
    estp_acc_list = []
    times = 3

    #print(f"####### Run {i} for seed {seed}")
    #print(i)
    seed = 0
    set_random_seed(seed)
    if logs:
        logger = TBLogger(name=f"{dataset_name}_loss_{loss_fn}_rpr_{replace_rate}_nh_{num_hidden}_nl_{num_layers}_lr_{lr}_mp_{max_epoch}_mpf_{max_epoch_f}_wd_{weight_decay}_wdf_{weight_decay_f}_{encoder_type}_{decoder_type}")
    else:
        logger = None
    model = build_model(args)
    device = 1
    model.to(device)
    optimizer = create_optimizer(optim_type, model, lr, weight_decay)

    #         if use_scheduler:
    #             logging.info("Use schedular")
    #             scheduler = lambda epoch :( 1 + np.cos((epoch) * np.pi / max_epoch) ) * 0.5
    #             # scheduler = lambda epoch: epoch / warmup_steps if epoch < warmup_steps \
    #                     # else ( 1 + np.cos((epoch - warmup_steps) * np.pi / (max_epoch - warmup_steps))) * 0.5
    #             scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=scheduler)
    #         else:
    #             scheduler = None
    #训练模型
    scheduler = None
    x = graph.ndata["feat"]
    if not load_model:
        model = pretrain(model, graph, x, optimizer, max_epoch, device, scheduler, num_classes, lr_f, weight_decay_f, max_epoch_f, linear_prob, logger)
        #model = model.cpu()
    x = graph.ndata["feat"]
    #model.to(device)
    embedding = model.embed(graph.to(device), x.to(device))
    new_pred = kMeans_use(embedding.cpu().detach().numpy(),num_classes)
    adata.obs["pre"] = new_pred
    score = adjusted_rand_score(adata.obs.lay_num.values, new_pred )
    first_result = score
    test = model.embed(graph.to(device), x.to(device))
    test_new_pred = kMeans_use(test.cpu().detach().numpy(),num_classes)
    score = adjusted_rand_score(adata.obs.lay_num.values, test_new_pred )
    adata.obs["second_pre"] = test_new_pred
    second_result = score
    #可调参数
    dataset_name = sample_name
    graph_devise = choose_parameter["graph_devise"]
    threshold_num = threshold
    node_num = graph.num_nodes()
    edge_num = graph.num_edges()
    feature_dim = choose_parameter["feature_dim"]
    feature_dim_num = pca_n_comps
    norm = args.norm
    mask_rate =  args.mask_rate
    encode_network = args.encoder
    decode_network = args.decoder
    num_hidden = args.num_hidden
    num_layers = args.num_layers
    activation = args.activation
    max_epoch = args.max_epoch
    lr = args.lr
    #first_result = 0.1

    result_in = [dataset_name,graph_devise,threshold_num,node_num,edge_num,
                feature_dim,feature_dim_num,mask_rate,encode_network,decode_network,norm,
                num_hidden,num_layers,activation,max_epoch,lr,first_result,second_result]
    time = time + 1
    #print(result_in)
    #print(columns)
    #print(time % 100 == 1)
    result_Series = pd.Series(result_in,index=columns)
    result_pd = result_pd.append(result_Series,ignore_index=True)
    if(time % 100 == 1):
        in_result_file = result_file + "/" +sample_name + "_" + str(time) + "_9.13_result.csv"
        result_pd.to_csv(in_result_file)
in_result_file = result_file + "/" +sample_name + "_" + str(time) + "_9.14_en_de_result.csv"
result_pd.to_csv(in_result_file)