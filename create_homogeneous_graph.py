import pandas as pd 
import itertools
# 011022.py
import torch
import random
from torch_geometric.data import Data

from torch_geometric.data import InMemoryDataset, download_url
from torch_geometric.utils import add_self_loops
import glob
import os
import re
from datetime import datetime
import numpy as np
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import colors
from matplotlib import cm
import seaborn as sns
import pickle
from HybridHomogeneous import DATE_TIME
from edge_strategies import feature_anomaly_edges_automatic, knn_graph, radius_graph
from read_embeddings import get_embeddings_df
# from dp_models.class_data_lr_homogeneous_node_graph import train 
"""# train lstm: dp190922.py
# load_embeddings.py
# Saved:  ./train_embedding_2022-09-29 21:32:26.pkl
# In:  /home/dproios/work/create_EHR_gra           
# Getting embeddings for:  val                     
# Embeddings from val_embedding out of 6371:   2%|▏
#            | 100/6371 [00:12<12:34,  8.31it/s]
# Saved:  ./val_embedding_2022-09-29 21:32:26.pkl
# In:  /home/dproios/work/create_EHR_gra           
# Getting embeddings for:  test                    
# Embeddings from test_embedding out of 6281:   2%|
# ▏           | 99/6281 [00:12<13:23,  7.70it/s]
# Saved:  ./test_embedding_2022-09-29 21:32:26.pkl
"""


"""###### output exampple of of read_lstm_embeddings
# read_embedings.py"""


class MyOwnDataset(InMemoryDataset):
    def __init__(self, root, node_type_embeddings='lstm', edge_strategy_name='quantile', 
                 k = 10, 
                 n_edges = None ,
                 distance_euclidean=True, transform=None, pre_transform=None, pre_filter=None):
                #  edge_index, train_X, train_y, val_X, val_y, test_X, test_y,
        # self.edge_index, self.train_X, self.train_y, self.val_X, self.val_y, self.test_X, self.test_y = edge_index, train_X, train_y, val_X, val_y, test_X, test_y
        self.edge_strategy_name= edge_strategy_name
        self.k =k
        self.n_edges=n_edges
        self.distance_euclidean=distance_euclidean
        self.node_type_embeddings= node_type_embeddings
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    def raw_file_names(self):
        return ['some_file_1', 'some_file_2']

    @property
    def processed_file_names(self):
        return ['data.pt']

    def process(self):
        print('Read Nodes ')
        
        train_df, val_df, test_df = get_embeddings_df()

        # pd.DataFrame([(item[0], item[1].shape) for item in result.items()])
        # import pdb; pdb.set_trace()

        if self.node_type_embeddings == 'lstm':
            train_X = train_df['lstm_embedding']
            val_X = val_df['lstm_embedding']
            test_X = test_df['lstm_embedding']
            train_y = train_df['ys']
            val_y = val_df['ys']
            test_y = test_df['ys']
        if self.node_type_embeddings == 'grnn':
            train_X = train_df['grnn_embedding']
            val_X = val_df['grnn_embedding']
            test_X = test_df['grnn_embedding']
            train_y = train_df['ys']
            val_y = val_df['ys']
            test_y = test_df['ys']
        if self.node_type_embeddings == 'stat':
            train_X = train_df['stat_features']
            val_X = val_df['stat_features']
            test_X = test_df['stat_features']
            train_y = train_df['ys']
            val_y = val_df['ys']
            test_y = test_df['ys']
            
            
        X_all = torch.cat((torch.Tensor(train_X),torch.Tensor(val_X),torch.Tensor(test_X),), 0)
        
        print('Edges')
        print('edge_strategy_name: ', self.edge_strategy_name)
        # if self.edge_strategy_name=='fully_connected':
        if self.edge_strategy_name=='random':
            # edge_strategy = feature_anomaly_edges_automatic
            n_nodes = len(X_all)
            A = [(X // n_nodes, X % n_nodes) for X in random.choices(range(n_nodes**2), k=self.n_edges)]
            print(f'Created random {len(A)} edges')
            # import pdb; pdb.set_trace()
            # edge_index_mod = convert_list_tuple_string_to_int(A)
            edge_index_torch = torch.tensor(A).T
            edge_index_mod_self, edge_attr=add_self_loops(edge_index_torch)
            print('Create graph')
            graph = Data(
                x=X_all, 
                edge_index=edge_index_mod_self,
            )

        if self.edge_strategy_name=='knn_graph':
            print('Create graph')
            edge_strategy = knn_graph(train_X, k=self.k, loop=False, distance_cosine=self.distance_euclidean)
            graph = edge_strategy(X_all)
            graph.x = graph.pos
            del graph.pos
        
        print('Create labels ')
        all_Y = torch.cat([torch.tensor(train_y),torch.tensor(val_y),torch.tensor(test_y)])
        train_mask = torch.cat([torch.ones(len(train_X)), torch.zeros(len(val_X)), torch.zeros(len(test_X))], 0)
        val_mask = torch.cat([torch.zeros(len(train_X)), torch.ones(len(val_X)), torch.zeros(len(test_X))], 0)
        test_mask = torch.cat([torch.zeros(len(train_X)), torch.zeros(len(val_X)), torch.ones(len(test_X))], 0)
        graph.y = all_Y
        graph.train_mask = train_mask.bool()
        graph.val_mask = val_mask.bool()
        graph.test_mask = test_mask.bool()
        data_list = [graph]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        # import pdb; pdb.set_trace()
        torch.save((data, slices), self.processed_paths[0])
        print('Processed data to: ', self.processed_paths[0])
    
import argparse

if __name__ == '__main__':
    # parse argument edge_strategy_name with values allowed: 'quantile', 'knn_graph'
    parser = argparse.ArgumentParser()
    parser.add_argument('--edge_strategy_name', type=str, default='knn_graph')
    parser.add_argument('--node_embeddings_type', type=str, default='lstm') # lstm_pca_6 # lstm_pca50
    parser.add_argument('--k', type=int, default='10')
    parser.add_argument('--n_edges', type=int, default=300_000)
    # import pdb; pdb.set_trace()
    # add boolean argument to the parser and store false if specified
    parser.add_argument('--distance_euclidean', action='store_false')

    print('Arguments: ', parser.parse_args())
    
    args = parser.parse_args()
    edge_strategy_name = args.edge_strategy_name
    k = args.k
    distance_euclidean_else_cosine = args.distance_euclidean
    name_distance = 'cosine' if distance_euclidean_else_cosine else 'euclidean'
    node_embeddings_type = args.node_embeddings_type
    
    print('edge_strategy_name: ', edge_strategy_name)
    DAY_MONTH_YEAR = datetime.now().strftime("%d_%m_%Y")
    FOLDER_NAME = f'data_e_{edge_strategy_name}_nf_{node_embeddings_type}_{DAY_MONTH_YEAR}'
    dataset = MyOwnDataset(
        root=FOLDER_NAME,
        node_type_embeddings = node_embeddings_type,
        edge_strategy_name=edge_strategy_name,
        k=k,
        n_edges=args.n_edges, # for random only
    )
    data = dataset[0]
    print('Created file: ',FOLDER_NAME)
    
    
# python 27_11_22_create_graph.py --edge_strategy_name random --node_embeddings_type lstm  --n_edges 20_000_000
# python create_homogeneous_graph.py --edge_strategy_name random --node_embeddings_type lstm  --n_edges 1_000_000