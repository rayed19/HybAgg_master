# --- train_egaiR.py ---
# Import a graph of embeddings of a GNN algorithm as input and use train_egai.py
# We added also F1 scores (micro, macro) outputs 
# Added part of codes are quoted by '#R' comment and ends with '#End R' for multi-line modifications 
# Author: Rihab AYED

import numpy as np
import scipy.sparse as sp
import sys
import pickle as pkl
import networkx as nx
import math
from time import perf_counter
from sklearn.metrics import f1_score

import torch
import torch.nn as nn
from torch.nn import Module
import torch.nn.functional as F
import torch.optim as optim

from utils import load_citation, sgc_precompute, set_seed
from models import get_model
from metrics import accuracy, f1 # R : F1 Added 
from module_egai import generate_new_trainset ,total_ssl

# R : K-NN graph construction 
# from construct_Knn_graphR import build_embeddings_graph  #  import the aggregated embeddings graph builder # R 
from construct_raw_Knn_DBLPR import build_embeddings_graph # import the raw embeddings graph builder # R 
# R : metapath graph construction 
# from construct_Meta_graphR import build_embeddings_graph # import the embededings graph builder # R 
# from construct_Wmeta_graphR import build_embeddings_graph# weighted edges of embeddings with minimum weight to include
# Training settings
lr = 0.01
weight_decay = 5e-4
hidden = 16
dropout = 0.5
# dataset = "cora"
model_name = "GCN"
feature = "mul"  # choices=['mul', 'cat', 'adj'],
normalization = "NormAdj"
per = -1
cuda = False

# R: for default dataset (e.g. cora) 
# adj_0, adj, features, labels, idx_train, idx_val, idx_test = load_citation(dataset, normalization, cuda) # R : desactivated 

# --- R : Load MAGNN embeddings graph instead of Cora ---
# Build the K-NN based graph for DBLP or IMDB 
features, labels, adj, adj_0, G_, edges_list, idx_train, idx_val, idx_test = build_embeddings_graph()
'''features, labels, adj, adj_0, G_, edges_list, idx_train, idx_val, idx_test = build_embeddings_graph(
    data_dir="data/dblp",                                                                                                   use_embeddings=False, 
    cuda=False) ''' # for construct_Meta_graphR

# Build the metapath based graph for DBLP or IMDB 
'''features, labels, adj, adj_0, G_, edges_list, idx_train, idx_val, idx_test = \
    build_embeddings_graph(data_dir='data/dblp',
    use_embeddings=True,
    cuda=False,
    test_size=0.4,
    val_fraction=0.5,
    k=None,
    )
'''
print("Graph loaded for training!")
print("Features:", features.shape)
print("Edges:", len(edges_list))
print("Train size:", len(idx_train))

# End R 

G_ = nx.from_numpy_array(adj_0.toarray())
edges = G_.edges()
edges_list = [i for i in edges]

def train_regression(model,train_features,train_labels,val_features,val_labels, dropout,adj_aug):
    model.train()
    optimizer.zero_grad()
    output = model(train_features, adj_aug)
    loss_train = F.cross_entropy(output[idx_train], train_labels)
    loss_train.backward()
    optimizer.step()

    with torch.no_grad():
        model.eval()
        output_val = model(val_features, adj)
        loss_val = F.cross_entropy(output_val[idx_val], val_labels)

    return model, loss_val, output
# R : added F1 scores : 
'''def test_regression(model, test_features, idx_test, test_labels):
    model.eval()
    output = model(test_features, adj)
    return accuracy(output[idx_test], test_labels)
'''
def test_regression(model, test_features, idx_test, test_labels):
    model.eval()
    output = model(test_features, adj)
    preds = output[idx_test].max(1)[1].type_as(test_labels)

    acc = accuracy(output[idx_test], test_labels)
    micro_f1, macro_f1 = f1(output[idx_test], test_labels)

    print(" Test Accuracy: {:.4f} | Micro-F1: {:.4f} | Macro-F1: {:.4f}".format(acc, micro_f1, macro_f1))
    return acc, micro_f1, macro_f1

# End R 


# train

# percent_list = [0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2] # R : desactivated 
# ratio_list = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9] # R : desactivated 
# Option A (équilibrée) : 3 percents × 3 ratios × 2 seeds = 18
percent_list = [0.8, 0.5, 0.2] # R : fraction of training nodes used for pseudo-label augmentation
ratio_list = [0.1, 0.5, 0.8] # R : ratio of edge pruning 
# End R 

# result_array = np.zeros((8, 9, 10)) # R 
# Last dimension = 3 for Accuracy, Micro-F1, Macro-F1
result_array = np.zeros((len(percent_list), len(ratio_list), 2, 3))
train_list  = [int(i) for i in idx_train]
for seed_index, seed in enumerate([42, 44]):  # R : 2
# for seed_index, seed in enumerate(range(42, 52)): # R : 10 seeds     
    # End R 
    for per_index, percent in enumerate(percent_list):
        for ratio_index, ratio in enumerate(ratio_list):
            set_seed(seed, cuda)
            print("Percent: {} Ratio: {} Seed:{} ".
                      format(percent, ratio, seed))
            model = get_model("GCN", features.size(1),
                  labels.max().item() + 1, hidden, dropout, cuda)
            optimizer = torch.optim.Adam([
        dict(params=model.gc1.parameters(), weight_decay=5e-4),
        dict(params=model.gc2.parameters(), weight_decay=0)
    ],
                                 lr=0.01)

            cost_val = []
            early_stopping = 10

            for epoch in range(200):
                if epoch < 1:
                    adj_aug = adj
                else:
                    adj_aug = total_ssl(G_, edges_list, list_nbunch, y_pse, output,
                                        ratio) # We added G_ (Rihab AYED) 
                model, loss_val, output = train_regression(
                    model, features, labels[idx_train], features,
                    labels[idx_val], dropout, adj_aug)
                cost_val.append(loss_val)

                if epoch > early_stopping and cost_val[-1] > np.mean(cost_val[-(early_stopping+1):-1]):
                    print("Early stopping...")
                    break


                y_pse, train_list_seg = generate_new_trainset(
                    percent, output, labels, idx_train)
                list_nbunch = train_list_seg + train_list

            # acc_test = test_regression(model, features, idx_test, labels[idx_test])
            acc_test, micro_f1, macro_f1 = test_regression(model, features, idx_test, labels[idx_test])
                
            # result_array[per_index, ratio_index, seed_index] = acc_test # R           
            result_array[per_index, ratio_index, seed_index, 0] = acc_test # R
            result_array[per_index, ratio_index, seed_index, 1] = micro_f1 # R 
            result_array[per_index, ratio_index, seed_index, 2] = macro_f1 # R
            
            np.save("result_array.npy", result_array) # We added this line to save the array
            print(" Test Accuracy: {:.4f} ".
                      format( acc_test))

# R : save the embeddings of EGAI : 
# ---- Save learned embeddings ----
model.eval()
with torch.no_grad():
    # Get embeddings from the penultimate layer (hidden representation)
    hidden_embeddings = model.gc1(features, adj)
    hidden_embeddings = F.relu(hidden_embeddings)  # activation same as in forward()

    # Convert to numpy
    embeddings_np = hidden_embeddings.cpu().numpy()

    # (BASIC) Save all node embeddings
    np.save("embeddings_knn_egai_dblp.npy", embeddings_np)
    # (BASIC) Save labels 
    np.save("labels_knn_egai_dblp.npy", labels)
    print("Saved EGAI embeddings → embeddings_knn_egai_dblp.npy with shape", embeddings_np.shape)
    
    # (OPTIONAL) SAVE ALSO FINAL PSEUDO-LABELS GENERATED BY EGAI 
    final_labels = labels.clone()
    for i, node in enumerate(train_list_seg):
        final_labels[node] = y_pse[i]
    np.save("predlabels_knn_egai_dblp.npy", final_labels.cpu().numpy())
    print("Saved pseudo labels + labels → predlabels_knn_egai_dblp.npy")

    # (OPTIONAL) SAVE TEST EMBEDDINGS ONLY (like MAGNN)
    test_embeddings = embeddings_np[idx_test]
    test_labels = labels[idx_test]
    np.save("embeddings_knn_egai_dblp_test.npy", test_embeddings)
    np.save("predlabels_knn_egai_dblp_test.npy", final_labels[idx_test].cpu().numpy())
    np.save("labels_knn_egai_dblp_test.npy", test_labels)
    print("Saved test embeddings → embeddings_knn_egai_dblp_test.npy with shape", test_embeddings.shape)
# End R 
print("Finished") # R 
