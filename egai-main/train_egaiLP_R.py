# --- train_egaiR_lp.py ---
# EGAI for link prediction
# Author: Rihab AYED (adapted for LP)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
from utils import set_seed
from module_egai import total_ssl  # For pseudo-labels if needed
from construct_Meta_graphRUnified import build_embeddings_graph
from sklearn.metrics import roc_auc_score, average_precision_score
import random

# --- SETTINGS ---
hidden = 16
dropout = 0.5
cuda = False
# percent_list = [0.8, 0.2]  # fraction of training nodes used for pseudo-label augmentation
# ratio_list = [0.1, 0.5]    # ratio for SSL edges
percent_list = [0.8, 0.5, 0.2] # R : fraction of training nodes used for pseudo-label augmentation
ratio_list = [0.1, 0.5, 0.8] # R : ratio of edge pruning 
# --- LOAD GRAPH ---
features, labels, adj, adj_0, G_, edges_list, idx_train, idx_val, idx_test = \
    build_embeddings_graph(
        data_dir="data/dblp",
        use_embeddings=True,
        normalization="NormAdj",
        cuda=False
    )

print("Graph loaded for link prediction!")
print("Features:", features.shape)
print("Edges:", len(edges_list))
train_edges = np.array(edges_list)[idx_train]
val_edges = np.array(edges_list)[idx_val]
test_edges = np.array(edges_list)[idx_test]

# --- Create negative samples ---
def negative_edges(num_edges, num_nodes, exclude_edges):
    """Random negative edge sampler"""
    neg_edges = []
    existing = set([tuple(e) for e in exclude_edges])
    while len(neg_edges) < num_edges:
        i, j = random.randint(0, num_nodes-1), random.randint(0, num_nodes-1)
        if i != j and (i,j) not in existing and (j,i) not in existing:
            neg_edges.append([i,j])
    return np.array(neg_edges)

neg_train_edges = negative_edges(len(train_edges), features.shape[0], edges_list)
neg_val_edges   = negative_edges(len(val_edges), features.shape[0], edges_list)
neg_test_edges  = negative_edges(len(test_edges), features.shape[0], edges_list)

# --- Define simple EGAI model for embeddings ---
class SimpleEGAI(nn.Module):
    def __init__(self, in_feats, hidden, dropout):
        super().__init__()
        self.gc1 = nn.Linear(in_feats, hidden)
        self.gc2 = nn.Linear(hidden, hidden)
        self.dropout = dropout

    def forward(self, x, adj):
        h = torch.relu(self.gc1(x))
        h = F.dropout(h, self.dropout, training=self.training)
        h = self.gc2(h)
        return h

# --- TRAINING LOOP ---
num_nodes = features.shape[0]
features = features.float()
device = torch.device("cuda" if cuda else "cpu")
features = features.to(device)
model = SimpleEGAI(features.shape[1], hidden, dropout).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
adj = adj.to(device)

for seed in [42, 44]:
    set_seed(seed, cuda)
    print(f"Seed: {seed}")
    for percent in percent_list:
        for ratio in ratio_list:
            print(f"Percent: {percent}, Ratio: {ratio}")
            for epoch in range(50):
                model.train()
                optimizer.zero_grad()
                
                # Compute node embeddings
                h = model(features, adj)
                
                # Positive edges (training)
                h_pos_i = h[train_edges[:,0]]
                h_pos_j = h[train_edges[:,1]]
                pos_score = torch.sigmoid((h_pos_i * h_pos_j).sum(dim=1))
                
                # Negative edges (training)
                h_neg_i = h[neg_train_edges[:,0]]
                h_neg_j = h[neg_train_edges[:,1]]
                neg_score = torch.sigmoid((h_neg_i * h_neg_j).sum(dim=1))
                
                # Loss
                loss = - torch.mean(torch.log(pos_score + 1e-8) + torch.log(1 - neg_score + 1e-8))
                loss.backward()
                optimizer.step()
                
                if epoch % 10 == 0:
                    print(f"Epoch {epoch} | Loss: {loss.item():.4f}")
            
            # --- Evaluation ---
            model.eval()
            with torch.no_grad():
                h = model(features, adj)
                
                # Validation
                h_pos_i = h[val_edges[:,0]]
                h_pos_j = h[val_edges[:,1]]
                pos_score = torch.sigmoid((h_pos_i * h_pos_j).sum(dim=1))
                
                h_neg_i = h[neg_val_edges[:,0]]
                h_neg_j = h[neg_val_edges[:,1]]
                neg_score = torch.sigmoid((h_neg_i * h_neg_j).sum(dim=1))
                
                y_true = np.array([1]*len(val_edges) + [0]*len(neg_val_edges))
                y_score = torch.cat([pos_score, neg_score]).cpu().numpy()
                
                auc = roc_auc_score(y_true, y_score)
                ap  = average_precision_score(y_true, y_score)
                print(f"Val AUC: {auc:.4f}, AP: {ap:.4f}")
                
                # Test
                h_pos_i = h[test_edges[:,0]]
                h_pos_j = h[test_edges[:,1]]
                pos_score = torch.sigmoid((h_pos_i * h_pos_j).sum(dim=1))
                
                h_neg_i = h[neg_test_edges[:,0]]
                h_neg_j = h[neg_test_edges[:,1]]
                neg_score = torch.sigmoid((h_neg_i * h_neg_j).sum(dim=1))
                
                y_true = np.array([1]*len(test_edges) + [0]*len(neg_test_edges))
                y_score = torch.cat([pos_score, neg_score]).cpu().numpy()
                
                auc = roc_auc_score(y_true, y_score)
                ap  = average_precision_score(y_true, y_score)
                print(f"Test AUC: {auc:.4f}, AP: {ap:.4f}")
# --- train_egaiR_lp.py ---
# EGAI for link prediction
# Author: Rihab AYED (adapted for LP)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
from utils import set_seed
from module_egai import total_ssl  # For pseudo-labels if needed
from construct_Meta_graphRUnified import build_embeddings_graph
from sklearn.metrics import roc_auc_score, average_precision_score
import random

# --- SETTINGS ---
hidden = 16
dropout = 0.5
cuda = False
# percent_list = [0.8, 0.2]  # fraction of training nodes used for pseudo-label augmentation
# ratio_list = [0.1, 0.5]    # ratio for SSL edges
percent_list = [0.8, 0.5, 0.2] # R : fraction of training nodes used for pseudo-label augmentation
ratio_list = [0.1, 0.5, 0.8] # R : ratio of edge pruning 
# --- LOAD GRAPH ---
features, labels, adj, adj_0, G_, edges_list, idx_train, idx_val, idx_test = \
    build_embeddings_graph(
        data_dir="data/dblp",
        use_embeddings=True,
        normalization="NormAdj",
        cuda=False
    )

print("Graph loaded for link prediction!")
print("Features:", features.shape)
print("Edges:", len(edges_list))
train_edges = np.array(edges_list)[idx_train]
val_edges = np.array(edges_list)[idx_val]
test_edges = np.array(edges_list)[idx_test]

# --- Create negative samples ---
def negative_edges(num_edges, num_nodes, exclude_edges):
    """Random negative edge sampler"""
    neg_edges = []
    existing = set([tuple(e) for e in exclude_edges])
    while len(neg_edges) < num_edges:
        i, j = random.randint(0, num_nodes-1), random.randint(0, num_nodes-1)
        if i != j and (i,j) not in existing and (j,i) not in existing:
            neg_edges.append([i,j])
    return np.array(neg_edges)

neg_train_edges = negative_edges(len(train_edges), features.shape[0], edges_list)
neg_val_edges   = negative_edges(len(val_edges), features.shape[0], edges_list)
neg_test_edges  = negative_edges(len(test_edges), features.shape[0], edges_list)

# --- Define simple EGAI model for embeddings ---
class SimpleEGAI(nn.Module):
    def __init__(self, in_feats, hidden, dropout):
        super().__init__()
        self.gc1 = nn.Linear(in_feats, hidden)
        self.gc2 = nn.Linear(hidden, hidden)
        self.dropout = dropout

    def forward(self, x, adj):
        h = torch.relu(self.gc1(x))
        h = F.dropout(h, self.dropout, training=self.training)
        h = self.gc2(h)
        return h

# --- TRAINING LOOP ---
num_nodes = features.shape[0]
features = features.float()
device = torch.device("cuda" if cuda else "cpu")
features = features.to(device)
model = SimpleEGAI(features.shape[1], hidden, dropout).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
adj = adj.to(device)

for seed in [42, 44]:
    set_seed(seed, cuda)
    print(f"Seed: {seed}")
    for percent in percent_list:
        for ratio in ratio_list:
            print(f"Percent: {percent}, Ratio: {ratio}")
            for epoch in range(50):
                model.train()
                optimizer.zero_grad()
                
                # Compute node embeddings
                h = model(features, adj)
                
                # Positive edges (training)
                h_pos_i = h[train_edges[:,0]]
                h_pos_j = h[train_edges[:,1]]
                pos_score = torch.sigmoid((h_pos_i * h_pos_j).sum(dim=1))
                
                # Negative edges (training)
                h_neg_i = h[neg_train_edges[:,0]]
                h_neg_j = h[neg_train_edges[:,1]]
                neg_score = torch.sigmoid((h_neg_i * h_neg_j).sum(dim=1))
                
                # Loss
                loss = - torch.mean(torch.log(pos_score + 1e-8) + torch.log(1 - neg_score + 1e-8))
                loss.backward()
                optimizer.step()
                
                if epoch % 10 == 0:
                    print(f"Epoch {epoch} | Loss: {loss.item():.4f}")
            
            # --- Evaluation ---
            model.eval()
            with torch.no_grad():
                h = model(features, adj)
                
                # Validation
                h_pos_i = h[val_edges[:,0]]
                h_pos_j = h[val_edges[:,1]]
                pos_score = torch.sigmoid((h_pos_i * h_pos_j).sum(dim=1))
                
                h_neg_i = h[neg_val_edges[:,0]]
                h_neg_j = h[neg_val_edges[:,1]]
                neg_score = torch.sigmoid((h_neg_i * h_neg_j).sum(dim=1))
                
                y_true = np.array([1]*len(val_edges) + [0]*len(neg_val_edges))
                y_score = torch.cat([pos_score, neg_score]).cpu().numpy()
                
                auc = roc_auc_score(y_true, y_score)
                ap  = average_precision_score(y_true, y_score)
                print(f"Val AUC: {auc:.4f}, AP: {ap:.4f}")
                
                # Test
                h_pos_i = h[test_edges[:,0]]
                h_pos_j = h[test_edges[:,1]]
                pos_score = torch.sigmoid((h_pos_i * h_pos_j).sum(dim=1))
                
                h_neg_i = h[neg_test_edges[:,0]]
                h_neg_j = h[neg_test_edges[:,1]]
                neg_score = torch.sigmoid((h_neg_i * h_neg_j).sum(dim=1))
                
                y_true = np.array([1]*len(test_edges) + [0]*len(neg_test_edges))
                y_score = torch.cat([pos_score, neg_score]).cpu().numpy()
                
                auc = roc_auc_score(y_true, y_score)
                ap  = average_precision_score(y_true, y_score)
                print(f"Test AUC: {auc:.4f}, AP: {ap:.4f}")
