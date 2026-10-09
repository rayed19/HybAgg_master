import numpy as np
import scipy.sparse as sp
import torch
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
from utils import preprocess_citation, sparse_mx_to_torch_sparse_tensor

def build_knn_graph(path, k, normalization="NormAdj", cuda=False):

    # ----------------------------
    # Load SMCD embeddings (WITH SPLITS)
    # ----------------------------
    data = np.load(path, allow_pickle=True)

    X = data["embeddings"]
    y = data["labels"]

    idx_train = data["idx_train"]
    idx_val   = data["idx_val"]
    idx_test  = data["idx_test"]

    N = X.shape[0]

    # ----------------------------
    # KNN graph
    # ----------------------------
    sim = cosine_similarity(X)

    adj = np.zeros_like(sim)

    for i in range(N):
        idx = np.argsort(sim[i])[-(k + 1):-1]
        adj[i, idx] = sim[i, idx]

    # symmetrize
    adj_0 = sp.csr_matrix(adj)
    adj_0 = adj_0 + adj_0.T.multiply(adj_0.T > adj_0) - adj_0.multiply(adj_0.T > adj_0)

    adj, _ = preprocess_citation(adj_0, sp.csr_matrix(X), normalization)

    # ----------------------------
    # convert
    # ----------------------------
    features = torch.FloatTensor(X)
    labels = torch.LongTensor(y)

    adj = sparse_mx_to_torch_sparse_tensor(adj).float()

    idx_train = torch.LongTensor(idx_train)
    idx_val   = torch.LongTensor(idx_val)
    idx_test  = torch.LongTensor(idx_test)

    G = nx.from_numpy_array(adj_0.toarray())
    edges = list(G.edges())

    if cuda:
        features = features.cuda()
        labels = labels.cuda()
        adj = adj.cuda()
        idx_train = idx_train.cuda()
        idx_val = idx_val.cuda()
        idx_test = idx_test.cuda()

    return features, labels, adj, adj_0, G, edges, idx_train, idx_val, idx_test