''' Rihab AYED :
Handles nodes and neighbors with zero features safely.

Filters weak edges via min_weight.

Applies k-NN pruning after filtering.

Compatible with both raw features and embeddings. ''' 

import numpy as np
import scipy.sparse as sp
import networkx as nx
import torch
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from utils import preprocess_citation, sparse_mx_to_torch_sparse_tensor
import os

def load_adjlist_txt(path):
    adj_dict = {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [int(x) for x in line.split()]
            node = parts[0]
            neighs = parts[1:] if len(parts) > 1 else []
            adj_dict[node] = neighs
    return adj_dict

def symmetrize(adj):
    return adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

def build_embeddings_graph(
    data_dir="data/dblpsmcd",
    k=None,                   # optional KNN pruning
    min_weight=0.1,           # minimum edge weight to keep
    test_size=0.4,
    val_fraction=0.5,
    normalization="NormAdj",
    use_embeddings=True,      # switch between embeddings or raw features
    cuda=False
):
    print("=== Building Weighted Meta-Path Graph (MAGNN → EGAI) ===")

    # -----------------------------
    # Load features
    # -----------------------------
    if use_embeddings:
        f = np.load(os.path.join(data_dir, "embeddings_smcd_imdb_64_aspaper_mpprob0.3.npz"))
        features_np = f["embeddings"].astype(np.float32)
        labels_np   = f["labels"]
        print("Using semantic embeddings") # MAGNN or SMCD embeddings 
    else:
        # f = np.load(os.path.join(data_dir, "features_0.npz")) # for magnn 
        f = np.load(os.path.join(data_dir, "m_feat.npz")) # for smcd        
        features_np = sp.csr_matrix((f["data"], f["indices"], f["indptr"]), shape=f["shape"]).toarray().astype(np.float32)
        labels_np   = np.load(os.path.join(data_dir, "labels.npy"))
        print("Using raw features")

    num_nodes = features_np.shape[0]
    print(f"Loaded features → {features_np.shape}, labels → {labels_np.shape}")

    # -----------------------------
    # Merge meta-paths ( MAGNN or SMCD)
    # -----------------------------
    # meta_dir = os.path.join(data_dir, "0") # for magnn 
    meta_dir = os.path.join(data_dir, "mp") # for smcd 
    meta_files = [x for x in os.listdir(meta_dir) if x.endswith(".npz") and x.startswith("a")]
    # meta_files = [x for x in os.listdir(meta_dir) if x.endswith(".adjlist")] # for magnn 
    merged = {i: set() for i in range(num_nodes)}
    for fi in meta_files:
        adjlist = load_adjlist_txt(os.path.join(meta_dir, fi))
        for node, neighs in adjlist.items():
            if node < num_nodes:
                valid = [n for n in neighs if n < num_nodes]
                merged[node].update(valid)

    # -----------------------------
    # Compute weighted adjacency safely
    # -----------------------------
    row, col, data = [], [], []
    for i, neighs in merged.items():
        if len(neighs) == 0:
            continue

        neighs_arr = np.array(list(neighs))  # convert to numpy array for safe indexing

        # safeguard zero-row features for the node itself
        if np.all(features_np[i] == 0):
            features_np[i] = 1e-10

        # safeguard zero-row features for neighbors
        zero_neighbor_rows = np.where(np.all(features_np[neighs_arr] == 0, axis=1))[0]
        if len(zero_neighbor_rows) > 0:
            features_np[neighs_arr[zero_neighbor_rows]] = 1e-10

        # compute cosine similarity
        sim = cosine_similarity(features_np[i:i+1, :], features_np[neighs_arr]).flatten()

        # filter by minimum weight
        mask = sim >= min_weight
        if np.any(mask):
            filtered_neighbors = neighs_arr[mask]
            filtered_sim = sim[mask]

            # optional KNN pruning
            if k is not None and len(filtered_neighbors) > k:
                top_idx = np.argsort(filtered_sim)[-k:]
                filtered_neighbors = filtered_neighbors[top_idx]
                filtered_sim = filtered_sim[top_idx]

            # extend final adjacency lists
            row.extend([i]*len(filtered_neighbors))
            col.extend(filtered_neighbors)
            data.extend(filtered_sim)

    # -----------------------------
    # Build sparse adjacency
    # -----------------------------
    adj_0 = sp.csr_matrix((data, (row, col)), shape=(num_nodes, num_nodes))
    adj_0 = symmetrize(adj_0)
    print(f"Weighted adjacency built → {adj_0.nnz} edges")

    # -----------------------------
    # Normalize adjacency
    # -----------------------------
    adj_norm, _ = preprocess_citation(
        adj_0,
        sp.csr_matrix(features_np),
        normalization=normalization
    )

    # -----------------------------
    # Torch tensors
    # -----------------------------
    features = torch.FloatTensor(features_np)
    labels = torch.LongTensor(labels_np)
    adj = sparse_mx_to_torch_sparse_tensor(adj_norm).float()

    # -----------------------------
    # NetworkX graph for stats/debugging
    # -----------------------------
    G_ = nx.from_scipy_sparse_matrix(adj_0.tocoo())
    edges_list = list(G_.edges())

    # -----------------------------
    # Train/val/test split
    # -----------------------------
    idx_all = np.arange(num_nodes)
    idx_train, idx_temp, y_train, y_temp = train_test_split(
        idx_all, labels, test_size=test_size, stratify=labels, random_state=42
    )
    idx_val, idx_test, _, _ = train_test_split(
        idx_temp, y_temp, test_size=val_fraction, stratify=y_temp, random_state=42
    )

    idx_train = torch.LongTensor(idx_train)
    idx_val   = torch.LongTensor(idx_val)
    idx_test  = torch.LongTensor(idx_test)

    # -----------------------------
    # CUDA
    # -----------------------------
    if cuda:
        features = features.cuda()
        labels = labels.cuda()
        adj = adj.cuda()
        idx_train = idx_train.cuda()
        idx_val = idx_val.cuda()
        idx_test = idx_test.cuda()

    print(f"Graph ready: {num_nodes} nodes, {len(edges_list)} edges, {labels.max().item()+1} classes")
    return features, labels, adj, adj_0, G_, edges_list, idx_train, idx_val, idx_test

# -----------------------------
# Optional test
# -----------------------------
if __name__ == "__main__":
    build_embeddings_graph()  # raw features or semantic embeddings 
