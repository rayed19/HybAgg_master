# R : construct a graph from raw preprocessed data or from MAGNN embeddings using metapaths of MAGNN  
import os
import numpy as np
import scipy.sparse as sp
import torch
import networkx as nx
from sklearn.model_selection import train_test_split
from utils import preprocess_citation, sparse_mx_to_torch_sparse_tensor

# -----------------------------
# Utility: read MAGNN .adjlist
# -----------------------------
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

# -----------------------------
# Symmetrize adjacency
# -----------------------------
def symmetrize(adj):
    return adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

# -----------------------------
# Main loader
# -----------------------------
def build_embeddings_graph(
    data_dir="data/dblp",
    k=None,                    # optional top-k pruning
    test_size=0.4,
    val_fraction=0.5,
    normalization="NormAdj",
    use_embeddings=False,
    cuda=False
):
    print("=== Loading MAGNN Meta-Path Graph for EGAI ===")

    # -----------------------------
    # Load features
    # -----------------------------
    if use_embeddings:
        f = np.load(os.path.join(data_dir, "embeddings_DBLP.npz"))
        features_np = f["embeddings"].astype(np.float32)
        labels_np   = f["labels"]
        print(f"Using MAGNN embeddings → {features_np.shape}")
    else:
        f = np.load(os.path.join(data_dir, "features_0.npz"))
        features_np = sp.csr_matrix((f["data"], f["indices"], f["indptr"]),
                                     shape=f["shape"]).toarray().astype(np.float32)
        labels_np   = np.load(os.path.join(data_dir, "labels.npy"))
        print(f"Using raw features → {features_np.shape}")

    num_nodes = features_np.shape[0]

    # -----------------------------
    # Merge MAGNN meta-paths
    # -----------------------------
    meta_dir = os.path.join(data_dir, "0")
    meta_files = [x for x in os.listdir(meta_dir) if x.endswith(".adjlist")]
    merged = {i: set() for i in range(num_nodes)}

    for fi in meta_files:
        adjlist = load_adjlist_txt(os.path.join(meta_dir, fi))
        for node, neighs in adjlist.items():
            if node < num_nodes:
                valid = [n for n in neighs if n < num_nodes]
                merged[node].update(valid) # unique neighbors 

    # -----------------------------
    # Optional top-k pruning : arbitrary selection based on node ID order
    # -----------------------------
    if k is not None:
        print(f"Applying KNN-style top-{k} pruning...")
        for node, neighs in merged.items():
            if len(neighs) > k:
                merged[node] = set(sorted(list(neighs))[:k])

    # -----------------------------
    # Build adjacency matrix
    # -----------------------------
    row, col = [], []
    for i, neighs in merged.items():
        for j in neighs:
            row.append(i)
            col.append(j)

    adj_0 = sp.csr_matrix((np.ones(len(row)), (row, col)), shape=(num_nodes, num_nodes))
    adj_0 = symmetrize(adj_0)
    print(f"Adjacency built → {adj_0.nnz} edges")

    # -----------------------------
    # Normalize adjacency
    # -----------------------------
    # preprocess_citation handles zero-row safety
    adj_norm, _ = preprocess_citation(adj_0, sp.csr_matrix(features_np), normalization=normalization)

    # -----------------------------
    # Train/val/test split
    # -----------------------------
    idx_all = np.arange(num_nodes)
    idx_train, idx_temp, _, y_temp = train_test_split(
        idx_all, labels_np, test_size=test_size, stratify=labels_np, random_state=42
    )
    idx_val, idx_test, _, _ = train_test_split(
        idx_temp, y_temp, test_size=val_fraction, stratify=y_temp, random_state=42
    )

    idx_train = torch.LongTensor(idx_train)
    idx_val   = torch.LongTensor(idx_val)
    idx_test  = torch.LongTensor(idx_test)

    # -----------------------------
    # Torch tensors
    # -----------------------------
    features = torch.FloatTensor(features_np)
    labels = torch.LongTensor(labels_np)
    adj = sparse_mx_to_torch_sparse_tensor(adj_norm).float()

    # -----------------------------
    # NetworkX graph
    # -----------------------------
    G_ = nx.from_scipy_sparse_matrix(adj_0.tocoo())
    edges_list = list(G_.edges())

    # -----------------------------
    # Move to CUDA
    # -----------------------------
    if cuda:
        features = features.cuda()
        labels = labels.cuda()
        adj = adj.cuda()
        idx_train = idx_train.cuda()
        idx_val = idx_val.cuda()
        idx_test = idx_test.cuda()

    print(f"Graph ready: {num_nodes} nodes, {len(edges_list)} edges, {labels.max().item()+1} classes")
    print(f"Splits → Train: {len(idx_train)}, Val: {len(idx_val)}, Test: {len(idx_test)}")

    return features, labels, adj, adj_0, G_, edges_list, idx_train, idx_val, idx_test

# -----------------------------
# Optional test
# -----------------------------
if __name__ == "__main__":
    build_embeddings_graph()
