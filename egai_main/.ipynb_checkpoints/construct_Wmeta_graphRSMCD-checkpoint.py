import numpy as np
import scipy.sparse as sp
import torch
import networkx as nx
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from utils import preprocess_citation, sparse_mx_to_torch_sparse_tensor
import os


def load_npz_adj(path):
    data = np.load(path, allow_pickle=True)
    return csr_matrix(
        (data["data"], data["indices"], data["indptr"]),
        shape=data["shape"]
    )


def symmetrize(adj):
    return adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)


def build_wmeta_graph(path_embeddings, meta_dir,
                     k,
                     min_weight,
                     normalization="NormAdj",
                     cuda=False):

    print("=== Building WMETA Graph (NPZ version) ===")

    # ----------------------------
    # Load embeddings (WITH SPLITS)
    # ----------------------------
    data = np.load(path_embeddings, allow_pickle=True)

    X = data["embeddings"].astype(np.float32)
    y = data["labels"]

    idx_train = data["idx_train"]
    idx_val   = data["idx_val"]
    idx_test  = data["idx_test"]

    N = X.shape[0]

    # SAFE COPY (avoid modifying original embeddings)
    X_safe = X.copy()

    print(f"Loaded embeddings: {X.shape}")

    # ----------------------------
    # Merge meta-paths (NPZ)
    # ----------------------------
    meta_files = [f for f in os.listdir(meta_dir) if f.endswith(".npz")]

    merged = {i: set() for i in range(N)}

    for fi in meta_files:
        A = load_npz_adj(os.path.join(meta_dir, fi))

        for i in range(N):
            start = A.indptr[i]
            end = A.indptr[i+1]

            neighs = A.indices[start:end]

            # FIX 1: index safety
            valid = neighs[neighs < N]

            merged[i].update(valid)

    # ----------------------------
    # WMETA weighting
    # ----------------------------
    row, col, data_w = [], [], []

    for i in range(N):
        neighs = list(merged[i])
        if len(neighs) == 0:
            continue

        neighs = np.array(neighs)

        # FIX 2: avoid zero vectors safely
        if np.all(X_safe[i] == 0):
            X_safe[i] = 1e-10

        zero_rows = np.where(np.all(X_safe[neighs] == 0, axis=1))[0]
        if len(zero_rows) > 0:
            X_safe[neighs[zero_rows]] = 1e-10

        # cosine similarity
        sim = cosine_similarity(X_safe[i:i+1], X_safe[neighs]).flatten()

        # filter weak edges
        mask = sim >= min_weight
        neighs = neighs[mask]
        sim = sim[mask]

        if len(neighs) == 0:
            continue

        # optional KNN pruning
        if k is not None and len(neighs) > k:
            idx = np.argsort(sim)[-k:]
            neighs = neighs[idx]
            sim = sim[idx]

        row.extend([i]*len(neighs))
        col.extend(neighs)
        data_w.extend(sim)

    # ----------------------------
    # Build adjacency
    # ----------------------------
    adj_0 = sp.csr_matrix((data_w, (row, col)), shape=(N, N))
    adj_0 = symmetrize(adj_0)

    print(f"Weighted adjacency built → {adj_0.nnz} edges")

    # ----------------------------
    # Normalize
    # ----------------------------
    adj, _ = preprocess_citation(adj_0, sp.csr_matrix(X_safe), normalization)

    # ----------------------------
    # Convert to torch
    # ----------------------------
    features = torch.FloatTensor(X)
    labels = torch.LongTensor(y)

    adj = sparse_mx_to_torch_sparse_tensor(adj).float()

    idx_train = torch.LongTensor(idx_train)
    idx_val   = torch.LongTensor(idx_val)
    idx_test  = torch.LongTensor(idx_test)

    # FIX 3: efficient graph construction
    G = nx.from_scipy_sparse_matrix(adj_0)
    edges = list(G.edges())

    # ----------------------------
    # CUDA
    # ----------------------------
    if cuda:
        features = features.cuda()
        labels = labels.cuda()
        adj = adj.cuda()
        idx_train = idx_train.cuda()
        idx_val = idx_val.cuda()
        idx_test = idx_test.cuda()

    print(f"Graph ready: {N} nodes, {len(edges)} edges")

    return features, labels, adj, adj_0, G, edges, idx_train, idx_val, idx_test