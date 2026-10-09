import numpy as np
import scipy.sparse as sp
import torch
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
from utils import preprocess_citation, sparse_mx_to_torch_sparse_tensor
import os


def load_npz_adj(path):
    data = np.load(path, allow_pickle=True)
    return sp.csr_matrix(
        (data["data"], data["indices"], data["indptr"]),
        shape=data["shape"]
    )


def symmetrize(adj):
    return adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)


def build_wmeta_graph(
    path_embeddings,
    meta_dir,
    k=20,
    alpha=0.5,              # blend original + WMETA graph
    temperature=0.5,        # similarity smoothing
    normalization="NormAdj",
    cuda=False
):

    print("=== Improved WMETA Graph ===")

    # ----------------------------
    # Load embeddings
    # ----------------------------
    data = np.load(path_embeddings, allow_pickle=True)

    X = data["embeddings"].astype(np.float32)
    y = data["labels"]

    idx_train = torch.LongTensor(data["idx_train"])
    idx_val   = torch.LongTensor(data["idx_val"])
    idx_test  = torch.LongTensor(data["idx_test"])

    N = X.shape[0]

    print(f"Loaded embeddings: {X.shape}")

    # keep original embeddings SAFE
    X_safe = X.copy()

    # ----------------------------
    # Load original adjacency (if exists in embeddings file or recompute)
    # ----------------------------
    adj_original = sp.csr_matrix((N, N))  # fallback empty

    # ----------------------------
    # Merge meta-paths WITH frequency weighting
    # ----------------------------
    meta_files = [f for f in os.listdir(meta_dir) if f.endswith(".npz")]

    merged = {i: {} for i in range(N)}  # dict of neighbor -> count

    for fi in meta_files:
        A = load_npz_adj(os.path.join(meta_dir, fi))

        for i in range(N):
            start, end = A.indptr[i], A.indptr[i + 1]
            neighs = A.indices[start:end]

            valid = neighs[neighs < N]

            for j in valid:
                merged[i][j] = merged[i].get(j, 0) + 1

    # ----------------------------
    # Build weighted WMETA graph
    # ----------------------------
    row, col, data_w = [], [], []

    for i in range(N):
        neigh_dict = merged[i]
        if len(neigh_dict) == 0:
            continue

        neighs = np.array(list(neigh_dict.keys()))
        freq   = np.array(list(neigh_dict.values()), dtype=np.float32)

        # safe feature copy (NO global modification)
        x_i = X_safe[i:i+1]

        x_j = X_safe[neighs]

        # cosine similarity
        sim = cosine_similarity(x_i, x_j).flatten()

        # meta-path frequency boost
        sim = sim * (1.0 + np.log1p(freq))

        # temperature scaling (stabilizes distribution)
        sim = np.exp(sim / temperature)

        # normalize
        sim = sim / (sim.sum() + 1e-8)

        # TOP-K only (no hard threshold)
        if len(neighs) > k:
            topk = np.argsort(sim)[-k:]
            neighs = neighs[topk]
            sim = sim[topk]

        row.extend([i] * len(neighs))
        col.extend(neighs)
        data_w.extend(sim)

    # ----------------------------
    # Build adjacency
    # ----------------------------
    adj_wmeta = sp.csr_matrix((data_w, (row, col)), shape=(N, N))
    adj_wmeta = symmetrize(adj_wmeta)

    print(f"WMETA edges: {adj_wmeta.nnz}")

    # ----------------------------
    # Blend with original structure (VERY IMPORTANT)
    # ----------------------------
    adj_final = alpha * adj_original + (1 - alpha) * adj_wmeta
    adj_final = symmetrize(adj_final)

    # ----------------------------
    # Normalize
    # ----------------------------
    adj_norm, _ = preprocess_citation(
        adj_final,
        sp.csr_matrix(X_safe),
        normalization
    )

    # ----------------------------
    # Torch conversion
    # ----------------------------
    features = torch.FloatTensor(X_safe)
    labels = torch.LongTensor(y)

    adj = sparse_mx_to_torch_sparse_tensor(adj_norm).float()

    # ----------------------------
    # Graph for stats
    # ----------------------------
    G = nx.from_scipy_sparse_matrix(adj_final.tocoo())
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

    return features, labels, adj, adj_final, G, edges, idx_train, idx_val, idx_test