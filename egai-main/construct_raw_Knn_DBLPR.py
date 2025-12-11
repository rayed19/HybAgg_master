# R : construct a k-NN graph for raw target node type features of DBLP 
import numpy as np
import scipy.sparse as sp
import networkx as nx
import torch
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from utils import preprocess_citation, sparse_mx_to_torch_sparse_tensor

def build_embeddings_graph(k=10, test_size=0.4, val_fraction=0.5, normalization="NormAdj", cuda=False):
    """
    Build a PyTorch-compatible graph from DBLP authors features (MAGNN preprocessed),
    using a k-NN similarity graph.
    """
    # --- Load sparse features and labels ---
    data = np.load("data/dblp/features_0.npz")
    print(data.files)  # should be ['indices', 'indptr', 'format', 'shape', 'data']

    embeddings = sp.csr_matrix((data['data'], data['indices'], data['indptr']), shape=data['shape'])

    # FIX 1: Convert to float BEFORE any normalization
    embeddings = embeddings.astype(np.float32)

    print(f"Loaded sparse feature matrix: {embeddings.shape}")

    labels = np.load("data/dblp/labels.npy")
    num_nodes = embeddings.shape[0]

    # --- Build k-NN graph (cosine distance) ---
    print("Building k-NN graph...")
    nbrs = NearestNeighbors(n_neighbors=k + 1, metric='cosine').fit(embeddings)
    distances, indices = nbrs.kneighbors(embeddings)

    adj_matrix = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    for i, neighbors in enumerate(indices):
        for j in neighbors[1:]:  # skip self
            adj_matrix[i, j] = 1.0

    # --- Make adjacency symmetric ---
    adj_0 = sp.csr_matrix(adj_matrix)

    # FIX 2: Ensure adjacency is float
    adj_0 = adj_0.astype(np.float32)

    adj_0 = adj_0 + adj_0.T.multiply(adj_0.T > adj_0) - adj_0.multiply(adj_0.T > adj_0)

    # --- Normalize adjacency safely ---
    adj, _ = preprocess_citation(adj_0, embeddings, normalization=normalization)

    # --- Build NetworkX graph ---
    G_ = nx.from_scipy_sparse_array(adj_0)
    edges_list = list(G_.edges())

    # --- Features and labels (convert to dense for PyTorch) ---
    features = torch.FloatTensor(embeddings.toarray())
    labels = torch.LongTensor(labels)

    # --- Train/val/test split ---
    idx_all = np.arange(num_nodes)
    idx_train, idx_temp, _, y_temp = train_test_split(
        idx_all, labels, test_size=test_size, stratify=labels, random_state=42)
    idx_val, idx_test, _, _ = train_test_split(
        idx_temp, y_temp, test_size=val_fraction, stratify=y_temp, random_state=42)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    # --- Convert adjacency to torch sparse tensor ---
    adj = sparse_mx_to_torch_sparse_tensor(adj).float()

    # --- Move to GPU if requested ---
    if cuda:
        features = features.cuda()
        labels = labels.cuda()
        adj = adj.cuda()
        idx_train = idx_train.cuda()
        idx_val = idx_val.cuda()
        idx_test = idx_test.cuda()

    return features, labels, adj, adj_0, G_, edges_list, idx_train, idx_val, idx_test


# --- Optional test ---
if __name__ == "__main__":
    features, labels, adj, adj_0, G_, edges_list, idx_train, idx_val, idx_test = build_embeddings_graph()
    print(f" Graph built: {features.shape[0]} nodes, {len(edges_list)} edges, {labels.max().item()+1} classes")
