# R : construct k-NN graph from MAGNN embeddings 
import numpy as np
import scipy.sparse as sp
import networkx as nx
import torch
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from utils import preprocess_citation, sparse_mx_to_torch_sparse_tensor

def build_embeddings_graph(k=10, test_size=0.4, val_fraction=0.5, normalization="NormAdj", cuda=False):
    """
    Build a PyTorch-compatible graph from MAGNN embeddings, closely mimicking load_citation.
    Returns:
        features: torch.FloatTensor (N x F)
        labels: torch.LongTensor (N)
        adj: torch.sparse.FloatTensor normalized adjacency
        adj_0: scipy.sparse adjacency (original)
        G_: NetworkX graph
        edges_list: list of edges
        idx_train, idx_val, idx_test: torch.LongTensor indices
    """
    # --- Load embeddings and labels ---
    data = np.load("data/embeddings_DBLP.npz")
    embeddings = data["embeddings"]        # shape [N, F]
    labels = data["labels"]                # shape [N,]
    num_nodes = embeddings.shape[0]

    # --- Construct k-NN similarity graph ---
    sim_matrix = cosine_similarity(embeddings)
    adj_matrix = np.zeros_like(sim_matrix)
    for i in range(num_nodes):
        top_k = np.argsort(sim_matrix[i])[-(k+1):-1]  # exclude self
        adj_matrix[i, top_k] = sim_matrix[i, top_k]

    # --- Original adjacency (symmetric like load_citation) ---
    adj_0 = sp.csr_matrix(adj_matrix)
    adj_0 = adj_0 + adj_0.T.multiply(adj_0.T > adj_0) - adj_0.multiply(adj_0.T > adj_0)

    # --- Normalize adjacency ---
    adj, _ = preprocess_citation(adj_0, sp.csr_matrix(embeddings), normalization=normalization)

    # --- NetworkX graph ---
    G_ = nx.from_numpy_array(adj_0.toarray())
    edges_list = list(G_.edges())

    # --- Features and labels ---
    features = torch.FloatTensor(embeddings)
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

    # --- Convert adjacency to torch sparse ---
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
    print(f"Graph built: {features.shape[0]} nodes, {len(edges_list)} edges, {labels.max().item()+1} classes")
