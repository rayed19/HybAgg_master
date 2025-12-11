import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import umap

# --------------------
# Load embeddings and labels
# --------------------
Z = np.load("embeddings_knn_egai_dblp_test.npy")  # your embeddings
labels = np.load("labels_knn_egai_dblp_test.npy")  # your labels

print("Embeddings shape:", Z.shape)
print("Labels shape:", labels.shape)

# --------------------
# t-SNE
# --------------------
tsne = TSNE(
    n_components=2,
    perplexity=30,
    learning_rate=200,
    init='pca',
    max_iter=1000,
    random_state=42
)
Z_tsne = tsne.fit_transform(Z)

plt.figure(figsize=(8,6))
plt.scatter(Z_tsne[:,0], Z_tsne[:,1], c=labels, s=8, cmap='tab10')
plt.title("t-SNE Visualization of Embeddings")
plt.colorbar()
plt.tight_layout()
plt.savefig("tsne_testnodes_knn_egai_embeddings.png", dpi=300)
plt.show()

# --------------------
# UMAP
# --------------------
reducer = umap.UMAP(n_components=2, random_state=42)
Z_umap = reducer.fit_transform(Z)

plt.figure(figsize=(8,6))
plt.scatter(Z_umap[:,0], Z_umap[:,1], c=labels, s=8, cmap='tab10')
plt.title("UMAP Visualization of Embeddings")
plt.colorbar()
plt.tight_layout()
plt.savefig("umap_testnodes_knn_egai_embeddings.png", dpi=300)
plt.show()

# --------------------
# Keep script window open
# --------------------
input("Press Enter to exit...")
