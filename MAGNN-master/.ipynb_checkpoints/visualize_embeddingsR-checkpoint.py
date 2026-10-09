import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import umap

# --------------------
# Load embeddings
# --------------------
data = np.load("embeddings_DBLP.npz")  # change path if needed
Z = data["embeddings"]
labels = data["labels"]

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
    max_iter=1000,      # updated for scikit-learn >=1.4
    random_state=42
)
Z_tsne = tsne.fit_transform(Z)

plt.figure(figsize=(8,6))
plt.scatter(Z_tsne[:,0], Z_tsne[:,1], c=labels, s=8, cmap='tab10')
plt.title("t-SNE Visualization of MAGNN Embeddings")
plt.colorbar()
plt.tight_layout()
plt.savefig("tsne_embeddings.png", dpi=300)
plt.show()

# --------------------
# UMAP
# --------------------
reducer = umap.UMAP(n_components=2, random_state=42)
Z_umap = reducer.fit_transform(Z)

plt.figure(figsize=(8,6))
plt.scatter(Z_umap[:,0], Z_umap[:,1], c=labels, s=8, cmap='tab10')
plt.title("UMAP Visualization of MAGNN Embeddings")
plt.colorbar()
plt.tight_layout()
plt.savefig("umap_embeddings.png", dpi=300)
plt.show()

# --------------------
# Keep script window open
# --------------------
input("Press Enter to exit...")
