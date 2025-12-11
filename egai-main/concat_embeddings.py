# --- concat_embeddings_magnn_egai.py --- (not used yet, requires the same configuration of the two GNN algorithms) 
# Naive Concatenation of MAGNN + EGAI embeddings into a single feature file
# Author: Rihab AYED

import numpy as np
import os

def concat_embeddings(
    data_dir="data/imdb",
    magnn_file="embeddings_IMDB.npz",
    egai_file="embeddings_egai_IMDB.npz",
    output_file="embeddings_concat_IMDB.npz"
):
    """
    Concatenate embeddings from MAGNN and EGAI into one .npz file.
    Both must correspond to the same set of nodes (same order).
    """

    magnn_path = os.path.join(data_dir, magnn_file)
    egai_path = os.path.join(data_dir, egai_file)
    output_path = os.path.join(data_dir, output_file)

    print("=== Concatenating Embeddings ===")
    print(f"→ MAGNN file: {magnn_path}")
    print(f"→ EGAI file:  {egai_path}")

    if not os.path.exists(magnn_path):
        raise FileNotFoundError(f"MAGNN embeddings not found: {magnn_path}")
    if not os.path.exists(egai_path):
        raise FileNotFoundError(f"EGAI embeddings not found: {egai_path}")

    # --- Load both embeddings ---
    f1 = np.load(magnn_path)
    emb_magnn = f1["embeddings"]
    f2 = np.load(egai_path)
    emb_egai = f2["embeddings"]

    print(f"Loaded MAGNN embeddings → {emb_magnn.shape}")
    print(f"Loaded EGAI embeddings  → {emb_egai.shape}")

    # --- Check node alignment ---
    if emb_magnn.shape[0] != emb_egai.shape[0]:
        raise ValueError(
            f"Node count mismatch: MAGNN={emb_magnn.shape[0]}, EGAI={emb_egai.shape[0]}"
        )

    # --- Concatenate along feature dimension ---
    emb_concat = np.concatenate([emb_magnn, emb_egai], axis=1)
    print(f" Concatenated embeddings shape → {emb_concat.shape}")

    # --- Save to new .npz file ---
    np.savez_compressed(output_path, embeddings=emb_concat)
    print(f" Saved concatenated embeddings → {output_path}")


# --- Run example ---
if __name__ == "__main__":
    concat_embeddings(
        data_dir="data/imdb",
        magnn_file="embeddings_IMDB.npz",
        egai_file="embeddings_egai_IMDB.npz",
        output_file="embeddings_concat_IMDB.npz"
    )
