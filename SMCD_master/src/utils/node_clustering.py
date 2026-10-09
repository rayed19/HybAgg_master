import numpy as np
import torch
from .logreg import LogReg
import torch.nn as nn
from sklearn.metrics import f1_score, roc_auc_score, normalized_mutual_info_score, adjusted_rand_score, accuracy_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize



def node_clustering(embeddings, true_labels, n_clusters=3, normalize_emb=True, save_vis=True):


    if normalize_emb:
        embeddings = normalize(embeddings, norm='l2', axis=1)  # L2归一化


    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)  #
    pred_labels = kmeans.fit_predict(embeddings)  #


    true_labels_int = np.argmax(true_labels, axis=1)  # [N,]（0/1/2）


    nmi = normalized_mutual_info_score(true_labels_int, pred_labels)
    ari = adjusted_rand_score(true_labels_int, pred_labels)
    acc = accuracy_score(true_labels_int, pred_labels)

    # 4. 输出结果
    cluster_results = {
        "pred_labels": pred_labels,
        "true_labels_int": true_labels_int,
        "nmi": nmi,
        "ari": ari,
        "acc": acc,
        "centers": kmeans.cluster_centers_
    }
    print(f"聚类评估结果：")
    print(f"NMI: {nmi:.4f}, ARI: {ari:.4f}, ACC: {acc:.4f}")


