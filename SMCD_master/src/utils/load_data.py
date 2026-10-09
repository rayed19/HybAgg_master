import numpy as np
import scipy.sparse as sp
import torch as th
from sklearn.preprocessing import OneHotEncoder
import pickle
import os


def encode_onehot(labels):
    labels = labels.reshape(-1, 1)
    enc = OneHotEncoder()
    enc.fit(labels)
    labels_onehot = enc.transform(labels).toarray()

    return labels_onehot


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features.todense()


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = th.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = th.from_numpy(sparse_mx.data)
    shape = th.Size(sparse_mx.shape)
    return th.sparse.FloatTensor(indices, values, shape)


def load_acm(type_num):
    path = "../data/acm/"
    label = np.load(path + "labels.npy").astype('int32')
    label = encode_onehot(label)
    feat_p = sp.load_npz(path + "p_feat.npz")
    feat_a = sp.load_npz(path + "a_feat.npz")
    feat_s = sp.load_npz(path + "s_feat.npz")
   
    pap = sp.load_npz(path + "mp/pap.npz")
    psp = sp.load_npz(path + "mp/psp.npz")
    mp_dict = {"pap": pap, "psp": psp}
    mp_dict = {mp: sparse_mx_to_torch_sparse_tensor(mp_dict[mp]) for mp in mp_dict.keys()}
    apa = sp.load_npz(path + "mp/apa.npz")
    sps = sp.load_npz(path + "mp/sps.npz")
    pspap = sp.load_npz(path + "mp/pspap.npz")
    assist_mp_dict = {"apa": apa, "sps": sps, "pspap": pspap}
    assist_mp_dict = {amp: sparse_mx_to_torch_sparse_tensor(assist_mp_dict[amp]) for amp in assist_mp_dict.keys()}


    train = np.load(path + "merged_train.npy")
    val = np.load(path + "merged_val.npy")
    test = np.load(path + "merged_test.npy")
    train = th.LongTensor(train)
    val = th.LongTensor(val)
    test = th.LongTensor(test)

    label = th.FloatTensor(label)
    feat_p = th.FloatTensor(preprocess_features(feat_p))
    feat_a = th.FloatTensor(preprocess_features(feat_a))
    feat_s = th.FloatTensor(preprocess_features(feat_s))
    feat_dic = {'p':feat_p, 'a':feat_a, 's':feat_s}
    type_range = {'p': range(0, 3025), 
                  'a': range(3025, 3025+5912), 
                  's': range(3025+5912, 3025+5912+57)}


    return feat_dic, type_range, mp_dict, assist_mp_dict, label, train, val, test


def load_dblp(type_num):

    path = "../data/dblp/"
    label = np.load(path + "labels.npy").astype('int32')
    label = encode_onehot(label)
    feat_a = sp.load_npz(path + "a_feat.npz")
    feat_p = sp.load_npz(path + "p_feat.npz")
    feat_c = sp.load_npz(path + "c_feat.npz")

    apa = sp.load_npz(path + "mp/apa.npz")
    apcpa = sp.load_npz(path + "mp/apcpa.npz")
    mp_dict = {"apa": apa, "apcpa": apcpa}
    mp_dict = {mp: sparse_mx_to_torch_sparse_tensor(mp_dict[mp]) for mp in mp_dict.keys()}
    cpc = sp.load_npz(path + "mp/cpc.npz")
    pap = sp.load_npz(path + "mp/pap.npz")
    pcp = sp.load_npz(path + "mp/pcp.npz")
    assist_mp_dict = {"cpc": cpc, "pcp": pap, "pcp": pcp}
    assist_mp_dict = {amp: sparse_mx_to_torch_sparse_tensor(assist_mp_dict[amp]) for amp in assist_mp_dict.keys()}


    train = np.load(path + "merged_train.npy")
    val = np.load(path + "merged_val.npy")
    test = np.load(path + "merged_test.npy")
    train = th.LongTensor(train)
    val = th.LongTensor(val)
    test = th.LongTensor(test)
    
    label = th.FloatTensor(label)
    feat_a = th.FloatTensor(preprocess_features(feat_a))
    feat_p = th.FloatTensor(preprocess_features(feat_p))
    feat_c = th.FloatTensor(preprocess_features(feat_c))

    feat_dic = {'a':feat_a, 'p':feat_p, 'c':feat_c}
    type_range = {'a': range(0, 4057), 
                  'p': range(4057, 4057+14328), 
                  'c': range(4057+14328, 4057+14328+20),}

    return feat_dic, type_range, mp_dict, assist_mp_dict, label, train, val, test


def load_imdb(type_num):

    path = "../data/imdb/"
    label = np.load(path + "labels.npy").astype('int32')
    label = encode_onehot(label)
    feat_m = sp.load_npz(path + "m_feat.npz").astype("float32")
    feat_d = sp.load_npz(path + "d_feat.npz")
    feat_a = sp.load_npz(path + "a_feat.npz")

    mam = sp.load_npz(path + "mp/mam.npz")
    mdm = sp.load_npz(path + "mp/mdm.npz")
    mp_dict = {"mam": mam, "mdm": mdm}
    mp_dict = {mp: sparse_mx_to_torch_sparse_tensor(mp_dict[mp]) for mp in mp_dict.keys()}
    ama = sp.load_npz(path + "mp/ama.npz")
    dmd = sp.load_npz(path + "mp/dmd.npz")
    mamdm = sp.load_npz(path + "mp/mamdm.npz")
    assist_mp_dict = {"ama": ama, "dmd": dmd, "mamdm": mamdm}
    assist_mp_dict = {amp: sparse_mx_to_torch_sparse_tensor(assist_mp_dict[amp]) for amp in assist_mp_dict.keys()}

    train = np.load(path + "merged_train.npy")
    val = np.load(path + "merged_val.npy")
    test = np.load(path + "merged_test.npy")
    train = th.LongTensor(train)
    val = th.LongTensor(val)
    test = th.LongTensor(test)

    label = th.FloatTensor(label)
    feat_m = th.FloatTensor(preprocess_features(feat_m))
    feat_d = th.FloatTensor(preprocess_features(feat_d))
    feat_a = th.FloatTensor(preprocess_features(feat_a))
    feat_dic = {'m':feat_m, 'd':feat_d, 'a':feat_a}
    type_range = {'m': range(0, 4661), 
                  'd': range(4661, 4661+2270), 
                  'a': range(4661+2270, 4661+2270+5841)}


    return feat_dic, type_range, mp_dict, assist_mp_dict, label, train, val, test


def load_data(dataset, type_num):
    if dataset == "acm":
        data = load_acm(type_num)
    elif dataset == "dblp":
        data = load_dblp(type_num)
    elif dataset == "imdb":
        data = load_imdb(type_num)
    return data
