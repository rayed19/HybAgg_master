# R commented lines are added or modified code from Rihab AYED 
import time
import argparse

import torch.nn.functional as F
import torch.sparse
import numpy as np
import dgl

from utils.pytorchtools import EarlyStopping
from utils.data import load_IMDB_data
from utils.tools import evaluate_results_nc
from model import MAGNN_nc
import psutil, os # R 

# Params
out_dim = 3
dropout_rate = 0.5
lr = 0.005
weight_decay = 0.001
etypes_lists = [[[0, 1], [2, 3]],
                [[1, 0], [1, 2, 3, 0]],
                [[3, 2], [3, 0, 1, 2]]]

def run_model_IMDB(feats_type, num_layers, hidden_dim, num_heads, attn_vec_dim, rnn_type,
                   num_epochs, patience, repeat, save_postfix):
    # R : added for memory tracking 
    process = psutil.Process(os.getpid())
    ram_before = process.memory_info().rss / 1e6  # MB
    start_time_total = time.time()
    #if torch.cuda.is_available():
      #  torch.cuda.reset_peak_memory_stats()
    # End R 
    
    nx_G_lists, edge_metapath_indices_lists, features_list, adjM, type_mask, labels, train_val_test_idx = load_IMDB_data()
    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') # R : desactivated because the requirements are low for my used cuda 
    device = torch.device("cpu") # R 
    features_list = [torch.FloatTensor(features.todense()).to(device) for features in features_list]
    # R comment : Feature ablation / feature availability scenarios # The following code of MAGNN  is used for several scenarios : 
    # R : feats_type == 0, Use real node features for all node types
    # R : feats_type == 1, Keep features only for target node type, replace others with dummy 10-dim vectors
    # R : feats_type == 2, Identity features for non-target node types (each node gets a one-hot / identity vector)
    # R : feats_type == 3, Identity features for all node types
    if feats_type == 0:
        in_dims = [features.shape[1] for features in features_list]
    elif feats_type == 1:
        in_dims = [features_list[0].shape[1]] + [10] * (len(features_list) - 1)
        for i in range(1, len(features_list)):
            features_list[i] = torch.zeros((features_list[i].shape[0], 10)).to(device)
    elif feats_type == 2:
        in_dims = [features.shape[0] for features in features_list]
        in_dims[0] = features_list[0].shape[1]
        for i in range(1, len(features_list)):
            dim = features_list[i].shape[0]
            indices = np.vstack((np.arange(dim), np.arange(dim)))
            indices = torch.LongTensor(indices)
            values = torch.FloatTensor(np.ones(dim))
            features_list[i] = torch.sparse.FloatTensor(indices, values, torch.Size([dim, dim])).to(device)
    elif feats_type == 3:
        in_dims = [features.shape[0] for features in features_list]
        for i in range(len(features_list)):
            dim = features_list[i].shape[0]
            indices = np.vstack((np.arange(dim), np.arange(dim)))
            indices = torch.LongTensor(indices)
            values = torch.FloatTensor(np.ones(dim))
            features_list[i] = torch.sparse.FloatTensor(indices, values, torch.Size([dim, dim])).to(device)
    edge_metapath_indices_lists = [[torch.LongTensor(indices).to(device) for indices in indices_list] for indices_list in
                                   edge_metapath_indices_lists]
    labels = torch.LongTensor(labels).to(device)
    g_lists = []
    for nx_G_list in nx_G_lists:
        g_lists.append([])
        for nx_G in nx_G_list:
            g = dgl.DGLGraph(multigraph=True)
            g.add_nodes(nx_G.number_of_nodes())
            g.add_edges(*list(zip(*sorted(map(lambda tup: (int(tup[0]), int(tup[1])), nx_G.edges())))))
            g_lists[-1].append(g)
    train_idx = train_val_test_idx['train_idx']
    val_idx = train_val_test_idx['val_idx']
    test_idx = train_val_test_idx['test_idx']

    svm_macro_f1_lists = []
    svm_micro_f1_lists = []
    nmi_mean_list = []
    nmi_std_list = []
    ari_mean_list = []
    ari_std_list = []
    for _ in range(repeat):
        net = MAGNN_nc(num_layers, [2, 2, 2], 4, etypes_lists, in_dims, hidden_dim, out_dim, num_heads, attn_vec_dim,
                       rnn_type, dropout_rate)
        net.to(device)
        optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
        # R comment : Only movie nodes (type 0) are used for: Training, Validation, Testing. Embedding output (the saved    embeddings_IMDB_small.npz)
        target_node_indices = np.where(type_mask == 0)[0]

        # training loop
        net.train() # R comment : this call is redundant ? 
        early_stopping = EarlyStopping(patience=patience, verbose=True, save_path='checkpoint/checkpoint_{}.pt'.format(save_postfix))
        dur1 = []
        dur2 = []
        dur3 = []
        for epoch in range(num_epochs):
            t0 = time.time()

            # training forward
            net.train()
            logits, embeddings = net((g_lists, features_list, type_mask, edge_metapath_indices_lists), target_node_indices)
            logp = F.log_softmax(logits, 1)
            train_loss = F.nll_loss(logp[train_idx], labels[train_idx])

            t1 = time.time()
            dur1.append(t1 - t0)

            # autograd
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            t2 = time.time()
            dur2.append(t2 - t1)

            # validation forward
            net.eval()
            with torch.no_grad():
                logits, embeddings = net((g_lists, features_list, type_mask, edge_metapath_indices_lists), target_node_indices)
                logp = F.log_softmax(logits, 1)
                val_loss = F.nll_loss(logp[val_idx], labels[val_idx])

            t3 = time.time()
            dur3.append(t3 - t2)

            # print info
            print(
                "Epoch {:05d} | Train_Loss {:.4f} | Val_Loss {:.4f} | Time1(s) {:.4f} | Time2(s) {:.4f} | Time3(s) {:.4f}".format(
                    epoch, train_loss.item(), val_loss.item(), np.mean(dur1), np.mean(dur2), np.mean(dur3)))
            # early stopping
            early_stopping(val_loss, net)
            if early_stopping.early_stop:
                print('Early stopping!')
                break


        # testing with evaluate_results_nc
        net.load_state_dict(torch.load('checkpoint/checkpoint_{}.pt'.format(save_postfix)))
        net.eval()
        with torch.no_grad():
            logits, embeddings = net((g_lists, features_list, type_mask, edge_metapath_indices_lists), target_node_indices)
            svm_macro_f1_list, svm_micro_f1_list, nmi_mean, nmi_std, ari_mean, ari_std = evaluate_results_nc(
                embeddings[test_idx].cpu().numpy(), labels[test_idx].cpu().numpy(), num_classes=out_dim)
                        
            # R : Save embeddings and labels for EGAI or other downstream tasks
            np.savez(f'embeddings_{save_postfix}.npz', embeddings=embeddings.cpu().numpy(), labels=labels[target_node_indices].cpu().numpy())
            print(f'Embeddings and labels saved to embeddings_{save_postfix}.npz')
            print("svm_macro_f1_lists:", svm_macro_f1_lists)
            print("shape:", np.array(svm_macro_f1_lists).shape)
            svm_macro_f1_lists.append(svm_macro_f1_list)
            svm_micro_f1_lists.append(svm_micro_f1_list)
            nmi_mean_list.append(nmi_mean)
            nmi_std_list.append(nmi_std)
            ari_mean_list.append(ari_mean)
            ari_std_list.append(ari_std)
            # End R 

    # R : report total time and memory ---
    end_time_total = time.time()
    ram_after = process.memory_info().rss / 1e6
    print(f"\n=== MAGNN summary ===")
    print(f"training time: {end_time_total - start_time_total:.2f} s (~{(end_time_total - start_time_total)/60:.2f} min)")
    print(f"RAM consumption: {ram_after - ram_before:.1f} MB")
    #if torch.cuda.is_available(): # not used
       # print(f"GPU max used memory: {torch.cuda.max_memory_allocated()/1e6:.1f} MB")
    # end R 
    # print out a summary of the evaluations
    svm_macro_f1_lists = np.transpose(svm_macro_f1_lists, (1,0,2))
    svm_micro_f1_lists = np.transpose(svm_micro_f1_lists, (1,0,2))
        
    nmi_mean_list = np.array(nmi_mean_list)
    nmi_std_list = np.array(nmi_std_list)
    ari_mean_list = np.array(ari_mean_list)
    ari_std_list = np.array(ari_std_list)
    print('----------------------------------------------------------------')
    print('SVM tests summary')
    print('Macro-F1: ' + ', '.join(['{:.6f}~{:.6f} ({:.1f})'.format(
        macro_f1[:, 0].mean(), macro_f1[:, 1].mean(), train_size) for macro_f1, train_size in
        zip(svm_macro_f1_lists, [0.8, 0.6, 0.4, 0.2])]))
    print('Micro-F1: ' + ', '.join(['{:.6f}~{:.6f} ({:.1f})'.format(
        micro_f1[:, 0].mean(), micro_f1[:, 1].mean(), train_size) for micro_f1, train_size in
        zip(svm_micro_f1_lists, [0.8, 0.6, 0.4, 0.2])]))
    print('K-means tests summary')
    print('NMI: {:.6f}~{:.6f}'.format(nmi_mean_list.mean(), nmi_std_list.mean()))
    print('ARI: {:.6f}~{:.6f}'.format(ari_mean_list.mean(), ari_std_list.mean()))
''' R : 
# Safe Configuration for CPU / Low-RAM Machines
if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='MAGNN testing for the IMDB dataset (CPU-friendly)')
    ap.add_argument('--feats-type', type=int, default=1,
                    help='0 - loaded features; '
                         '1 - only target node features (zero vec for others); '
                         '2 - only target node features (id vec for others); '
                         '3 - all id vec. Default is 1 for CPU efficiency.')
    ap.add_argument('--layers', type=int, default=1,
                    help='Number of layers. Default is 1 for lower memory use.')
    ap.add_argument('--hidden-dim', type=int, default=32,
                    help='Hidden dimension size. Default 32 (was 64).')
    ap.add_argument('--num-heads', type=int, default=4,
                    help='Number of attention heads. Default 4 (was 8).')
    ap.add_argument('--attn-vec-dim', type=int, default=64,
                    help='Attention vector dimension. Default 64 (was 128).')
    ap.add_argument('--rnn-type', default='RotatE0',
                    help='Type of the aggregator. Default RotatE0.')
    ap.add_argument('--epoch', type=int, default=30,
                    help='Number of epochs. Default 30 (was 100).')
    ap.add_argument('--patience', type=int, default=5,
                    help='Patience for early stopping. Default 5 (was 10).')
    ap.add_argument('--repeat', type=int, default=1,
                    help='Repeat count. Default 1 (avoid extra memory).')
    ap.add_argument('--save-postfix', default='IMDB_small',
                    help='Postfix for saved model/results. Default IMDB_small.')
'''
# End R 
# R : default configuration 
if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='MRGNN testing for the IMDB dataset')
    ap.add_argument('--feats-type', type=int, default=2,
                    help='Type of the node features used. ' +
                         '0 - loaded features; ' +
                         '1 - only target node features (zero vec for others); ' +
                         '2 - only target node features (id vec for others); ' +
                         '3 - all id vec. Default is 2.')
    ap.add_argument('--layers', type=int, default=2, help='Number of layers. Default is 2.')
    ap.add_argument('--hidden-dim', type=int, default=64, help='Dimension of the node hidden state. Default is 64.')
    ap.add_argument('--num-heads', type=int, default=8, help='Number of the attention heads. Default is 8.')
    ap.add_argument('--attn-vec-dim', type=int, default=128, help='Dimension of the attention vector. Default is 128.')
    ap.add_argument('--rnn-type', default='RotatE0', help='Type of the aggregator. Default is RotatE0.')
    ap.add_argument('--epoch', type=int, default=100, help='Number of epochs. Default is 100.')
    # ap.add_argument('--patience', type=int, default=10, help='Patience. Default is 10.') # R 
    ap.add_argument('--patience', type=int, default=30, help='Patience. Default is 30.') # R 
    ap.add_argument('--repeat', type=int, default=1, help='Repeat the training and testing for N times. Default is 1.') # change it to 10 runs later .. 
    ap.add_argument('--save-postfix', default='IMDB', help='Postfix for the saved model and result. Default is IMDB.')
    args = ap.parse_args()
    run_model_IMDB(args.feats_type, args.layers, args.hidden_dim, args.num_heads, args.attn_vec_dim, args.rnn_type,
                   args.epoch, args.patience, args.repeat, args.save_postfix)
