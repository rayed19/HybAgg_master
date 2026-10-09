import argparse
import sys

argv = sys.argv
dataset = argv[1]


def acm_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="acm")
    # parser.add_argument('--ratio', type=int, default=[1, 3, 5])
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--nei_mask', type=bool, default=True)
    parser.add_argument('--mp_mask', type=bool, default=True)

    # The parameters of train
    parser.add_argument('--epochs', type=int, default=200)
    
    # The parameters of evaluation(testing)
    parser.add_argument('--eva_lr', type=float, default=0.05)
    parser.add_argument('--eva_wd', type=float, default=0)
    
    # The parameters of training process
    parser.add_argument('--lr', type=float, default=0.0010)
    parser.add_argument('--l2_coef', type=float, default=0)
    
    # model-specific parameters
    parser.add_argument('--hidden_dim', type=int, default=64) #
    parser.add_argument('--d_emb_size', type=int, default=8) #
    parser.add_argument('--norm', type=bool, default=True)
    parser.add_argument('--steps', type=int, default=80)# √
    parser.add_argument('--noise_scale', type=float, default=1e-4)
    parser.add_argument('--noise_min', type=float, default=0.0001)
    parser.add_argument('--noise_max', type=float, default=0.001)
    parser.add_argument('--sampling_steps', type=int, default=0)  #
    parser.add_argument('--dims', type=str, default='[64]') #

    parser.add_argument('--mp_lam', type=float, default=0.8)
    parser.add_argument('--tau', type=float, default=0.8)#
    parser.add_argument('--feat_drop', type=float, default=0.3)
    parser.add_argument('--attn_drop', type=float, default=0.5)
    parser.add_argument('--lam', type=float, default=0.5)#
    parser.add_argument('--interest_type', type=str, default='p') # R : target node type 
    parser.add_argument('--alpha', type=float, default=0.5) #
    parser.add_argument('--alpha2', type=float, default=0.5)  #

    parser.add_argument('--ic_lam', type=float, default=1.0)
    parser.add_argument('--cg_lam', type=float, default=0.00001)

    # CL parameters
    parser.add_argument('--mp_prob', type=float, default=0.2)#
    parser.add_argument('--nei_rate', type=float, default=0.1)#

    args, _ = parser.parse_known_args()
    args.type_num = [3025, 5912, 57]  # the number of every node type
    return args


def dblp_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="dblp")
    # parser.add_argument('--ratio', type=int, default=[1, 3, 5])
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0) # default 0 
    parser.add_argument('--nei_mask', type=bool, default=True)
    parser.add_argument('--mp_mask', type=bool, default=True)

    # The parameters of train
    # parser.add_argument('--epochs', type=int, default=950) # R : takes too long 
    parser.add_argument('--epochs', type=int, default=950) # R : epoch 200 ?? 
    
    # The parameters of evaluation
    parser.add_argument('--eva_lr', type=float, default=0.05)
    parser.add_argument('--eva_wd', type=float, default=0)
    
    # The parameters of learning process
    parser.add_argument('--lr', type=float, default=0.0008) # default : 0.0008
    parser.add_argument('--l2_coef', type=float, default=0)
    
    # model-specific parameters
    parser.add_argument('--hidden_dim', type=int, default=16) # R : to test 
    # parser.add_argument('--hidden_dim', type=int, default=64) 
    parser.add_argument('--d_emb_size', type=int, default=8)
    parser.add_argument('--norm', type=bool, default=True)
    parser.add_argument('--steps', type=int, default=50) # √
    parser.add_argument('--noise_scale', type=float, default=1e-4)
    parser.add_argument('--noise_min', type=float, default=0.0001)
    parser.add_argument('--noise_max', type=float, default=0.001)
    parser.add_argument('--sampling_steps', type=int, default=0)  #
    parser.add_argument('--dims', type=str, default='[16]') # R : SMCD default='[64]'

    parser.add_argument('--mp_lam', type=float, default=0.8)
    parser.add_argument('--tau', type=float, default=0.9)#
    parser.add_argument('--feat_drop', type=float, default=0.4)
    parser.add_argument('--attn_drop', type=float, default=0.35)
    parser.add_argument('--lam', type=float, default=0.5) #
    parser.add_argument('--interest_type', type=str, default='a')
    parser.add_argument('--alpha', type=float, default=0.5)  #
    parser.add_argument('--alpha2', type=float, default=0.5)  #

    parser.add_argument('--ic_lam', type=float, default=1.0)
    parser.add_argument('--cg_lam', type=float, default=0.00001)

    # CL parameters
    parser.add_argument('--mp_prob', type=float, default=0.7)
    parser.add_argument('--nei_rate', type=float, default=0.1)

    args, _ = parser.parse_known_args()
    args.type_num = [4057, 14328, 20]  # the number of every node type
    return args


def imdb_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="imdb")
    # parser.add_argument('--ratio', type=int, default=[1, 3, 5])
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--nei_mask', type=bool, default=True)
    parser.add_argument('--mp_mask', type=bool, default=True)

    # The parameters of train
    # parser.add_argument('--epochs', type=int, default=950)#1150 # R : desactivated 
    parser.add_argument('--epochs', type=int, default=950) # R : 200 
    
    # The parameters of evaluation
    parser.add_argument('--eva_lr', type=float, default=0.25)
    parser.add_argument('--eva_wd', type=float, default=0)
    
    # The parameters of learning process
    parser.add_argument('--lr', type=float, default=0.001) # R : Learning rate
    parser.add_argument('--l2_coef', type=float, default=0.0)
    
    # model-specific parameters
    parser.add_argument('--hidden_dim', type=int, default=16)
    parser.add_argument('--d_emb_size', type=int, default=8)
    parser.add_argument('--norm', type=bool, default=True)
    parser.add_argument('--steps', type=int, default=50) #  R : diffusion steps 
    parser.add_argument('--noise_scale', type=float, default=1e-4)
    parser.add_argument('--noise_min', type=float, default=0.0001)
    parser.add_argument('--noise_max', type=float, default=0.001)
    parser.add_argument('--sampling_steps', type=int, default=50)  # 
    parser.add_argument('--dims', type=str, default='[16]')

    # parser.add_argument('--mp_lam', type=float, default=0.3) # R desactivated
    parser.add_argument('--mp_lam', type=float, default=0.8) # R : 0.8 like the paper  (auxiliary meta-path contribution) 
 
    parser.add_argument('--tau', type=float, default=0.5)#
    parser.add_argument('--feat_drop', type=float, default=0.4)
    parser.add_argument('--attn_drop', type=float, default=0.3)
    parser.add_argument('--lam', type=float, default=0.5)#
    parser.add_argument('--interest_type', type=str, default='m')
    parser.add_argument('--alpha', type=float, default=0.5)  #
    parser.add_argument('--alpha2', type=float, default=0.5)  #

    # parser.add_argument('--ic_lam', type=float, default=0.4) # R desactivated 
    parser.add_argument('--ic_lam', type=float, default=0.5) # R : modified like the paper (IC self-supervised loss) 
    parser.add_argument('--cg_lam', type=float, default=0.000005)

    # CL parameters
    # parser.add_argument('--mp_prob', type=float, default=1.0) 
    parser.add_argument('--mp_prob', type=float, default=0.3) # R : a meta-path masking ratio like the paper (e.g., mp_prob=0.5) {0.1, 0.3, 0.5, 0.7, 0.9}
    parser.add_argument('--nei_rate', type=float, default=0.5)
    
    args, _ = parser.parse_known_args()
    args.type_num = [4661, 2270, 5841]  # the number of every node type
    return args


def set_params():
    if dataset == "acm":
        args = acm_params()
    elif dataset == "dblp":
        args = dblp_params()
    elif dataset == "imdb":
        args = imdb_params()
    return args
