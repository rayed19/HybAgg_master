if __name__ == '__main__':
    seeds = [0, 1, 2, 3, 4]  # 5 independent runs
    macro_f1_all, micro_f1_all, auc_all = [], [], []

    for seed in seeds:
        print(f"\n===== Run with seed {seed} =====")
        args.seed = seed
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # Train and get embeddings
        feat_dic, type_range, mp_dict, assist_mp_dict, label, idx_train, idx_val, idx_test = \
            load_data(args.dataset, args.type_num)
        classes_num = label.shape[-1]

        hyper_dict = {
            "encoder1": Mp_attn_encoder,
            "encoder2": None,
            "GaussianDiffusion": GaussianDiffusion,
            "Denoise": Denoise,
            "device": device,
            "num_classes": classes_num,
            "hidden_dim": args.hidden_dim,
            "d_emb_size": args.d_emb_size,
            "norm": args.norm,
            "steps": args.steps,
            "noise_scale": args.noise_scale,
            "noise_min": args.noise_min,
            "noise_max": args.noise_max,
            "sampling_steps": args.sampling_steps,
            "dims": args.dims,
            "feats_dim_dict": {k: feat_dic[k].shape[1] for k in feat_dic},
            "feat_drop": args.feat_drop,
            "attn_drop": args.attn_drop,
            "mp_name": list(mp_dict.keys()),
            "assist_mp_name": list(assist_mp_dict.keys()),
            "type_range": type_range,
            "tau": args.tau,
            "lam": args.lam,
            "alpha": args.alpha,
            "alpha2": args.alpha2,
            "mp_lam": args.mp_lam,
            "ic_lam": args.ic_lam,
            "cg_lam": args.cg_lam,
            "interest_type": args.interest_type,
            "nei_mask": args.nei_mask,
            "nei_rate": args.nei_rate,
        }

        h = Para(hyper_dict)
        model = HeCL(h)
        optimiser = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2_coef)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimiser, lr_lambda=lambda epoch: (0.9*epoch/10 + 0.1) if epoch < 10 else 1
        )

        if torch.cuda.is_available():
            model.cuda()
            feat_dic = {k: feat_dic[k].cuda() for k in feat_dic}
            mp_dict = {k: mp_dict[k].cuda() for k in mp_dict}
            assist_mp_dict = {k: assist_mp_dict[k].cuda() for k in assist_mp_dict}
            label = label.cuda()
            idx_train, idx_val, idx_test = idx_train.cuda(), idx_val.cuda(), idx_test.cuda()

        data_dict = {
            "feat_dic": feat_dic,
            "mp_dict": mp_dict,
            "assist_mp_dict": assist_mp_dict,
            "rebuild_mp": mp_dict,
            "labels": label
        }

        # Train
        for epoch in range(args.epochs + 1):
            model.train()
            optimiser.zero_grad()
            d = Para(data_dict)
            loss = model(d)
            loss.backward()
            optimiser.step()
            scheduler.step()

        # Eval
        model.eval()
        embeds = model.get_embeds(d)
        macro_f1, micro_f1, auc = evaluate(embeds, idx_train, idx_val, idx_test, label, classes_num, device,
                                           args.dataset, args.eva_lr, args.eva_wd)
        macro_f1_all.append(macro_f1)
        micro_f1_all.append(micro_f1)
        auc_all.append(auc)

    print("\n===== Final Results over 5 runs =====")
    print(f"Macro-F1: mean {np.mean(macro_f1_all):.4f}, std {np.std(macro_f1_all):.4f}")
    print(f"Micro-F1: mean {np.mean(micro_f1_all):.4f}, std {np.std(micro_f1_all):.4f}")
    print(f"AUC     : mean {np.mean(auc_all):.4f}, std {np.std(auc_all):.4f}")