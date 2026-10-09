import torch
import torch.nn as nn
import torch.nn.functional as F
from .contrast import Contrast
import torch.utils.data as dataloader
import torch.utils.data as data

class HeCL(nn.Module):
    def __init__(self, h):
        super(HeCL, self).__init__()
        self.type_range = h.type_range
        self.interest_type = h.interest_type
        self.hidden_dim = h.hidden_dim#
        self.fc_dict = nn.ModuleDict({k: nn.Linear(h.feats_dim_dict[k], h.hidden_dim, bias=True)
                                      for k in h.feats_dim_dict})
        for k in self.fc_dict:
            nn.init.xavier_normal_(self.fc_dict[k].weight, gain=1.414)

        if h.feat_drop > 0:
            self.feat_drop = nn.Dropout(h.feat_drop)
        else:
            self.feat_drop = lambda x: x
        
        self.encoder1 = h.encoder1(h)
        self.encoder2 = h.encoder2(h) if h.encoder2 else self.encoder1
        self.contrast = Contrast(h)

        self.diffusion_model = h.GaussianDiffusion(h.noise_scale, h.noise_min, h.noise_max, h.steps).to(h.device)
        out_dims = eval(h.dims) + [h.hidden_dim]
        in_dims = out_dims[::-1]
        self.denoise_model = h.Denoise(in_dims, out_dims, h.d_emb_size, norm=h.norm).to(h.device)

        self.h = h
        self.sampling_steps = h.sampling_steps

    def forward(self, d, full=False):
        h_all = []
        for k in self.type_range:
            h_all.append(F.elu(self.feat_drop(self.fc_dict[k](d.feat_dic[k]))))
        d.h = torch.cat(h_all, dim=0)


        z1, z1_assist = self.encoder1(d, full=False)
        z1_assist = z1 + self.h.mp_lam * z1_assist
        z1_diff_loss, z1_output = self.diffusion_model.training_losses(self.denoise_model, z1, z1_assist)
        z1_output = z1_output[self.type_range[self.interest_type]]


        mp_list = self.h.mp_name[:]
        for mp in mp_list:
            diffusionData = DiffusionData(d.mp_dict[mp].to_dense())
            diffusionLoader = dataloader.DataLoader(diffusionData, batch_size=1024, shuffle=True, num_workers=0)
            mp_out_dims = eval(self.h.dims) + [d.mp_dict[mp].shape[1]]
            mp_in_dims = mp_out_dims[::-1]

            self.mp_denoise_model = self.h.Denoise(mp_in_dims, mp_out_dims, self.h.d_emb_size, norm=self.h.norm).cuda()
            with torch.no_grad():
                all_edges = []

                for _, batch in enumerate(diffusionLoader):
                    batch_item, batch_index = batch
                    batch_item, batch_index = batch_item.cuda(), batch_index.cuda()

                    denoised_batch = self.diffusion_model.p_sample(self.mp_denoise_model, batch_item, self.h.sampling_steps)


                    all_edges.append(denoised_batch)

                if all_edges:

                    full_denoised = torch.cat(all_edges, dim=0)

                    assert full_denoised.shape == d.mp_dict[mp].shape, \
                        f"拼接形状不匹配：去噪后 {full_denoised.shape} vs 原始 {d.mp_dict[mp].shape}"
                    threshold = 1e-1  # 自定义阈值（根据数据分布调整）
                    non_zero_mask = torch.abs(full_denoised) > threshold  # 大于阈值的元素视为有效
                    non_zero_indices = torch.nonzero(non_zero_mask, as_tuple=False)  # 有效元素索引 (N, 2)
                    non_zero_values = full_denoised[non_zero_mask]  # 有效元素值


                    if non_zero_indices.numel() > 0:  #

                        indices = non_zero_indices.t().contiguous()
                        sparse_mp = torch.sparse.FloatTensor(
                            indices,
                            non_zero_values,
                            d.mp_dict[mp].shape
                        ).coalesce()
                        d.rebuild_mp[mp] = sparse_mp
        z2= self.encoder2.diffusion_forward(d, full=False)

        z2= z2[self.type_range[self.interest_type]]


        d.z1 = z1_output
        d.z2 = z2

        clloss = self.contrast(d)
        loss= z1_diff_loss.mean() + clloss
        return loss

    def get_embeds(self, d):
        h_all = []
        for k in self.type_range:#
            h_all.append(F.elu(self.feat_drop(self.fc_dict[k](d.feat_dic[k]))))
        d.h = torch.cat(h_all, dim=0)

        z1 = self.encoder1.forward2(d)#
        z1 = z1[self.type_range[self.interest_type]]
        
        return z1.detach()

class DiffusionData(data.Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        item = self.data[index]
        return item, index

    def __len__(self):
        return len(self.data)