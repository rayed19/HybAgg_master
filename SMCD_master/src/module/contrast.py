import torch
import torch.nn as nn
import torch.nn.functional as F

def InfoNCE(sim_matrix):

    pos_matrix = torch.eye(sim_matrix.shape[0]).to(sim_matrix.device)

    sim_matrix_norm = sim_matrix / (torch.sum(sim_matrix, dim=1).view(-1, 1) + 1e-8)

    loss = -torch.log((sim_matrix_norm * pos_matrix).sum(dim=-1)).mean()

    return loss


class Contrast(nn.Module):
    def __init__(self, h):
        super(Contrast, self).__init__()
        self.proj = nn.Sequential(
            nn.Linear(h.hidden_dim, h.hidden_dim),
            nn.ELU(),
            nn.Linear(h.hidden_dim, h.hidden_dim)
        )


        self.classifier = nn.Sequential(
            nn.Linear(h.hidden_dim, h.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(h.hidden_dim // 2, h.num_classes)
        )
        self.tau = h.tau
        self.lam = h.lam
        self.h = h
        for model in self.proj:
            if isinstance(model, nn.Linear):
                nn.init.xavier_normal_(model.weight, gain=1.414)
        for model in self.classifier:
            if isinstance(model, nn.Linear):
                nn.init.xavier_normal_(model.weight, gain=1.414)

    def sim(self, z1, z2):

        z1_norm = torch.norm(z1, dim=-1, keepdim=True)
        z2_norm = torch.norm(z2, dim=-1, keepdim=True)

        dot_numerator = torch.mm(z1, z2.t())

        dot_denominator = torch.mm(z1_norm, z2_norm.t())
        sim_matrix = torch.exp(dot_numerator / dot_denominator / self.tau)
        return sim_matrix

    def normalize(self, Z):

        mu = torch.mean(Z, dim=0, keepdim=True)
        sigma = torch.std(Z, dim=0, keepdim=True)
        return (Z - mu) / (sigma + 1e-8)

    def covariance_loss(self, Z_norm):

        corr = torch.matmul(Z_norm.T, Z_norm)

        off_diag = corr - torch.diag(torch.diag(corr))  #
        return torch.sum(off_diag ** 2)

    def category_distance(self, Z, Cp_indices, Cq_indices):

        if len(Cp_indices) == 0 or len(Cq_indices) == 0:
            return torch.tensor(0.0, device=Z.device)
        Zp = Z[Cp_indices]  # [|Cp|, D]
        Zq = Z[Cq_indices]  # [|Cq|, D]
        mean_inner = torch.mean(torch.matmul(Zp, Zq.T))
        return 1 - mean_inner

    def compute_view_lcg(self, Z, C_list, C0_indices, K):

        s = 0.0
        for Ci in C_list:
            if len(Ci) < 2:
                continue
            s += self.category_distance(Z, Ci, Ci)
        s /= K


        d = 0.0
        count_pairs = 0
        for i in range(K):
            for j in range(i + 1, K):
                d += self.category_distance(Z, C_list[i], C_list[j])
                count_pairs += 1
        d = d / count_pairs if count_pairs > 0 else 1e-8


        u = 0.0
        if len(C0_indices) > 0:
            for Ci in C_list:
                u += self.category_distance(Z, C0_indices, Ci)
            u /= K
        return s, d, u

    def category_guided_loss(self, Za, Zb, onehot_labels):

        N, K_total = onehot_labels.shape
        device = onehot_labels.device


        label_sums = onehot_labels.sum(dim=1)
        has_label = (label_sums > 1e-6)
        category_indices = torch.zeros(N, dtype=torch.long, device=device)
        if has_label.any():

            class_ids = torch.argmax(onehot_labels[has_label], dim=1)  # 0~2
            category_indices[has_label] = class_ids + 1  # 1~3


        C0_indices = torch.where(category_indices == 0)[0]
        unique_labels = torch.unique(category_indices)
        valid_K = len(unique_labels) - 1
        if valid_K <= 0:
            return torch.tensor(0.0, device=device)


        C_indices = {c.item(): torch.where(category_indices == c)[0]
                     for c in unique_labels if c.item() != 0}
        C_list = list(C_indices.values())


        s_za, d_za, u_za = self.compute_view_lcg(Za, C_list, C0_indices, valid_K)
        s_zb, d_zb, u_zb = self.compute_view_lcg(Zb, C_list, C0_indices, valid_K)


        eps = 1e-8
        lcg = (s_za / (d_za + eps)) + (s_zb / (d_zb + eps)) + self.h.alpha2 * (u_za + u_zb)
        return lcg

    def forward(self, d):

        z_proj1 = self.proj(d.z1)
        z_proj2 = self.proj(d.z2)

        matrix_1to2 = self.sim(z_proj1, z_proj2)
        matrix_2to1 = matrix_1to2.t()

        lori_1to2 = InfoNCE(matrix_1to2)
        lori_2to1 = InfoNCE(matrix_2to1)

        loss = self.lam * lori_1to2 + (1 - self.lam) * lori_2to1


        supervised_loss = torch.tensor(0.0, device=d.z1.device)
        if hasattr(d, 'labels') and d.labels is not None:
            labels = d.labels
            device = labels.device

            labeled_mask = (labels.sum(dim=1) > 1e-6)
            labeled_indices = torch.nonzero(labeled_mask, as_tuple=False).squeeze(dim=1)

            if labeled_indices.numel() > 0:

                z_fused = (d.z1 + d.z2) / 2
                labeled_embeddings = z_fused[labeled_indices]
                labeled_labels = labels[labeled_indices]


                preds = self.classifier(labeled_embeddings)


                log_probs = F.log_softmax(preds, dim=1)
                supervised_loss = -torch.mean(torch.sum(labeled_labels * log_probs, dim=1))


        Za_norm = self.normalize(d.z1)
        Zb_norm = self.normalize(d.z2)

        cross_corr = torch.matmul(Za_norm.T, Zb_norm)
        invariance = torch.sum(torch.diag(cross_corr))
        cov_za = self.covariance_loss(Za_norm)
        cov_zb = self.covariance_loss(Zb_norm)
        total_covariance = cov_za + cov_zb

        denominator = invariance - self.h.alpha * total_covariance
        eps = 1e-8

        lic_loss = 1.0 / (denominator + eps)


        lcg_loss = torch.tensor(0.0, device=d.z1.device)
        if hasattr(d, 'labels') and d.labels is not None:
            lcg_loss = self.category_guided_loss(d.z1, d.z2, d.labels)

        total_loss = supervised_loss + self.h.ic_lam * lic_loss + (1-self.h.ic_lam) * lcg_loss

        return total_loss


