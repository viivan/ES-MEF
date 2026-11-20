
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import config as cfg


class AEDSFusion(nn.Module):

    def __init__(self, n_branches=3, n_classes=3, eps=1e-7, mode="sa_ds", eta=0.1):
        super().__init__()
        self.n_branches = n_branches
        self.n_classes = n_classes
        self.eps = eps
        self.mode = mode
        self.eta = eta
        self.W = nn.Parameter(torch.zeros(n_branches, n_classes))

    def prob_to_bpa(self, P):

        return P / P.sum(dim=-1, keepdim=True).clamp(min=self.eps)

    def combine_two(self, m1, t1, m2, t2):

        B, C = m1.shape
        combined = m1 * m2 + m1 * t2.unsqueeze(1) + t1.unsqueeze(1) * m2
        combined_theta = t1 * t2
        total = (combined.sum(dim=-1) + combined_theta).clamp(min=self.eps)
        return combined / total.unsqueeze(1), combined_theta / total

    def forward(self, probs_list):
        B, C = probs_list[0].shape

        if self.mode == "mean":
            P_final = torch.stack(probs_list, dim=0).mean(dim=0)
            return P_final, [torch.ones(B, device=P_final.device)] * len(probs_list)

        K = len(probs_list)
        if self.mode == "sa":
            w_bar = torch.full((K, C), 1.0 / K, device=probs_list[0].device)
        else:
            w_bar = torch.softmax(self.W, dim=0)

        alpha_list, masses, thetas = [], [], []
        for k in range(K):
            m = self.prob_to_bpa(probs_list[k])

            if self.mode == "ds":
                alpha = torch.ones(B, device=m.device)
            else:
                H = -(m * torch.log(m.clamp(min=self.eps))).sum(dim=-1)
                H_bar = H / math.log(C)
                alpha = torch.exp(-self.eta * H_bar)

            m_disc = alpha.unsqueeze(1) * m
            theta_disc = 1.0 - alpha

            m_tilde = w_bar[k].unsqueeze(0) * m_disc

            denom = m_tilde.sum(dim=-1, keepdim=True) + theta_disc.unsqueeze(1)
            m_final = m_tilde / denom.clamp(min=self.eps)
            theta_final = theta_disc / denom.squeeze(-1).clamp(min=self.eps)

            alpha_list.append(alpha)
            masses.append(m_final)
            thetas.append(theta_final)

        M, T = masses[0], thetas[0]
        for k in range(1, K):
            M, T = self.combine_two(M, T, masses[k], thetas[k])

        P_final = M / (1.0 - T.unsqueeze(1)).clamp(min=self.eps)

        return P_final, alpha_list


class HierarchicalClassifiers(nn.Module):

    def __init__(self, d_model=128, hidden_dim=64, n_classes=5, dropout=cfg.DROPOUT):
        super().__init__()
        def _head():
            return nn.Sequential(
                nn.Linear(d_model, hidden_dim), nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, n_classes),
            )

        self.cls_ele = _head()
        self.cls_vib = _head()
        self.cls_aco = _head()

        self.cls_multi = nn.Sequential(
            nn.Linear(d_model, hidden_dim * 2), nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, f_ele, f_vib, f_aco, f_multi):
        return (
            self.cls_ele(f_ele),
            self.cls_vib(f_vib),
            self.cls_aco(f_aco),
            self.cls_multi(f_multi),
        )
