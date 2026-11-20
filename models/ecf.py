
import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossModalAttention(nn.Module):

    def __init__(self, d_model=128, n_heads=4, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, H_ele, H_target):
        B, N, d = H_ele.shape
        Q = self.W_q(H_ele).view(B, N, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(H_target).view(B, N, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(H_target).view(B, N, self.n_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)
        attn = self.dropout(F.softmax(scores, dim=-1))
        out = torch.matmul(attn, V)
        return self.W_o(out.transpose(1, 2).contiguous().view(B, N, d))


class ElectricalQualityGate(nn.Module):

    def __init__(self, d_model=128):
        super().__init__()
        self.estimator = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(inplace=True),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid(),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, F_original, F_guided, H_ele_pooled, return_q=False):
        
        q = self.estimator(H_ele_pooled)
        q = q.unsqueeze(1)
        out = self.norm(q * F_guided + (1 - q) * F_original)
        if return_q:
            return out, q.squeeze(1).squeeze(1)
        return out


class ECFModule(nn.Module):

    def __init__(self, d_model=128, n_heads=4, dropout=0.1):
        super().__init__()
        self.cross_vib = CrossModalAttention(d_model, n_heads, dropout)
        self.gate_vib = ElectricalQualityGate(d_model)
        self.cross_aco = CrossModalAttention(d_model, n_heads, dropout)
        self.gate_aco = ElectricalQualityGate(d_model)

    def forward(self, H_ele, H_vib, H_aco, aco_mask=None, return_q=False):

        H_ele_pooled = H_ele.mean(dim=1)

        F_vib = self.cross_vib(H_ele, H_vib)
        H_vib_out, q_vib = self.gate_vib(H_vib, F_vib, H_ele_pooled, return_q=True)

        F_aco = self.cross_aco(H_ele, H_aco)
        H_aco_out, q_aco = self.gate_aco(H_aco, F_aco, H_ele_pooled, return_q=True)

        if aco_mask is not None:
            H_aco_out = H_aco_out * aco_mask.view(-1, 1, 1)

        if return_q:
            return H_vib_out, H_aco_out, {"q_vib": q_vib, "q_aco": q_aco}
        return H_vib_out, H_aco_out
