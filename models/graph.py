
import torch
import torch.nn as nn
import torch.nn.functional as F


def build_local_adj(n_nodes, w_local=3):
    adj = torch.zeros(n_nodes, n_nodes)
    for i in range(n_nodes):
        for j in range(max(0, i - w_local), min(n_nodes, i + w_local + 1)):
            if i != j:
                adj[i, j] = 1.0
    return adj


def build_global_adj(
    n_nodes,
    gamma=0.5,
    n_mods=3,
    w_local=3,
    use_intra_temporal=True,
    use_sync=True,
    use_delay=True,
):

    N = n_nodes
    total = n_mods * N

    A_intra = torch.zeros(total, total)
    if use_intra_temporal:
        local = build_local_adj(N, w_local)
        for m in range(n_mods):
            A_intra[m * N:(m + 1) * N, m * N:(m + 1) * N] = local

    A_sync = torch.zeros(total, total)
    if use_sync:
        for t in range(N):
            for m1 in range(n_mods):
                for m2 in range(n_mods):
                    if m1 != m2:
                        A_sync[m1 * N + t, m2 * N + t] = 1.0

    A_causal = torch.zeros(total, total)
    if use_delay:
        for t in range(N - 1):
            for m1 in range(n_mods):
                for m2 in range(m1 + 1, n_mods):
                    A_causal[m1 * N + t, m2 * N + t + 1] = gamma
                    A_causal[m2 * N + t, m1 * N + t + 1] = gamma

    return [A_intra, A_sync, A_causal]


def add_self_loops(adj_list):
    result = []
    for A in adj_list:
        I = torch.eye(A.size(0), device=A.device)
        result.append(A + I)
    return result


def normalize_adj(adj_list):
    result = []
    for A in adj_list:
        deg = A.sum(dim=1).clamp(min=1)
        d_inv_sqrt = torch.diag(deg.pow(-0.5))
        result.append(d_inv_sqrt @ A @ d_inv_sqrt)
    return result


class ResidualGCNLayer(nn.Module):

    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.W = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, H, A_norm):
        """H: (B, N, d)，A_norm: (N, N) 或 (B, N, N)。"""
        msg = torch.bmm(A_norm, H) if A_norm.dim() == 3 else torch.matmul(A_norm.unsqueeze(0), H)
        out = self.W(msg)
        out = F.relu(out)
        out = self.dropout(out)
        return out + H


class ResidualGCN(nn.Module):

    def __init__(self, d_model=128, n_layers=3, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            ResidualGCNLayer(d_model, dropout) for _ in range(n_layers)
        ])
        self.n_layers = n_layers

    def forward(self, H, A_norm):
        for layer in self.layers:
            H = layer(H, A_norm)
        return H


class RGCNLayer(nn.Module):

    def __init__(self, d_model=128, n_relations=3, dropout=0.1):
        super().__init__()
        self.W = nn.ModuleList([nn.Linear(d_model, d_model, bias=False) for _ in range(n_relations)])
        self.W_self = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.n_relations = n_relations

    def forward(self, H, adj_norm_list):
    
        out = self.W_self(H)
        for r in range(self.n_relations):
            A = adj_norm_list[r]
            msg = torch.bmm(A.unsqueeze(0), H) if A.dim() == 2 else torch.matmul(A, H)
            out = out + self.W[r](msg)
        out = F.relu(out)
        out = self.dropout(out)
        return out + H


class RGCN(nn.Module):
    

    def __init__(self, d_model=128, n_relations=3, n_layers=2, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            RGCNLayer(d_model, n_relations, dropout) for _ in range(n_layers)
        ])

    def forward(self, H, adj_norm_list):
        for layer in self.layers:
            H = layer(H, adj_norm_list)
        return H


class AttentionReadout(nn.Module):


    def __init__(self, d_model=128):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, d_model) * 0.02)

    def forward(self, H):
        """H: (B, N, d) -> (B, d)。"""
        B, N, d = H.shape
        q = self.query.unsqueeze(0).expand(B, 1, d)
        scores = torch.bmm(q, H.transpose(1, 2)) / (d ** 0.5)
        attn = F.softmax(scores, dim=-1)
        out = torch.bmm(attn, H).squeeze(1)
        return out


class MultiModalReadout(nn.Module):

    def __init__(self, d_model=128, n_mods=3):
        super().__init__()
        self.n_mods = n_mods
        self.readout_ele = AttentionReadout(d_model)
        self.readout_vib = AttentionReadout(d_model)
        self.readout_aco = AttentionReadout(d_model)
        self.readout_multi = AttentionReadout(d_model)

    def forward(self, H_loc_ele, H_loc_vib, H_loc_aco, H_glo):
        f_ele = self.readout_ele(H_loc_ele)
        f_vib = self.readout_vib(H_loc_vib)
        f_aco = self.readout_aco(H_loc_aco)
        f_multi = self.readout_multi(H_glo)
        return f_ele, f_vib, f_aco, f_multi
