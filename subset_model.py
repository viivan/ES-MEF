import torch
import torch.nn as nn
import torch.nn.functional as F

import config as cfg
from models.extractors import SensorLSTM, MultiScale1DCNN, Acoustic1DCNN, WindowEncoder
from models.graph import (
    ResidualGCN, RGCN, AttentionReadout,
    build_local_adj, build_global_adj, add_self_loops, normalize_adj,
)
from models.ecf import ECFModule
from models.fusion import AEDSFusion

ALL_MODALITIES = ("ele", "vib", "aco")


def _make_encoder(mod, d_model):
    if mod == "ele":
        return WindowEncoder(SensorLSTM(in_channels=3, hidden=cfg.LSTM_HIDDEN,
                                        layers=cfg.LSTM_LAYERS, out_dim=d_model), d_model)
    if mod == "vib":
        return WindowEncoder(MultiScale1DCNN(in_channels=4, base_ch=cfg.CNN_CHANNELS[0],
                                             kernels=cfg.CNN_KERNELS, out_dim=d_model), d_model)
    return WindowEncoder(Acoustic1DCNN(in_channels=1, base_ch=cfg.CNN_CHANNELS[0],
                                       out_dim=d_model), d_model)


class SubsetESMEF(nn.Module):

    def __init__(self, modalities=ALL_MODALITIES, d_model=cfg.D_MODEL,
                 n_classes=cfg.NUM_CLASSES, hidden_dim=cfg.HIDDEN_DIM,
                 fusion_mode="sa_ds", use_ecf=True):
        super().__init__()
        mods = tuple(m for m in ALL_MODALITIES if m in modalities)
        if not mods:
            raise ValueError("至少选择一个模态")
        self.modalities = mods
        self.n_classes = n_classes
        self.w_local = cfg.W_LOCAL
        self.gamma = cfg.GAMMA
        self.use_ecf = use_ecf and ("ele" in mods) and len(mods) > 1

        for m in mods:
            setattr(self, f"enc_{m}", _make_encoder(m, d_model))
            setattr(self, f"gcn_{m}", ResidualGCN(d_model, cfg.GCN_LAYERS, cfg.GCN_DROPOUT))
            setattr(self, f"readout_{m}", AttentionReadout(d_model))
            setattr(self, f"head_{m}", nn.Sequential(
                nn.Linear(d_model, hidden_dim), nn.ReLU(inplace=True),
                nn.Dropout(cfg.DROPOUT), nn.Linear(hidden_dim, n_classes)))

        self.ecf = ECFModule(d_model, cfg.ACF_HEADS, cfg.ACF_DROPOUT) if self.use_ecf else None
        self.rgcn = RGCN(d_model, n_relations=3, n_layers=cfg.RGCN_LAYERS,
                         dropout=cfg.GCN_DROPOUT)
        self.readout_multi = AttentionReadout(d_model)
        self.head_multi = nn.Sequential(
            nn.Linear(d_model, hidden_dim * 2), nn.ReLU(inplace=True),
            nn.Dropout(cfg.DROPOUT),
            nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(inplace=True),
            nn.Dropout(cfg.DROPOUT), nn.Linear(hidden_dim, n_classes))
        self.aeds = AEDSFusion(len(mods), n_classes, mode=fusion_mode, eta=cfg.ETA) \
            if len(mods) > 1 else None
        self._cache = {}

    def _adjs(self, n_nodes, device):
        key = (n_nodes, device, len(self.modalities))
        if key not in self._cache:
            A_loc = build_local_adj(n_nodes, self.w_local).to(device)
            A_loc = normalize_adj(add_self_loops([A_loc]))[0]
            A_glo = build_global_adj(n_nodes, self.gamma, n_mods=len(self.modalities),
                                     w_local=self.w_local)
            A_glo = normalize_adj([a.to(device) for a in A_glo])
            self._cache[key] = (A_loc, A_glo)
        return self._cache[key]

    def forward(self, cur, vib, aco, aco_mask=None, return_q=False):
        inputs = {"ele": cur, "vib": vib, "aco": aco}
        N = cur.shape[1]
        B = cur.shape[0]
        device = cur.device

        H = {m: getattr(self, f"enc_{m}")(inputs[m]) for m in self.modalities}

        q_dict = {}
        if self.use_ecf:
            ele_pooled = H["ele"].mean(dim=1)
            for m in self.modalities:
                if m == "ele":
                    continue
                cross = self.ecf.cross_vib if m == "vib" else self.ecf.cross_aco
                gate = self.ecf.gate_vib if m == "vib" else self.ecf.gate_aco
                guided = cross(H["ele"], H[m])
                H[m], q = gate(H[m], guided, ele_pooled, return_q=True)
                if m == "aco" and aco_mask is not None:
                    H[m] = H[m] * aco_mask.view(-1, 1, 1)
                q_dict[f"q_{m}"] = q

        A_loc, A_glo = self._adjs(N, device)
        H_loc = {m: getattr(self, f"gcn_{m}")(H[m], A_loc) for m in self.modalities}

        H_glo = torch.cat([H_loc[m] for m in self.modalities], dim=1)
        A_batch = [a.unsqueeze(0).expand(B, -1, -1) for a in A_glo]
        H_glo = self.rgcn(H_glo, A_batch)

        f = {m: getattr(self, f"readout_{m}")(H_loc[m]) for m in self.modalities}
        f_multi = self.readout_multi(H_glo)

        logits = {m: getattr(self, f"head_{m}")(f[m]) for m in self.modalities}
        logit_multi = self.head_multi(f_multi)

        probs = {m: F.softmax(logits[m], dim=-1) for m in self.modalities}
        if len(self.modalities) == 1:
            only = self.modalities[0]
            M_final = probs[only]
            alpha_list = [torch.ones(B, device=device)]
        else:
            M_final, alpha_list = self.aeds([probs[m] for m in self.modalities])

        out = {
            "M_final": M_final,
            "logit_multi": logit_multi,
            "alpha": alpha_list,
            "probs": probs,
        }
        for m in self.modalities:
            out[f"logit_{m}"] = logits[m]
            out[f"P_{m}"] = probs[m]
        if return_q:
            out["q"] = q_dict
        return out
