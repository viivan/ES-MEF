import torch
import torch.nn as nn
import torch.nn.functional as F

import config as cfg
from models.extractors import SensorLSTM, MultiScale1DCNN, Acoustic1DCNN, WindowEncoder
from models.graph import (
    ResidualGCN, RGCN, MultiModalReadout,
    build_local_adj, build_global_adj,
    add_self_loops, normalize_adj,
)
from models.ecf import ECFModule
from models.fusion import AEDSFusion, HierarchicalClassifiers


class ESMEF(nn.Module):

    def __init__(
        self,
        d_model=cfg.D_MODEL,
        n_classes=cfg.NUM_CLASSES,
        w_local=cfg.W_LOCAL,
        gcn_layers=cfg.GCN_LAYERS,
        rgcn_layers=cfg.RGCN_LAYERS,
        gamma=cfg.GAMMA,
        gcn_dropout=cfg.GCN_DROPOUT,
        acf_heads=cfg.ACF_HEADS,
        acf_dropout=cfg.ACF_DROPOUT,
        hidden_dim=cfg.HIDDEN_DIM,
        n_mods=3,
        cur_channels=3,
        fusion_mode="sa_ds",
        use_ecf=True,
        use_local_graph=True,
        use_global_graph=True,
        use_intra_temporal=True,
        use_sync=True,
        use_delay=True,
        active_modality=None,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_classes = n_classes
        self.n_nodes = cfg.NODE_WINDOW
        self.w_local = w_local
        self.gamma = gamma
        self.n_mods = n_mods
        self.use_ecf = use_ecf
        self.use_local_graph = use_local_graph
        self.use_global_graph = use_global_graph
        self.use_intra_temporal = use_intra_temporal
        self.use_sync = use_sync
        self.use_delay = use_delay
        if active_modality not in (None, "ele", "vib", "aco"):
            raise ValueError("active_modality must be None, 'ele', 'vib', or 'aco'")
        self.active_modality = active_modality

        self.enc_ele = WindowEncoder(
            SensorLSTM(in_channels=cur_channels, hidden=cfg.LSTM_HIDDEN,
                       layers=cfg.LSTM_LAYERS, out_dim=d_model),
            out_dim=d_model,
        )
        self.enc_vib = WindowEncoder(
            MultiScale1DCNN(in_channels=4, base_ch=cfg.CNN_CHANNELS[0],
                            kernels=cfg.CNN_KERNELS, out_dim=d_model),
            out_dim=d_model,
        )
        self.enc_aco = WindowEncoder(
            Acoustic1DCNN(in_channels=1, base_ch=cfg.CNN_CHANNELS[0],
                          out_dim=d_model),
            out_dim=d_model,
        )

        self.ecf = ECFModule(d_model=d_model, n_heads=acf_heads, dropout=acf_dropout)

        self.gcn_ele = ResidualGCN(d_model, gcn_layers, gcn_dropout)
        self.gcn_vib = ResidualGCN(d_model, gcn_layers, gcn_dropout)
        self.gcn_aco = ResidualGCN(d_model, gcn_layers, gcn_dropout)

        self.rgcn = RGCN(d_model, n_relations=3, n_layers=rgcn_layers, dropout=gcn_dropout)

        self.readout = MultiModalReadout(d_model, n_mods=n_mods)

        self.classifiers = HierarchicalClassifiers(d_model, hidden_dim, n_classes)

        self.aeds = AEDSFusion(n_branches=3, n_classes=n_classes, mode=fusion_mode, eta=cfg.ETA)

        self._cache = {}

    def _get_adjs(self, n_nodes, device):
        key = (
            n_nodes, device, self.w_local, self.use_intra_temporal,
            self.use_sync, self.use_delay,
        )
        if key not in self._cache:
            A_loc = build_local_adj(n_nodes, self.w_local).to(device)
            A_loc_norm = normalize_adj(add_self_loops([A_loc]))[0]

            A_glo_list = build_global_adj(
                n_nodes,
                self.gamma,
                n_mods=self.n_mods,
                w_local=self.w_local,
                use_intra_temporal=self.use_intra_temporal,
                use_sync=self.use_sync,
                use_delay=self.use_delay,
            )
            A_glo_list = [A.to(device) for A in A_glo_list]
            A_glo_norm = normalize_adj(A_glo_list)

            self._cache[key] = (A_loc_norm, A_glo_norm)
        return self._cache[key]

    def forward(self, cur_wins, vib_wins, aco_wins, aco_mask=None, return_q=False):
        B, N = cur_wins.shape[:2]
        device = cur_wins.device

        H_ele = self.enc_ele(cur_wins)
        H_vib = self.enc_vib(vib_wins)
        H_aco = self.enc_aco(aco_wins)

        if self.active_modality is not None:
            H_vib_denoised = H_vib
            H_aco_denoised = H_aco
            q_dict = {"q_vib": torch.zeros(B, device=device),
                      "q_aco": torch.zeros(B, device=device)}
        elif self.use_ecf:
            if return_q:
                H_vib_denoised, H_aco_denoised, q_dict = self.ecf(H_ele, H_vib, H_aco, aco_mask, return_q=True)
            else:
                H_vib_denoised, H_aco_denoised = self.ecf(H_ele, H_vib, H_aco, aco_mask)
        else:
            H_vib_denoised, H_aco_denoised = H_vib, H_aco
            if return_q:
                q_dict = {"q_vib": torch.zeros(B, device=device),
                          "q_aco": torch.zeros(B, device=device)}

        A_loc_norm, A_glo_norm = self._get_adjs(N, device)
        if self.use_local_graph:
            H_ele_local = self.gcn_ele(H_ele, A_loc_norm)
            H_vib_local = self.gcn_vib(H_vib_denoised, A_loc_norm)
            H_aco_local = self.gcn_aco(H_aco_denoised, A_loc_norm)
        else:
            H_ele_local = H_ele
            H_vib_local = H_vib_denoised
            H_aco_local = H_aco_denoised

        H_glo = torch.cat([H_ele_local, H_vib_local, H_aco_local], dim=1)
        if self.active_modality is not None:
            active_idx = {"ele": 0, "vib": 1, "aco": 2}[self.active_modality]
            f_multi = [H_ele_local, H_vib_local, H_aco_local][active_idx].mean(dim=1)
        else:
            if self.use_global_graph:
                A_glo_batch = [A.unsqueeze(0).expand(B, -1, -1) for A in A_glo_norm]
                H_glo = self.rgcn(H_glo, A_glo_batch)

            _, _, _, f_multi = self.readout(
                H_ele_local, H_vib_local, H_aco_local, H_glo,
            )

        f_ele, f_vib, f_aco, _ = self.readout(
            H_ele_local, H_vib_local, H_aco_local, H_glo,
        )

        logit_ele, logit_vib, logit_aco, logit_multi = \
            self.classifiers(f_ele, f_vib, f_aco, f_multi)

        P_ele = F.softmax(logit_ele, dim=-1)
        P_vib = F.softmax(logit_vib, dim=-1)
        P_aco = F.softmax(logit_aco, dim=-1)
        if self.active_modality is None:
            M_final, alpha_list = self.aeds([P_ele, P_vib, P_aco])
        else:
            active_prob = {"ele": P_ele, "vib": P_vib, "aco": P_aco}[self.active_modality]
            M_final = active_prob
            alpha_list = [torch.ones(B, device=device)]

        out = {
            "logit_ele": logit_ele, "logit_vib": logit_vib,
            "logit_aco": logit_aco, "logit_multi": logit_multi,
            "P_ele": P_ele, "P_vib": P_vib, "P_aco": P_aco,
            "P_multi": F.softmax(logit_multi, dim=-1),
            "M_final": M_final, "alpha": alpha_list,
            "f_ele": f_ele, "f_vib": f_vib, "f_aco": f_aco,
            "f_multi": f_multi,
        }
        if return_q:
            out["q"] = q_dict
        return out
