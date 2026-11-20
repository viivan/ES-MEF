
import argparse
import copy
import json
import os
import random
import sys
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.special import digamma, gammaln

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

import config as cfg
from data_loader_multimodal import get_multimodal_dataloaders_temporal
from models.extractors import SensorLSTM, MultiScale1DCNN, Acoustic1DCNN, WindowEncoder
from es_mef import ESMEF

PROJECT_ROOT = os.environ.get("ESMEF_ROOT", ".")
SAVE_DIR = os.path.join(PROJECT_ROOT, "results", "baselines_all")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

N_CLASSES = 3
SEEDS = [42, 123, 456]
LR = 5e-4
WEIGHT_DECAY = 1e-2
BATCH_SIZE = 16
MAX_EPOCHS = 60
PATIENCE = 10
EPS = 1e-8


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def class_weights(train_loader):
    counts = Counter()
    for _, _, _, _, labels in train_loader:
        counts.update(labels.tolist())
    total = sum(counts.values())
    w = torch.tensor([total / (N_CLASSES * counts.get(i, 1)) for i in range(N_CLASSES)],
                     dtype=torch.float32)
    return (w / (w.min() + 1e-7)).clamp(max=5.0).to(DEVICE)


@torch.no_grad()
def collect_probs(model, loader, forward_fn):
    model.eval()
    probs, labels = [], []
    for batch in loader:
        out = forward_fn(model, batch)
        if isinstance(out, dict):
            out = out["probability"]
        probs.append(F.softmax(out, dim=-1).cpu().numpy())
        labels.append(batch[-1].numpy())
    return np.concatenate(probs), np.concatenate(labels)


def report(probs, labels):
    preds = probs.argmax(axis=-1)
    return {
        "acc": float(accuracy_score(labels, preds)),
        "macro_f1": float(f1_score(labels, preds, average="macro", zero_division=0)),
        "per_class_f1": f1_score(labels, preds, labels=list(range(N_CLASSES)),
                                 average=None, zero_division=0).tolist(),
        "confusion_matrix": confusion_matrix(labels, preds,
                                             labels=list(range(N_CLASSES))).tolist(),
    }


def train_loop(model, train_loader, val_loader, forward_fn, loss_fn,
               lr=LR, wd=WEIGHT_DECAY, epochs=MAX_EPOCHS, patience=PATIENCE,
               clip=5.0, select="loss"):

    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=8, min_lr=1e-6)
    best = float("inf")
    best_state = None
    bad = 0

    for epoch in range(1, epochs + 1):
        model.train()
        for batch in train_loader:
            opt.zero_grad()
            out = forward_fn(model, batch)
            loss = loss_fn(out, batch[-1].to(DEVICE), epoch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            opt.step()

        model.eval()
        val_loss, ys, ps = 0.0, [], []
        with torch.no_grad():
            for batch in val_loader:
                out = forward_fn(model, batch)
                val_loss += F.cross_entropy(out, batch[-1].to(DEVICE)).item() * batch[-1].size(0)
                ps.extend(out.argmax(-1).cpu().tolist())
                ys.extend(batch[-1].tolist())
        val_loss /= max(1, len(val_loader))
        sched.step(val_loss)
        score = val_loss if select == "loss" else -f1_score(ys, ps, average="macro", zero_division=0)
        if score < best:
            best, bad = score, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad += 1
        if bad >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model


def ce_loss(out, labels, epoch=None):
    return F.cross_entropy(out, labels)


# 1. CFCNN [52]
class MultiscaleShrinkageEncoder(nn.Module):


    def __init__(self, channels, d_model=32):
        super().__init__()
        self.conv_small = nn.Conv1d(channels, 16, kernel_size=3, padding=1)
        self.conv_large = nn.Conv1d(channels, 16, kernel_size=7, padding=3)
        self.tau_small = nn.Parameter(torch.tensor(0.05))
        self.tau_large = nn.Parameter(torch.tensor(0.05))
        self.attn = nn.Sequential(nn.Linear(32, 8), nn.GELU(), nn.Linear(8, 2))
        self.proj = nn.Linear(16, d_model)

    @staticmethod
    def soft_shrink(x, tau):
        return torch.sign(x) * torch.clamp(torch.abs(x) - tau.abs(), min=0.0)

    def forward(self, x):
        a = F.adaptive_avg_pool1d(self.conv_small(x), 32)
        b = F.adaptive_avg_pool1d(self.conv_large(x), 32)
        a = self.soft_shrink(a, self.tau_small)
        b = self.soft_shrink(b, self.tau_large)
        fused = torch.cat([a, b], dim=1)
        w = torch.softmax(self.attn(fused.mean(dim=-1)), dim=-1)
        fused = w[:, 0:1].unsqueeze(-1) * a + w[:, 1:2].unsqueeze(-1) * b
        return self.proj(fused.mean(dim=-1))


class CFM(nn.Module):


    def __init__(self, d_model=32, n_mods=3, n_classes=N_CLASSES):
        super().__init__()
        self.tokens = nn.Linear(d_model, d_model)
        self.encoder = nn.TransformerEncoderLayer(d_model=d_model, nhead=4,
                                                  dim_feedforward=64, dropout=0.1,
                                                  batch_first=True)
        self.gate = nn.Sequential(nn.Linear(n_mods * d_model, d_model),
                                  nn.GELU(), nn.Dropout(0.1))
        self.decision = nn.Linear(d_model, n_classes)

    def forward(self, feats):
        tokens = torch.stack([self.tokens(f) for f in feats], dim=1)
        att = self.encoder(tokens)
        flat = att.flatten(start_dim=1)
        fused = self.gate(flat) * torch.sigmoid(self.gate(flat))
        return self.decision(fused)


class CFCNN(nn.Module):
    def __init__(self, d_model=32):
        super().__init__()
        self.enc_cur = MultiscaleShrinkageEncoder(channels=3, d_model=d_model)
        self.enc_vib = MultiscaleShrinkageEncoder(channels=4, d_model=d_model)
        self.enc_aco = MultiscaleShrinkageEncoder(channels=1, d_model=d_model)
        self.cfm = CFM(d_model=d_model, n_mods=3, n_classes=N_CLASSES)

    @staticmethod
    def _per_node(x):
        B, N = x.shape[0], x.shape[1]
        return x.reshape(B * N, x.shape[2], x.shape[3])

    def forward(self, cur, vib, aco, mask=None):
        B = cur.shape[0]
        f_cur = self.enc_cur(self._per_node(cur)).reshape(B, -1, 32).mean(dim=1)
        f_vib = self.enc_vib(self._per_node(vib)).reshape(B, -1, 32).mean(dim=1)
        aco_d = aco[:, :, :, ::10]
        f_aco = self.enc_aco(self._per_node(aco_d)).reshape(B, -1, 32).mean(dim=1)
        return self.cfm([f_cur, f_vib, f_aco])


def fwd_cfcnn(model, batch):
    cur, vib, aco, mask, _ = batch
    return model(cur.to(DEVICE), vib.to(DEVICE), aco.to(DEVICE), mask.to(DEVICE))


def loss_cfcnn(out, labels, epoch):
    smoothing = 0.10 * (1.0 - (epoch - 1) / MAX_EPOCHS) + 0.01
    return F.cross_entropy(out, labels, label_smoothing=smoothing)


def run_cfcnn(train_loader, val_loader, test_loader, seed):
    model = CFCNN().to(DEVICE)
    model = train_loop(model, train_loader, val_loader, fwd_cfcnn, loss_cfcnn,
                       lr=LR, wd=WEIGHT_DECAY, epochs=MAX_EPOCHS, patience=PATIENCE)
    probs, labels = collect_probs(model, test_loader, fwd_cfcnn)
    return report(probs, labels)


# 2. CNN-LSTM-Concat [54]
class Electrical3Class(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.lstm = nn.LSTM(3, hidden, 1, batch_first=True)
        self.cls = nn.Linear(hidden, N_CLASSES)

    def extract(self, x):
        B, N, C, T = x.shape
        x = x.reshape(B * N, C, T).permute(0, 2, 1)
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return out.reshape(B, N, -1).mean(dim=1)

    def forward(self, x):
        return self.cls(self.extract(x))


class Vibration3Class(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.lstm = nn.LSTM(4, hidden, 1, batch_first=True)
        self.cls = nn.Linear(hidden, N_CLASSES)

    def extract(self, x):
        B, N, C, T = x.shape
        x = x.reshape(B * N, C, T).permute(0, 2, 1)
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return out.reshape(B, N, -1).mean(dim=1)

    def forward(self, x):
        return self.cls(self.extract(x))


class Acoustic3Class(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, 7, padding=3), nn.ReLU(), nn.MaxPool1d(4),
            nn.Conv1d(32, 64, 5, padding=2), nn.ReLU(), nn.MaxPool1d(4),
            nn.Conv1d(64, 128, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool1d(1),
        )
        self.cls = nn.Linear(128, N_CLASSES)

    def extract(self, x):
        B, N, C, T = x.shape
        x = x.reshape(B * N, C, T)
        x = self.conv(x).squeeze(-1)
        return x.reshape(B, N, -1).mean(dim=1)

    def forward(self, x):
        return self.cls(self.extract(x))


class LinearClassifier(nn.Module):
    def __init__(self, in_dim=256, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm1d(in_dim),
            nn.Linear(in_dim, hidden), nn.ReLU(inplace=True), nn.Dropout(0.3),
            nn.Linear(hidden, N_CLASSES),
        )

    def forward(self, x):
        return self.net(x)


def run_concat(train_loader, val_loader, test_loader, seed):

    mods = [("cur", Electrical3Class(), 0), ("vib", Vibration3Class(), 1),
            ("aco", Acoustic3Class(), 2)]
    fused = LinearClassifier()

    for name, net, idx in mods:
        net = net.to(DEVICE)
        def fwd(m, batch, idx=idx):
            return m(batch[idx].to(DEVICE))
        net = train_loop(net, train_loader, val_loader, fwd, ce_loss, select="f1")
        setattr(fused, f"feat_{name}", net)

    def fwd_fused(m, batch):
        feats = [getattr(m, f"feat_{n}").extract(batch[i].to(DEVICE))
                 for n, _, i in mods]
        return m(torch.cat(feats, dim=-1))

    fused = fused.to(DEVICE)
    fused = train_loop(fused, train_loader, val_loader, fwd_fused, ce_loss, select="f1")
    probs, labels = collect_probs(fused, test_loader, fwd_fused)
    return report(probs, labels)


# 3. AMTFNet [56]
class MultiscaleDepthwiseStem(nn.Module):


    def __init__(self, in_ch, d_model=64, pool_len=16):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, 32, 7, stride=2, padding=3), nn.ReLU(),
            nn.Conv1d(32, 64, 5, stride=2, padding=2), nn.ReLU(),
            nn.AdaptiveAvgPool1d(pool_len),
        )
        self.dw = nn.ModuleList([
            nn.Conv1d(64, 64, k, padding=k // 2, groups=64) for k in (3, 5, 7)
        ])
        self.point = nn.Conv1d(64, 64, 1)
        self.inst_norm = nn.InstanceNorm1d(64 * 3, affine=True)
        self.mix = nn.Linear(64 * 3, d_model)

    def forward(self, x):
        z = self.conv(x)
        scales = [F.relu(self.point(F.relu(dw(z)))) for dw in self.dw]
        feat = torch.cat(scales, dim=1)
        feat = self.inst_norm(feat)
        feat = torch.amax(feat, dim=-1)
        return self.mix(feat)


class ModalityEncoder(nn.Module):
    def __init__(self, in_ch, d_model=64, gru_hidden=64):
        super().__init__()
        self.stem = MultiscaleDepthwiseStem(in_ch, d_model)
        self.gru = nn.GRU(d_model, gru_hidden, batch_first=True)
        self.out_dim = gru_hidden

    def forward(self, x):
        B, N = x.shape[0], x.shape[1]
        h = self.stem(x.reshape(-1, *x.shape[2:]))
        h = h.reshape(B, N, -1)
        h, _ = self.gru(h)
        return h


class TemporalAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.w = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, 1, bias=False)

    def forward(self, h):
        scores = self.v(torch.tanh(self.w(h))).squeeze(-1)
        attn = torch.softmax(scores, dim=-1)
        return (h * attn.unsqueeze(-1)).sum(dim=1), attn


class AMTFNet(nn.Module):
    def __init__(self, chans=(3, 4, 1), d_model=64, hidden=64):
        super().__init__()
        self.encoders = nn.ModuleList([ModalityEncoder(c, d_model, hidden) for c in chans])
        self.temporal_attn = TemporalAttention(hidden)
        fusion_dim = len(chans) * hidden
        self.cls = nn.Sequential(
            nn.Linear(fusion_dim, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, N_CLASSES),
        )

    def forward(self, cur, vib, aco, mask=None):
        feats = []
        for enc, x in zip(self.encoders, (cur, vib, aco)):
            h = enc(x)
            v, _ = self.temporal_attn(h)
            feats.append(v)
        return self.cls(torch.cat(feats, dim=-1))


def fwd_amtfnet(model, batch):
    cur, vib, aco, mask, _ = batch
    return model(cur.to(DEVICE), vib.to(DEVICE), aco.to(DEVICE))


def run_amtfnet(train_loader, val_loader, test_loader, seed):
    model = AMTFNet(chans=(3, 4, 1)).to(DEVICE)
    model = train_loop(model, train_loader, val_loader, fwd_amtfnet, ce_loss)
    probs, labels = collect_probs(model, test_loader, fwd_amtfnet)
    return report(probs, labels)


# 4. DEWVS
def stat_features(x):

    B = x.shape[0]
    h = x.reshape(B, -1, x.shape[-1]).float()
    mean = h.mean(-1)
    std = h.std(-1)
    rms = (h ** 2).mean(-1).sqrt()
    centered = h - mean.unsqueeze(-1)
    skew = (centered ** 3).mean(-1) / (std ** 3 + 1e-8)
    kurt = (centered ** 4).mean(-1) / (std ** 4 + 1e-8)
    return torch.cat([mean, std, rms, skew, kurt], dim=-1)


class StatModel(nn.Module):
    def __init__(self, in_dim, n_classes=N_CLASSES):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, 32), nn.GELU(),
                                 nn.Linear(32, n_classes))

    def forward(self, feats):
        return self.net(feats)


class DeepModel(nn.Module):
    def __init__(self, channels, n_classes=N_CLASSES):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(channels, 16, 3, padding=1), nn.GELU(),
            nn.Conv1d(16, 16, 5, padding=2), nn.GELU(),
            nn.AdaptiveAvgPool1d(1))
        self.head = nn.Linear(16, n_classes)

    def forward(self, x):
        B = x.shape[0]
        h = x.reshape(B, -1, x.shape[-1]).float()
        return self.head(self.net(h).squeeze(-1))


def run_dewvs(train_loader, val_loader, test_loader, seed):

    cur0, vib0, aco0, _, _ = next(iter(train_loader))
    dims = {"cur": stat_features(cur0).shape[-1],
            "vib": stat_features(vib0).shape[-1],
            "aco": stat_features(aco0).shape[-1]}
    deep_ch = {"cur": 24, "vib": 32, "aco": 8}

    specs = [("cur_stat", "stat", 0), ("cur_deep", "deep", 0),
             ("vib_stat", "stat", 1), ("vib_deep", "deep", 1),
             ("aco_stat", "stat", 2), ("aco_deep", "deep", 2)]

    def make_input(kind, idx, batch):
        x = batch[idx].to(DEVICE)
        if kind == "stat":
            return stat_features(x)
        if idx == 2:
            x = x[:, :, :, ::10]
        return x

    models, val_probs = [], []
    for name, kind, idx in specs:
        mod = StatModel(dims[["cur", "vib", "aco"][idx]]) if kind == "stat" \
            else DeepModel(deep_ch[["cur", "vib", "aco"][idx]])
        mod = mod.to(DEVICE)
        fwd = lambda m, b, k=kind, i=idx: m(make_input(k, i, b))
        mod = train_loop(mod, train_loader, val_loader, fwd, ce_loss, select="f1")
        p, _ = collect_probs(mod, val_loader, fwd)
        models.append((name, mod, fwd))
        val_probs.append(p)

    _, val_Y = collect_probs(models[0][1], val_loader, models[0][2])
    Mx = np.zeros((len(models), N_CLASSES))
    for i, P in enumerate(val_probs):
        pred = P.argmax(-1)
        for c in range(N_CLASSES):
            idx_c = val_Y == c
            Mx[i, c] = (pred[idx_c] == c).mean() if idx_c.sum() else 0.0
    Mx = np.clip(Mx, 1e-3, None)

    test_P = np.stack([collect_probs(m, test_loader, f)[0] for _, m, f in models])
    _, test_Y = collect_probs(models[0][1], test_loader, models[0][2])
    W = np.einsum("imc,ic->im", test_P, Mx)
    W = W / (W.sum(axis=0, keepdims=True) + 1e-12)
    votes = test_P.argmax(-1)
    n = test_P.shape[1]
    scores = np.zeros((n, N_CLASSES))
    for i in range(len(models)):
        for c in range(N_CLASSES):
            scores[:, c] += W[i, :] * (votes[i] == c)
    prob_mass = np.einsum("im,imc->mc", W, test_P)
    final = scores + 1e-6 * prob_mass / (prob_mass.max() + 1e-12)
    preds = final.argmax(-1)

    probs = np.zeros((n, N_CLASSES))
    probs[np.arange(n), preds] = 1.0
    return report(probs, test_Y)


# 5. MSCAF [59]
class MSCAFDirichletHead(nn.Module):
    def __init__(self, dim, n_classes, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim, n_classes),
        )

    def forward(self, features):
        evidence = F.softplus(self.net(features)) + 1e-6
        alpha = evidence + 1.0
        strength = alpha.sum(dim=-1, keepdim=True)
        return evidence, alpha, alpha / strength


def _js_divergence(p, q):
    mean = 0.5 * (p + q)
    return 0.5 * (
        (p * (p.clamp_min(EPS).log() - mean.clamp_min(EPS).log())).sum(-1)
        + (q * (q.clamp_min(EPS).log() - mean.clamp_min(EPS).log())).sum(-1)
    )


def evidence_to_mass(alpha):
    classes = alpha.size(-1)
    strength = alpha.sum(dim=-1, keepdim=True).clamp_min(EPS)
    belief = alpha / strength
    uncertainty = classes / strength
    return belief * (1.0 - uncertainty), uncertainty.squeeze(-1)


def adaptive_source_weights(probabilities, uncertainties, mask=None):
    agreements = []
    for i in range(3):
        source_agreement = []
        for j in range(3):
            if i != j:
                source_agreement.append(1.0 - _js_divergence(probabilities[i], probabilities[j]))
        agreements.append(torch.stack(source_agreement, dim=0).mean(dim=0))
    agreement = torch.stack(agreements, dim=1).clamp(0.0, 1.0)
    quality = (1.0 - uncertainties).clamp_min(0.0)
    raw = quality * torch.exp(agreement - uncertainties)
    if mask is not None:
        raw[:, 2] = raw[:, 2] * mask
    weights = raw / raw.sum(dim=1, keepdim=True).clamp_min(EPS)
    return weights, agreement


def ds_combine(beliefs, uncertainties, weights):
    mass = beliefs[0] * weights[:, 0:1]
    unknown = 1.0 - mass.sum(dim=-1, keepdim=True)
    for i in range(1, len(beliefs)):
        next_mass = beliefs[i] * weights[:, i:i + 1]
        next_unknown = 1.0 - next_mass.sum(dim=-1, keepdim=True)
        conflict = (mass.unsqueeze(-1) * next_mass.unsqueeze(-2)).sum(dim=(-1, -2))
        denominator = (1.0 - conflict).clamp_min(1e-6)
        combined = (
            mass * next_mass.sum(dim=-1, keepdim=True)
            + next_mass * mass.sum(dim=-1, keepdim=True)
            + mass * next_mass
        ) / denominator.unsqueeze(-1)
        combined_unknown = unknown * next_unknown / denominator.unsqueeze(-1)
        mass = combined
        unknown = combined_unknown
    total = mass.sum(dim=-1, keepdim=True) + unknown
    mass = mass / total.clamp_min(EPS)
    unknown = (unknown / total.clamp_min(EPS)).squeeze(-1)
    probability = mass / (1.0 - unknown.unsqueeze(-1)).clamp_min(EPS)
    return probability, unknown


class MSCAFModel(nn.Module):
    def __init__(self, d_model=128, n_classes=N_CLASSES, dropout=0.2):
        super().__init__()
        self.n_classes = n_classes
        self.elec_encoder = WindowEncoder(
            SensorLSTM(3, hidden=64, layers=1, out_dim=d_model, dropout=dropout), d_model)
        self.vib_encoder = WindowEncoder(
            MultiScale1DCNN(4, base_ch=16, out_dim=d_model, dropout=dropout), d_model)
        self.aco_encoder = WindowEncoder(
            Acoustic1DCNN(1, base_ch=16, out_dim=d_model, dropout=dropout), d_model)
        self.heads = nn.ModuleList([
            MSCAFDirichletHead(d_model, n_classes, dropout) for _ in range(3)
        ])

    def forward(self, cur, vib, aco, mask=None):
        features = [
            self.elec_encoder(cur).mean(dim=1),
            self.vib_encoder(vib).mean(dim=1),
            self.aco_encoder(aco).mean(dim=1),
        ]
        outputs = [head(f) for head, f in zip(self.heads, features)]
        evidences = [o[0] for o in outputs]
        alphas = [o[1] for o in outputs]
        probabilities = [o[2] for o in outputs]
        uncertainties = torch.stack([evidence_to_mass(a)[1] for a in alphas], dim=1)
        weights, agreement = adaptive_source_weights(probabilities, uncertainties, mask)
        beliefs = [evidence_to_mass(a)[0] for a in alphas]
        final_probability, final_uncertainty = ds_combine(beliefs, uncertainties, weights)
        return {
            "evidence": evidences, "alpha": alphas,
            "source_probabilities": probabilities, "beliefs": beliefs,
            "uncertainty": uncertainties, "source_weights": weights,
            "agreement": agreement,
            "probability": final_probability,
            "uncertainty_final": final_uncertainty,
        }


def dirichlet_kl(alpha):
    classes = alpha.size(-1)
    prior = torch.ones_like(alpha)
    alpha_sum = alpha.sum(dim=-1)
    prior_sum = prior.sum(dim=-1)
    return (
        gammaln(alpha_sum) - gammaln(prior_sum)
        - (gammaln(alpha) - gammaln(prior)).sum(dim=-1)
        + ((alpha - prior) * (digamma(alpha) - digamma(alpha_sum.unsqueeze(-1)))).sum(dim=-1)
    )


def source_loss(alpha, labels, class_weight=None, kl_weight=1e-3):
    strength = alpha.sum(dim=-1)
    target_alpha = alpha.gather(1, labels.unsqueeze(1)).squeeze(1)
    nll = digamma(strength) - digamma(target_alpha)
    kl = dirichlet_kl(alpha)
    if class_weight is not None:
        sample_weight = class_weight[labels]
        nll = (nll * sample_weight).sum() / sample_weight.sum().clamp_min(EPS)
        kl = (kl * sample_weight).sum() / sample_weight.sum().clamp_min(EPS)
    else:
        nll = nll.mean()
        kl = kl.mean()
    return nll + kl_weight * kl


def mscaf_loss(output, labels, class_weight=None, kl_weight=1e-3, fusion_weight=1.0):
    source_total = sum(source_loss(a, labels, class_weight, kl_weight)
                       for a in output["alpha"])
    fused_nll = F.nll_loss(output["probability"].clamp_min(EPS).log(), labels,
                           weight=class_weight)
    return source_total + fusion_weight * fused_nll


def fwd_mscaf(model, batch):
    cur, vib, aco, mask, _ = batch
    return model(cur.to(DEVICE), vib.to(DEVICE), aco.to(DEVICE), mask.to(DEVICE))


def run_mscaf(train_loader, val_loader, test_loader, seed):
    model = MSCAFModel(d_model=128, n_classes=N_CLASSES).to(DEVICE)
    weight = class_weights(train_loader)
    opt = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=8, min_lr=1e-6)
    best, best_state, bad = float("inf"), None, 0

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        for batch in train_loader:
            opt.zero_grad()
            out = fwd_mscaf(model, batch)
            loss = mscaf_loss(out, batch[-1].to(DEVICE), weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                out = fwd_mscaf(model, batch)
                val_loss += mscaf_loss(out, batch[-1].to(DEVICE), weight).item() * batch[-1].size(0)
        val_loss /= max(1, len(val_loader))
        sched.step(val_loss)
        if val_loss < best:
            best, bad, best_state = val_loss, 0, copy.deepcopy(model.state_dict())
        else:
            bad += 1
        if bad >= PATIENCE:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    probs, labels = collect_probs(model, test_loader, fwd_mscaf)
    return report(probs, labels)


# 6. MMDG [58]
MMDG_DIM = 256
MMDG_EPOCHS = 50
MMDG_LR = 1e-3
MMDG_BATCH = 32
MMDG_LAMBDA_MMD = 0.5
MMDG_LAMBDA_COV = 0.5


def conv3x1(inp, oup, stride=1):
    return nn.Conv1d(inp, oup, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(inp, oup, stride=1):
    return nn.Conv1d(inp, oup, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv3x1(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x1(planes, planes)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


class ResNet1D(nn.Module):


    def __init__(self, in_channel=3):
        super().__init__()
        self.inplanes = 16
        self.pre_layer = nn.Sequential(
            nn.Conv1d(in_channel, self.inplanes, kernel_size=64, stride=8,
                      padding=32, bias=False),
            nn.BatchNorm1d(self.inplanes),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=5, stride=3, padding=2),
        )
        self.layer1 = self._make_layer(BasicBlock, 32, 1)
        self.layer2 = self._make_layer(BasicBlock, 64, 1, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 128, 1, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 256, 1, stride=2)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes, stride),
                nn.BatchNorm1d(planes),
            )
        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.pre_layer(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


class MultiModal1D(nn.Module):


    def __init__(self, chans=(3, 4, 1), dim=MMDG_DIM):
        super().__init__()
        self.encs = nn.ModuleList([ResNet1D(c) for c in chans])
        self.heads = nn.ModuleList([
            nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(),
                          nn.Linear(dim, dim), nn.ReLU())
            for _ in range(3)
        ])

    def forward(self, xs):
        return [head(enc(x)) for enc, head, x in zip(self.encs, self.heads, xs)]


class CrossAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)

    def forward(self, x_q, x_kv):
        x_q = self.norm_q(x_q)
        x_kv = self.norm_kv(x_kv)
        h = self.heads
        q = self.to_q(x_q)
        k, v = self.to_kv(x_kv).chunk(2, dim=-1)
        B, N = k.shape[0], k.shape[1]
        q = q.view(B, N, h, -1).permute(0, 2, 1, 3)
        k = k.view(B, N, h, -1).permute(0, 2, 1, 3)
        v = v.view(B, N, h, -1).permute(0, 2, 1, 3)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.dropout(self.attend(dots))
        out = torch.matmul(attn, v)
        return out.permute(0, 2, 1, 3).reshape(B, N, -1)


class DomainFeatureExtractor(nn.Module):


    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.cam_vib = CrossAttention(input_dim, heads=8, dim_head=32)
        self.cam_cur = CrossAttention(input_dim, heads=8, dim_head=32)
        self.cam_aud = CrossAttention(input_dim, heads=8, dim_head=32)
        self.fuse_trans = nn.Linear(256 * 3, 256, bias=False)

    def forward(self, feat_vib, feat_cur, feat_aud):
        fv = feat_vib.unsqueeze(-2)
        fc = feat_cur.unsqueeze(-2)
        fa = feat_aud.unsqueeze(-2)
        fused_vib_cur = self.cam_vib(fv, fc)
        fused_cur_aud = self.cam_cur(fc, fa)
        fused_aud_vib = self.cam_aud(fa, fv)
        return self.fuse_trans(torch.cat(
            (fused_vib_cur.squeeze(-2), fused_cur_aud.squeeze(-2),
             fused_aud_vib.squeeze(-2)), dim=-1))


def gaussian_kernel(x, y, gamma=1.0):
    x = x.view(x.size(0), -1)
    y = y.view(y.size(0), -1)
    xx = torch.mm(x, x.t())
    yy = torch.mm(y, y.t())
    xy = torch.mm(x, y.t())
    xx = xx.diag().unsqueeze(0).expand_as(xx)
    yy = yy.diag().unsqueeze(0).expand_as(yy)
    return torch.exp(-gamma * (xx.t() + yy - 2 * xy))


def mmd_loss(x, y):
    return (gaussian_kernel(x, x).mean() + gaussian_kernel(y, y).mean()
            - 2 * gaussian_kernel(x, y).mean())


def covariance_loss(x, y):
    x = x - x.mean(dim=0, keepdim=True)
    y = y - y.mean(dim=0, keepdim=True)
    cov = torch.mm(x.t(), y) / (x.size(0) - 1)
    return torch.norm(cov, p=2)


def collect_series(loader):

    xs = [[], [], []]
    ys = []
    for cur, vib, aco, _, y in loader:
        for i in range(cur.shape[0]):
            for m, w in enumerate((cur[i], vib[i], aco[i])):
                xs[m].append(w.permute(1, 0, 2).reshape(w.shape[1], -1))
            ys.append(int(y[i]))
    return [torch.stack(x) for x in xs], np.array(ys, dtype=np.int64)


def run_mmdg(train_loader, val_loader, test_loader, seed):
    x_tr, y_tr = collect_series(train_loader)
    x_va, y_va = collect_series(val_loader)
    x_te, y_te = collect_series(test_loader)

    model = MultiModal1D(chans=tuple(x.shape[1] for x in x_tr)).to(DEVICE)
    domain_extractor = DomainFeatureExtractor(MMDG_DIM, MMDG_DIM).to(DEVICE)
    classifier = nn.Linear(MMDG_DIM, N_CLASSES).to(DEVICE)

    params = list(model.parameters()) + list(domain_extractor.parameters()) + \
        list(classifier.parameters())
    opt = optim.Adam(params, lr=MMDG_LR)
    sched = optim.lr_scheduler.StepLR(opt, step_size=200, gamma=0.1)

    best_acc, best_state, bad = -1.0, None, 0
    n = len(y_tr)

    for epoch in range(1, MMDG_EPOCHS + 1):
        model.train(); domain_extractor.train(); classifier.train()
        perm = np.random.permutation(n)
        for i in range(0, n, MMDG_BATCH):
            idx = perm[i:i + MMDG_BATCH]
            xb = [x_m[idx].float().to(DEVICE) for x_m in x_tr]
            yb = torch.from_numpy(y_tr[idx]).to(DEVICE)

            feas = model(xb)
            halves = []
            for f in feas:
                d = f.size(1) // 2
                halves.append((f[:, :d], f[:, d:]))
            vib_inv, vib_spec = halves[0]
            cur_inv, cur_spec = halves[1]
            aud_inv, aud_spec = halves[2]

            mmd_total = (mmd_loss(vib_inv, cur_inv) + mmd_loss(vib_inv, aud_inv)
                         + mmd_loss(cur_inv, aud_inv))
            orth_total = (covariance_loss(vib_inv, vib_spec)
                          + covariance_loss(cur_inv, cur_spec)
                          + covariance_loss(aud_inv, aud_spec)
                          + covariance_loss(vib_spec, cur_spec)
                          + covariance_loss(vib_spec, aud_spec)
                          + covariance_loss(cur_spec, aud_spec))

            logits = classifier(domain_extractor(*feas))
            loss = (F.cross_entropy(logits, yb)
                    + MMDG_LAMBDA_MMD * mmd_total
                    + MMDG_LAMBDA_COV * orth_total)
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()

        model.eval(); domain_extractor.eval(); classifier.eval()
        preds = []
        with torch.no_grad():
            for i in range(0, len(y_va), MMDG_BATCH):
                xb = [x_m[i:i + MMDG_BATCH].float().to(DEVICE) for x_m in x_va]
                preds.append(classifier(domain_extractor(*model(xb))).cpu().numpy())
        val_acc = accuracy_score(y_va, np.concatenate(preds).argmax(-1))
        if val_acc > best_acc:
            best_acc, bad = val_acc, 0
            best_state = [{k: v.cpu().clone() for k, v in m.state_dict().items()}
                          for m in (model, domain_extractor, classifier)]
        else:
            bad += 1
        if bad >= 15:
            break

    for m, sd in zip((model, domain_extractor, classifier), best_state):
        m.load_state_dict(sd)

    model.eval(); domain_extractor.eval(); classifier.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(y_te), MMDG_BATCH):
            xb = [x_m[i:i + MMDG_BATCH].float().to(DEVICE) for x_m in x_te]
            preds.append(classifier(domain_extractor(*model(xb))).cpu().numpy())
    probs = F.softmax(torch.from_numpy(np.concatenate(preds)), dim=-1).numpy()
    return report(probs, y_te)


# 7. CWDS [21]
CWDS_SCORE_THRESHOLD = 1e-3


class CWDSClassifier(nn.Module):
    def __init__(self, channels, hidden_dim=64, bidirectional=False):
        super().__init__()
        self.lstm = nn.LSTM(channels, hidden_dim, batch_first=True,
                            bidirectional=bidirectional)
        out_dim = hidden_dim * (2 if bidirectional else 1)
        self.classifier = nn.Linear(out_dim, N_CLASSES)

    def forward(self, x):
        b, n, c, t = x.shape
        x = x.reshape(b * n, c, t).transpose(1, 2)
        x, _ = self.lstm(x)
        x = x[:, -1]
        x = x.reshape(b, n, -1).mean(dim=1)
        return self.classifier(x)


class CWDSAcoustic(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 32, 7, padding=3), nn.ReLU(), nn.MaxPool1d(4),
            nn.Conv1d(32, 64, 5, padding=2), nn.ReLU(), nn.MaxPool1d(4),
            nn.Conv1d(64, 128, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Linear(128, N_CLASSES)

    def forward(self, x):
        b, n, c, t = x.shape
        h = self.encoder(x.reshape(b * n, c, t)).flatten(1)
        return self.classifier(h.reshape(b, n, -1).mean(dim=1))


def threshold_and_normalize(probabilities, weights):
    mass = np.asarray(probabilities, dtype=np.float64) * np.asarray(weights)[None, :]
    mass[mass < CWDS_SCORE_THRESHOLD] = 0.0
    total = mass.sum(axis=-1, keepdims=True)
    valid = total[:, 0] > EPS
    mass[valid] /= total[valid]
    mass[~valid] = 1.0 / N_CLASSES
    return mass


def cwds_fuse(probabilities, weights):

    masses = np.stack([threshold_and_normalize(p, w)
                       for p, w in zip(probabilities, weights)])
    with np.errstate(divide="ignore"):
        log_product = np.log(masses).sum(axis=0)
    valid = np.isfinite(log_product).any(axis=1)
    fused = masses.mean(axis=0)
    if valid.any():
        scores = log_product[valid]
        scores = np.exp(scores - scores.max(axis=1, keepdims=True))
        fused[valid] = scores / scores.sum(axis=1, keepdims=True)
    return fused


def run_cwds(train_loader, val_loader, test_loader, seed):
    weight = class_weights(train_loader)
    branches = [("cur", CWDSClassifier(3), 0),
                ("vib", CWDSClassifier(4, bidirectional=True), 1),
                ("aco", CWDSAcoustic(), 2)]

    val_probs, test_probs, val_weights = [], [], []
    trained = []
    for name, net, idx in branches:
        net = net.to(DEVICE)
        fwd = lambda m, b, i=idx: m(b[i].to(DEVICE))
        net = train_loop(net, train_loader, val_loader, fwd,
                         lambda out, y, e: F.cross_entropy(out, y, weight=weight))
        trained.append((net, fwd))
        pv, yv = collect_probs(net, val_loader, fwd)
        pt, _ = collect_probs(net, test_loader, fwd)
        val_probs.append(pv)
        test_probs.append(pt)
        val_weights.append(f1_score(yv, pv.argmax(-1), labels=list(range(N_CLASSES)),
                                    average=None, zero_division=0))

    fused = cwds_fuse(test_probs, val_weights)
    _, y_test = collect_probs(trained[0][0], test_loader, trained[0][1])
    return report(fused, y_test)


# 8. MGCN-TMIF [53]
import hashlib

MGCN_EPOCHS = 100
MGCN_LR = 1e-3
MGCN_KNN_K = 5
MGCN_HIDDEN = 64
MGCN_KL_EPOCHS = 50.0


def fft_feature(window):

    spec = torch.fft.rfft(window, dim=-1).abs()
    return spec.mean(dim=0)


def build_node_graph(windows):

    feats = torch.stack([fft_feature(w) for w in windows]).numpy()
    fmin, fmax = feats.min(), feats.max()
    feats = (feats - fmin) / (fmax - fmin + 1e-7)
    N = len(windows)

    x = torch.from_numpy(feats).float()
    d2 = torch.cdist(x, x) ** 2
    d2.fill_diagonal_(float("inf"))
    k = min(MGCN_KNN_K, N - 1)
    neigh = d2.topk(k, largest=False).indices
    zeta = d2[d2.isfinite()].median().item() + 1e-7
    A = torch.zeros(N, N)
    for i in range(N):
        for j in neigh[i].tolist():
            A[i, j] = float(torch.exp(-d2[i, j] / (2 * zeta ** 2)))
    A = A + torch.eye(N)
    deg = A.sum(dim=1)
    d_inv_sqrt = torch.diag(deg.pow(-0.5))
    return torch.from_numpy(feats).float(), d_inv_sqrt @ A @ d_inv_sqrt


def collect_split_nodes(loader):

    node_ids = [{"seen": {}, "list": []} for _ in range(3)]
    triple_nodes, labels = [], []
    for cur, vib, aco, _, y in loader:
        b = cur.shape[0]
        for i in range(b):
            row = []
            for m, w in enumerate((cur[i], vib[i], aco[i])):
                series = w.permute(1, 0, 2).reshape(w.shape[1], -1)
                key = hashlib.sha1(series.numpy().tobytes()).hexdigest()
                table = node_ids[m]
                if key not in table["seen"]:
                    table["seen"][key] = len(table["list"])
                    table["list"].append(series.cpu())
                row.append(table["seen"][key])
            triple_nodes.append(row)
            labels.append(int(y[i]))
    return node_ids, triple_nodes, labels


class GConv(nn.Module):
    def __init__(self, in_f, out_f):
        super().__init__()
        self.lin = nn.Linear(in_f, out_f)

    def forward(self, x, a_hat):
        return a_hat @ self.lin(x)


class MGCNEncoder(nn.Module):

    def __init__(self, in_f, hidden=MGCN_HIDDEN):
        super().__init__()
        self.g1 = GConv(in_f, hidden)
        self.bn1 = nn.BatchNorm1d(hidden)
        self.g2 = GConv(hidden, hidden)
        self.bn2 = nn.BatchNorm1d(hidden)
        self.fc = nn.Linear(hidden, N_CLASSES)
        self.softplus = nn.Softplus()

    def forward(self, x, a_hat):
        h = F.leaky_relu(self.bn1(self.g1(x, a_hat)))
        h = F.leaky_relu(self.bn2(self.g2(h, a_hat)))
        return self.softplus(self.fc(h)) + 1.0


def reduced_dst(b1, u1, b2, u2):

    C = sum(b1[:, i] * b2[:, j]
            for i in range(N_CLASSES) for j in range(N_CLASSES) if i != j)
    denom = (1.0 - C).clamp(min=1e-7)
    b = (b1 * b2 + b1 * u2[:, None] + b2 * u1[:, None]) / denom[:, None]
    u = (u1 * u2) / denom
    return b, u


def fuse_opinions(ops):
    b, u = ops[0]
    for op in ops[1:]:
        b, u = reduced_dst(b, u, op[0], op[1])
    return b, u


def mgcn_dirichlet_loss(alpha, y, epoch):
    S = alpha.sum(dim=-1)
    psi_S = torch.digamma(S)
    psi_a = torch.digamma(alpha)
    y_onehot = F.one_hot(y, N_CLASSES).float()
    l_acc = (y_onehot * (psi_S[:, None] - psi_a)).sum(dim=-1)

    tilde = y_onehot + (1.0 - y_onehot) * alpha
    tS = tilde.sum(dim=-1)
    kl = (torch.lgamma(tS)
          - torch.lgamma(torch.tensor(float(N_CLASSES), device=alpha.device))
          - torch.lgamma(tilde).sum(dim=-1)
          + ((tilde - 1) * (torch.digamma(tilde)
                            - torch.digamma(tS)[:, None])).sum(dim=-1))
    lam = min(1.0, epoch / MGCN_KL_EPOCHS)
    return (l_acc + lam * kl).mean()


def run_mgcntmif(train_loader, val_loader, test_loader, seed):
    split_data = {}
    for name, loader in zip(("train", "val", "test"),
                            (train_loader, val_loader, test_loader)):
        node_ids, triples, labels = collect_split_nodes(loader)
        graphs = []
        for m in range(3):
            feats, a_hat = build_node_graph(node_ids[m]["list"])
            graphs.append((feats.to(DEVICE), a_hat.to(DEVICE)))
        split_data[name] = {
            "graphs": graphs,
            "triples": torch.tensor(triples).to(DEVICE),
            "labels": torch.tensor(labels).to(DEVICE),
        }

    encoders = [MGCNEncoder(split_data["train"]["graphs"][m][0].shape[1]).to(DEVICE)
                for m in range(3)]
    params = [p for enc in encoders for p in enc.parameters()]
    opt = torch.optim.Adam(params, lr=MGCN_LR)
    tr = split_data["train"]
    best_val, best_state, bad = float("inf"), None, 0

    def forward_split(sd):
        alphas = [enc(sd["graphs"][m][0], sd["graphs"][m][1])
                  for m, enc in enumerate(encoders)]
        ops = []
        for m in range(3):
            a = alphas[m]
            node = sd["triples"][:, m]
            S = a.sum(dim=-1)
            b = (a - 1.0).clamp(min=0.0) / S[:, None]
            u = N_CLASSES / S
            ops.append((b[node], u[node]))
        bf, uf = fuse_opinions(ops)
        S_f = N_CLASSES / (uf + 1e-7)
        return bf * S_f[:, None], alphas

    for epoch in range(1, MGCN_EPOCHS + 1):
        for enc in encoders:
            enc.train()
        alpha_f, alphas = forward_split(tr)
        y = tr["labels"]
        loss = mgcn_dirichlet_loss(alpha_f, y, epoch)
        for m, a in enumerate(alphas):
            loss = loss + mgcn_dirichlet_loss(a[tr["triples"][:, m]], y, epoch)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, max_norm=5.0)
        opt.step()

        for enc in encoders:
            enc.eval()
        with torch.no_grad():
            va_f, _ = forward_split(split_data["val"])
            val_loss = F.nll_loss(F.log_softmax(va_f, dim=-1),
                                  split_data["val"]["labels"]).item()
        if val_loss < best_val:
            best_val, bad = val_loss, 0
            best_state = [{k: v.cpu().clone() for k, v in enc.state_dict().items()}
                          for enc in encoders]
        else:
            bad += 1
        if bad >= 20:
            break

    for enc, st in zip(encoders, best_state):
        enc.load_state_dict(st)
        enc.eval()
    with torch.no_grad():
        te_f, _ = forward_split(split_data["test"])
    probs = F.softmax(te_f, dim=-1).cpu().numpy()
    return report(probs, split_data["test"]["labels"].cpu().numpy())


# 9. SConv-ViT [55]
from scipy.signal import stft

CSST_IMG = 224
CSST_EPOCHS = 80


class SpatialAttention(nn.Module):
    def __init__(self, reduction_factor):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=reduction_factor)
        self.conv = nn.Conv2d(2, 1, kernel_size=3, padding=1)
        self.gelu = nn.GELU()
        self.factor = reduction_factor

    def forward(self, x):
        pooled = self.pool(x)
        avg_out = torch.mean(pooled, dim=1, keepdim=True)
        max_out, _ = torch.max(pooled, dim=1, keepdim=True)
        att = self.gelu(self.conv(torch.cat([avg_out, max_out], dim=1)))
        return F.interpolate(att, scale_factor=self.factor, mode="nearest")


class DownsampleLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=kernel_size, stride=stride)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.gelu = nn.GELU()

    def forward(self, x):
        return self.gelu(self.batch_norm(self.conv(x)))


class SConvLayer(nn.Module):
    def __init__(self, in_channels, hidden, groups, reduction_factor):
        super().__init__()
        self.groups = groups
        self.conv1 = nn.Conv2d(in_channels, in_channels, 5, padding=2,
                               groups=in_channels)
        self.batch_norm1 = nn.BatchNorm2d(in_channels)
        self.gelu = nn.GELU()
        self.conv2 = nn.Conv2d(in_channels, hidden, 1, groups=groups)
        self.batch_norm2 = nn.BatchNorm2d(hidden)
        self.conv3 = nn.Conv2d(hidden, in_channels, 1, groups=groups)
        self.batch_norm3 = nn.BatchNorm2d(in_channels)
        self.spatial_attention = SpatialAttention(reduction_factor)

    def _shuffle(self, x):
        b, c, h, w = x.size()
        cpg = c // self.groups
        x = x.view(b, self.groups, cpg, h, w)
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        return x.view(b, -1, h, w)

    def forward(self, x):
        residual = x
        out = self.gelu(self.batch_norm1(self.conv1(x)))
        out = self.gelu(self.batch_norm2(self.conv2(out)))
        out = self._shuffle(out)
        out = self.gelu(self.batch_norm3(self.conv3(out)))
        return out * self.spatial_attention(out) + residual


class SConv(nn.Module):
    def __init__(self, in_channels, hidden, groups, reduction_factor, depth):
        super().__init__()
        self.layers = nn.ModuleList([
            SConvLayer(in_channels, hidden, groups, reduction_factor)
            for _ in range(depth)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kw):
        return self.fn(x, **kw) + x


class LayerNormalize(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kw):
        return self.fn(self.norm(x), **kw)


class MLP_Block(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.nn1 = nn.Linear(dim, hidden_dim)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.nn2 = nn.Linear(hidden_dim, dim)

    def forward(self, x):
        return self.dropout(self.nn2(self.dropout(self.gelu(self.nn1(x)))))


class Attn(nn.Module):
    def __init__(self, dim, heads=8, dropout=0.1):
        super().__init__()
        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        self.to_qkv = nn.Linear(dim, dim * 3, bias=True)
        self.nn1 = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        b, n, _ = x.shape
        h = self.heads
        d = x.shape[-1] // h
        qkv = self.to_qkv(x).reshape(b, n, 3, h, d).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        dots = (q @ k.transpose(-1, -2)) * self.scale
        if mask is not None:
            dots = dots.masked_fill(~mask, float("-inf"))
        out = dots.softmax(dim=-1) @ v
        out = out.transpose(1, 2).reshape(b, n, -1)
        return self.dropout(self.nn1(out))


class ViTBlock(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(LayerNormalize(dim, Attn(dim, heads=heads, dropout=dropout))),
                Residual(LayerNormalize(dim, MLP_Block(dim, mlp_dim, dropout=dropout))),
            ]))

    def forward(self, x, mask=None):
        for attention, mlp in self.layers:
            x = mlp(attention(x, mask=mask))
        return x


class CSST_Net(nn.Module):
    def __init__(self, *, num_classes, num_patches, dim, depth, heads,
                 mlp_dim, linear_dim, dropout=0.1):
        super().__init__()
        self.downsample_1 = DownsampleLayer(3, 12, 2, 2)
        self.sconv_1 = SConv(12, 48, 4, 16, 1)
        self.downsample_2 = DownsampleLayer(12, 24, 2, 2)
        self.sconv_2 = SConv(24, 96, 4, 8, 1)
        self.downsample_3 = DownsampleLayer(24, 48, 2, 2)
        self.sconv_3 = SConv(48, 192, 4, 4, 1)
        self.downsample_4 = DownsampleLayer(48, 96, 2, 2)
        self.sconv_4 = SConv(96, 384, 4, 2, 1)
        self.downsample_5 = DownsampleLayer(96, 192, 2, 2)
        self.patch_conv = nn.Conv2d(dim, dim, 1, stride=1)
        self.pos_embedding = nn.Parameter(torch.empty(1, num_patches + 1, dim))
        torch.nn.init.normal_(self.pos_embedding, std=.02)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.dropout = nn.Dropout(dropout)
        self.transformer = ViTBlock(dim, depth, heads, mlp_dim, dropout)
        self.nn1 = nn.Linear(dim, linear_dim)
        self.nn2 = nn.Linear(linear_dim, num_classes)

    def forward(self, x, mask=None):
        out = self.sconv_1(self.downsample_1(x))
        out = self.sconv_2(self.downsample_2(out))
        out = self.sconv_3(self.downsample_3(out))
        out = self.sconv_4(self.downsample_4(out))
        out = self.patch_conv(self.downsample_5(out))
        out = out.flatten(2).transpose(1, 2)
        cls_tokens = self.cls_token.expand(out.shape[0], -1, -1)
        out = self.dropout(torch.cat((cls_tokens, out), dim=1) + self.pos_embedding)
        out = self.transformer(out, mask)
        return self.nn2(self.nn1(out[:, 0]))


def _tf_image(series, nperseg):

    _, _, z = stft(np.asarray(series, dtype=np.float64),
                   nperseg=min(nperseg, len(series)),
                   noverlap=min(nperseg, len(series)) // 2)
    mag = np.log1p(np.abs(z))
    mag = (mag - mag.mean()) / (mag.std() + 1e-7)
    t = torch.from_numpy(mag[None, None]).float()
    return F.interpolate(t, size=(CSST_IMG, CSST_IMG), mode="bilinear",
                         align_corners=False).squeeze(0)


def build_split_images(loader):

    imgs, labels = [], []
    for cur, vib, aco, _, y in loader:
        for i in range(cur.shape[0]):
            s_cur = cur[i].permute(1, 0, 2).reshape(cur.shape[2], -1)
            s_vib = vib[i].permute(1, 0, 2).reshape(vib.shape[2], -1)
            s_aco = aco[i].permute(1, 0, 2).reshape(aco.shape[2], -1)
            ch0 = _tf_image(s_cur.mean(0).numpy(),
                            max(64, min(256, s_cur.shape[-1] // 4)))
            ch1 = _tf_image(s_vib.mean(0).numpy(),
                            max(64, min(256, s_vib.shape[-1] // 4)))
            ch2 = _tf_image(s_aco.mean(0).numpy(),
                            max(256, min(2048, s_aco.shape[-1] // 8)))
            imgs.append(torch.cat([ch0, ch1, ch2], dim=0))
            labels.append(int(y[i]))
    return torch.stack(imgs).float(), np.array(labels, dtype=np.int64)


def run_sconvvit(train_loader, val_loader, test_loader, seed):
    x_tr, y_tr = build_split_images(train_loader)
    x_va, y_va = build_split_images(val_loader)
    x_te, y_te = build_split_images(test_loader)

    model = CSST_Net(num_classes=N_CLASSES, num_patches=49, dim=192, depth=1,
                     heads=8, mlp_dim=768, linear_dim=64).to(DEVICE)
    weight = class_weights(train_loader)
    opt = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=8, min_lr=1e-6)
    best, best_state, bad = float("inf"), None, 0
    n = len(y_tr)

    for epoch in range(1, CSST_EPOCHS + 1):
        model.train()
        perm = np.random.permutation(n)
        for i in range(0, n, BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            xb = x_tr[idx].to(DEVICE)
            yb = torch.from_numpy(y_tr[idx]).to(DEVICE)
            opt.zero_grad()
            loss = F.cross_entropy(model(xb), yb, weight=weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for i in range(0, len(y_va), BATCH_SIZE):
                xb = x_va[i:i + BATCH_SIZE].to(DEVICE)
                yb = torch.from_numpy(y_va[i:i + BATCH_SIZE]).to(DEVICE)
                val_loss += F.cross_entropy(model(xb), yb, weight=weight).item() * len(yb)
        val_loss /= max(1, len(y_va))
        sched.step(val_loss)
        if val_loss < best:
            best, bad = val_loss, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad += 1
        if bad >= 15:
            break

    model.load_state_dict(best_state)
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(y_te), BATCH_SIZE):
            preds.append(model(x_te[i:i + BATCH_SIZE].to(DEVICE)).cpu().numpy())
    probs = F.softmax(torch.from_numpy(np.concatenate(preds)), dim=-1).numpy()
    return report(probs, y_te)


# 10. MENN-SWR [57]
class EvidentialHead(nn.Module):
    def __init__(self, d_model, hidden_dim, n_classes, dropout=cfg.DROPOUT):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim), nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes),
            nn.Softplus(),
        )

    def forward(self, f):
        return self.net(f)


def evidence_to_belief(g):
    K = g.shape[-1]
    S = (g + 1.0).sum(dim=-1, keepdim=True)
    return g / S, K / S.squeeze(-1)


def edl_loss(g, labels, n_classes, anneal_coef):
    alpha = g + 1.0
    S = alpha.sum(dim=-1, keepdim=True)
    y = F.one_hot(labels, n_classes).float()

    E = alpha / S
    Var = alpha * (S - alpha) / (S ** 2 * (S + 1.0))
    mse = ((y - E) ** 2 + Var).sum(dim=-1).mean()

    alpha_tilde = y + (1.0 - y) * alpha
    S_tilde = alpha_tilde.sum(dim=-1, keepdim=True)
    log_gamma_K = torch.tensor(float(n_classes), dtype=g.dtype, device=g.device).lgamma()
    kl = (torch.lgamma(S_tilde.squeeze(-1))
          - log_gamma_K
          - torch.lgamma(alpha_tilde).sum(dim=-1)
          + ((alpha_tilde - 1.0) * (torch.digamma(alpha_tilde)
                                    - torch.digamma(S_tilde))).sum(dim=-1))
    return mse + anneal_coef * kl.mean()


class MENNSWRFusion(nn.Module):


    def __init__(self, n_elesors=3, w_hat=None, eps=1e-7):
        super().__init__()
        self.n_elesors = n_elesors
        self.eps = eps
        if w_hat is None:
            w_hat = [1.0 / n_elesors] * n_elesors
        self.register_buffer("w_hat", torch.tensor(w_hat, dtype=torch.float32))

    def forward(self, beta_list, u_list):
        w = self.w_hat
        L = len(beta_list)
        B = beta_list[0].shape[0]
        device = beta_list[0].device

        alpha_m = w[0] * beta_list[0]
        alpha_theta = (1.0 - w[0]) * torch.ones(B, device=device)
        alpha_tilde_theta = w[0] * u_list[0]

        for i in range(1, L):
            am_i = w[i] * beta_list[i]
            at_i = (1.0 - w[i]) * torch.ones(B, device=device)
            ati_i = w[i] * u_list[i]

            conflict = alpha_m.sum(-1) * am_i.sum(-1) - (alpha_m * am_i).sum(-1)
            Ke = 1.0 / (1.0 - conflict).clamp(min=self.eps)

            new_am = Ke.unsqueeze(1) * (
                alpha_m * am_i
                + alpha_m * at_i.unsqueeze(1)
                + alpha_theta.unsqueeze(1) * am_i
            )
            new_at = Ke * (alpha_theta * at_i + alpha_tilde_theta * at_i + alpha_theta * ati_i)
            new_ati = Ke * (alpha_tilde_theta * ati_i)
            alpha_m, alpha_theta, alpha_tilde_theta = new_am, new_at, new_ati

        denom = (alpha_m.sum(-1) + alpha_theta).clamp(min=self.eps)
        return alpha_m / denom.unsqueeze(1), alpha_theta / denom


def compute_elesor_stats(loader, delta=0.5):

    abs_means = [[], [], []]
    for cur, vib, aco, _mask, _labels in loader:
        for i, t in enumerate([cur, vib, aco]):
            flat = t.reshape(t.shape[0], -1).float()
            abs_means[i].append(flat.abs().mean(dim=1))

    w_list, r_list = [], []
    for i in range(3):
        b = torch.cat(abs_means[i])
        a_bar = b.mean()
        sa = b.std(unbiased=False)
        va = (sa / a_bar.abs().clamp(min=1e-6)).clamp(max=1.0)
        w_list.append(1.0 - va)

        b_bar = b.mean()
        d = (b - b_bar).abs()
        D = d.mean()
        r_d = (D / d.max().clamp(min=1e-6)).clamp(max=1.0)
        r_list.append(delta * 1.0 + (1.0 - delta) * r_d)

    w = torch.stack(w_list)
    w = w / w.sum().clamp(min=1e-6)
    r = torch.stack(r_list).clamp(min=1e-6, max=1.0)
    return w, r, w / (1.0 + w - r)


class MENNSWRModel(ESMEF):


    def __init__(self, n_classes=N_CLASSES, w_hat=None):
        super().__init__(n_classes=n_classes, fusion_mode="sa_ds")
        d = cfg.D_MODEL
        self.evi_ele = EvidentialHead(d, cfg.HIDDEN_DIM, n_classes)
        self.evi_vib = EvidentialHead(d, cfg.HIDDEN_DIM, n_classes)
        self.evi_aco = EvidentialHead(d, cfg.HIDDEN_DIM, n_classes)
        self.er = MENNSWRFusion(3, w_hat)

    def forward(self, cur_wins, vib_wins, aco_wins, aco_mask=None, return_q=False):
        B, N = cur_wins.shape[:2]
        device = cur_wins.device

        H_ele = self.enc_ele(cur_wins)
        H_vib = self.enc_vib(vib_wins)
        H_aco = self.enc_aco(aco_wins)

        H_vib_d, H_aco_d = self.ecf(H_ele, H_vib, H_aco, aco_mask)

        A_loc_norm, A_glo_norm = self._get_adjs(N, device)
        H_ele_l = self.gcn_ele(H_ele, A_loc_norm)
        H_vib_l = self.gcn_vib(H_vib_d, A_loc_norm)
        H_aco_l = self.gcn_aco(H_aco_d, A_loc_norm)

        H_glo = torch.cat([H_ele_l, H_vib_l, H_aco_l], dim=1)
        A_glo_batch = [A.unsqueeze(0).expand(B, -1, -1) for A in A_glo_norm]
        H_glo = self.rgcn(H_glo, A_glo_batch)

        f_ele, f_vib, f_aco, _ = self.readout(H_ele_l, H_vib_l, H_aco_l, H_glo)
        g_list = [self.evi_ele(f_ele), self.evi_vib(f_vib), self.evi_aco(f_aco)]
        betas, us = [], []
        for g in g_list:
            beta, u = evidence_to_belief(g)
            betas.append(beta)
            us.append(u)
        prob, unc = self.er(betas, us)
        return {"probability": prob, "evidence": g_list, "uncertainty": unc}


def fwd_menn(model, batch):
    cur, vib, aco, mask, _ = batch
    return model(cur.to(DEVICE), vib.to(DEVICE), aco.to(DEVICE), mask.to(DEVICE))


def run_mennswr(train_loader, val_loader, test_loader, seed):
    _, _, w_hat = compute_elesor_stats(train_loader)
    model = MENNSWRModel(n_classes=N_CLASSES, w_hat=w_hat.tolist()).to(DEVICE)
    opt = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=8, min_lr=1e-6)
    best, best_state, bad = float("inf"), None, 0

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        anneal = min(1.0, epoch / 10.0)
        for batch in train_loader:
            out = fwd_menn(model, batch)
            y = batch[-1].to(DEVICE)
            loss = sum(edl_loss(g, y, N_CLASSES, anneal) for g in out["evidence"]) / 3.0
            loss = loss + F.nll_loss(out["probability"].clamp_min(EPS).log(), y)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                out = fwd_menn(model, batch)
                y = batch[-1].to(DEVICE)
                vl = sum(edl_loss(g, y, N_CLASSES, anneal) for g in out["evidence"]) / 3.0
                val_loss += (vl + F.nll_loss(out["probability"].clamp_min(EPS).log(), y)).item() * y.size(0)
        val_loss /= max(1, len(val_loader))
        sched.step(val_loss)
        if val_loss < best:
            best, bad = val_loss, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad += 1
        if bad >= PATIENCE:
            break

    model.load_state_dict(best_state)
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            out = fwd_menn(model, batch)
            preds.append(out["probability"].cpu().numpy())
            labels.append(batch[-1].numpy())
    return report(np.concatenate(preds), np.concatenate(labels))


METHODS = {
    "cfcnn": run_cfcnn,
    "dewvs": run_dewvs,
    "mgcntmif": run_mgcntmif,
    "concat": run_concat,
    "sconvvit": run_sconvvit,
    "amtfnet": run_amtfnet,
    "mennswr": run_mennswr,
    "mmdg": run_mmdg,
    "cwds": run_cwds,
    "mscaf": run_mscaf,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", default="all", choices=list(METHODS) + ["all"])
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    args = parser.parse_args()

    os.makedirs(SAVE_DIR, exist_ok=True)
    if args.method != "all" and METHODS.get(args.method) is None:
        raise SystemExit(f"{args.method} 尚未实现")
    targets = ([k for k, v in METHODS.items() if v is not None]
               if args.method == "all" else [args.method])
    summary = {}

    for name in targets:
        rows = []
        for seed in args.seeds:
            seed_everything(seed)
            train_loader, val_loader, test_loader = get_multimodal_dataloaders_temporal(
                batch_size=BATCH_SIZE)
            res = METHODS[name](train_loader, val_loader, test_loader, seed)
            rows.append({"seed": seed, **res})
            print(f"[{name}] seed={seed} Acc={res['acc'] * 100:.2f}% "
                  f"F1={res['macro_f1'] * 100:.2f}%", flush=True)
        accs = np.array([r["acc"] for r in rows])
        f1s = np.array([r["macro_f1"] for r in rows])
        summary[name] = {
            "acc_mean": float(accs.mean()), "acc_std": float(accs.std()),
            "f1_mean": float(f1s.mean()), "f1_std": float(f1s.std()),
            "seeds": rows,
        }
        print(f"{name:<10s} Acc {accs.mean() * 100:.2f}±{accs.std() * 100:.2f}  "
              f"F1 {f1s.mean() * 100:.2f}±{f1s.std() * 100:.2f}", flush=True)

    out = os.path.join(SAVE_DIR, "baselines_all_summary.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
