
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score

from data_loader_multimodal import (
    get_multimodal_dataloaders_temporal as get_multimodal_dataloaders)
from es_mef import ESMEF

PROJECT_ROOT = os.environ.get("ESMEF_ROOT", ".")
SAVE_DIR = os.path.join(PROJECT_ROOT, "results", "missing_modality_ablation_temporal")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CKPT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    PROJECT_ROOT, "results", "main_esmef_decoupled", "best_full_s42.pt")


class NoGate(nn.Module):

    def __init__(self, gate):
        super().__init__()
        self.norm = gate.norm

    def forward(self, F_original, F_guided, H_ele_pooled, return_q=True):
        out = self.norm(F_original)
        q = torch.zeros(F_guided.shape[0], device=F_guided.device)
        return out, q


def disable_gate(model):
    model.ecf.gate_vib = NoGate(model.ecf.gate_vib)
    model.ecf.gate_aco = NoGate(model.ecf.gate_aco)
    return model


@torch.no_grad()
def evaluate(model, loader, missing=None):
    model.eval()
    all_preds, all_labels = [], []
    q_vib_l, q_aco_l = [], []
    a_ele_l, a_vib_l, a_aco_l = [], [], []
    for cur, vib, aco, mask, labels in loader:
        cur = cur.to(DEVICE)
        vib = vib.to(DEVICE)
        aco = aco.to(DEVICE)
        mask = mask.to(DEVICE)
        labels = labels.to(DEVICE)

        if missing == "electrical":
            cur = torch.zeros_like(cur)
        elif missing == "vibration":
            vib = torch.zeros_like(vib)
        elif missing == "acoustic":
            aco = torch.zeros_like(aco)
            mask = torch.zeros_like(mask)

        out = model(cur, vib, aco, mask, return_q=True)
        all_preds.append(out["M_final"].argmax(dim=-1).cpu().numpy())
        all_labels.append(labels.cpu().numpy())
        q_vib_l.append(out["q"]["q_vib"].cpu().numpy())
        q_aco_l.append(out["q"]["q_aco"].cpu().numpy())
        a_ele_l.append(out["alpha"][0].cpu().numpy())
        a_vib_l.append(out["alpha"][1].cpu().numpy())
        a_aco_l.append(out["alpha"][2].cpu().numpy())

    acc = float(accuracy_score(np.concatenate(all_labels), np.concatenate(all_preds)))
    return {
        "acc": acc,
        "q_vib": float(np.mean(np.concatenate(q_vib_l))),
        "q_aco": float(np.mean(np.concatenate(q_aco_l))),
        "alpha_ele": float(np.mean(np.concatenate(a_ele_l))),
        "alpha_vib": float(np.mean(np.concatenate(a_vib_l))),
        "alpha_aco": float(np.mean(np.concatenate(a_aco_l))),
    }


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    _, _, test_loader = get_multimodal_dataloaders()
    ckpt = torch.load(CKPT, map_location=DEVICE, weights_only=False)

    model_full = ESMEF(n_classes=3, fusion_mode="sa_ds").to(DEVICE)
    model_full.load_state_dict(ckpt["model_state_dict"])
    model_full.eval()

    model_nogate = ESMEF(n_classes=3, fusion_mode="sa_ds").to(DEVICE)
    model_nogate.load_state_dict(ckpt["model_state_dict"])
    disable_gate(model_nogate)
    model_nogate.eval()

    scenarios = [
        ("none", None),
        ("electrical", "electrical"),
        ("vibration", "vibration"),
        ("acoustic", "acoustic"),
    ]

    results = {}
    print(f"{'scenario':<14} {'full':>7} {'no_gate':>8} {'gap':>7} | "
          f"{'q_vib':>6} {'q_aco':>6} {'a_ele':>6} {'a_vib':>6} {'a_aco':>6}")
    print("-" * 92)
    for name, miss in scenarios:
        rf = evaluate(model_full, test_loader, missing=miss)
        rn = evaluate(model_nogate, test_loader, missing=miss)
        gap = rf["acc"] - rn["acc"]
        results[name] = {"full": rf, "no_gate": rn, "gap": gap}
        print(f"{name:<14} {rf['acc'] * 100:6.2f}% {rn['acc'] * 100:7.2f}% "
              f"{gap * 100:+6.2f}% | {rf['q_vib']:.3f} {rf['q_aco']:.3f} "
              f"{rf['alpha_ele']:.3f} {rf['alpha_vib']:.3f} {rf['alpha_aco']:.3f}")

    out = os.path.join(SAVE_DIR, "missing_modality_ablation.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
