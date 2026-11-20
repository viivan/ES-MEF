

import argparse
import json
import os
import random
import sys
import warnings
from collections import Counter

warnings.filterwarnings("ignore")

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch.optim.lr_scheduler import ReduceLROnPlateau

import config as cfg
from data_loader_multimodal import get_multimodal_dataloaders_temporal
from subset_model import SubsetESMEF

PROJECT_ROOT = os.environ.get("ESMEF_ROOT", ".")
STAGE1_DIR = os.path.join(PROJECT_ROOT, "results", "single_branch_esmef_3seed_temporal")
SAVE_DIR = os.path.join(PROJECT_ROOT, "results", "modality_subset")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

N_CLASSES = 3
SEEDS = [42, 123, 456]
FROZEN_PREFIXES = ("enc_", "gcn_")

COMBOS = {
    "ele": ("ele",),
    "vib": ("vib",),
    "aco": ("aco",),
    "ele+vib": ("ele", "vib"),
    "ele+aco": ("ele", "aco"),
    "vib+aco": ("vib", "aco"),
}


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_class_weights(train_loader):
    counts = Counter()
    for _, _, _, _, labels in train_loader:
        counts.update(labels.tolist())
    total = sum(counts.values())
    w = torch.tensor([total / (N_CLASSES * counts.get(i, 1)) for i in range(N_CLASSES)],
                     dtype=torch.float32)
    return (w / (w.min() + 1e-7)).clamp(max=5.0).to(DEVICE)


def transplant(model, mods, seed):
    mapping = {}
    for m in mods:
        mapping[f"enc_{m}."] = (m, f"enc_{m}.")
        mapping[f"gcn_{m}."] = (m, f"gcn_{m}.")
        mapping[f"readout_{m}."] = (m, f"readout.readout_{m}.")
        mapping[f"head_{m}."] = (m, f"classifiers.cls_{m}.")

    sd_all = {}
    for m in mods:
        path = os.path.join(STAGE1_DIR, f"best_{m}_s{seed}.pt")
        if not os.path.exists(path):
            raise FileNotFoundError(f"stage-1 checkpoint missing: {path}")
        sd_all[m] = torch.load(path, map_location="cpu",
                               weights_only=False)["model_state_dict"]

    target = model.state_dict()
    copied = 0
    for key in list(target):
        for my_prefix, (m, src_prefix) in mapping.items():
            if key.startswith(my_prefix):
                src_key = src_prefix + key[len(my_prefix):]
                if src_key in sd_all[m]:
                    target[key].copy_(sd_all[m][src_key])
                    copied += 1
                break
    return copied


def freeze_backbone(model):
    frozen, trainable = [], []
    for name, p in model.named_parameters():
        if name.startswith(FROZEN_PREFIXES):
            p.requires_grad_(False)
            frozen.append(name)
        else:
            trainable.append(name)
    return frozen, trainable


def set_frozen_eval(model):
    for name, module in model.named_children():
        if name.startswith(FROZEN_PREFIXES):
            module.eval()


@torch.no_grad()
def evaluate(model, loader, criterion, weight, lam1):
    model.eval()
    preds, labels = [], []
    total_loss, total = 0.0, 0
    for cur, vib, aco, mask, y in loader:
        y_dev = y.to(DEVICE)
        out = model(cur.to(DEVICE), vib.to(DEVICE), aco.to(DEVICE), mask.to(DEVICE))
        loss = F.nll_loss(torch.log(out["M_final"].clamp(min=1e-7)), y_dev, weight=weight)
        if lam1 > 0:
            loss = loss + lam1 * sum(
                criterion(out[f"logit_{m}"], y_dev) for m in model.modalities)
        loss = loss + cfg.LAMBDA_2 * criterion(out["logit_multi"], y_dev)
        total_loss += loss.item() * y.size(0)
        total += y.size(0)
        preds.append(out["M_final"].argmax(-1).cpu().numpy())
        labels.append(y.numpy())
    preds = np.concatenate(preds)
    labels = np.concatenate(labels)
    return {
        "loss": total_loss / max(total, 1),
        "acc": float(accuracy_score(labels, preds)),
        "macro_f1": float(f1_score(labels, preds, average="macro", zero_division=0)),
        "per_class_f1": f1_score(labels, preds, labels=list(range(N_CLASSES)),
                                 average=None, zero_division=0).tolist(),
        "confusion_matrix": confusion_matrix(labels, preds,
                                             labels=list(range(N_CLASSES))).tolist(),
        "preds": preds, "labels": labels,
    }


def run_combo(key, mods, seed, loaders, weight):
    seed_everything(seed)
    train_loader, val_loader, test_loader = loaders
    os.makedirs(SAVE_DIR, exist_ok=True)

    model = SubsetESMEF(modalities=mods, n_classes=N_CLASSES).to(DEVICE)
    transplant(model, mods, seed)
    frozen, trainable = freeze_backbone(model)

    criterion = nn.CrossEntropyLoss(weight=weight)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = optim.AdamW(params, lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
    sched = ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=10, min_lr=1e-6)

    lam1 = 0.0 if len(mods) == 1 else cfg.LAMBDA_1
    lam2 = cfg.LAMBDA_2
    ckpt_path = os.path.join(SAVE_DIR, f"best_{key.replace('+', '_')}_s{seed}.pt")
    best_val, bad = float("inf"), 0

    for epoch in range(1, cfg.EPOCHS + 1):
        model.train()
        set_frozen_eval(model)
        for cur, vib, aco, mask, y in train_loader:
            cur, vib = cur.to(DEVICE), vib.to(DEVICE)
            aco, mask, y = aco.to(DEVICE), mask.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            out = model(cur, vib, aco, mask)
            loss = F.nll_loss(torch.log(out["M_final"].clamp(min=1e-7)), y, weight=weight)
            if lam1 > 0:
                loss = loss + lam1 * sum(
                    criterion(out[f"logit_{m}"], y) for m in mods)
            loss = loss + lam2 * criterion(out["logit_multi"], y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=5.0)
            opt.step()

        val = evaluate(model, val_loader, criterion, weight, lam1)
        val_loss = val["loss"]
        sched.step(val_loss)
        if val_loss < best_val:
            best_val, bad = val_loss, 0
            torch.save({"model_state_dict": model.state_dict()}, ckpt_path)
        else:
            bad += 1
        if bad >= cfg.PATIENCE:
            break

    state = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    test = evaluate(model, test_loader, criterion, weight, lam1)
    print(f"  [{key}] seed={seed} Acc={test['acc'] * 100:.2f}% "
          f"F1={test['macro_f1'] * 100:.2f}%", flush=True)
    return {"seed": seed, "test_acc": test["acc"], "test_macro_f1": test["macro_f1"],
            "test_per_class_f1": test["per_class_f1"],
            "confusion_matrix": test["confusion_matrix"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--combos", nargs="+", default=list(COMBOS),
                        choices=list(COMBOS))
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    args = parser.parse_args()

    os.makedirs(SAVE_DIR, exist_ok=True)
    loaders = get_multimodal_dataloaders_temporal()
    weight = compute_class_weights(loaders[0])
    summary = {}

    for key in args.combos:
        mods = COMBOS[key]
        print(f"\n### {key}  modalities={mods}", flush=True)
        rows = [run_combo(key, mods, seed, loaders, weight) for seed in args.seeds]
        accs = np.array([r["test_acc"] for r in rows])
        f1s = np.array([r["test_macro_f1"] for r in rows])
        summary[key] = {
            "modalities": list(mods),
            "acc_mean": float(accs.mean()), "acc_std": float(accs.std()),
            "f1_mean": float(f1s.mean()), "f1_std": float(f1s.std()),
            "seeds": rows,
        }

    out = os.path.join(SAVE_DIR, "modality_subset_summary.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 66)
    for key, v in summary.items():
        print(f"{key:<10s} Acc {v['acc_mean'] * 100:6.2f}±{v['acc_std'] * 100:5.2f}  "
              f"F1 {v['f1_mean'] * 100:6.2f}±{v['f1_std'] * 100:5.2f}")
    print("=" * 66)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
