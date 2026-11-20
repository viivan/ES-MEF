
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
from es_mef import ESMEF

PROJECT_ROOT = os.environ.get("ESMEF_ROOT", ".")
STAGE1_DIR = os.path.join(PROJECT_ROOT, "results", "single_branch_esmef_3seed_temporal")
SAVE_DIR = os.path.join(PROJECT_ROOT, "results", "graph_ablation_decoupled")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

N_CLASSES = 3
SEEDS = [42, 123, 456]
MODALITIES = ("ele", "vib", "aco")
FROZEN_PREFIXES = ("enc_", "gcn_")

VARIANTS = [
    ("full", "ES-MEF (LG-STG)", {}),
    ("no_lgstg", "w/o LG-STG", {"use_local_graph": False,
                                "use_global_graph": False}),
    ("no_sync", "w/o SE", {"use_sync": False}),
    ("no_delay", "w/o DDE", {"use_delay": False}),
    ("no_ie", "w/o IE", {"use_intra_temporal": False}),
]


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
    weights = torch.tensor(
        [total / (N_CLASSES * counts.get(i, 1)) for i in range(N_CLASSES)],
        dtype=torch.float32,
    )
    weights = weights / (weights.min() + 1e-7)
    return weights.clamp(max=5.0).to(DEVICE)


def transplant_branch_weights(model, seed):
    for mod in MODALITIES:
        path = os.path.join(STAGE1_DIR, f"best_{mod}_s{seed}.pt")
        if not os.path.exists(path):
            raise FileNotFoundError(f"stage-1 checkpoint missing: {path}")
        sd = torch.load(path, map_location="cpu",
                        weights_only=False)["model_state_dict"]
        prefixes = (f"enc_{mod}.", f"gcn_{mod}.",
                    f"readout.readout_{mod}.", f"classifiers.cls_{mod}.")
        target = model.state_dict()
        for key in target:
            if key.startswith(prefixes) and key in sd:
                target[key] = sd[key]
        model.load_state_dict(target)


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
def evaluate(model, loader, criterion, weight):
    model.eval()
    all_preds, all_labels = [], []
    total_loss, total = 0.0, 0

    for cur, vib, aco, mask, labels in loader:
        cur, vib = cur.to(DEVICE), vib.to(DEVICE)
        aco, mask = aco.to(DEVICE), mask.to(DEVICE)
        labels = labels.to(DEVICE)
        out = model(cur, vib, aco, mask)
        loss = (
            F.nll_loss(
                torch.log(out["M_final"].clamp(min=1e-7)),
                labels, weight=weight,
            )
            + cfg.LAMBDA_1 * (
                criterion(out["logit_ele"], labels)
                + criterion(out["logit_vib"], labels)
                + criterion(out["logit_aco"], labels)
            )
            + cfg.LAMBDA_2 * criterion(out["logit_multi"], labels)
        )
        total_loss += loss.item() * cur.size(0)
        total += cur.size(0)
        all_preds.append(out["M_final"].argmax(dim=-1).cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)
    return {
        "loss": total_loss / max(total, 1),
        "acc": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro", zero_division=0),
        "per_f1": f1_score(labels, preds, average=None,
                           labels=list(range(N_CLASSES)),
                           zero_division=0).tolist(),
        "preds": preds, "labels": labels,
    }


def run_variant(key, model_kwargs, seed, train_loader, val_loader,
                test_loader, weight):
    seed_everything(seed)
    os.makedirs(SAVE_DIR, exist_ok=True)

    model = ESMEF(n_classes=N_CLASSES, fusion_mode="sa_ds",
                     use_ecf=True, **model_kwargs).to(DEVICE)
    transplant_branch_weights(model, seed)
    frozen, trainable = freeze_backbone(model)

    criterion = nn.CrossEntropyLoss(weight=weight)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(params, lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5,
                                  patience=10, min_lr=1e-6)

    ckpt_path = os.path.join(SAVE_DIR, f"best_{key}_s{seed}.pt")
    best_val = float("inf")
    patience_cnt = 0
    history = {"epoch": [], "train_loss": [], "train_acc": [],
               "val_loss": [], "val_acc": [], "val_f1": []}

    for epoch in range(1, cfg.EPOCHS + 1):
        model.train()
        set_frozen_eval(model)

        total_loss, correct, total = 0.0, 0, 0
        for cur, vib, aco, mask, labels in train_loader:
            cur, vib = cur.to(DEVICE), vib.to(DEVICE)
            aco, mask = aco.to(DEVICE), mask.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            out = model(cur, vib, aco, mask)
            loss = (
                F.nll_loss(
                    torch.log(out["M_final"].clamp(min=1e-7)),
                    labels, weight=weight,
                )
                + cfg.LAMBDA_1 * (
                    criterion(out["logit_ele"], labels)
                    + criterion(out["logit_vib"], labels)
                    + criterion(out["logit_aco"], labels)
                )
                + cfg.LAMBDA_2 * criterion(out["logit_multi"], labels)
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=5.0)
            optimizer.step()

            total_loss += loss.item() * cur.size(0)
            preds = out["M_final"].argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / max(total, 1)
        val = evaluate(model, val_loader, criterion, weight)
        scheduler.step(val["loss"])

        if epoch == 1 or epoch % 5 == 0:
            print(f"  epoch {epoch:3d} val_loss={val['loss']:.4f} "
                  f"val_acc={val['acc'] * 100:.1f}%", flush=True)

        history["epoch"].append(epoch)
        history["train_loss"].append(round(total_loss / max(total, 1), 6))
        history["train_acc"].append(round(train_acc, 6))
        history["val_loss"].append(round(val["loss"], 6))
        history["val_acc"].append(round(val["acc"], 6))
        history["val_f1"].append(round(val["macro_f1"], 6))

        if val["loss"] < best_val:
            best_val = val["loss"]
            patience_cnt = 0
            torch.save({"epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "best_val_loss": best_val}, ckpt_path)
        else:
            patience_cnt += 1
        if patience_cnt >= cfg.PATIENCE:
            break

    checkpoint = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    test = evaluate(model, test_loader, criterion, weight)

    print(f"  [{key}] seed={seed}: Acc={test['acc'] * 100:.2f}% "
          f"F1={test['macro_f1'] * 100:.2f}%", flush=True)

    return {
        "seed": seed,
        "test_acc": test["acc"],
        "test_macro_f1": test["macro_f1"],
        "test_per_class_f1": test["per_f1"],
        "confusion_matrix": confusion_matrix(
            test["labels"], test["preds"],
            labels=list(range(N_CLASSES))).tolist(),
        "history": history,
    }


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    print("=" * 70, flush=True)
    print(f"Decoupled graph ablation (LG-STG), temporal split, "
          f"{len(SEEDS)} seeds, device={DEVICE}", flush=True)
    print("=" * 70, flush=True)

    loaders = get_multimodal_dataloaders_temporal()
    weight = compute_class_weights(loaders[0])

    summary = {}
    for key, display, model_kwargs in VARIANTS:
        print(f"\n### Variant: {display}  {model_kwargs or '(intact)'}",
              flush=True)
        seeds_detail = [
            run_variant(key, model_kwargs, seed,
                        loaders[0], loaders[1], loaders[2], weight)
            for seed in SEEDS
        ]
        accs = np.array([d["test_acc"] for d in seeds_detail])
        f1s = np.array([d["test_macro_f1"] for d in seeds_detail])
        per = np.array([d["test_per_class_f1"] for d in seeds_detail])
        summary[key] = {
            "display": display,
            "model_kwargs": model_kwargs,
            "acc_mean": float(accs.mean()),
            "acc_std": float(accs.std()),
            "f1_mean": float(f1s.mean()),
            "f1_std": float(f1s.std()),
            "per_f1_means": per.mean(axis=0).tolist(),
            "per_f1_stds": per.std(axis=0).tolist(),
            "seeds": seeds_detail,
        }

    out_path = os.path.join(SAVE_DIR, "graph_ablation_decoupled_summary.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 80)
    print(f"Decoupled graph ablation summary (mean ± std over {len(SEEDS)} seeds)")
    print("=" * 80)
    for key, display, _ in VARIANTS:
        v = summary[key]
        per_str = "  ".join(f"{m * 100:5.2f}" for m in v["per_f1_means"])
        print(f"{display:<16s} {v['acc_mean'] * 100:>6.2f}±{v['acc_std'] * 100:<5.2f}"
              f" {v['f1_mean'] * 100:>6.2f}   {per_str}")
    print("=" * 80)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
