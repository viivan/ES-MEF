
import argparse
import copy
import json
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, recall_score

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from data_loader_multimodal import (
    get_multimodal_dataloaders_temporal as get_multimodal_dataloaders)
from es_mef import ESMEF

PROJECT_ROOT = os.environ.get("ESMEF_ROOT", ".")
OUT_DIR = os.path.join(PROJECT_ROOT, "results", "robustness")
DEFAULT_CHECKPOINT = os.path.join(
    PROJECT_ROOT, "results", "graph_ablation_multiseed", "best_full_s42.pt"
)
DEFAULT_CHECKPOINT_TEMPLATE = os.path.join(
    PROJECT_ROOT, "results", "graph_ablation_multiseed", "best_full_s{seed}.pt"
)
DEFAULT_SEEDS = [42, 123, 456]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_CLASSES = 3

MISSING_SCENARIOS = [
    ("complete", None),
    ("missing_electrical", "electrical"),
    ("missing_vibration", "vibration"),
    ("missing_acoustic", "acoustic"),
]
DROPOUT_LEVELS = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9]
SNR_LEVELS = [90, 30, 20, 10, 5, 0]
BIAS_LEVELS = [0.0, 0.1, 0.3, 0.5, 0.7, 1.0]


class NoGate(nn.Module):


    def __init__(self, gate):
        super().__init__()
        self.norm = copy.deepcopy(gate.norm)

    def forward(self, original, guided, electrical_summary, return_q=True):
        output = self.norm(original)
        q = torch.zeros(original.size(0), device=original.device)
        return (output, q) if return_q else output


def disable_gate(model):
    model.ecf.gate_vib = NoGate(model.ecf.gate_vib)
    model.ecf.gate_aco = NoGate(model.ecf.gate_aco)
    return model


def load_model(checkpoint, use_gate=True):
    model = ESMEF(n_classes=N_CLASSES, fusion_mode="sa_ds").to(DEVICE)
    state = torch.load(checkpoint, map_location=DEVICE, weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    if not use_gate:
        disable_gate(model)
    model.eval()
    return model


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def degrade_dropout(cur, level, generator):

    if level == 0:
        return cur
    keep = torch.rand(cur.shape, generator=generator, device=cur.device) > level
    return cur * keep.to(cur.dtype)


def degrade_snr(cur, snr_db, generator):

    if snr_db is None or snr_db >= 90:
        return cur
    signal_power = cur.pow(2).mean(dim=tuple(range(1, cur.dim())), keepdim=True).clamp(min=1e-8)
    noise_power = signal_power / (10.0 ** (snr_db / 10.0))
    noise = torch.randn(cur.shape, generator=generator, device=cur.device) * noise_power.sqrt()
    return cur + noise


def degrade_bias(cur, level, generator):

    if level == 0:
        return cur
    std = cur.std(dim=tuple(range(1, cur.dim())), keepdim=True).clamp(min=1e-8)
    sign = torch.where(
        torch.rand((cur.size(0), 1, 1, 1), generator=generator, device=cur.device) > 0.5,
        1.0,
        -1.0,
    )
    return cur + sign * level * std


def apply_missing(cur, vib, aco, mask, missing):
    if missing == "electrical":
        cur = torch.zeros_like(cur)
    elif missing == "vibration":
        vib = torch.zeros_like(vib)
    elif missing == "acoustic":
        aco = torch.zeros_like(aco)
        mask = torch.zeros_like(mask)
    return cur, vib, aco, mask


@torch.no_grad()
def evaluate(model, loader, transform=None, missing=None, seed=42):
    model.eval()
    preds, labels = [], []
    q_vib, q_aco, alpha_ele = [], [], []
    for batch_index, (cur, vib, aco, mask, target) in enumerate(loader):
        cur = cur.to(DEVICE)
        vib = vib.to(DEVICE)
        aco = aco.to(DEVICE)
        mask = mask.to(DEVICE)
        target = target.to(DEVICE)
        cur, vib, aco, mask = apply_missing(cur, vib, aco, mask, missing)
        if transform is not None:
            generator = torch.Generator(device=DEVICE).manual_seed(seed + batch_index)
            cur = transform(cur, generator)
        output = model(cur, vib, aco, mask, return_q=True)
        preds.append(output["M_final"].argmax(dim=-1).cpu().numpy())
        labels.append(target.cpu().numpy())
        q_vib.extend(output["q"]["q_vib"].cpu().numpy())
        q_aco.extend(output["q"]["q_aco"].cpu().numpy())
        alpha_ele.extend(output["alpha"][0].cpu().numpy())
    preds = np.concatenate(preds)
    labels = np.concatenate(labels)
    per_f1 = f1_score(labels, preds, labels=[0, 1, 2], average=None, zero_division=0)
    recalls = recall_score(labels, preds, labels=[0, 1, 2], average=None, zero_division=0)
    return {
        "acc": float(accuracy_score(labels, preds)),
        "macro_f1": float(f1_score(labels, preds, average="macro", zero_division=0)),
        "f1_normal": float(per_f1[0]),
        "f1_minor": float(per_f1[1]),
        "f1_critical": float(per_f1[2]),
        "critical_recall": float(recalls[2]),
        "q_vib_mean": float(np.mean(q_vib)),
        "q_aco_mean": float(np.mean(q_aco)),
        "alpha_ele_mean": float(np.mean(alpha_ele)),
    }


def run_missing(models, loader, seed):
    results = {}
    for name, missing in MISSING_SCENARIOS:
        results[name] = {}
        for model_name, model in models.items():
            results[name][model_name] = evaluate(model, loader, missing=missing, seed=seed)
    return results


def run_degradation(models, loader, seed, corruption_repeats=3):
    corruption_seeds = [seed * 1000 + repeat for repeat in range(corruption_repeats)]
    configs = {
        "dropout": [(float(x), lambda cur, g, x=x: degrade_dropout(cur, x, g)) for x in DROPOUT_LEVELS],
        "gaussian_snr": [(float(x), lambda cur, g, x=x: degrade_snr(cur, x, g)) for x in SNR_LEVELS],
        "bias": [(float(x), lambda cur, g, x=x: degrade_bias(cur, x, g)) for x in BIAS_LEVELS],
    }
    results = {}
    for kind, levels in configs.items():
        results[kind] = []
        for level, transform in levels:
            row = {"level": level, "corruption_seeds": corruption_seeds, "models": {}}
            for model_name, model in models.items():
                repeated = [
                    evaluate(model, loader, transform=transform, seed=corruption_seed)
                    for corruption_seed in corruption_seeds
                ]
                row["models"][model_name] = summarize_metric_rows(repeated)
            results[kind].append(row)
    return results


def summarize_metric_rows(rows):
    metrics = rows[0].keys()
    summary = {}
    for metric in metrics:
        values = np.asarray([row[metric] for row in rows], dtype=float)
        summary[f"{metric}_mean"] = float(values.mean())
        summary[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    return summary


def summarize_missing(per_seed):
    summary = {}
    for scenario, _ in MISSING_SCENARIOS:
        summary[scenario] = {}
        for model_name in ("ES-MEF", "w/o ECF gate"):
            rows = [item["missing"][scenario][model_name] for item in per_seed]
            summary[scenario][model_name] = summarize_metric_rows(rows)
    return summary


def summarize_degradation(per_seed):
    summary = {}
    levels_by_kind = {
        "dropout": DROPOUT_LEVELS,
        "gaussian_snr": SNR_LEVELS,
        "bias": BIAS_LEVELS,
    }
    for kind, levels in levels_by_kind.items():
        summary[kind] = []
        for index, level in enumerate(levels):
            row = {"level": float(level), "models": {}}
            for model_name in ("ES-MEF", "w/o ECF gate"):
                seed_rows = [
                    item["degradation"][kind][index]["models"][model_name]
                    for item in per_seed
                ]
                metrics = {
                    key.removesuffix("_mean")
                    for key in seed_rows[0]
                    if key.endswith("_mean")
                }
                aggregate = {}
                for metric in metrics:
                    means = np.asarray(
                        [item[f"{metric}_mean"] for item in seed_rows], dtype=float
                    )
                    within_vars = np.asarray(
                        [item.get(f"{metric}_std", 0.0) ** 2 for item in seed_rows],
                        dtype=float,
                    )
                    aggregate[f"{metric}_mean"] = float(means.mean())
                    aggregate[f"{metric}_std"] = float(
                        np.sqrt(within_vars.mean() + means.var(ddof=1))
                        if len(means) > 1
                        else np.sqrt(within_vars.mean())
                    )
                row["models"][model_name] = aggregate
            summary[kind].append(row)
    return summary


def run_one_seed(checkpoint, seed, loader, corruption_repeats=3):
    seed_everything(seed)
    models = {
        "ES-MEF": load_model(checkpoint, use_gate=True),
        "w/o ECF gate": load_model(checkpoint, use_gate=False),
    }
    return {
        "seed": seed,
        "checkpoint": checkpoint,
        "missing": run_missing(models, loader, seed),
        "degradation": run_degradation(models, loader, seed, corruption_repeats),
    }


def resolve_checkpoints(args):
    if args.checkpoint is not None:
        if args.seeds is None:
            return [(args.seed, args.checkpoint)]
        if len(args.seeds) != 1:
            raise ValueError("--checkpoint can only be used with one seed")
        return [(args.seeds[0], args.checkpoint)]
    seeds = args.seeds if args.seeds is not None else DEFAULT_SEEDS
    return [(seed, args.checkpoint_template.format(seed=seed)) for seed in seeds]


def write_json(path, metadata, results):
    with open(path, "w", encoding="utf-8") as file:
        json.dump({"metadata": metadata, "results": results}, file, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=None,
                        help="")
    parser.add_argument("--checkpoint-template", default=DEFAULT_CHECKPOINT_TEMPLATE,
                        help=" {seed} ")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--corruption-repeats", type=int, default=3,
                        help="")
    args = parser.parse_args()
    if args.corruption_repeats < 1:
        raise ValueError("--corruption-repeats must be at least 1")
    checkpoints = resolve_checkpoints(args)
    missing_checkpoints = [path for _, path in checkpoints if not os.path.isfile(path)]
    if missing_checkpoints:
        raise FileNotFoundError("Missing checkpoint(s): " + ", ".join(missing_checkpoints))
    os.makedirs(OUT_DIR, exist_ok=True)
    _, _, test_loader = get_multimodal_dataloaders()
    per_seed = [
        run_one_seed(checkpoint, seed, test_loader, args.corruption_repeats)
        for seed, checkpoint in checkpoints
    ]
    seeds = [item["seed"] for item in per_seed]
    metadata = {
        "checkpoints": {str(item["seed"]): item["checkpoint"] for item in per_seed},
        "seeds": seeds,
        "device": str(DEVICE),
        "test_only_degradation": True,
        "missing_scenarios": [name for name, _ in MISSING_SCENARIOS],
        "dropout_levels": DROPOUT_LEVELS,
        "snr_levels_db": SNR_LEVELS,
        "bias_levels_std": BIAS_LEVELS,
        "corruption_repeats": args.corruption_repeats,
        "corruption_seed_scheme": "checkpoint_seed * 1000 + repeat_index",
    }
    missing_results = {
        "per_seed": [
            {"seed": item["seed"], "checkpoint": item["checkpoint"], "results": item["missing"]}
            for item in per_seed
        ],
        "summary": summarize_missing(per_seed),
    }
    degradation_results = {
        "per_seed": [
            {"seed": item["seed"], "checkpoint": item["checkpoint"], "results": item["degradation"]}
            for item in per_seed
        ],
        "summary": summarize_degradation(per_seed),
    }
    write_json(
        os.path.join(OUT_DIR, "modality_missing_multiseed_results.json"),
        metadata,
        missing_results,
    )
    write_json(
        os.path.join(OUT_DIR, "electrical_degradation_multiseed_results.json"),
        metadata,
        degradation_results,
    )
    print(f"Saved robustness results to {OUT_DIR}")
    for scenario, row in summarize_missing(per_seed).items():
        full = row["ES-MEF"]
        print(
            f"{scenario:>20}: Acc={full['acc_mean'] * 100:.2f}±{full['acc_std'] * 100:.2f}% "
            f"Macro-F1={full['macro_f1_mean'] * 100:.2f}±{full['macro_f1_std'] * 100:.2f}% "
            f"Critical-Recall={full['critical_recall_mean'] * 100:.2f}±{full['critical_recall_std'] * 100:.2f}%"
        )


if __name__ == "__main__":
    main()
