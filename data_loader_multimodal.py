import json
import os
import re
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from scipy import signal

import config as cfg

DATA_DIR = os.path.join(os.environ.get("ESMEF_ROOT", "."), "data")
ELEC_NPZ_PATH = os.path.join(DATA_DIR, "electrical", "electrical_augmented.npz")
VIB_DIR = os.path.join(DATA_DIR, "vibration")
VIB_LABEL_PATH = os.path.join(VIB_DIR, "labels.json")
ACO_DIR = os.path.join(DATA_DIR, "acoustic", "augmented")


def _resolve_aco_labels():
    base = os.path.join(DATA_DIR, "acoustic")
    preferred = os.path.join(base, "labels_3class_augmented.json")
    if os.path.exists(preferred):
        return preferred
    for name in sorted(os.listdir(base)):
        if name.startswith("labels_") and name.endswith("3class_augmented.json"):
            return os.path.join(base, name)
    return preferred


ACO_LABEL_PATH = _resolve_aco_labels()

NODE_WINDOW = 8
ELEC_SAMPLES_PER_NODE = 8
VIB_SAMPLES_PER_NODE = 25
ACO_SR_TARGET = 4000
ACO_WIN_SEC = 4.0
ACO_WIN_SAMPLES = int(ACO_WIN_SEC * ACO_SR_TARGET)
ACO_SAMPLES_PER_NODE = ACO_WIN_SAMPLES // NODE_WINDOW


def load_electrical_samples(split):
    data = np.load(ELEC_NPZ_PATH)
    elec = data["electrical"]
    labels = data["labels"]
    template_ids = data.get("template_id", np.arange(len(elec)))

    unique_tids = np.unique(template_ids)
    n = len(unique_tids)
    train_end = int(n * 0.6)
    val_end = int(n * 0.8)

    if split == "all":
        mask = np.ones(len(elec), dtype=bool)
    elif split == "train":
        mask = np.isin(template_ids, unique_tids[:train_end])
    elif split == "val":
        mask = np.isin(template_ids, unique_tids[train_end:val_end])
    else:
        mask = np.isin(template_ids, unique_tids[val_end:])

    samples = []
    for i in np.where(mask)[0]:
        samples.append({
            "data": elec[i, :, :, :].copy().astype(np.float32),
            "label": int(labels[i]),
        })
    return samples


def load_vibration_samples(split):
    with open(VIB_LABEL_PATH, "r", encoding="utf-8") as f:
        vib_labels = json.load(f)

    samples = []
    for fname, info in vib_labels["labels"].items():
        if split != "all" and not fname.startswith(split + "/"):
            continue
        fp = os.path.join(VIB_DIR, fname)
        raw = np.load(fp).astype(np.float32)
        c, total = raw.shape[0], raw.shape[1]
        t_per = total // NODE_WINDOW
        vib = raw[:, :t_per * NODE_WINDOW].reshape(c, NODE_WINDOW, t_per).transpose(1, 0, 2)
        samples.append({
            "data": vib.astype(np.float32),
            "label": info["label"],
        })
    return samples


def load_acoustic_samples(split):
    with open(ACO_LABEL_PATH, "r", encoding="utf-8") as f:
        aco_labels = json.load(f)

    base_to_files = defaultdict(list)
    base_label = {}
    for fname, info in aco_labels["labels"].items():
        base = re.sub(r'_aug\d+\.npy$', '.npy', fname)
        base_to_files[base].append(fname)
        base_label[base] = info["label"]

    label_bases = {0: [], 1: [], 2: []}
    for base in sorted(base_to_files.keys()):
        lbl = base_label[base]
        if lbl in label_bases:
            label_bases[lbl].append(base)

    selected_files = set()
    for lbl in [0, 1, 2]:
        bases = label_bases[lbl]
        n = len(bases)
        if n == 0:
            continue
        train_e = int(n * 0.6)
        val_e = int(n * 0.8)
        if split == "train":
            lo, hi = 0, train_e
        elif split == "val":
            lo, hi = train_e, val_e
        elif split == "test":
            lo, hi = val_e, n
        else:
            lo, hi = 0, n
        for bi in range(lo, hi):
            for f in base_to_files[bases[bi]]:
                selected_files.add(f)

    samples = []
    for fname in sorted(selected_files):
        fp = os.path.join(ACO_DIR, fname)
        x = np.load(fp).astype(np.float32)
        x = signal.decimate(x, 4, ftype='iir')
        if len(x) >= ACO_WIN_SAMPLES:
            start = (len(x) - ACO_WIN_SAMPLES) // 2
            x = x[start:start + ACO_WIN_SAMPLES]
        else:
            x = np.pad(x, (0, ACO_WIN_SAMPLES - len(x)), mode='reflect')
        x = x[:ACO_WIN_SAMPLES]
        aco = x.reshape(NODE_WINDOW, 1, ACO_SAMPLES_PER_NODE).copy()
        samples.append({
            "data": aco.astype(np.float32),
            "label": aco_labels["labels"][fname]["label"],
        })
    return samples


class MultiModalDataset(Dataset):

    _SPLIT_SEED = {"train": 0, "val": 1, "test": 2}

    def __init__(self, split="train", pairing="label_matched", vib_loader=None):
        self.split = split
        self.pairing = pairing
        if pairing not in {"label_matched", "independent"}:
            raise ValueError(f"Unsupported pairing mode: {pairing}")
        self.rng = np.random.RandomState(cfg.RANDOM_SEED + self._SPLIT_SEED[split])

        cur_all = load_electrical_samples(split)
        vib_all = (vib_loader or load_vibration_samples)(split)
        aco_all = load_acoustic_samples(split)

        self.cur_all = cur_all
        self.vib_all = vib_all
        self.aco_all = aco_all
        self.cur_by_label = defaultdict(list)
        self.vib_by_label = defaultdict(list)
        self.aco_by_label = defaultdict(list)

        for s in cur_all:
            self.cur_by_label[s["label"]].append(s)
        for s in vib_all:
            self.vib_by_label[s["label"]].append(s)
        for s in aco_all:
            self.aco_by_label[s["label"]].append(s)

        self._label_index = {}
        if pairing == "independent":
            self.labels = sorted(set(self.cur_by_label) & set(self.vib_by_label) & set(self.aco_by_label))
            self.n_per_label = {
                lbl: min(len(self.cur_by_label[lbl]), len(self.vib_by_label[lbl]),
                         len(self.aco_by_label[lbl]))
                for lbl in self.labels
            }
        else:
            self.labels = sorted(set(list(self.cur_by_label.keys())
                                     + list(self.vib_by_label.keys())
                                     + list(self.aco_by_label.keys())))
            self.n_per_label = {}
            for lbl in self.labels:
                n_cur = len(self.cur_by_label.get(lbl, []))
                n_vib = len(self.vib_by_label.get(lbl, []))
                n_aco = len(self.aco_by_label.get(lbl, []))
                self.n_per_label[lbl] = max(n_cur, n_vib, n_aco, 1)

        self.n_samples = sum(self.n_per_label.values())
        counts = {lbl: self.n_per_label[lbl] for lbl in self.labels}
        names = {0: "Normal", 1: "Minor", 2: "Critical"}
        parts = ", ".join(f"{names[l]}:{counts[l]}" for l in sorted(counts))
        print(f"[multi {split}] {self.n_samples} triples  ({parts}) | "
              f"elec:{len(cur_all)} vib:{len(vib_all)} aco:{len(aco_all)}")

        offset = 0
        for lbl in sorted(self.labels):
            self._label_index[lbl] = (offset, offset + self.n_per_label[lbl])
            offset += self.n_per_label[lbl]

        self._pairs = None
        if split != "train":
            self._pairs = [self._draw_triple(idx) for idx in range(self.n_samples)]

    def _label_for_idx(self, idx):
        for lbl in sorted(self.labels):
            start, end = self._label_index[lbl]
            if start <= idx < end:
                return lbl
        return self.labels[-1]

    def _get_random(self, pool):
        if not pool:
            return None
        idx = self.rng.randint(0, len(pool))
        return pool[idx]

    def _draw_triple(self, idx):
        label = self._label_for_idx(idx)
        if self.pairing == "independent":
            local_idx = idx - self._label_index[label][0]
            return (self.cur_by_label[label][local_idx],
                    self.vib_by_label[label][local_idx],
                    self.aco_by_label[label][local_idx])
        cur_pool = self.cur_by_label.get(label, [])
        vib_pool = self.vib_by_label.get(label, [])
        aco_pool = self.aco_by_label.get(label, [])
        return (self._get_random(cur_pool),
                self._get_random(vib_pool),
                self._get_random(aco_pool))

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        label = self._label_for_idx(idx)
        if self._pairs is not None:
            cur_s, vib_s, aco_s = self._pairs[idx]
        else:
            cur_s, vib_s, aco_s = self._draw_triple(idx)

        eps = 1e-8

        if cur_s is not None:
            cur = torch.from_numpy(cur_s["data"].copy())
            cur = (cur - cur.mean(dim=-1, keepdim=True)) / (cur.std(dim=-1, keepdim=True) + eps)
        else:
            cur = torch.zeros(NODE_WINDOW, 3, ELEC_SAMPLES_PER_NODE)

        if vib_s is not None:
            vib = torch.from_numpy(vib_s["data"].copy())
            vib = (vib - vib.mean(dim=-1, keepdim=True)) / (vib.std(dim=-1, keepdim=True) + eps)
        else:
            vib = torch.zeros(NODE_WINDOW, 4, VIB_SAMPLES_PER_NODE)

        if aco_s is not None:
            aco = torch.from_numpy(aco_s["data"].copy())
            aco = (aco - aco.mean(dim=-1, keepdim=True)) / (aco.std(dim=-1, keepdim=True) + eps)
            aco_mask = 1.0
        else:
            aco = torch.zeros(NODE_WINDOW, 1, ACO_SAMPLES_PER_NODE)
            aco_mask = 0.0

        return (
            cur, vib, aco,
            torch.tensor(aco_mask, dtype=torch.float32),
            torch.tensor(label, dtype=torch.long),
        )


def get_multimodal_dataloaders(batch_size=None, pairing="label_matched"):
    if batch_size is None:
        batch_size = min(cfg.BATCH_SIZE, 8)
    loaders = {}
    for sp in ["train", "val", "test"]:
        ds = MultiModalDataset(split=sp, pairing=pairing)
        loaders[sp] = DataLoader(
            ds, batch_size=batch_size, shuffle=(sp == "train"),
            num_workers=0, pin_memory=True,
        )
    return loaders["train"], loaders["val"], loaders["test"]


VIB_GUARD = 2


def load_vibration_samples_temporal(split):
    with open(VIB_LABEL_PATH, "r", encoding="utf-8") as f:
        vib_labels = json.load(f)

    per_class = defaultdict(list)
    unparsed = []
    for fname, info in vib_labels["labels"].items():
        m = re.search(r"vib_(\d+)_(\d+)\.npy$", fname)
        if not m:
            unparsed.append(fname)
            continue
        c, idx = int(m.group(1)), int(m.group(2))
        per_class[c].append((idx, fname, int(info["label"])))

    if unparsed:
        print(f"[temporal-vib] {len(unparsed)} files did not match vib_(c)_(idx).npy: {unparsed[:3]}")

    chosen = []
    stats = {}
    for c in sorted(per_class):
        items = sorted(per_class[c], key=lambda t: t[0])
        n = len(items)
        te, ve = int(n * 0.6), int(n * 0.8)
        ranges = {
            "train": (0, te),
            "val": (te + VIB_GUARD, ve),
            "test": (ve + VIB_GUARD, n),
            "all": (0, n),
        }
        lo, hi = ranges[split]
        for idx, fname, label in items[lo:hi]:
            chosen.append((fname, label))
        if split != "all":
            stats[c] = (n, hi - lo)

    samples = []
    for fname, label in chosen:
        fp = os.path.join(VIB_DIR, fname)
        raw = np.load(fp).astype(np.float32)
        ch, total = raw.shape[0], raw.shape[1]
        t_per = total // NODE_WINDOW
        vib = raw[:, :t_per * NODE_WINDOW].reshape(
            ch, NODE_WINDOW, t_per).transpose(1, 0, 2)
        samples.append({"data": vib.astype(np.float32), "label": label})

    if split != "all":
        detail = ", ".join(f"c{c}:{stats[c][1]}/{stats[c][0]}" for c in sorted(stats))
        print(f"[temporal-vib] {split}: {len(samples)} windows "
              f"(guard={VIB_GUARD}, per-class kept/total: {detail})")
    return samples


class TemporalMultiModalDataset(MultiModalDataset):

    def __init__(self, split="train", pairing="label_matched"):
        super().__init__(split=split, pairing=pairing,
                         vib_loader=load_vibration_samples_temporal)


def get_multimodal_dataloaders_temporal(batch_size=None, pairing="label_matched"):
    if batch_size is None:
        batch_size = min(cfg.BATCH_SIZE, 8)
    loaders = {}
    for sp in ["train", "val", "test"]:
        ds = TemporalMultiModalDataset(split=sp, pairing=pairing)
        loaders[sp] = DataLoader(
            ds, batch_size=batch_size, shuffle=(sp == "train"),
            num_workers=0, pin_memory=True,
        )
    return loaders["train"], loaders["val"], loaders["test"]


if __name__ == "__main__":
    tl, vl, ts = get_multimodal_dataloaders_temporal()
    cur, vib, aco, mask, label = next(iter(tl))
    print(f"cur: {cur.shape}  vib: {vib.shape}  aco: {aco.shape}")
    print(f"mask: {mask}  label: {label[:8]}")
