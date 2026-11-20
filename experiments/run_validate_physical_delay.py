from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd


ID_COLUMNS = ["Elevator_ID", "Timestamp", "Fault_Label"]
MODALITY_PATTERNS = {
    "electrical": ("current", "voltage", "electrical", "stator"),
    "vibration": ("vibration", "accelerometer", "accel_", "accelx", "accely", "accelz"),
    "acoustic": ("acoustic", "sound", "audio", "microphone", "mic_"),
}


def load_config(config_path: Path):
    spec = importlib.util.spec_from_file_location("esmef_config", config_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load config: {config_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def robust_standardize(values: np.ndarray) -> np.ndarray | None:
    values = np.asarray(values, dtype=np.float64)
    values = values - np.median(values)
    scale = 1.4826 * np.median(np.abs(values))
    if not np.isfinite(scale) or scale < 1e-12:
        scale = np.std(values)
    if not np.isfinite(scale) or scale < 1e-12:
        return None
    return np.clip(values / scale, -8.0, 8.0)


def corr_at_lag(current_delta: np.ndarray, vibration_delta: np.ndarray,
                lag_samples: int) -> float:
    if lag_samples > 0:
        x, y = current_delta[:-lag_samples], vibration_delta[lag_samples:]
    elif lag_samples < 0:
        x, y = current_delta[-lag_samples:], vibration_delta[:lag_samples]
    else:
        x, y = current_delta, vibration_delta
    if len(x) < 3 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def estimate_block(current: np.ndarray, vibration: np.ndarray, fs: float,
                   max_lag_sec: float):
    x = robust_standardize(np.diff(current))
    y = robust_standardize(np.diff(vibration))
    if x is None or y is None or len(x) != len(y):
        return None

    max_lag = min(int(round(max_lag_sec * fs)), max(0, len(x) // 4))
    candidates = []
    for lag in range(-max_lag, max_lag + 1):
        r = corr_at_lag(x, y, lag)
        if np.isfinite(r):
            candidates.append((lag, r))
    if not candidates:
        return None
    lag, corr = max(candidates, key=lambda pair: abs(pair[1]))
    return lag / fs, corr


def continuous_runs(group: pd.DataFrame, gap_limit_sec: float):
    times = group["Timestamp"].to_numpy(dtype="datetime64[ns]")
    if len(times) == 0:
        return
    gaps = np.diff(times).astype("timedelta64[ns]").astype(np.int64) / 1e9
    cuts = np.flatnonzero((gaps <= 0) | (gaps > gap_limit_sec)) + 1
    bounds = np.r_[0, cuts, len(group)]
    for start, end in zip(bounds[:-1], bounds[1:]):
        if end > start:
            yield group.iloc[start:end]


def detect_signal_columns(columns):
    detected = {name: [] for name in MODALITY_PATTERNS}
    ignored = []
    for column in columns:
        if column in ID_COLUMNS:
            continue
        normalized = column.lower()
        matches = [
            modality for modality, patterns in MODALITY_PATTERNS.items()
            if any(pattern in normalized for pattern in patterns)
        ]
        if len(matches) == 1:
            detected[matches[0]].append(column)
        else:
            ignored.append(column)
    return detected, ignored


def build_signal_pairs(detected, columns):
    channels = {key: [(name, name) for name in names]
                for key, names in detected.items()}
    vib_axes = detected["vibration"]
    axis_groups = {}
    for name in vib_axes:
        axis_groups.setdefault(name.rsplit("_", 1)[0], []).append(name)
    for group in axis_groups.values():
        if len(group) >= 2 and all(name in columns for name in group):
            magnitude_name = "Vibration_Magnitude_" + "_".join(group)
            channels["vibration"].append((magnitude_name, "__magnitude__:" + ",".join(group)))

    pairs = []
    for source_modality, target_modality in (
        ("electrical", "vibration"),
        ("electrical", "acoustic"),
        ("vibration", "acoustic"),
    ):
        for source_label, source_column in channels[source_modality]:
            for target_label, target_column in channels[target_modality]:
                pairs.append((source_modality, source_label, source_column,
                              target_modality, target_label, target_column))
    return pairs


def get_signal(frame: pd.DataFrame, signal_spec: str) -> np.ndarray:
    if signal_spec.startswith("__magnitude__:"):
        names = signal_spec.split(":", 1)[1].split(",")
        matrix = frame[names].to_numpy(dtype=np.float64)
        return np.sqrt(np.sum(matrix ** 2, axis=1))
    return frame[signal_spec].to_numpy(dtype=np.float64)


def analyze(args):
    config = load_config(args.config)
    csv_path = args.csv or Path(config.CSV_PATH)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    header = pd.read_csv(csv_path, nrows=0).columns.tolist()
    missing_ids = [column for column in ID_COLUMNS if column not in header]
    if missing_ids:
        raise ValueError(f"CSV is missing required indexing columns: {missing_ids}")
    detected, _ = detect_signal_columns(header)
    signal_columns = sorted({name for names in detected.values() for name in names})
    if not detected["electrical"] or not detected["vibration"]:
        raise ValueError(
            "At least one electrical column and one vibration column are needed. "
            f"Detected modalities: {detected}"
        )
    read_columns = ID_COLUMNS + signal_columns
    frame = pd.read_csv(csv_path, usecols=read_columns, parse_dates=["Timestamp"])
    frame = frame.dropna(subset=read_columns)
    frame = frame.sort_values(["Elevator_ID", "Timestamp"], kind="stable")

    positive_deltas = []
    for _, elevator in frame.groupby("Elevator_ID", sort=False):
        dt = elevator["Timestamp"].diff().dt.total_seconds().to_numpy()
        valid = dt[np.isfinite(dt) & (dt > 0)]
        if len(valid):
            positive_deltas.append(valid)
    if not positive_deltas:
        raise ValueError("Could not infer a positive sampling interval from timestamps")
    sample_interval = float(np.median(np.concatenate(positive_deltas)))
    fs = 1.0 / sample_interval
    gap_limit = args.gap_limit_sec or 1.5 * sample_interval

    signal_pairs = build_signal_pairs(detected, set(frame.columns))
    block_samples = max(4, int(round(args.block_sec * fs)))
    hop = max(1, int(round(block_samples * (1.0 - args.overlap))))
    records = []
    group_cols = ["Elevator_ID", "Fault_Label"]
    for (elevator_id, label), group in frame.groupby(group_cols, sort=False):
        for run_id, run in enumerate(continuous_runs(group, gap_limit)):
            if len(run) < block_samples:
                continue
            timestamps = run["Timestamp"].to_numpy()
            signal_cache = {}
            for pair in signal_pairs:
                source_modality, source_label, source_spec, target_modality, target_label, target_spec = pair
                if source_spec not in signal_cache:
                    signal_cache[source_spec] = get_signal(run, source_spec)
                if target_spec not in signal_cache:
                    signal_cache[target_spec] = get_signal(run, target_spec)
                source = signal_cache[source_spec]
                target = signal_cache[target_spec]
                if len(source) < block_samples or len(target) < block_samples:
                    continue
                for start in range(0, len(run) - block_samples + 1, hop):
                    end = start + block_samples
                    estimate = estimate_block(
                        source[start:end], target[start:end], fs, args.max_lag_sec,
                    )
                    if estimate is None:
                        continue
                    lag_sec, corr = estimate
                    records.append({
                        "elevator_id": elevator_id,
                        "fault_label": label,
                        "run_id": run_id,
                        "source_modality": source_modality,
                        "source_signal": source_label,
                        "target_modality": target_modality,
                        "target_signal": target_label,
                        "start_time": str(timestamps[start]),
                        "end_time": str(timestamps[end - 1]),
                        "lag_sec": lag_sec,
                        "correlation": corr,
                        "abs_correlation": abs(corr),
                        "n_samples": block_samples,
                    })

    if not records:
        raise ValueError(
            "No analysis blocks were produced. Reduce --block-sec or inspect the data."
        )

    result = pd.DataFrame(records)
    result["within_one_dde_step"] = result["lag_sec"].abs() <= float(config.STRIDE_SEC)
    result["meets_corr_threshold"] = result["abs_correlation"] >= args.min_abs_corr

    summaries = []
    summary_groups = [("overall", result)]
    summary_groups.extend(
        (f"{source}_to_{target}", subset)
        for (source, target), subset in result.groupby(
            ["source_modality", "target_modality"]
        )
    )
    for key, subset in summary_groups:
        summaries.append({
            "group": str(key),
            "blocks": int(len(subset)),
            "median_lag_sec": float(subset["lag_sec"].median()),
            "median_abs_lag_sec": float(subset["lag_sec"].abs().median()),
            "median_abs_correlation": float(subset["abs_correlation"].median()),
            "fraction_within_one_dde_step": float(subset["within_one_dde_step"].mean()),
            "fraction_within_step_and_corr_threshold": float(
                (subset["within_one_dde_step"] & subset["meets_corr_threshold"]).mean()
            ),
        })

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "delay_estimates.csv"
    summary_path = output_dir / "delay_summary.json"
    figure_path = output_dir / "delay_distribution.png"
    result.to_csv(result_path, index=False, encoding="utf-8-sig")

    summary = {
        "csv_path": str(csv_path),
        "rows_used": int(len(frame)),
        "sampling_interval_sec": sample_interval,
        "sampling_rate_hz": fs,
        "analysis_block_sec": args.block_sec,
        "max_lag_search_sec": args.max_lag_sec,
        "dde_node_step_sec_from_config_stride": float(config.STRIDE_SEC),
        "groups": summaries,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
        axes[0].hist(result["lag_sec"], bins=31, color="#238b8e", edgecolor="white")
        axes[0].axvline(0, color="black", linewidth=1)
        axes[0].axvline(float(config.STRIDE_SEC), color="#d95f02", linestyle="--",
                        label=f"+1 DDE step ({config.STRIDE_SEC:g} s)")
        axes[0].axvline(-float(config.STRIDE_SEC), color="#d95f02", linestyle="--")
        axes[0].set(xlabel="Peak lag (s); positive means target follows source",
                    ylabel="Analysis blocks", title="Estimated lag distribution")
        axes[0].legend(frameon=False)
        axes[1].scatter(result["lag_sec"], result["abs_correlation"], s=13,
                        alpha=0.55, color="#526b9b", edgecolors="none")
        axes[1].axvline(float(config.STRIDE_SEC), color="#d95f02", linestyle="--")
        axes[1].axvline(-float(config.STRIDE_SEC), color="#d95f02", linestyle="--")
        axes[1].set(xlabel="Peak lag (s)", ylabel="Absolute correlation",
                    title="Peak association by block", ylim=(0, 1))
        fig.savefig(figure_path, dpi=220)
        plt.close(fig)
    except ImportError:
        figure_path = None

    print(f"Sampling interval: {sample_interval:.6g} s ({fs:.4g} Hz)")
    print(f"Configured DDE step: {config.STRIDE_SEC:g} s")
    print(f"Analysis blocks: {len(result)}")
    print(f"Results: {result_path}")
    print(f"Summary: {summary_path}")
    if figure_path:
        print(f"Figure: {figure_path}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-dir", type=Path,
        default=Path("."),
        help="Folder containing config.py",
    )
    parser.add_argument("--csv", type=Path, default=None,
                        help="Override config.CSV_PATH")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Output folder (default: <project-dir>/physical_delay_analysis)")
    parser.add_argument("--block-sec", type=float, default=60.0,
                        help="Duration of each local correlation block")
    parser.add_argument("--overlap", type=float, default=0.5,
                        help="Fractional overlap between blocks, from 0 to <1")
    parser.add_argument("--max-lag-sec", type=float, default=8.0,
                        help="Search range in both lag directions")
    parser.add_argument("--min-abs-corr", type=float, default=0.2,
                        help="Threshold used only for descriptive summary")
    parser.add_argument("--gap-limit-sec", type=float, default=None,
                        help="Split a sequence when timestamp gap exceeds this value")
    args = parser.parse_args()
    args.config = args.project_dir / "config.py"
    if args.output_dir is None:
        args.output_dir = args.project_dir / "physical_delay_analysis"
    if not 0 <= args.overlap < 1:
        parser.error("--overlap must be in [0, 1)")
    return args


if __name__ == "__main__":
    analyze(parse_args())
