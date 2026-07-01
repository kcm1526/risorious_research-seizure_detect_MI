import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


ROOT = Path(__file__).resolve().parent
BANDS = (
    ("delta", 0.5, 4.0),
    ("theta", 4.0, 8.0),
    ("alpha", 8.0, 13.0),
    ("beta", 13.0, 30.0),
    ("gamma", 30.0, 80.0),
    ("high", 80.0, 120.0),
)


def read_jsonl(path: Path) -> List[Dict[str, object]]:
    if not path.exists():
        return []
    rows = []
    with path.open() as fp:
        for line in fp:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_csv(path: Path, rows: List[Dict[str, object]]):
    if not rows:
        path.write_text("")
        return
    columns = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def mask_ranges(mask: np.ndarray) -> List[Tuple[int, int]]:
    ranges = []
    start = None
    for index, value in enumerate(mask.astype(bool)):
        if value and start is None:
            start = index
        elif not value and start is not None:
            ranges.append((start, index))
            start = None
    if start is not None:
        ranges.append((start, len(mask)))
    return ranges


def merge_ranges(ranges: Sequence[Tuple[int, int]], max_gap: int) -> List[Tuple[int, int]]:
    if not ranges:
        return []
    merged = [list(ranges[0])]
    for start, end in ranges[1:]:
        if start - merged[-1][1] <= max_gap:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(int(start), int(end)) for start, end in merged]


def map_mask_ranges(
    mask: np.ndarray,
    target_samples: int,
    fs: int,
    min_duration_sec: float,
    merge_gap_sec: float,
) -> List[Tuple[int, int]]:
    scale = target_samples / len(mask)
    ranges = []
    for start, end in mask_ranges(mask):
        ranges.append((int(math.floor(start * scale)), int(math.ceil(end * scale))))
    ranges = merge_ranges(ranges, max_gap=int(round(merge_gap_sec * fs)))
    min_len = max(1, int(round(min_duration_sec * fs)))
    return [
        (max(0, start), min(target_samples, end))
        for start, end in ranges
        if end - start >= min_len
    ]


def top_time_context_ranges(
    scores: np.ndarray,
    target_samples: int,
    fs: int,
    top_k: int,
    context_sec: float,
) -> List[Tuple[int, int]]:
    flat = np.asarray(scores).reshape(-1)
    if flat.size == 0:
        return []
    indices = np.argsort(flat)[-min(top_k, flat.size) :][::-1]
    half_context = max(1, int(round(context_sec * fs / 2.0)))
    scale = target_samples / len(flat)
    ranges = []
    for index in indices:
        center = int(round((index + 0.5) * scale))
        ranges.append((max(0, center - half_context), min(target_samples, center + half_context)))
    return merge_ranges(sorted(ranges), max_gap=int(round(context_sec * fs / 4.0)))


def band_power_fractions(power: np.ndarray, freqs: np.ndarray) -> Dict[str, float]:
    total_mask = (freqs >= 0.5) & (freqs <= 120.0)
    total = float(power[total_mask].sum())
    result = {}
    for name, low, high in BANDS:
        mask = (freqs >= low) & (freqs < high)
        result[f"{name}_frac"] = float(power[mask].sum() / total) if total > 0 else 0.0
    return result


def spectral_entropy(power: np.ndarray, freqs: np.ndarray) -> float:
    mask = (freqs >= 0.5) & (freqs <= 80.0)
    values = power[mask].astype(float)
    total = float(values.sum())
    if total <= 0 or len(values) <= 1:
        return 0.0
    probs = values / total
    return float(-np.sum(probs * np.log(probs + 1e-12)) / math.log(len(probs)))


def analyze_eeg_segment(
    eeg: np.ndarray,
    labels: np.ndarray,
    baseline: np.ndarray,
    masked_pred: np.ndarray,
    latent_pred: np.ndarray,
    start: int,
    end: int,
    fs: int,
) -> Dict[str, float]:
    segment = np.asarray(eeg[:, start:end], dtype=float)
    label_segment = np.asarray(labels[start:end], dtype=float)
    baseline_segment = np.asarray(baseline[start:end], dtype=float)
    masked_segment = np.asarray(masked_pred[start:end], dtype=float)
    latent_segment = np.asarray(latent_pred[start:end], dtype=float)

    if segment.shape[-1] < 4:
        segment = np.pad(segment, ((0, 0), (0, max(0, 4 - segment.shape[-1]))))

    centered = segment - segment.mean(axis=1, keepdims=True)
    window = np.hanning(centered.shape[-1])
    if not window.any():
        window = np.ones(centered.shape[-1])
    spectrum = np.fft.rfft(centered * window[None, :], axis=1)
    power_by_channel = np.abs(spectrum) ** 2
    power = power_by_channel.mean(axis=0)
    freqs = np.fft.rfftfreq(centered.shape[-1], d=1.0 / fs)

    freq_mask = (freqs >= 0.5) & (freqs <= 80.0)
    if freq_mask.any() and power[freq_mask].sum() > 0:
        dominant_freq = float(freqs[freq_mask][np.argmax(power[freq_mask])])
    else:
        dominant_freq = 0.0

    rhythmic_mask = (freqs >= 0.5) & (freqs <= 30.0)
    rhythmic_total = float(power[rhythmic_mask].sum())
    rhythmicity = float(power[rhythmic_mask].max() / rhythmic_total) if rhythmic_total > 0 else 0.0

    diff1 = np.diff(centered, axis=1)
    diff2 = np.diff(centered, n=2, axis=1)
    line_length = float(np.mean(np.abs(diff1))) if diff1.size else 0.0
    sharpness = float(np.mean(np.abs(diff2)) / (np.mean(np.abs(diff1)) + 1e-8)) if diff2.size else 0.0
    channel_rms = np.sqrt(np.mean(centered**2, axis=1))
    channel_focus = float(channel_rms.max() / (channel_rms.sum() + 1e-8))

    result = {
        "start_sample": int(start),
        "end_sample": int(end),
        "start_sec_in_window": float(start / fs),
        "end_sec_in_window": float(end / fs),
        "duration_sec": float((end - start) / fs),
        "label_fraction": float(label_segment.mean()) if label_segment.size else 0.0,
        "baseline_mean": float(baseline_segment.mean()) if baseline_segment.size else 0.0,
        "masked_mean": float(masked_segment.mean()) if masked_segment.size else 0.0,
        "latent_mean": float(latent_segment.mean()) if latent_segment.size else 0.0,
        "masked_delta": float(masked_segment.mean() - baseline_segment.mean()) if baseline_segment.size else 0.0,
        "latent_delta": float(latent_segment.mean() - baseline_segment.mean()) if baseline_segment.size else 0.0,
        "rms_amplitude": float(np.sqrt(np.mean(centered**2))),
        "peak_to_peak": float(np.percentile(centered, 99) - np.percentile(centered, 1)),
        "line_length": line_length,
        "sharpness_index": sharpness,
        "channel_focus_index": channel_focus,
        "dominant_freq_hz": dominant_freq,
        "rhythmicity_index": rhythmicity,
        "spectral_entropy": spectral_entropy(power, freqs),
    }
    result.update(band_power_fractions(power, freqs))
    primary, tags = clinical_candidate_tags(result)
    result["clinical_candidate"] = primary
    result["clinical_candidate_tags"] = ";".join(tags)
    result["phase_candidate"] = phase_candidate(result)
    return result


def clinical_candidate_tags(metrics: Dict[str, float]) -> Tuple[str, List[str]]:
    tags = []
    delta_theta = metrics.get("delta_frac", 0.0) + metrics.get("theta_frac", 0.0)
    beta_gamma = metrics.get("beta_frac", 0.0) + metrics.get("gamma_frac", 0.0) + metrics.get("high_frac", 0.0)
    dominant = metrics.get("dominant_freq_hz", 0.0)
    rhythmicity = metrics.get("rhythmicity_index", 0.0)
    entropy = metrics.get("spectral_entropy", 1.0)
    sharpness = metrics.get("sharpness_index", 0.0)

    if beta_gamma >= 0.55 and dominant >= 18.0:
        tags.append("fast_activity_or_artifact_candidate")
    if rhythmicity >= 0.18 and 2.0 <= dominant <= 12.0 and entropy <= 0.78:
        tags.append("rhythmic_discharge_candidate")
    if delta_theta >= 0.62 and dominant < 7.0:
        tags.append("slow_activity_candidate")
    if sharpness >= 1.35 and beta_gamma < 0.55:
        tags.append("sharp_transient_candidate")

    if not tags:
        tags.append("mixed_or_uncertain_candidate")
    return tags[0], tags


def phase_candidate(metrics: Dict[str, float]) -> str:
    label_fraction = metrics.get("label_fraction", 0.0)
    baseline = metrics.get("baseline_mean", 0.0)
    if label_fraction >= 0.5:
        return "ictal"
    if label_fraction > 0.0:
        return "seizure_edge_or_mixed"
    if baseline >= 0.5:
        return "model_positive_background"
    return "background"


def weighted_average(rows: List[Dict[str, object]], key: str, weight_key: str = "attribution_score") -> float:
    total = 0.0
    weight_total = 0.0
    for row in rows:
        value = row.get(key)
        weight = row.get(weight_key, 1.0)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            weight = float(weight) if isinstance(weight, (int, float)) and math.isfinite(float(weight)) else 1.0
            total += float(value) * max(weight, 1e-12)
            weight_total += max(weight, 1e-12)
    return total / weight_total if weight_total > 0 else float("nan")


def summarize_features(feature_rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    grouped = defaultdict(list)
    for row in feature_rows:
        grouped[int(row["latent_feature"])] .append(row)

    summary = []
    numeric_keys = [
        "attribution_score",
        "activation_mean_abs",
        "label_fraction",
        "baseline_mean",
        "masked_mean",
        "latent_mean",
        "masked_delta",
        "latent_delta",
        "rms_amplitude",
        "peak_to_peak",
        "line_length",
        "sharpness_index",
        "channel_focus_index",
        "dominant_freq_hz",
        "rhythmicity_index",
        "spectral_entropy",
    ] + [f"{name}_frac" for name, _, _ in BANDS]

    for feature, rows in grouped.items():
        tag_counts = Counter(row["clinical_candidate"] for row in rows)
        phase_counts = Counter(row["phase_candidate"] for row in rows)
        item = {
            "latent_feature": feature,
            "count": len(rows),
            "total_attribution_score": float(sum(float(row["attribution_score"]) for row in rows)),
            "primary_clinical_candidate": tag_counts.most_common(1)[0][0],
            "clinical_candidate_counts": json.dumps(dict(tag_counts), sort_keys=True),
            "primary_phase_candidate": phase_counts.most_common(1)[0][0],
            "phase_candidate_counts": json.dumps(dict(phase_counts), sort_keys=True),
        }
        for key in numeric_keys:
            item[f"mean_{key}"] = weighted_average(rows, key)
        summary.append(item)
    summary.sort(key=lambda row: row["total_attribution_score"], reverse=True)
    return summary


def load_examples(examples_dir: Path) -> List[Tuple[Path, Dict[str, object]]]:
    manifest_rows = read_jsonl(examples_dir / "manifest.jsonl")
    if manifest_rows:
        examples = []
        for row in manifest_rows:
            artifact = Path(str(row["artifact"]))
            if artifact.exists():
                examples.append((artifact, row))
        return examples
    return [(path, {"example_id": index, "artifact": str(path)}) for index, path in enumerate(sorted(examples_dir.glob("example_*.npz")))]


def analyze_example(
    npz_path: Path,
    metadata: Dict[str, object],
    args,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    data = np.load(npz_path)
    eeg = data["input"]
    labels = data["label"]
    baseline = data["baseline_prediction"]
    masked_pred = data["masked_prediction"]
    latent_pred = data["latent_prediction"]
    input_mask_time = data["input_mask_time"].astype(bool)
    input_time_attr = data["input_attribution_time"]
    latent_mask_time = data["latent_mask_time"].astype(bool)
    latent_time_attr = data["latent_attribution_time"]
    latent_channel_attr = data["latent_attribution_channel"]
    latent_attr = data["latent_attribution"]
    latent = data["latent"]
    fs = int(data["fs"]) if "fs" in data.files else args.fs
    target_samples = eeg.shape[-1]
    example_id = int(metadata.get("example_id", len(str(npz_path))))

    segment_specs = []
    for source_name, mask in [
        ("input_mask_selected", input_mask_time),
        ("latent_mask_selected", latent_mask_time),
    ]:
        ranges = map_mask_ranges(
            mask,
            target_samples=target_samples,
            fs=fs,
            min_duration_sec=args.min_segment_sec,
            merge_gap_sec=args.merge_gap_sec,
        )
        for start, end in ranges[: args.max_segments_per_source]:
            segment_specs.append((source_name, start, end))

    for source_name, scores in [
        ("top_input_attribution_context", input_time_attr),
        ("top_latent_attribution_context", latent_time_attr),
    ]:
        ranges = top_time_context_ranges(
            scores,
            target_samples=target_samples,
            fs=fs,
            top_k=args.top_time_contexts,
            context_sec=args.context_sec,
        )
        for start, end in ranges[: args.max_segments_per_source]:
            segment_specs.append((source_name, start, end))

    segment_rows = []
    seen = set()
    for source_name, start, end in segment_specs:
        key = (source_name, start, end)
        if key in seen or end <= start:
            continue
        seen.add(key)
        row = analyze_eeg_segment(eeg, labels, baseline, masked_pred, latent_pred, start, end, fs)
        row.update(
            {
                "example_id": example_id,
                "artifact": str(npz_path),
                "source": source_name,
                "edf": metadata.get("edf", ""),
            }
        )
        segment_rows.append(row)

    feature_rows = []
    top_features = np.argsort(latent_channel_attr)[-min(args.top_features, len(latent_channel_attr)) :][::-1]
    latent_step_samples = target_samples / latent.shape[-1]
    half_context = max(1, int(round(args.context_sec * fs / 2.0)))
    for feature in top_features:
        feature_attr = latent_attr[feature]
        top_step = int(np.argmax(feature_attr))
        center = int(round((top_step + 0.5) * latent_step_samples))
        start = max(0, center - half_context)
        end = min(target_samples, center + half_context)
        row = analyze_eeg_segment(eeg, labels, baseline, masked_pred, latent_pred, start, end, fs)
        row.update(
            {
                "example_id": example_id,
                "artifact": str(npz_path),
                "edf": metadata.get("edf", ""),
                "latent_feature": int(feature),
                "top_latent_step": top_step,
                "attribution_score": float(latent_channel_attr[feature]),
                "feature_peak_attribution": float(feature_attr[top_step]),
                "activation_mean": float(latent[feature, :].mean()),
                "activation_mean_abs": float(np.abs(latent[feature, :]).mean()),
                "activation_at_top_step": float(latent[feature, top_step]),
            }
        )
        feature_rows.append(row)

    return segment_rows, feature_rows


def write_feature_cards(path: Path, feature_summary: List[Dict[str, object]], limit: int):
    lines = [
        "# Clinical MI Feature Cards",
        "",
        "이 파일은 latent feature 번호를 임상 EEG 후보 의미로 번역하기 위한 요약입니다.",
        "태그는 자동 휴리스틱이므로 진단명이 아니라 임상의 검토를 위한 후보입니다.",
        "",
    ]
    for row in feature_summary[:limit]:
        bands = ", ".join(
            f"{name}={row.get(f'mean_{name}_frac', float('nan')):.2f}"
            for name, _, _ in BANDS
        )
        lines.extend(
            [
                f"## Latent feature {row['latent_feature']}",
                "",
                f"- Primary clinical candidate: `{row['primary_clinical_candidate']}`",
                f"- Primary phase candidate: `{row['primary_phase_candidate']}`",
                f"- Count: {row['count']}",
                f"- Total attribution score: {row['total_attribution_score']:.6g}",
                f"- Mean dominant frequency: {row['mean_dominant_freq_hz']:.2f} Hz",
                f"- Mean rhythmicity index: {row['mean_rhythmicity_index']:.3f}",
                f"- Mean label fraction: {row['mean_label_fraction']:.3f}",
                f"- Mean latent intervention delta: {row['mean_latent_delta']:.3f}",
                f"- Band fractions: {bands}",
                "",
                "Interpretation:",
                "- 이 feature가 반복적으로 어떤 EEG 구간에서 켜지는지 top examples를 임상의가 확인해야 합니다.",
                "- `latent_delta`가 음수이면 이 feature 주변 구간을 suppress했을 때 seizure probability가 내려간 경향입니다.",
                "- `fast_activity_or_artifact_candidate`는 artifact 가능성을, `rhythmic_discharge_candidate`는 ictal rhythmic pattern 가능성을 우선 검토합니다.",
                "",
            ]
        )
    path.write_text("\n".join(lines))


def write_guide(path: Path):
    path.write_text(
        """# Clinical MI Visualization Guide

이 분석은 latent feature 번호를 직접 해석하지 않고, 그 feature가 반복적으로 반응한 EEG 구간의 morphology를 요약합니다.

## clinical_candidate

- `rhythmic_discharge_candidate`: 특정 주파수대 에너지가 비교적 집중되어 있고 rhythmicity가 높은 구간입니다. 임상적으로 ictal rhythmic discharge 가능성을 검토할 후보입니다.
- `slow_activity_candidate`: delta/theta 비중이 높은 구간입니다. slowing, postictal slowing, 또는 저주파 ictal activity 후보입니다.
- `fast_activity_or_artifact_candidate`: beta/gamma/high-frequency 비중이 높은 구간입니다. fast activity 또는 muscle/electrode artifact 가능성을 검토합니다.
- `sharp_transient_candidate`: 변화가 급격한 transient가 두드러진 후보입니다. spike/sharp-like morphology인지 확인합니다.
- `mixed_or_uncertain_candidate`: 자동 지표만으로 명확하지 않은 구간입니다.

## 주요 figure의 의미

1. `01_clinical_tag_counts`
   - top latent feature들이 어떤 임상 EEG 후보 태그에 많이 연결되는지 보여줍니다.
   - artifact 후보가 많으면 모델이 artifact에 민감할 가능성을 점검해야 합니다.

2. `02_feature_bandpower_heatmap`
   - latent feature별로 어떤 frequency band의 EEG 구간과 연결되는지 보여줍니다.
   - delta/theta가 높으면 slow/rhythmic pattern, gamma/high가 높으면 fast activity 또는 artifact 후보입니다.

3. `03_feature_phase_and_effect`
   - x축은 해당 feature가 연결된 구간이 seizure label과 얼마나 겹치는지, y축은 latent intervention 후 prediction 변화입니다.
   - 오른쪽 아래 방향, 즉 ictal overlap이 높고 intervention delta가 음수이면 seizure evidence feature 후보입니다.

4. `04_prediction_effect_by_clinical_tag`
   - clinical candidate별로 latent intervention이 prediction을 올렸는지 내렸는지 보여줍니다.
   - 특정 태그를 suppress했을 때 FP가 줄거나 seizure prediction이 줄면, 그 태그가 모델 행동에 causal하게 연결되어 있을 가능성이 있습니다.

주의: 이 결과는 임상 진단이 아니라 모델 해석용 후보 생성입니다. 최종 임상 의미는 EEG 전문의의 morphology review와 함께 판단해야 합니다.
"""
    )


def run_analysis(args):
    examples_dir = Path(args.examples_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    examples = load_examples(examples_dir)
    if not examples:
        raise FileNotFoundError(f"No MI example artifacts found in {examples_dir}")

    segment_rows = []
    feature_rows = []
    for npz_path, metadata in examples:
        if args.max_examples is not None and len({row.get("example_id") for row in feature_rows}) >= args.max_examples:
            break
        print(f"Analyzing {npz_path}")
        segments, features = analyze_example(npz_path, metadata, args)
        segment_rows.extend(segments)
        feature_rows.extend(features)

    feature_summary = summarize_features(feature_rows)
    write_csv(output_dir / "clinical_mi_segments.csv", segment_rows)
    write_csv(output_dir / "clinical_mi_feature_examples.csv", feature_rows)
    write_csv(output_dir / "clinical_mi_feature_summary.csv", feature_summary)
    write_feature_cards(output_dir / "clinical_mi_feature_cards.md", feature_summary, args.feature_card_limit)
    write_guide(output_dir / "clinical_mi_figure_guide.md")
    with (output_dir / "clinical_mi_summary.json").open("w") as fp:
        json.dump(
            {
                "num_examples": len(examples),
                "num_segments": len(segment_rows),
                "num_feature_rows": len(feature_rows),
                "num_feature_summary_rows": len(feature_summary),
                "bands": BANDS,
                "note": "Clinical candidate tags are heuristic model-interpretation aids, not diagnoses.",
            },
            fp,
            indent=2,
        )
    print(f"Saved clinical MI analysis to {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="Bridge MI example artifacts to clinical EEG candidate patterns.")
    parser.add_argument("--examples_dir", type=str, default=str(ROOT / "mi_results/examples"))
    parser.add_argument("--output_dir", type=str, default=str(ROOT / "mi_results/clinical_mi"))
    parser.add_argument("--fs", type=int, default=256)
    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument("--top_features", type=int, default=16)
    parser.add_argument("--top_time_contexts", type=int, default=6)
    parser.add_argument("--context_sec", type=float, default=3.0)
    parser.add_argument("--min_segment_sec", type=float, default=0.4)
    parser.add_argument("--merge_gap_sec", type=float, default=0.35)
    parser.add_argument("--max_segments_per_source", type=int, default=8)
    parser.add_argument("--feature_card_limit", type=int, default=20)
    return parser.parse_args()


if __name__ == "__main__":
    run_analysis(parse_args())
