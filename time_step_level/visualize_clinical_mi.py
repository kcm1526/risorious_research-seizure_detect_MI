import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List


ROOT = Path(__file__).resolve().parent
BAND_KEYS = ("delta", "theta", "alpha", "beta", "gamma", "high")
TAG_COLORS = {
    "rhythmic_discharge_candidate": "#3B6EA8",
    "slow_activity_candidate": "#7A5195",
    "fast_activity_or_artifact_candidate": "#D18F2F",
    "sharp_transient_candidate": "#BF616A",
    "mixed_or_uncertain_candidate": "#6B7280",
}


def read_csv_rows(path: Path) -> List[Dict[str, object]]:
    if not path.exists():
        return []
    rows = []
    with path.open(newline="") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            parsed = {}
            for key, value in row.items():
                if value is None or value == "":
                    parsed[key] = value
                    continue
                try:
                    parsed[key] = float(value)
                except ValueError:
                    parsed[key] = value
            rows.append(parsed)
    return rows


def finite(value, default=0.0):
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return default


def style_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#D8DEE9", linewidth=0.8, alpha=0.75)
    ax.set_axisbelow(True)


def tag_color(tag: str) -> str:
    return TAG_COLORS.get(str(tag), "#6B7280")


def save_figure(fig, output_dir: Path, name: str, dpi: int):
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_dir / f"{name}.png", dpi=dpi, bbox_inches="tight")
    fig.savefig(output_dir / f"{name}.pdf", bbox_inches="tight")


def top_summary_rows(rows: List[Dict[str, object]], top_n: int) -> List[Dict[str, object]]:
    return sorted(rows, key=lambda row: finite(row.get("total_attribution_score")), reverse=True)[:top_n]


def plot_tag_counts(feature_summary, segment_rows, output_dir, dpi):
    import matplotlib.pyplot as plt

    feature_counter = Counter(str(row.get("primary_clinical_candidate", "")) for row in feature_summary)
    segment_counter = Counter(str(row.get("clinical_candidate", "")) for row in segment_rows)
    tags = sorted(set(feature_counter) | set(segment_counter))
    x = range(len(tags))

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
    for ax, counter, title in [
        (axes[0], feature_counter, "Latent feature clinical candidates"),
        (axes[1], segment_counter, "Selected EEG segment clinical candidates"),
    ]:
        values = [counter[tag] for tag in tags]
        ax.bar(x, values, color=[tag_color(tag) for tag in tags], edgecolor="white", linewidth=0.8)
        ax.set_xticks(list(x))
        ax.set_xticklabels(tags, rotation=25, ha="right")
        ax.set_ylabel("Count")
        ax.set_title(title)
        style_axis(ax)
    save_figure(fig, output_dir, "01_clinical_tag_counts", dpi)
    plt.close(fig)


def plot_bandpower_heatmap(feature_summary, output_dir, dpi, top_n):
    import matplotlib.pyplot as plt
    import numpy as np

    rows = top_summary_rows(feature_summary, top_n)
    if not rows:
        return
    matrix = []
    labels = []
    for row in rows:
        matrix.append([finite(row.get(f"mean_{band}_frac")) for band in BAND_KEYS])
        labels.append(f"F{int(row['latent_feature'])}")
    matrix = np.asarray(matrix)

    fig, ax = plt.subplots(figsize=(10, max(5, len(rows) * 0.32)))
    image = ax.imshow(matrix, aspect="auto", cmap="YlGnBu", vmin=0, vmax=max(0.05, matrix.max()))
    ax.set_xticks(range(len(BAND_KEYS)))
    ax.set_xticklabels(BAND_KEYS)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_title("Top latent features mapped to EEG frequency bands")
    ax.set_xlabel("EEG frequency band fraction")
    ax.set_ylabel("Latent feature")
    cbar = fig.colorbar(image, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Weighted mean band fraction")
    save_figure(fig, output_dir, "02_feature_bandpower_heatmap", dpi)
    plt.close(fig)


def plot_phase_and_effect(feature_summary, output_dir, dpi, top_n):
    import matplotlib.pyplot as plt

    rows = top_summary_rows(feature_summary, top_n)
    if not rows:
        return

    fig, ax = plt.subplots(figsize=(9, 6.2))
    for row in rows:
        tag = str(row.get("primary_clinical_candidate", ""))
        x = finite(row.get("mean_label_fraction"))
        y = finite(row.get("mean_latent_delta"))
        size = max(80, min(600, 100 + math.log10(finite(row.get("total_attribution_score"), 1e-8) + 1e-8) * 70 + 600))
        ax.scatter(x, y, s=size, color=tag_color(tag), alpha=0.82, edgecolor="white", linewidth=1.0)
        ax.annotate(f"F{int(row['latent_feature'])}", (x, y), xytext=(5, 4), textcoords="offset points", fontsize=8)

    ax.axhline(0, color="#2E3440", linewidth=0.9)
    ax.axvline(0.5, color="#4C566A", linestyle="--", linewidth=0.9)
    ax.set_title("Clinical phase alignment vs latent intervention effect")
    ax.set_xlabel("Mean seizure-label overlap of linked EEG segments")
    ax.set_ylabel("Mean prediction change after latent intervention")
    style_axis(ax)

    legend_handles = []
    for tag, color in TAG_COLORS.items():
        legend_handles.append(plt.Line2D([0], [0], marker="o", color="w", label=tag, markerfacecolor=color, markersize=8))
    ax.legend(handles=legend_handles, frameon=False, fontsize=8, loc="best")
    save_figure(fig, output_dir, "03_feature_phase_and_effect", dpi)
    plt.close(fig)


def mean_by_tag(rows: Iterable[Dict[str, object]], value_key: str, tag_key: str) -> Dict[str, float]:
    sums = defaultdict(float)
    counts = defaultdict(float)
    for row in rows:
        tag = str(row.get(tag_key, ""))
        value = row.get(value_key)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            sums[tag] += float(value)
            counts[tag] += 1.0
    return {tag: sums[tag] / counts[tag] for tag in sums if counts[tag] > 0}


def plot_prediction_effect_by_tag(segment_rows, feature_rows, output_dir, dpi):
    import matplotlib.pyplot as plt
    import numpy as np

    segment_latent = mean_by_tag(segment_rows, "latent_delta", "clinical_candidate")
    feature_latent = mean_by_tag(feature_rows, "latent_delta", "clinical_candidate")
    segment_masked = mean_by_tag(segment_rows, "masked_delta", "clinical_candidate")
    tags = sorted(set(segment_latent) | set(feature_latent) | set(segment_masked))
    if not tags:
        return

    x = np.arange(len(tags))
    width = 0.25
    fig, ax = plt.subplots(figsize=(12, 5.4))
    ax.bar(x - width, [segment_masked.get(tag, 0.0) for tag in tags], width=width, color="#D18F2F", label="Input masking delta")
    ax.bar(x, [segment_latent.get(tag, 0.0) for tag in tags], width=width, color="#3E8F6B", label="Latent intervention delta, selected segments")
    ax.bar(x + width, [feature_latent.get(tag, 0.0) for tag in tags], width=width, color="#3B6EA8", label="Latent intervention delta, top features")
    ax.axhline(0, color="#2E3440", linewidth=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(tags, rotation=25, ha="right")
    ax.set_ylabel("Mean prediction change vs original")
    ax.set_title("Prediction effect grouped by clinical EEG candidate")
    ax.legend(frameon=False)
    style_axis(ax)
    save_figure(fig, output_dir, "04_prediction_effect_by_clinical_tag", dpi)
    plt.close(fig)


def plot_feature_score_table(feature_summary, output_dir, top_n):
    rows = top_summary_rows(feature_summary, top_n)
    lines = [
        "# Top Clinical MI Latent Features",
        "",
        "| Feature | Clinical candidate | Phase | Attr score | Label overlap | Dominant Hz | Rhythmicity | Latent delta | Band signature |",
        "|---:|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        bands = sorted(
            [(band, finite(row.get(f"mean_{band}_frac"))) for band in BAND_KEYS],
            key=lambda item: item[1],
            reverse=True,
        )[:2]
        signature = ", ".join(f"{band} {value:.2f}" for band, value in bands)
        lines.append(
            "| {feature} | `{tag}` | `{phase}` | {score:.4g} | {label:.2f} | {freq:.2f} | {rhythm:.2f} | {delta:.3f} | {signature} |".format(
                feature=int(row["latent_feature"]),
                tag=row.get("primary_clinical_candidate", ""),
                phase=row.get("primary_phase_candidate", ""),
                score=finite(row.get("total_attribution_score")),
                label=finite(row.get("mean_label_fraction")),
                freq=finite(row.get("mean_dominant_freq_hz")),
                rhythm=finite(row.get("mean_rhythmicity_index")),
                delta=finite(row.get("mean_latent_delta")),
                signature=signature,
            )
        )
    (output_dir / "clinical_mi_top_feature_table.md").write_text("\n".join(lines) + "\n")


def write_visualization_guide(output_dir: Path):
    guide = """# Clinical MI Visualization Meaning

이 visualization은 latent feature 번호를 임상적으로 해석 가능한 EEG 후보 패턴과 연결하기 위한 보조 자료입니다.

## 01_clinical_tag_counts

Top latent features와 selected EEG segments가 어떤 임상 후보 태그로 분류되는지 보여줍니다.
예를 들어 `fast_activity_or_artifact_candidate`가 많으면 모델 내부 feature가 artifact-like fast activity에 민감할 가능성을 점검해야 합니다.

## 02_feature_bandpower_heatmap

각 latent feature가 주로 어떤 EEG frequency band와 연결되는지 보여줍니다.
delta/theta가 높으면 slow activity 또는 low-frequency rhythmic pattern 후보이고, beta/gamma/high가 높으면 fast activity 또는 artifact 후보입니다.

## 03_feature_phase_and_effect

x축은 해당 feature가 연결된 EEG 구간이 seizure label과 얼마나 겹치는지입니다.
y축은 latent intervention 후 seizure probability 변화입니다.

- 오른쪽 아래: seizure 구간과 잘 겹치고, suppress 시 prediction이 내려가는 feature입니다. seizure-evidence feature 후보입니다.
- 왼쪽 아래: label과는 덜 겹치지만 suppress 시 prediction이 내려가는 feature입니다. false positive 또는 artifact-driven evidence 후보일 수 있습니다.
- 위쪽: suppress 후 prediction이 올라간 경우라서 inhibitory 또는 복잡한 보상 효과를 의심합니다.

## 04_prediction_effect_by_clinical_tag

임상 후보 태그별로 input masking 또는 latent intervention이 prediction을 얼마나 바꿨는지 보여줍니다.
특정 태그에서 delta가 음수이면, 그 후보 패턴이 모델의 seizure prediction을 지탱하고 있었을 가능성이 있습니다.

주의: 모든 clinical candidate는 자동 휴리스틱입니다. 진단명이 아니며, EEG 전문의의 morphology review를 위한 후보 생성으로 사용해야 합니다.
"""
    (output_dir / "clinical_mi_visualization_guide.md").write_text(guide)


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize clinical-bridge MI analysis outputs.")
    parser.add_argument("--analysis_dir", type=str, default=str(ROOT / "mi_results/clinical_mi"))
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--top_n", type=int, default=25)
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        import matplotlib  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "matplotlib is required for visualization. Install it with `pip install matplotlib` "
            "or run inside the full time_step_level environment."
        ) from exc

    analysis_dir = Path(args.analysis_dir)
    output_dir = Path(args.output_dir) if args.output_dir else analysis_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    feature_summary = read_csv_rows(analysis_dir / "clinical_mi_feature_summary.csv")
    feature_rows = read_csv_rows(analysis_dir / "clinical_mi_feature_examples.csv")
    segment_rows = read_csv_rows(analysis_dir / "clinical_mi_segments.csv")
    if not feature_summary and not segment_rows:
        raise FileNotFoundError(f"No clinical MI CSV outputs found in {analysis_dir}")

    plot_tag_counts(feature_summary, segment_rows, output_dir, args.dpi)
    plot_bandpower_heatmap(feature_summary, output_dir, args.dpi, args.top_n)
    plot_phase_and_effect(feature_summary, output_dir, args.dpi, args.top_n)
    plot_prediction_effect_by_tag(segment_rows, feature_rows, output_dir, args.dpi)
    plot_feature_score_table(feature_summary, output_dir, args.top_n)
    write_visualization_guide(output_dir)
    print(f"Saved clinical MI figures to {output_dir}")


if __name__ == "__main__":
    main()
