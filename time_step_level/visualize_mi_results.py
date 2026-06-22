import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional


ROOT = Path(__file__).resolve().parent

METHOD_LABELS = {
    "attribution_only": "Attribution only",
    "attribution_guided_masking": "Attribution-guided masking",
    "mi_guided_latent_intervention": "MI-guided latent intervention",
}

METHOD_COLORS = {
    "attribution_only": "#3B6EA8",
    "attribution_guided_masking": "#D18F2F",
    "mi_guided_latent_intervention": "#3E8F6B",
}


def load_summary(path: Path) -> Dict[str, Dict[str, Dict[str, float]]]:
    with path.open() as fp:
        data = json.load(fp)
    methods = data.get("methods", data)
    if not isinstance(methods, dict):
        raise ValueError(f"Cannot find method results in {path}")
    return methods


def load_per_recording(path: Optional[Path]) -> List[Dict[str, object]]:
    if path is None or not path.exists():
        return []
    rows = []
    with path.open(newline="") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            parsed = {}
            for key, value in row.items():
                if key in ("file", "method"):
                    parsed[key] = value
                    continue
                try:
                    parsed[key] = float(value)
                except (TypeError, ValueError):
                    parsed[key] = value
            rows.append(parsed)
    return rows


def method_order(methods: Dict[str, object]) -> List[str]:
    preferred = [
        "attribution_only",
        "attribution_guided_masking",
        "mi_guided_latent_intervention",
    ]
    ordered = [method for method in preferred if method in methods]
    ordered.extend(method for method in methods if method not in ordered)
    return ordered


def label(method: str) -> str:
    return METHOD_LABELS.get(method, method.replace("_", " ").title())


def color(method: str) -> str:
    return METHOD_COLORS.get(method, "#6B7280")


def get_metric(methods: Dict[str, object], method: str, group: str, metric: str) -> float:
    value = methods.get(method, {}).get(group, {}).get(metric, float("nan"))
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def finite_values(values: Iterable[float]) -> List[float]:
    return [value for value in values if isinstance(value, (int, float)) and math.isfinite(value)]


def annotate_bars(ax, bars, fmt="{:.3f}", rotation=0):
    for bar in bars:
        height = bar.get_height()
        if not math.isfinite(height):
            continue
        ax.annotate(
            fmt.format(height),
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
            rotation=rotation,
        )


def style_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#D8DEE9", linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)


def save_figure(fig, output_dir: Path, name: str, dpi: int):
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_dir / f"{name}.png", dpi=dpi, bbox_inches="tight")
    fig.savefig(output_dir / f"{name}.pdf", bbox_inches="tight")


def plot_detection_scores(methods, order, output_dir, dpi):
    import matplotlib.pyplot as plt
    import numpy as np

    score_metrics = ["f1", "sensitivity", "precision"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    width = 0.22
    x = np.arange(len(score_metrics))

    for ax, group in zip(axes, ["sample", "event"]):
        for idx, method in enumerate(order):
            offset = (idx - (len(order) - 1) / 2) * width
            values = [get_metric(methods, method, group, metric) for metric in score_metrics]
            bars = ax.bar(
                x + offset,
                values,
                width=width,
                label=label(method),
                color=color(method),
                edgecolor="white",
                linewidth=0.8,
            )
            annotate_bars(ax, bars)
        ax.set_title(f"{group.title()} detection scores")
        ax.set_xticks(x)
        ax.set_xticklabels([metric.replace("_", " ").title() for metric in score_metrics])
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Score")
        style_axis(ax)

    axes[1].legend(frameon=False, loc="upper right")
    save_figure(fig, output_dir, "01_detection_scores", dpi)
    plt.close(fig)


def plot_fp_rates(methods, order, output_dir, dpi):
    import matplotlib.pyplot as plt
    import numpy as np

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    x = np.arange(len(order))

    for ax, group in zip(axes, ["sample", "event"]):
        values = [get_metric(methods, method, group, "fpRate") for method in order]
        bars = ax.bar(
            x,
            values,
            color=[color(method) for method in order],
            edgecolor="white",
            linewidth=0.8,
        )
        annotate_bars(ax, bars, fmt="{:.1f}", rotation=25)
        ax.set_title(f"{group.title()} false-positive rate")
        ax.set_xticks(x)
        ax.set_xticklabels([label(method) for method in order], rotation=18, ha="right")
        ax.set_yscale("log")
        ax.set_ylabel("FP rate, log scale")
        style_axis(ax)

    save_figure(fig, output_dir, "02_false_positive_rates", dpi)
    plt.close(fig)


def plot_prediction_delta(methods, order, output_dir, dpi):
    import matplotlib.pyplot as plt
    import numpy as np

    metrics = [
        ("mean_delta", "Mean delta"),
        ("mean_delta_on_background", "Background delta"),
        ("mean_delta_on_seizure", "Seizure delta"),
        ("mean_abs_delta", "Abs delta"),
        ("mean_abs_delta_on_background", "Background abs delta"),
        ("mean_abs_delta_on_seizure", "Seizure abs delta"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))

    signed = metrics[:3]
    absolute = metrics[3:]
    width = 0.22

    for ax, selected, title in zip(
        axes,
        [signed, absolute],
        ["Signed prediction shift", "Magnitude of prediction shift"],
    ):
        x = np.arange(len(selected))
        for idx, method in enumerate(order):
            offset = (idx - (len(order) - 1) / 2) * width
            values = [get_metric(methods, method, "delta", metric) for metric, _ in selected]
            bars = ax.bar(
                x + offset,
                values,
                width=width,
                label=label(method),
                color=color(method),
                edgecolor="white",
                linewidth=0.8,
            )
            annotate_bars(ax, bars, fmt="{:.3f}")
        ax.axhline(0, color="#2E3440", linewidth=0.9)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels([name for _, name in selected], rotation=18, ha="right")
        ax.set_ylabel("Prediction probability change")
        style_axis(ax)

    axes[1].legend(frameon=False, loc="upper left")
    save_figure(fig, output_dir, "03_prediction_delta", dpi)
    plt.close(fig)


def plot_attribution_diagnostics(methods, order, output_dir, dpi):
    import matplotlib.pyplot as plt
    import numpy as np

    metrics = [
        ("attribution_label_auc", "Attribution-label AUC"),
        ("attribution_mass_on_seizure", "Mass on seizure"),
        ("normalized_attribution_entropy", "Normalized entropy"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    x = np.arange(len(order))

    label_rate_values = [
        get_metric(methods, method, "attribution", "label_positive_rate")
        for method in order
    ]
    label_rate = finite_values(label_rate_values)
    label_rate = label_rate[0] if label_rate else None

    for ax, (metric, title) in zip(axes, metrics):
        values = [get_metric(methods, method, "attribution", metric) for method in order]
        bars = ax.bar(
            x,
            values,
            color=[color(method) for method in order],
            edgecolor="white",
            linewidth=0.8,
        )
        annotate_bars(ax, bars)
        if metric == "attribution_mass_on_seizure" and label_rate is not None:
            ax.axhline(
                label_rate,
                color="#BF616A",
                linestyle="--",
                linewidth=1.2,
                label="Label positive rate",
            )
            ax.legend(frameon=False, fontsize=8)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels([label(method) for method in order], rotation=18, ha="right")
        ax.set_ylim(0, min(1.05, max(1.0, max(finite_values(values + ([label_rate] if label_rate else [])), default=1.0) * 1.2)))
        style_axis(ax)

    save_figure(fig, output_dir, "04_attribution_diagnostics", dpi)
    plt.close(fig)


def plot_precision_sensitivity_tradeoff(methods, order, output_dir, dpi):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
    for ax, group in zip(axes, ["sample", "event"]):
        for method in order:
            precision = get_metric(methods, method, group, "precision")
            sensitivity = get_metric(methods, method, group, "sensitivity")
            f1 = get_metric(methods, method, group, "f1")
            fp_rate = get_metric(methods, method, group, "fpRate")
            if not all(math.isfinite(value) for value in [precision, sensitivity, f1]):
                continue
            size = 120
            if math.isfinite(fp_rate):
                size = max(80, min(420, 80 + math.log10(max(fp_rate, 1.0)) * 80))
            ax.scatter(
                sensitivity,
                precision,
                s=size,
                color=color(method),
                alpha=0.85,
                edgecolor="white",
                linewidth=1.2,
                label=label(method),
            )
            ax.annotate(
                f"F1={f1:.3f}",
                (sensitivity, precision),
                xytext=(6, 5),
                textcoords="offset points",
                fontsize=8,
            )
        ax.set_title(f"{group.title()} precision-sensitivity trade-off")
        ax.set_xlabel("Sensitivity")
        ax.set_ylabel("Precision")
        ax.set_xlim(0, 1.02)
        ax.set_ylim(0, 1.02)
        style_axis(ax)

    axes[1].legend(frameon=False, loc="lower left")
    save_figure(fig, output_dir, "05_precision_sensitivity_tradeoff", dpi)
    plt.close(fig)


def values_by_method(rows, order, metric):
    grouped = {method: [] for method in order}
    for row in rows:
        method = row.get("method")
        value = row.get(metric)
        if method in grouped and isinstance(value, (int, float)) and math.isfinite(value):
            grouped[method].append(value)
    return grouped


def plot_per_recording_boxplots(rows, order, output_dir, dpi):
    if not rows:
        return

    import matplotlib.pyplot as plt

    box_metrics = [
        ("event_f1", "Per-recording event F1", "Event F1"),
        ("sample_f1", "Per-recording sample F1", "Sample F1"),
        ("mean_delta_on_seizure", "Per-recording seizure prediction shift", "Mean delta on seizure"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))

    for ax, (metric, title, ylabel) in zip(axes, box_metrics):
        grouped = values_by_method(rows, order, metric)
        data = [grouped[method] for method in order]
        if not any(data):
            ax.set_visible(False)
            continue
        bp = ax.boxplot(data, patch_artist=True, labels=[label(method) for method in order])
        for patch, method in zip(bp["boxes"], order):
            patch.set_facecolor(color(method))
            patch.set_alpha(0.75)
        for median in bp["medians"]:
            median.set_color("#2E3440")
            median.set_linewidth(1.5)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", labelrotation=18)
        style_axis(ax)

    save_figure(fig, output_dir, "06_per_recording_distributions", dpi)
    plt.close(fig)


def write_markdown_summary(methods, order, output_dir):
    lines = [
        "# MI Results Visualization Summary",
        "",
        "| Method | Sample F1 | Event F1 | Event Sens. | Event Prec. | Event FP rate | Mean delta seizure | Attr AUC | Attr mass / label rate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in order:
        label_rate = get_metric(methods, method, "attribution", "label_positive_rate")
        mass = get_metric(methods, method, "attribution", "attribution_mass_on_seizure")
        enrichment = mass / label_rate if label_rate and math.isfinite(label_rate) else float("nan")
        lines.append(
            "| {method} | {sample_f1:.3f} | {event_f1:.3f} | {event_sens:.3f} | "
            "{event_prec:.3f} | {event_fp:.2f} | {seizure_delta:.3f} | "
            "{attr_auc:.3f} | {enrichment:.2f}x |".format(
                method=label(method),
                sample_f1=get_metric(methods, method, "sample", "f1"),
                event_f1=get_metric(methods, method, "event", "f1"),
                event_sens=get_metric(methods, method, "event", "sensitivity"),
                event_prec=get_metric(methods, method, "event", "precision"),
                event_fp=get_metric(methods, method, "event", "fpRate"),
                seizure_delta=get_metric(methods, method, "delta", "mean_delta_on_seizure"),
                attr_auc=get_metric(methods, method, "attribution", "attribution_label_auc"),
                enrichment=enrichment,
            )
        )
    (output_dir / "figure_summary.md").write_text("\n".join(lines) + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize mi_experiments.py summary outputs.")
    parser.add_argument("--summary", type=str, default=str(ROOT / "mi_results/summary.json"))
    parser.add_argument("--per_recording", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=str(ROOT / "mi_results/figures"))
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def main():
    args = parse_args()
    summary_path = Path(args.summary)
    output_dir = Path(args.output_dir)
    per_recording_path = Path(args.per_recording) if args.per_recording else summary_path.parent / "per_recording_results.csv"

    try:
        import matplotlib  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "matplotlib is required for visualization. Install it with `pip install matplotlib` "
            "or run inside the full time_step_level environment."
        ) from exc

    methods = load_summary(summary_path)
    order = method_order(methods)
    rows = load_per_recording(per_recording_path)

    output_dir.mkdir(parents=True, exist_ok=True)
    plot_detection_scores(methods, order, output_dir, args.dpi)
    plot_fp_rates(methods, order, output_dir, args.dpi)
    plot_prediction_delta(methods, order, output_dir, args.dpi)
    plot_attribution_diagnostics(methods, order, output_dir, args.dpi)
    plot_precision_sensitivity_tradeoff(methods, order, output_dir, args.dpi)
    plot_per_recording_boxplots(rows, order, output_dir, args.dpi)
    write_markdown_summary(methods, order, output_dir)

    print(f"Saved figures to {output_dir}")


if __name__ == "__main__":
    main()
