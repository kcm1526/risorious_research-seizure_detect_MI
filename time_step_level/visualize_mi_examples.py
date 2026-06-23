import argparse
import json
import math
from pathlib import Path
from typing import Dict, List

import numpy as np


ROOT = Path(__file__).resolve().parent


def load_manifest(path: Path) -> List[Dict[str, object]]:
    if not path.exists():
        return []
    rows = []
    with path.open() as fp:
        for line in fp:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def finite_percentile(values: np.ndarray, percentile: float, fallback: float = 1.0) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return fallback
    value = float(np.percentile(np.abs(finite), percentile))
    return value if value > 0 else fallback


def mask_ranges(mask: np.ndarray) -> List[tuple]:
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


def shade_mask_ranges(ax, mask: np.ndarray, total_seconds: float, color: str, label: str):
    if mask.size == 0:
        return
    scale = total_seconds / mask.size
    first = True
    for start, end in mask_ranges(mask):
        ax.axvspan(
            start * scale,
            end * scale,
            color=color,
            alpha=0.18,
            linewidth=0,
            label=label if first else None,
        )
        first = False


def plot_top_latent_channels(ax, scores: np.ndarray, mask: np.ndarray, top_k: int):
    scores = np.asarray(scores)
    if scores.size == 0:
        ax.set_visible(False)
        return
    top_k = min(top_k, scores.size)
    indices = np.argsort(scores)[-top_k:][::-1]
    colors = ["#BF616A" if mask[index] else "#3E8F6B" for index in indices]
    ax.bar(np.arange(top_k), scores[indices], color=colors, edgecolor="white", linewidth=0.8)
    ax.set_xticks(np.arange(top_k))
    ax.set_xticklabels([str(int(index)) for index in indices], rotation=45, ha="right")
    ax.set_title("Top latent feature dimensions")
    ax.set_ylabel("Mean attribution")
    ax.set_xlabel("Latent feature index")
    style_axis(ax)


def style_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#D8DEE9", linewidth=0.8, alpha=0.7)
    ax.set_axisbelow(True)


def add_colorbar(fig, image, ax, label: str):
    cbar = fig.colorbar(image, ax=ax, fraction=0.018, pad=0.01)
    cbar.set_label(label)


def plot_example(npz_path: Path, output_dir: Path, dpi: int, top_k: int):
    import matplotlib.pyplot as plt

    data = np.load(npz_path)
    eeg = data["input"]
    labels = data["label"]
    baseline = data["baseline_prediction"]
    input_attr = data["input_attribution"]
    input_time_attr = data["input_attribution_time"]
    input_mask_time = data["input_mask_time"].astype(bool)
    masked_pred = data["masked_prediction"]
    latent_attr = data["latent_attribution"]
    latent_time_attr = data["latent_attribution_time"]
    latent_channel_attr = data["latent_attribution_channel"]
    latent_mask_time = data["latent_mask_time"].astype(bool)
    latent_mask_channel = data["latent_mask_channel"].astype(bool)
    latent_pred = data["latent_prediction"]
    fs = int(data["fs"]) if "fs" in data.files else 256

    channels, samples = eeg.shape
    seconds = np.arange(samples) / fs
    total_seconds = samples / fs
    latent_seconds = np.linspace(0, total_seconds, latent_time_attr.shape[0], endpoint=False)

    fig, axes = plt.subplots(
        6,
        1,
        figsize=(16, 15),
        gridspec_kw={"height_ratios": [1.25, 1.35, 0.7, 1.1, 1.35, 0.95]},
    )

    eeg_limit = finite_percentile(eeg, 98)
    image = axes[0].imshow(
        eeg,
        aspect="auto",
        cmap="RdBu_r",
        vmin=-eeg_limit,
        vmax=eeg_limit,
        extent=[0, total_seconds, channels - 0.5, -0.5],
    )
    axes[0].set_title("Input EEG window")
    axes[0].set_ylabel("EEG channel")
    shade_mask_ranges(axes[0], input_mask_time, total_seconds, "#D18F2F", "Masked input time")
    add_colorbar(fig, image, axes[0], "z-scored signal")

    attr_limit = finite_percentile(input_attr, 99)
    image = axes[1].imshow(
        input_attr,
        aspect="auto",
        cmap="magma",
        vmin=0,
        vmax=attr_limit,
        extent=[0, total_seconds, channels - 0.5, -0.5],
    )
    axes[1].set_title("Input attribution heatmap: abs(gradient * input)")
    axes[1].set_ylabel("EEG channel")
    shade_mask_ranges(axes[1], input_mask_time, total_seconds, "#D18F2F", "Selected for masking")
    axes[1].legend(frameon=False, loc="upper right")
    add_colorbar(fig, image, axes[1], "Input attribution")

    axes[2].plot(seconds, input_time_attr, color="#D18F2F", linewidth=1.4)
    shade_mask_ranges(axes[2], input_mask_time, total_seconds, "#D18F2F", "Selected input mask")
    axes[2].set_title("Input attribution collapsed over channels")
    axes[2].set_ylabel("Mean attr.")
    axes[2].legend(frameon=False, loc="upper right")
    style_axis(axes[2])

    axes[3].plot(seconds, baseline, color="#3B6EA8", linewidth=1.5, label="Original prediction")
    axes[3].plot(seconds, masked_pred, color="#D18F2F", linewidth=1.2, label="After input masking")
    axes[3].plot(seconds, latent_pred, color="#3E8F6B", linewidth=1.2, label="After latent intervention")
    axes[3].fill_between(seconds, 0, labels, color="#BF616A", alpha=0.16, label="Seizure label")
    axes[3].axhline(0.8, color="#4C566A", linestyle="--", linewidth=0.9, label="Threshold")
    shade_mask_ranges(axes[3], input_mask_time, total_seconds, "#D18F2F", "Input masked")
    axes[3].set_ylim(-0.02, 1.02)
    axes[3].set_title("Prediction effect of the interventions")
    axes[3].set_ylabel("Seizure probability")
    axes[3].legend(frameon=False, ncol=3, loc="upper right")
    style_axis(axes[3])

    latent_limit = finite_percentile(latent_attr, 99)
    image = axes[4].imshow(
        latent_attr,
        aspect="auto",
        cmap="viridis",
        vmin=0,
        vmax=latent_limit,
        extent=[0, total_seconds, latent_attr.shape[0] - 0.5, -0.5],
    )
    axes[4].set_title("Transformer latent attribution heatmap")
    axes[4].set_ylabel("Latent feature")
    shade_mask_ranges(axes[4], latent_mask_time, total_seconds, "#3E8F6B", "Latent time touched")
    axes[4].legend(frameon=False, loc="upper right")
    add_colorbar(fig, image, axes[4], "Latent attribution")

    ax_last = axes[5]
    ax_last.plot(latent_seconds, latent_time_attr, color="#3E8F6B", linewidth=1.4, label="Latent time attribution")
    shade_mask_ranges(ax_last, latent_mask_time, total_seconds, "#3E8F6B", "Latent time touched")
    ax_last.set_title("Latent attribution collapsed over features")
    ax_last.set_xlabel("Seconds in window")
    ax_last.set_ylabel("Mean attr.")
    ax_last.legend(frameon=False, loc="upper right")
    style_axis(ax_last)

    inset = ax_last.inset_axes([0.71, 0.2, 0.27, 0.72])
    plot_top_latent_channels(inset, latent_channel_attr, latent_mask_channel, top_k)

    fig.suptitle(npz_path.stem, y=0.995, fontsize=14)
    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{npz_path.stem}.png", dpi=dpi, bbox_inches="tight")
    fig.savefig(output_dir / f"{npz_path.stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize saved MI example windows.")
    parser.add_argument("--examples_dir", type=str, default=str(ROOT / "mi_results/examples"))
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--top_k", type=int, default=16)
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

    examples_dir = Path(args.examples_dir)
    output_dir = Path(args.output_dir) if args.output_dir else examples_dir / "figures"
    manifest_rows = load_manifest(examples_dir / "manifest.jsonl")
    if manifest_rows:
        example_paths = [Path(row["artifact"]) for row in manifest_rows]
    else:
        example_paths = sorted(examples_dir.glob("example_*.npz"))

    if args.max_examples is not None:
        example_paths = example_paths[: args.max_examples]
    if not example_paths:
        raise SystemExit(f"No example_*.npz files found in {examples_dir}")

    for path in example_paths:
        plot_example(path, output_dir, args.dpi, args.top_k)
    print(f"Saved example figures to {output_dir}")


if __name__ == "__main__":
    main()
