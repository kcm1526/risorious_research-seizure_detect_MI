import argparse
import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from model import SeizureTransformer


ROOT = Path(__file__).resolve().parent
METHOD_ATTRIBUTION_ONLY = "attribution_only"
METHOD_ATTRIBUTION_MASKING = "attribution_guided_masking"
METHOD_LATENT_INTERVENTION = "mi_guided_latent_intervention"
METHODS = (
    METHOD_ATTRIBUTION_ONLY,
    METHOD_ATTRIBUTION_MASKING,
    METHOD_LATENT_INTERVENTION,
)


@dataclass
class ScoredPrediction:
    sample: Dict[str, float]
    event: Dict[str, float]
    postprocessed_positive_rate: float


class MeanAccumulator:
    def __init__(self):
        self.sums = defaultdict(float)
        self.counts = defaultdict(float)

    def add(self, values: Dict[str, float], weight: float = 1.0):
        for key, value in values.items():
            if value is None:
                continue
            if isinstance(value, float) and math.isnan(value):
                continue
            self.sums[key] += float(value) * weight
            self.counts[key] += weight

    def as_dict(self, prefix: str = "") -> Dict[str, float]:
        return {
            f"{prefix}{key}": self.sums[key] / self.counts[key]
            for key in sorted(self.sums)
            if self.counts[key] > 0
        }


class ScoreAccumulator:
    def __init__(self):
        from service.result import Result

        self.sample = Result()
        self.event = Result()
        self.files = 0

    def add(self, reference: np.ndarray, prediction: np.ndarray, threshold: float):
        from timescoring import scoring
        from timescoring.annotations import Annotation
        from service.post_process import morphological_filter_1d, remove_short_events
        from service.result import Result

        binary_output = (prediction > threshold).astype(int)
        binary_output = morphological_filter_1d(binary_output, operation="opening", kernel_size=5)
        binary_output = morphological_filter_1d(binary_output, operation="closing", kernel_size=5)
        binary_output = remove_short_events(binary_output, min_length=2.0, fs=256)

        hyp = Annotation(binary_output, 256)
        ref = Annotation(reference.astype(float), 256)
        sample_score = scoring.SampleScoring(ref, hyp)
        event_score = scoring.EventScoring(ref, hyp)

        sample_result = Result(sample_score)
        event_result = Result(event_score)
        self.sample += sample_result
        self.event += event_result
        self.files += 1

        sample_result.computeScores()
        event_result.computeScores()
        return ScoredPrediction(
            sample=result_to_dict(sample_result),
            event=result_to_dict(event_result),
            postprocessed_positive_rate=float(binary_output.mean()),
        )

    def summary(self) -> Dict[str, Dict[str, float]]:
        if self.files == 0:
            return {"sample": {}, "event": {}}
        self.sample.computeScores()
        self.event.computeScores()
        return {
            "sample": result_to_dict(self.sample),
            "event": result_to_dict(self.event),
        }


class SeizureTransformerAccess:
    def __init__(self, model: SeizureTransformer):
        self.model = model

    def encode_to_latent(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        encoded, skips = self.model.encoder(x)
        res_x = self.model.res_cnn_stack(encoded)
        latent = res_x.permute(2, 0, 1)
        latent = self.model.position_encoding(latent)
        latent = self.model.transformer_encoder(latent)
        latent = latent.permute(1, 2, 0)
        latent = latent + res_x
        return latent, skips

    def decode_from_latent(
        self,
        latent: torch.Tensor,
        skips: Optional[Sequence[torch.Tensor]],
    ) -> torch.Tensor:
        detection = self.model.decoder_d(latent, skips)
        detection = torch.sigmoid(self.model.conv_d(detection))
        return torch.squeeze(detection, dim=1)


def result_to_dict(result) -> Dict[str, float]:
    keys = ("f1", "sensitivity", "precision", "fpRate")
    values = {}
    for key in keys:
        value = getattr(result, key, float("nan"))
        try:
            values[key] = float(value)
        except (TypeError, ValueError):
            values[key] = float("nan")
    return values


def default_checkpoint_name(args) -> Path:
    return ROOT / "ckp" / (
        f"full_18_beta{args.beta}_alpha{args.alpha}_window{args.window_size}"
        f"_dim{args.dim_feedforward}_layer{args.num_layers}"
        f"_num_head{args.num_heads}_f1.pth"
    )


def resolve_path(path: Optional[str], base: Optional[Path] = None) -> Optional[Path]:
    if path is None:
        return None
    value = Path(path)
    if value.is_absolute():
        return value
    if base is None:
        base = Path.cwd()
    return base / value


def load_model(args, device: torch.device) -> SeizureTransformer:
    checkpoint_path = resolve_path(args.checkpoint) if args.checkpoint else default_checkpoint_name(args)
    if checkpoint_path is None or not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}. Pass --checkpoint or train the model first."
        )

    model = SeizureTransformer(
        in_channels=args.num_channel,
        in_samples=args.window_size,
        dim_feedforward=args.dim_feedforward,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        drop_rate=args.dropout,
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model", "model_state_dict"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                checkpoint = checkpoint[key]
                break
    checkpoint = strip_module_prefix(checkpoint)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    return model


def strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if not all(key.startswith("module.") for key in state_dict.keys()):
        return state_dict
    return {key[len("module.") :]: value for key, value in state_dict.items()}


def choose_device(device_arg: str, gpu: int) -> torch.device:
    if device_arg != "auto":
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device(f"cuda:{gpu}")
    return torch.device("cpu")


def make_objective(
    output: torch.Tensor,
    labels: Optional[torch.Tensor],
    strategy: str,
    threshold: float,
) -> torch.Tensor:
    if strategy == "all":
        return output.mean()

    if strategy == "prediction_positive":
        mask = output.detach() > threshold
        if mask.any():
            return output[mask].mean()
        return output.mean()

    if strategy == "label_positive":
        if labels is not None:
            mask = labels > 0.5
            if mask.any():
                return output[mask].mean()
        return output.mean()

    raise ValueError(f"Unknown objective strategy: {strategy}")


def top_fraction_mask(
    scores: torch.Tensor,
    fraction: float,
    largest: bool = True,
) -> torch.Tensor:
    if fraction <= 0:
        return torch.zeros_like(scores, dtype=torch.bool)
    flat = scores.reshape(scores.shape[0], -1)
    k = max(1, int(math.ceil(flat.shape[1] * fraction)))
    k = min(k, flat.shape[1])
    indices = torch.topk(flat, k=k, dim=1, largest=largest).indices
    mask = torch.zeros_like(flat, dtype=torch.bool)
    mask.scatter_(1, indices, True)
    return mask.reshape_as(scores)


def build_input_replacement(x: torch.Tensor, mode: str, noise_scale: float = 1.0) -> torch.Tensor:
    if mode == "zero":
        return torch.zeros_like(x)
    if mode == "channel_mean":
        return x.mean(dim=-1, keepdim=True).expand_as(x)
    if mode == "batch_channel_mean":
        return x.mean(dim=(0, 2), keepdim=True).expand_as(x)
    if mode == "noise":
        std = x.std(dim=-1, keepdim=True).clamp_min(1e-6)
        return torch.randn_like(x) * std * noise_scale
    raise ValueError(f"Unknown mask replacement mode: {mode}")


def input_attribution(
    model: SeizureTransformer,
    x: torch.Tensor,
    labels: Optional[torch.Tensor],
    objective_strategy: str,
    threshold: float,
    attribution_kind: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    model.zero_grad(set_to_none=True)
    x_attr = x.detach().clone().requires_grad_(True)
    output = model(x_attr)
    objective = make_objective(output, labels, objective_strategy, threshold)
    objective.backward()
    grad = x_attr.grad.detach()
    if attribution_kind == "grad_x_input":
        attribution = grad * x_attr.detach()
    elif attribution_kind == "saliency":
        attribution = grad
    else:
        raise ValueError(f"Unknown attribution kind: {attribution_kind}")
    return output.detach(), attribution.abs()


def attribution_guided_masking(
    model: SeizureTransformer,
    x: torch.Tensor,
    labels: Optional[torch.Tensor],
    args,
) -> Tuple[torch.Tensor, torch.Tensor]:
    _, attribution = input_attribution(
        model=model,
        x=x,
        labels=labels,
        objective_strategy=args.objective,
        threshold=args.threshold,
        attribution_kind=args.attribution_kind,
    )
    if args.mask_scope == "time":
        time_scores = attribution.mean(dim=1)
        time_mask = top_fraction_mask(
            time_scores,
            fraction=args.mask_fraction,
            largest=args.mask_target == "highest",
        )
        mask = time_mask.unsqueeze(1).expand_as(x)
    else:
        mask = top_fraction_mask(
            attribution,
            fraction=args.mask_fraction,
            largest=args.mask_target == "highest",
        )
    replacement = build_input_replacement(x, args.mask_replacement, args.noise_scale)
    x_masked = torch.where(mask, replacement, x)
    with torch.no_grad():
        return model(x_masked).detach(), attribution.detach()


def latent_attribution(
    access: SeizureTransformerAccess,
    x: torch.Tensor,
    labels: Optional[torch.Tensor],
    objective_strategy: str,
    threshold: float,
    attribution_kind: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[torch.Tensor]]:
    with torch.no_grad():
        latent, skips = access.encode_to_latent(x)
        skips = [skip.detach() for skip in skips]

    access.model.zero_grad(set_to_none=True)
    latent_attr = latent.detach().clone().requires_grad_(True)
    output = access.decode_from_latent(latent_attr, skips)
    objective = make_objective(output, labels, objective_strategy, threshold)
    objective.backward()
    grad = latent_attr.grad.detach()
    if attribution_kind == "grad_x_input":
        attribution = grad * latent_attr.detach()
    elif attribution_kind == "saliency":
        attribution = grad
    else:
        raise ValueError(f"Unknown attribution kind: {attribution_kind}")
    return output.detach(), latent_attr.detach(), attribution.abs(), skips


def intervene_latent(
    latent: torch.Tensor,
    attribution: torch.Tensor,
    args,
) -> torch.Tensor:
    if args.latent_scope == "time":
        scores = attribution.mean(dim=1)
        time_mask = top_fraction_mask(scores, fraction=args.latent_fraction, largest=True)
        mask = time_mask.unsqueeze(1).expand_as(latent)
    elif args.latent_scope == "channel":
        scores = attribution.mean(dim=2)
        channel_mask = top_fraction_mask(scores, fraction=args.latent_fraction, largest=True)
        mask = channel_mask.unsqueeze(2).expand_as(latent)
    else:
        mask = top_fraction_mask(attribution, fraction=args.latent_fraction, largest=True)

    if args.latent_intervention == "suppress":
        replacement = torch.zeros_like(latent)
        return torch.where(mask, replacement, latent)
    if args.latent_intervention == "mean":
        replacement = latent.mean(dim=2, keepdim=True).expand_as(latent)
        return torch.where(mask, replacement, latent)
    if args.latent_intervention == "amplify":
        return torch.where(mask, latent * (1.0 + args.latent_strength), latent)
    if args.latent_intervention == "invert":
        return torch.where(mask, -args.latent_strength * latent, latent)
    raise ValueError(f"Unknown latent intervention: {args.latent_intervention}")


def mi_guided_latent_intervention(
    access: SeizureTransformerAccess,
    x: torch.Tensor,
    labels: Optional[torch.Tensor],
    args,
) -> Tuple[torch.Tensor, torch.Tensor]:
    _, latent, attribution, skips = latent_attribution(
        access=access,
        x=x,
        labels=labels,
        objective_strategy=args.objective,
        threshold=args.threshold,
        attribution_kind=args.attribution_kind,
    )
    intervened = intervene_latent(latent, attribution, args)
    with torch.no_grad():
        return access.decode_from_latent(intervened, skips).detach(), attribution.detach()


def batch_delta_stats(
    baseline: np.ndarray,
    prediction: np.ndarray,
    labels: np.ndarray,
) -> Dict[str, float]:
    delta = prediction - baseline
    stats = {
        "mean_pred": float(prediction.mean()),
        "mean_baseline_pred": float(baseline.mean()),
        "mean_delta": float(delta.mean()),
        "mean_abs_delta": float(np.abs(delta).mean()),
    }
    positive = labels > 0.5
    negative = ~positive
    if positive.any():
        stats["mean_delta_on_seizure"] = float(delta[positive].mean())
        stats["mean_abs_delta_on_seizure"] = float(np.abs(delta[positive]).mean())
    if negative.any():
        stats["mean_delta_on_background"] = float(delta[negative].mean())
        stats["mean_abs_delta_on_background"] = float(np.abs(delta[negative]).mean())
    return stats


def attribution_alignment_stats(
    attribution: torch.Tensor,
    labels: torch.Tensor,
    expected_length: Optional[int] = None,
) -> Dict[str, float]:
    attr = attribution.detach()
    if attr.ndim == 3:
        attr = attr.mean(dim=1)
    if expected_length is not None and attr.shape[-1] != expected_length:
        attr = torch.nn.functional.interpolate(
            attr.unsqueeze(1),
            size=expected_length,
            mode="linear",
            align_corners=False,
        ).squeeze(1)

    attr_np = attr.detach().cpu().numpy()
    labels_np = labels.detach().cpu().numpy()
    stats = MeanAccumulator()
    for scores, target in zip(attr_np, labels_np):
        scores = np.abs(scores).astype(float)
        target = target > 0.5
        total = float(scores.sum())
        if total > 0:
            stats.add({"attribution_mass_on_seizure": float(scores[target].sum() / total)})
            probs = scores / total
            entropy = -float(np.sum(probs * np.log(probs + 1e-12)))
            stats.add({"normalized_attribution_entropy": entropy / math.log(len(probs))})
        auc = binary_auc(scores, target)
        if not math.isnan(auc):
            stats.add({"attribution_label_auc": auc})
        stats.add({"label_positive_rate": float(target.mean())})
    return stats.as_dict()


def binary_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    labels = labels.astype(bool)
    n_pos = int(labels.sum())
    n_neg = int((~labels).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty_like(sorted_scores, dtype=float)
    start = 0
    n = len(sorted_scores)
    while start < n:
        end = start + 1
        while end < n and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[start:end] = (start + 1 + end) / 2.0
        start = end
    original_ranks = np.empty_like(ranks)
    original_ranks[order] = ranks
    rank_sum_pos = float(original_ranks[labels].sum())
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def load_eval_files(data_dir: Path) -> Tuple[List[str], List[str]]:
    from service.handle_data import get_file

    file_list, label_list = get_file(str(data_dir))
    pairs = [(edf, label) for edf, label in zip(file_list, label_list) if Path(label).exists()]
    return [edf for edf, _ in pairs], [label for _, label in pairs]


def load_recording(edf_path: str, label_path: str):
    from service.handle_data import get_data_18

    return get_data_18(edf_path, label_path)


def build_recording_loader(data, labels, args):
    from service.result import get_testingdataloader_window

    return get_testingdataloader_window(
        data,
        labels,
        window_size=args.window_size,
        batch_size=args.batch_size,
        fs=args.fs,
    )


def run_recording(
    model: SeizureTransformer,
    access: SeizureTransformerAccess,
    edf_path: str,
    label_path: str,
    args,
    device: torch.device,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict[str, float]], Dict[str, Dict[str, float]], np.ndarray]:
    data, reference = load_recording(edf_path, label_path)
    if data.shape[1] < args.window_size:
        raise ValueError(f"Recording shorter than window_size: {edf_path}")

    reference_np = reference.detach().cpu().numpy().astype(float)
    dataloader = build_recording_loader(data, reference_np, args)

    outputs = {method: [] for method in args.methods}
    baseline_chunks = []
    attribution_stats = defaultdict(MeanAccumulator)

    for batch in dataloader:
        x, labels = batch
        x = x.float().to(device)
        labels = labels.float().to(device)

        with torch.no_grad():
            baseline = model(x).detach()
        baseline_chunks.append(baseline.cpu().numpy())

        if METHOD_ATTRIBUTION_ONLY in args.methods:
            attr_output, attribution = input_attribution(
                model=model,
                x=x,
                labels=labels,
                objective_strategy=args.objective,
                threshold=args.threshold,
                attribution_kind=args.attribution_kind,
            )
            outputs[METHOD_ATTRIBUTION_ONLY].append(attr_output.cpu().numpy())
            attribution_stats[METHOD_ATTRIBUTION_ONLY].add(
                attribution_alignment_stats(attribution, labels)
            )

        if METHOD_ATTRIBUTION_MASKING in args.methods:
            masked_output, attribution = attribution_guided_masking(model, x, labels, args)
            outputs[METHOD_ATTRIBUTION_MASKING].append(masked_output.cpu().numpy())
            attribution_stats[METHOD_ATTRIBUTION_MASKING].add(
                attribution_alignment_stats(attribution, labels)
            )

        if METHOD_LATENT_INTERVENTION in args.methods:
            intervened_output, latent_attr = mi_guided_latent_intervention(access, x, labels, args)
            outputs[METHOD_LATENT_INTERVENTION].append(intervened_output.cpu().numpy())
            attribution_stats[METHOD_LATENT_INTERVENTION].add(
                attribution_alignment_stats(latent_attr, labels, expected_length=labels.shape[-1])
            )

    baseline_pred = flatten_prediction(np.concatenate(baseline_chunks, axis=0), len(reference_np))
    predictions = {}
    method_stats = {}
    attr_summary = {}
    for method in args.methods:
        method_output = np.concatenate(outputs[method], axis=0)
        pred = flatten_prediction(method_output, len(reference_np))
        predictions[method] = pred
        method_stats[method] = batch_delta_stats(baseline_pred, pred, reference_np)
        attr_summary[method] = attribution_stats[method].as_dict(prefix="")

    return predictions, method_stats, attr_summary, reference_np


def flatten_prediction(prediction: np.ndarray, target_length: int) -> np.ndarray:
    return prediction.reshape(-1)[:target_length]


def write_csv(path: Path, rows: List[Dict[str, object]]):
    if not rows:
        return
    columns = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def save_npz(output_dir: Path, stem: str, reference: np.ndarray, predictions: Dict[str, np.ndarray]):
    arrays = {"reference": reference}
    arrays.update({f"prediction_{method}": pred for method, pred in predictions.items()})
    np.savez_compressed(output_dir / f"{stem}.npz", **arrays)


def safe_stem(edf_path: str, index: int) -> str:
    stem = Path(edf_path).stem
    return f"{index:04d}_{stem}"


def run_experiment(args):
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = choose_device(args.device, args.gpu)
    model = load_model(args, device)
    access = SeizureTransformerAccess(model)

    data_dir = resolve_path(args.data_dir)
    file_list, label_list = load_eval_files(data_dir)
    if args.max_files is not None:
        file_list = file_list[: args.max_files]
        label_list = label_list[: args.max_files]
    if not file_list:
        raise FileNotFoundError(f"No EDF/CSV_BI pairs found under {data_dir}")

    aggregate_scores = {method: ScoreAccumulator() for method in args.methods}
    aggregate_delta = {method: MeanAccumulator() for method in args.methods}
    aggregate_attr = {method: MeanAccumulator() for method in args.methods}
    rows = []
    skipped = []

    for index, (edf_path, label_path) in enumerate(zip(file_list, label_list)):
        print(f"[{index + 1}/{len(file_list)}] {edf_path}")
        try:
            predictions, delta_stats, attr_stats, reference = run_recording(
                model=model,
                access=access,
                edf_path=edf_path,
                label_path=label_path,
                args=args,
                device=device,
            )
        except Exception as exc:
            skipped.append({"edf": edf_path, "error": repr(exc)})
            print(f"  skipped: {exc}")
            continue

        if args.save_npz:
            save_npz(output_dir, safe_stem(edf_path, index), reference, predictions)

        for method, prediction in predictions.items():
            scored = aggregate_scores[method].add(reference, prediction, args.threshold)
            aggregate_delta[method].add(delta_stats[method])
            aggregate_attr[method].add(attr_stats[method])
            row = {
                "file": edf_path,
                "method": method,
                "postprocessed_positive_rate": scored.postprocessed_positive_rate,
            }
            row.update({f"sample_{key}": value for key, value in scored.sample.items()})
            row.update({f"event_{key}": value for key, value in scored.event.items()})
            row.update(delta_stats[method])
            row.update(attr_stats[method])
            rows.append(row)

    summary = {
        "config": vars(args),
        "num_files": len(file_list),
        "num_skipped": len(skipped),
        "skipped": skipped,
        "methods": {},
    }
    for method in args.methods:
        summary["methods"][method] = {
            **aggregate_scores[method].summary(),
            "delta": aggregate_delta[method].as_dict(),
            "attribution": aggregate_attr[method].as_dict(),
        }

    write_csv(output_dir / "per_recording_results.csv", rows)
    with (output_dir / "summary.json").open("w") as fp:
        json.dump(summary, fp, indent=2)

    print(json.dumps(summary["methods"], indent=2))
    print(f"Saved: {output_dir / 'summary.json'}")
    print(f"Saved: {output_dir / 'per_recording_results.csv'}")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compare attribution-only, attribution-guided masking, and "
            "MI-guided latent intervention on a trained SeizureTransformer."
        )
    )
    parser.add_argument("--data_dir", type=str, default=str(ROOT / "data/v2.0.3/edf/eval"))
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=str(ROOT / "mi_results"))
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--max_files", type=int, default=None)
    parser.add_argument("--save_npz", action="store_true")

    parser.add_argument("--window_size", type=int, default=15360)
    parser.add_argument("--num_channel", type=int, default=18)
    parser.add_argument("--fs", type=int, default=256)
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--beta", type=float, default=2.0)
    parser.add_argument("--dim_feedforward", type=int, default=2048)
    parser.add_argument("--num_layers", type=int, default=8)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--gpu", type=int, default=0)

    parser.add_argument(
        "--objective",
        choices=("label_positive", "prediction_positive", "all"),
        default="label_positive",
        help="Scalar target used for gradient attribution.",
    )
    parser.add_argument(
        "--attribution_kind",
        choices=("grad_x_input", "saliency"),
        default="grad_x_input",
    )

    parser.add_argument("--mask_fraction", type=float, default=0.1)
    parser.add_argument("--mask_scope", choices=("time", "element"), default="time")
    parser.add_argument("--mask_target", choices=("highest", "lowest"), default="highest")
    parser.add_argument(
        "--mask_replacement",
        choices=("zero", "channel_mean", "batch_channel_mean", "noise"),
        default="zero",
    )
    parser.add_argument("--noise_scale", type=float, default=1.0)

    parser.add_argument("--latent_fraction", type=float, default=0.1)
    parser.add_argument("--latent_scope", choices=("time", "channel", "element"), default="element")
    parser.add_argument(
        "--latent_intervention",
        choices=("suppress", "mean", "amplify", "invert"),
        default="suppress",
    )
    parser.add_argument("--latent_strength", type=float, default=1.0)
    return parser.parse_args()


if __name__ == "__main__":
    run_experiment(parse_args())
