from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from config import DEFAULT_CHARSET
from decode import decode_ctc_greedy
from metrics import compute_cer, compute_wer, detection_f1, exact_match_ratio


class ProgressLogger:
    def __init__(self, name: str, total: int, log_every: int = 50) -> None:
        self.name = name
        self.total = max(1, total)
        self.log_every = max(1, log_every)
        self.start = time.time()

    def eta(self, step: int) -> str:
        elapsed = time.time() - self.start
        rate = step / max(elapsed, 1e-6)
        remaining = max(0.0, (self.total - step) / max(rate, 1e-6))
        minutes, seconds = divmod(int(remaining), 60)
        hours, minutes = divmod(minutes, 60)
        if hours:
            return f"{hours:d}h{minutes:02d}m"
        return f"{minutes:02d}m{seconds:02d}s"

    def should_log(self, step: int) -> bool:
        return step == 1 or step == self.total or step % self.log_every == 0

    def log(self, step: int, message: str) -> None:
        print(f"{self.name} {step}/{self.total} eta {self.eta(step)} | {message}")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _amp_enabled(device: torch.device, amp: bool) -> bool:
    return amp and device.type == "cuda"


def _autocast(device: torch.device, amp: bool):
    return torch.autocast(device_type=device.type, enabled=_amp_enabled(device, amp))


def _grad_scaler(device: torch.device, amp: bool):
    enabled = _amp_enabled(device, amp)
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    epoch: int | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw_model = getattr(model, "_orig_mod", model)
    state: dict[str, Any] = {"model": raw_model.state_dict()}
    if optimizer is not None:
        state["optimizer"] = optimizer.state_dict()
    if epoch is not None:
        state["epoch"] = epoch
    if extra:
        state.update(extra)
    torch.save(state, path)


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    device: torch.device | str = "cpu",
    strict: bool = True,
) -> dict[str, Any]:
    ckpt = torch.load(path, map_location=device)
    state = ckpt.get("model", ckpt)
    model.load_state_dict(state, strict=strict)
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt


def train_detector(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    amp: bool = False,
    grad_accum_steps: int = 1,
    log_every: int = 50,
) -> dict[str, float]:
    model.train()
    scaler = _grad_scaler(device, amp)
    total_loss = 0.0
    progress = ProgressLogger(f"detector train epoch {epoch}", len(train_loader), log_every)
    optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(train_loader, start=1):
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        with _autocast(device, amp):
            logits = model(images)
            loss = F.binary_cross_entropy_with_logits(logits, masks) / grad_accum_steps
        scaler.scale(loss).backward()
        if step % grad_accum_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        total_loss += float(loss.detach()) * grad_accum_steps

        if log_every and progress.should_log(step):
            with torch.no_grad():
                preds = torch.sigmoid(logits.detach()) >= 0.5
                truth = masks >= 0.5
                pixel_tp = float((preds & truth).sum())
                pixel_fp = float((preds & ~truth).sum())
                pixel_fn = float((~preds & truth).sum())
            precision = pixel_tp / (pixel_tp + pixel_fp) if pixel_tp + pixel_fp else 0.0
            recall = pixel_tp / (pixel_tp + pixel_fn) if pixel_tp + pixel_fn else 0.0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            progress.log(step, f"loss {total_loss / step:.4f} precision {precision:.3f} recall {recall:.3f} f1 {f1:.3f}")

    if len(train_loader) % grad_accum_steps:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
    return {"loss": total_loss / max(1, len(train_loader))}


@torch.no_grad()
def validate_detector(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    amp: bool = False,
    threshold: float = 0.5,
) -> dict[str, float]:
    model.eval()
    losses: list[float] = []
    pixel_tp = pixel_fp = pixel_fn = 0.0
    progress = ProgressLogger("detector val", len(val_loader), log_every=20)
    for step, batch in enumerate(val_loader, start=1):
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        with _autocast(device, amp):
            logits = model(images)
            loss = F.binary_cross_entropy_with_logits(logits, masks)
        probs = torch.sigmoid(logits)
        preds = probs >= threshold
        truth = masks >= 0.5
        pixel_tp += float((preds & truth).sum())
        pixel_fp += float((preds & ~truth).sum())
        pixel_fn += float((~preds & truth).sum())
        losses.append(float(loss))
        precision = pixel_tp / (pixel_tp + pixel_fp) if pixel_tp + pixel_fp else 0.0
        recall = pixel_tp / (pixel_tp + pixel_fn) if pixel_tp + pixel_fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        if progress.should_log(step):
            progress.log(step, f"loss {float(np.mean(losses)):.4f} precision {precision:.3f} recall {recall:.3f} f1 {f1:.3f}")
    precision = pixel_tp / (pixel_tp + pixel_fp) if pixel_tp + pixel_fp else 0.0
    recall = pixel_tp / (pixel_tp + pixel_fn) if pixel_tp + pixel_fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"loss": float(np.mean(losses)) if losses else 0.0, "precision": precision, "recall": recall, "f1": f1}


def train_recognizer(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    amp: bool = False,
    grad_accum_steps: int = 1,
    log_every: int = 50,
) -> dict[str, float]:
    model.train()
    ctc_loss = nn.CTCLoss(blank=0, zero_infinity=True)
    scaler = _grad_scaler(device, amp)
    total_loss = 0.0
    progress = ProgressLogger(f"recognizer train epoch {epoch}", len(train_loader), log_every)
    optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(train_loader, start=1):
        images = batch["images"].to(device, non_blocking=True)
        targets = batch["targets"].to(device, non_blocking=True)
        target_lengths = batch["target_lengths"].to(device, non_blocking=True)
        with _autocast(device, amp):
            logits = model(images)
            log_probs = F.log_softmax(logits, dim=-1)
            input_lengths = torch.full(
                (images.size(0),),
                logits.size(0),
                dtype=torch.long,
                device=device,
            )
            loss = ctc_loss(log_probs, targets, input_lengths, target_lengths) / grad_accum_steps
        scaler.scale(loss).backward()
        if step % grad_accum_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        total_loss += float(loss.detach()) * grad_accum_steps

        if log_every and progress.should_log(step):
            with torch.no_grad():
                batch_predictions = decode_ctc_greedy(logits.detach())
            batch_targets = batch["texts"]
            cer = float(np.mean([compute_cer(p, t) for p, t in zip(batch_predictions, batch_targets)])) if batch_targets else 0.0
            exact = exact_match_ratio(batch_predictions, batch_targets) if batch_targets else 0.0
            progress.log(step, f"loss {total_loss / step:.4f} cer {cer:.3f} exact {exact:.3f}")

    if len(train_loader) % grad_accum_steps:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
    return {"loss": total_loss / max(1, len(train_loader))}


@torch.no_grad()
def validate_recognizer(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    charset: str = DEFAULT_CHARSET,
    amp: bool = False,
) -> dict[str, float]:
    model.eval()
    ctc_loss = nn.CTCLoss(blank=0, zero_infinity=True)
    losses: list[float] = []
    predictions: list[str] = []
    targets_text: list[str] = []
    progress = ProgressLogger("recognizer val", len(val_loader), log_every=20)

    for step, batch in enumerate(val_loader, start=1):
        images = batch["images"].to(device, non_blocking=True)
        targets = batch["targets"].to(device, non_blocking=True)
        target_lengths = batch["target_lengths"].to(device, non_blocking=True)
        with _autocast(device, amp):
            logits = model(images)
            log_probs = F.log_softmax(logits, dim=-1)
            input_lengths = torch.full((images.size(0),), logits.size(0), dtype=torch.long, device=device)
            loss = ctc_loss(log_probs, targets, input_lengths, target_lengths)
        losses.append(float(loss))
        predictions.extend(decode_ctc_greedy(logits, charset=charset))
        targets_text.extend(batch["texts"])
        cer = float(np.mean([compute_cer(p, t) for p, t in zip(predictions, targets_text)])) if targets_text else 0.0
        wer = float(np.mean([compute_wer(p, t) for p, t in zip(predictions, targets_text)])) if targets_text else 0.0
        exact = exact_match_ratio(predictions, targets_text)
        if progress.should_log(step):
            progress.log(step, f"loss {float(np.mean(losses)):.4f} cer {cer:.3f} wer {wer:.3f} exact {exact:.3f}")

    cer = float(np.mean([compute_cer(p, t) for p, t in zip(predictions, targets_text)])) if targets_text else 0.0
    wer = float(np.mean([compute_wer(p, t) for p, t in zip(predictions, targets_text)])) if targets_text else 0.0
    return {
        "loss": float(np.mean(losses)) if losses else 0.0,
        "cer": cer,
        "wer": wer,
        "exact_match": exact_match_ratio(predictions, targets_text),
    }


def save_predictions(predictions: list[dict[str, Any]], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps({"predictions": predictions}, indent=2))


def run_ocr_folder(*args, **kwargs) -> list[dict[str, Any]]:
    from infer import run_ocr_folder_infer

    return run_ocr_folder_infer(*args, **kwargs)


def benchmark(
    predictions_path: str | Path,
    labels_path: str | Path | None = None,
    detection_labels_path: str | Path | None = None,
) -> dict[str, Any]:
    predictions_data = json.loads(Path(predictions_path).read_text())
    predictions = predictions_data.get("predictions", predictions_data)
    report: dict[str, Any] = {"num_predictions": len(predictions)}
    print(f"benchmark loaded {len(predictions)} prediction(s)")

    if labels_path is None:
        report["gt_text_metrics"] = "not computed; no labels manifest was provided"
        print("benchmark text metrics: no ground-truth labels provided")
    else:
        label_items = json.loads(Path(labels_path).read_text())
        if isinstance(label_items, dict):
            label_items = label_items.get("samples") or label_items.get("items") or label_items.get("labels") or []
        labels = {Path(item["image"]).name: item.get("text", "") for item in label_items}
        pairs = [(p["text"], labels[p["image"]]) for p in predictions if p["image"] in labels]
        if pairs:
            report["cer"] = float(np.mean([compute_cer(p, t) for p, t in pairs]))
            report["wer"] = float(np.mean([compute_wer(p, t) for p, t in pairs]))
            report["exact_match"] = exact_match_ratio([p for p, _ in pairs], [t for _, t in pairs])
            print(
                "benchmark text metrics: "
                f"cer {report['cer']:.3f} wer {report['wer']:.3f} exact {report['exact_match']:.3f}"
            )
        else:
            report["gt_text_metrics"] = "not computed; no prediction filenames matched labels"
            print("benchmark text metrics: no prediction filenames matched labels")

    if detection_labels_path is not None:
        label_items = json.loads(Path(detection_labels_path).read_text())
        if isinstance(label_items, dict):
            label_items = label_items.get("pages") or label_items.get("items") or []
        labels = {Path(item["image"]).name: item.get("boxes", []) for item in label_items}
        scores = []
        for pred in predictions:
            if pred["image"] not in labels:
                continue
            pred_boxes = [tuple(r["box"]) for r in pred.get("regions", [])]
            gt_boxes = [tuple(b) for b in labels[pred["image"]]]
            scores.append(detection_f1(pred_boxes, gt_boxes))
        if scores:
            report["det_precision"] = float(np.mean([s["precision"] for s in scores]))
            report["det_recall"] = float(np.mean([s["recall"] for s in scores]))
            report["det_f1"] = float(np.mean([s["f1"] for s in scores]))
            print(
                "benchmark detection metrics: "
                f"precision {report['det_precision']:.3f} recall {report['det_recall']:.3f} f1 {report['det_f1']:.3f}"
            )
    return report
