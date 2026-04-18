from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw

from config import (
    DEFAULT_CHARSET,
    DEFAULT_DETECTOR_SIZE,
    DEFAULT_RECOGNIZER_HEIGHT,
    DEFAULT_RECOGNIZER_WIDTH,
    IMAGE_EXTENSIONS,
)
from decode import decode_ctc_greedy
from transforms import RecognitionTransform, normalize_tensor, pil_to_tensor, resize_page
from utils import ProgressLogger


def sort_boxes_reading_order(boxes: list[tuple[int, int, int, int]]) -> list[tuple[int, int, int, int]]:
    if not boxes:
        return []
    heights = [y2 - y1 for _, y1, _, y2 in boxes]
    row_tol = max(8, int(np.median(heights) * 0.6))
    boxes = sorted(boxes, key=lambda b: (b[1], b[0]))
    rows: list[list[tuple[int, int, int, int]]] = []
    for box in boxes:
        cy = (box[1] + box[3]) // 2
        for row in rows:
            row_cy = sum((b[1] + b[3]) // 2 for b in row) / len(row)
            if abs(cy - row_cy) <= row_tol:
                row.append(box)
                break
        else:
            rows.append([box])
    ordered: list[tuple[int, int, int, int]] = []
    for row in sorted(rows, key=lambda r: min(b[1] for b in r)):
        ordered.extend(sorted(row, key=lambda b: b[0]))
    return ordered


def _merge_overlapping_boxes(boxes: list[tuple[int, int, int, int]], y_overlap: float = 0.45) -> list[tuple[int, int, int, int]]:
    if not boxes:
        return []
    boxes = sort_boxes_reading_order(boxes)
    merged: list[tuple[int, int, int, int]] = []
    for box in boxes:
        if not merged:
            merged.append(box)
            continue
        x1, y1, x2, y2 = box
        mx1, my1, mx2, my2 = merged[-1]
        inter_y = max(0, min(y2, my2) - max(y1, my1))
        min_h = max(1, min(y2 - y1, my2 - my1))
        gap = x1 - mx2
        if inter_y / min_h >= y_overlap and gap < max(18, min_h * 2):
            merged[-1] = (min(mx1, x1), min(my1, y1), max(mx2, x2), max(my2, y2))
        else:
            merged.append(box)
    return merged


def _projection_runs(values: np.ndarray, threshold: float, max_gap: int = 2) -> list[tuple[int, int]]:
    active = values > threshold
    runs: list[tuple[int, int]] = []
    start: int | None = None
    last = -1
    for idx, is_active in enumerate(active):
        if is_active:
            if start is None:
                start = idx
            last = idx
        elif start is not None:
            runs.append((start, last + 1))
            start = None
    if start is not None:
        runs.append((start, last + 1))

    if not runs:
        return []
    merged = [runs[0]]
    for y1, y2 in runs[1:]:
        prev_y1, prev_y2 = merged[-1]
        if y1 - prev_y2 <= max_gap:
            merged[-1] = (prev_y1, y2)
        else:
            merged.append((y1, y2))
    return merged


def _line_boxes_from_crop(
    image: Image.Image,
    box: tuple[int, int, int, int],
    padding: int = 3,
    min_height: int = 6,
    min_width: int = 24,
) -> list[tuple[int, int, int, int]]:
    x1, y1, x2, y2 = box
    crop = np.asarray(image.crop(box).convert("L"))
    if crop.size == 0:
        return []

    blurred = cv2.GaussianBlur(crop, (3, 3), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    row_counts = binary.sum(axis=1) / 255.0
    if len(row_counts) >= 5:
        row_counts = cv2.GaussianBlur(row_counts.reshape(-1, 1), (1, 5), 0).ravel()

    threshold = max(2.0, crop.shape[1] * 0.018)
    runs = _projection_runs(row_counts, threshold=threshold, max_gap=2)

    line_boxes: list[tuple[int, int, int, int]] = []
    for local_y1, local_y2 in runs:
        if local_y2 - local_y1 < min_height:
            continue
        band = binary[local_y1:local_y2, :]
        col_counts = band.sum(axis=0) / 255.0
        col_runs = _projection_runs(col_counts, threshold=max(1.0, band.shape[0] * 0.08), max_gap=8)
        if col_runs:
            local_x1 = max(0, col_runs[0][0] - padding)
            local_x2 = min(crop.shape[1], col_runs[-1][1] + padding)
        else:
            local_x1, local_x2 = 0, crop.shape[1]
        if local_x2 - local_x1 < min_width:
            continue
        line_boxes.append(
            (
                max(0, x1 + local_x1),
                max(0, y1 + local_y1 - padding),
                min(image.width, x1 + local_x2),
                min(image.height, y1 + local_y2 + padding),
            )
        )
    return line_boxes


def split_regions_into_lines(
    image: Image.Image,
    boxes: list[tuple[int, int, int, int]],
    min_split_height: int = 42,
    min_region_width: int = 40,
    min_region_height: int = 8,
) -> list[tuple[int, int, int, int]]:
    line_boxes: list[tuple[int, int, int, int]] = []
    for box in boxes:
        x1, y1, x2, y2 = box
        height = y2 - y1
        if height >= min_split_height:
            split = _line_boxes_from_crop(image, box)
            if len(split) >= 2:
                line_boxes.extend(split)
                continue
        line_boxes.append(box)
    filtered = [
        box
        for box in line_boxes
        if box[2] - box[0] >= min_region_width and box[3] - box[1] >= min_region_height
    ]
    return sort_boxes_reading_order(filtered)


def detect_text_regions(
    detector: torch.nn.Module,
    image: Image.Image,
    device: torch.device,
    image_size: tuple[int, int] = DEFAULT_DETECTOR_SIZE,
    threshold: float = 0.35,
    min_area: int = 80,
    padding: int = 4,
) -> tuple[list[tuple[int, int, int, int]], np.ndarray]:
    original = image.convert("L")
    original_w, original_h = original.size
    model_input = resize_page(original, image_size)
    tensor = normalize_tensor(pil_to_tensor(model_input)).unsqueeze(0).to(device)

    detector.eval()
    with torch.no_grad():
        prob = torch.sigmoid(detector(tensor))[0, 0].detach().cpu().numpy()

    prob = cv2.resize(prob, (original_w, original_h), interpolation=cv2.INTER_LINEAR)
    mask = (prob >= threshold).astype(np.uint8) * 255
    kernel_w = max(9, original_w // 80)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_w, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes: list[tuple[int, int, int, int]] = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w * h < min_area or h < 5 or w < 8:
            continue
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(original_w, x + w + padding)
        y2 = min(original_h, y + h + padding)
        boxes.append((x1, y1, x2, y2))
    merged = _merge_overlapping_boxes(boxes)
    return split_regions_into_lines(original, merged), prob


def recognize_crop(
    recognizer: torch.nn.Module,
    image: Image.Image,
    device: torch.device,
    charset: str = DEFAULT_CHARSET,
    height: int = DEFAULT_RECOGNIZER_HEIGHT,
    width: int = DEFAULT_RECOGNIZER_WIDTH,
) -> str:
    transform = RecognitionTransform(height=height, width=width, augment=False)
    tensor = transform(image).unsqueeze(0).to(device)
    recognizer.eval()
    with torch.no_grad():
        logits = recognizer(tensor)
    return decode_ctc_greedy(logits, charset=charset)[0]


def save_overlay(image: Image.Image, boxes: list[tuple[int, int, int, int]], path: str | Path) -> None:
    overlay = image.convert("RGB")
    draw = ImageDraw.Draw(overlay)
    for box in boxes:
        draw.rectangle(box, outline=(220, 40, 40), width=2)
    overlay.save(path)


def run_ocr_image(
    image_path: str | Path,
    recognizer: torch.nn.Module,
    device: torch.device,
    detector: torch.nn.Module | None = None,
    skip_detector: bool = False,
    output_dir: str | Path | None = None,
    save_crops: bool = False,
    save_visualizations: bool = False,
    detector_size: tuple[int, int] = DEFAULT_DETECTOR_SIZE,
    threshold: float = 0.35,
    charset: str = DEFAULT_CHARSET,
    rec_height: int = DEFAULT_RECOGNIZER_HEIGHT,
    rec_width: int = DEFAULT_RECOGNIZER_WIDTH,
    split_lines_without_detector: bool = False,
) -> dict[str, Any]:
    image_path = Path(image_path)
    image = Image.open(image_path).convert("L")
    output_dir = Path(output_dir) if output_dir else image_path.parent / "ocr_outputs"
    crops_dir = output_dir / "crops" / image_path.stem
    vis_dir = output_dir / "visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)

    boxes: list[tuple[int, int, int, int]]
    used_line_split_fallback = False
    if skip_detector or detector is None:
        boxes = [(0, 0, image.width, image.height)]
        if split_lines_without_detector and image.height >= 120 and image.width >= 200:
            split_boxes = split_regions_into_lines(image, boxes)
            if split_boxes:
                boxes = split_boxes
                used_line_split_fallback = True
        prob_map = None
    else:
        boxes, prob_map = detect_text_regions(
            detector=detector,
            image=image,
            device=device,
            image_size=detector_size,
            threshold=threshold,
        )
        if not boxes:
            boxes = [(0, 0, image.width, image.height)]
        boxes = split_regions_into_lines(image, boxes)
        if len(boxes) <= 1 and image.height >= 120 and image.width >= 200:
            fallback_boxes = split_regions_into_lines(image, [(0, 0, image.width, image.height)])
            if len(fallback_boxes) > len(boxes):
                boxes = fallback_boxes
                used_line_split_fallback = True

    region_results: list[dict[str, Any]] = []
    texts: list[str] = []
    if save_crops:
        crops_dir.mkdir(parents=True, exist_ok=True)
    for idx, box in enumerate(boxes):
        crop = image.crop(box)
        text = recognize_crop(recognizer, crop, device, charset, rec_height, rec_width)
        texts.append(text)
        if save_crops:
            crop.save(crops_dir / f"region_{idx:03d}.png")
        region_results.append({"box": list(box), "text": text})

    if save_visualizations:
        vis_dir.mkdir(parents=True, exist_ok=True)
        save_overlay(image, boxes, vis_dir / f"{image_path.stem}_boxes.png")
        if prob_map is not None:
            heat = (np.clip(prob_map, 0, 1) * 255).astype(np.uint8)
            cv2.imwrite(str(vis_dir / f"{image_path.stem}_prob.png"), heat)

    return {
        "image": image_path.name,
        "path": str(image_path),
        "text": "\n".join(texts),
        "regions": region_results if (not skip_detector and detector is not None) or split_lines_without_detector else [],
        "used_line_split_fallback": used_line_split_fallback,
    }


def run_ocr_folder_infer(
    input_dir: str | Path,
    output_dir: str | Path,
    recognizer: torch.nn.Module,
    device: torch.device,
    detector: torch.nn.Module | None = None,
    skip_detector: bool = False,
    save_crops: bool = False,
    save_visualizations: bool = False,
    detector_size: tuple[int, int] = DEFAULT_DETECTOR_SIZE,
    threshold: float = 0.35,
    charset: str = DEFAULT_CHARSET,
    rec_height: int = DEFAULT_RECOGNIZER_HEIGHT,
    rec_width: int = DEFAULT_RECOGNIZER_WIDTH,
    log_every: int = 10,
    split_lines_without_detector: bool = False,
) -> list[dict[str, Any]]:
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    text_dir = output_dir / "texts"
    text_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(p for p in input_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)
    results: list[dict[str, Any]] = []
    progress = ProgressLogger("ocr folder", len(image_paths), log_every)
    total_regions = 0
    fallback_count = 0
    for step, image_path in enumerate(image_paths, start=1):
        result = run_ocr_image(
            image_path=image_path,
            recognizer=recognizer,
            detector=detector,
            device=device,
            skip_detector=skip_detector,
            output_dir=output_dir,
            save_crops=save_crops,
            save_visualizations=save_visualizations,
            detector_size=detector_size,
            threshold=threshold,
            charset=charset,
            rec_height=rec_height,
            rec_width=rec_width,
            split_lines_without_detector=split_lines_without_detector,
        )
        (text_dir / f"{image_path.stem}.txt").write_text(result["text"])
        results.append(result)
        region_count = len(result["regions"]) if result["regions"] else 1
        total_regions += region_count
        fallback_count += int(bool(result.get("used_line_split_fallback")))
        if progress.should_log(step):
            avg_regions = total_regions / step
            progress.log(
                step,
                f"images done {step} avg_regions {avg_regions:.1f} line_split_fallbacks {fallback_count} loss n/a accuracy n/a"
                " (no OCR labels provided)",
            )

    (output_dir / "predictions.json").write_text(json.dumps({"predictions": results}, indent=2))
    return results
