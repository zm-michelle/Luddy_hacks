from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from config import COMMON_FONT_DIRS, DEFAULT_PAGE_SIZE, FONT_FAMILIES, FONT_FILE_HINTS
from transforms import (
    add_gaussian_noise,
    add_salt_pepper,
    add_smudge,
    add_stain,
    add_uneven_illumination,
    mild_compression_artifacts,
)
from utils import ProgressLogger


OFFICE_WORDS = [
    "account",
    "analysis",
    "archive",
    "balance",
    "budget",
    "client",
    "contract",
    "department",
    "document",
    "estimate",
    "finance",
    "invoice",
    "ledger",
    "manager",
    "meeting",
    "office",
    "policy",
    "printed",
    "project",
    "quality",
    "receipt",
    "record",
    "report",
    "review",
    "schedule",
    "service",
    "shipping",
    "summary",
    "system",
    "total",
    "transfer",
    "vendor",
    "version",
]


@dataclass
class DegradationConfig:
    gaussian_noise: bool = True
    salt_pepper: bool = True
    blur: bool = True
    low_contrast: bool = True
    uneven_illumination: bool = True
    smudges: bool = True
    stains: bool = True
    faded_print: bool = True
    speckle: bool = True
    compression: bool = False
    severity: float = 0.7


class FontResolver:
    def __init__(self) -> None:
        self._font_paths = self._scan_fonts()
        self._cache: dict[tuple[str, int, bool], ImageFont.FreeTypeFont | ImageFont.ImageFont] = {}

    def _scan_fonts(self) -> list[Path]:
        paths: list[Path] = []
        for root in COMMON_FONT_DIRS:
            if root.exists():
                paths.extend(root.rglob("*.ttf"))
                paths.extend(root.rglob("*.otf"))
                paths.extend(root.rglob("*.ttc"))
        return paths

    def _find_path(self, family: str, bold: bool = False) -> Path | None:
        names = FONT_FAMILIES.get(family, FONT_FAMILIES["serif"])
        for name in names:
            hints = FONT_FILE_HINTS.get(name, [name.lower()])
            for path in self._font_paths:
                normalized = path.name.lower().replace("-", " ").replace("_", " ")
                if any(hint.lower().replace("-", " ").replace("_", " ") in normalized for hint in hints):
                    if bold and "bold" not in normalized:
                        continue
                    return path
            for path in self._font_paths:
                normalized = path.name.lower().replace("-", " ").replace("_", " ")
                if any(hint.lower().replace("-", " ").replace("_", " ") in normalized for hint in hints):
                    return path
        return None

    def get(self, family: str, size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
        key = (family, size, bold)
        if key in self._cache:
            return self._cache[key]
        path = self._find_path(family, bold=bold)
        try:
            font = ImageFont.truetype(str(path), size=size) if path else ImageFont.load_default()
        except OSError:
            font = ImageFont.load_default()
        self._cache[key] = font
        return font


def _font_size_for_style(style: str) -> int:
    if style == "footnote":
        return random.randint(14, 18)
    if style == "large":
        return random.randint(28, 40)
    return random.randint(20, 28)


def _random_line(max_words: int) -> str:
    words = random.randint(4, max_words)
    pieces: list[str] = []
    for i in range(words):
        word = random.choice(OFFICE_WORDS)
        r = random.random()
        if r < 0.12:
            word = word.upper()
        elif r < 0.28:
            word = word.capitalize()
        if random.random() < 0.08:
            word += str(random.randint(1, 99))
        pieces.append(word)
        if i < words - 1 and random.random() < 0.12:
            pieces[-1] += random.choice([",", ";", ":", "-"])
    line = " ".join(pieces)
    if random.random() < 0.55:
        line += random.choice([".", ".", ":", ";"])
    return line


def _text_bbox(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> tuple[int, int]:
    box = draw.textbbox((0, 0), text, font=font)
    return box[2] - box[0], box[3] - box[1]


def _apply_degradation(image: Image.Image, cfg: DegradationConfig) -> Image.Image:
    s = max(0.0, min(1.0, cfg.severity))
    out = image.convert("L")
    if cfg.faded_print and random.random() < 0.65:
        arr = np.asarray(out).astype(np.float32)
        dark = arr < 180
        arr[dark] = arr[dark] * (1 - 0.25 * s) + 255 * (0.25 * s)
        out = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    if cfg.low_contrast and random.random() < 0.75:
        arr = np.asarray(out).astype(np.float32)
        arr = 128 + (arr - 128) * random.uniform(0.55, 1.0)
        out = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    if cfg.uneven_illumination and random.random() < 0.55:
        out = add_uneven_illumination(out, strength=0.18 * s)
    if cfg.stains and random.random() < 0.45:
        out = add_stain(out, opacity=random.uniform(0.08, 0.22) * s)
    if cfg.smudges and random.random() < 0.35:
        out = add_smudge(out, strength=random.randint(3, 9))
    if cfg.blur and random.random() < 0.45:
        radius = random.uniform(0.25, 1.0 + s)
        arr = cv2.GaussianBlur(np.asarray(out), (0, 0), radius)
        out = Image.fromarray(arr)
    if cfg.gaussian_noise and random.random() < 0.75:
        out = add_gaussian_noise(out, sigma=random.uniform(3, 18) * s)
    if cfg.salt_pepper and random.random() < 0.55:
        out = add_salt_pepper(out, amount=random.uniform(0.002, 0.02) * s)
    if cfg.speckle and random.random() < 0.45:
        arr = np.asarray(out).copy()
        count = int(arr.size * random.uniform(0.0008, 0.004) * s)
        if count:
            ys = np.random.randint(0, arr.shape[0], count)
            xs = np.random.randint(0, arr.shape[1], count)
            arr[ys, xs] = np.random.randint(0, 80, count)
        out = Image.fromarray(arr)
    if cfg.compression and random.random() < 0.35:
        out = mild_compression_artifacts(out, quality=random.randint(35, 70))
    return out


def _visible_substring(text: str, font: ImageFont.ImageFont, start_px: int, end_px: int) -> str:
    if start_px <= 0 and end_px >= int(font.getlength(text)):
        return text
    spans: list[tuple[float, float, str]] = []
    cursor = 0.0
    for ch in text:
        next_cursor = cursor + float(font.getlength(ch))
        spans.append((cursor, next_cursor, ch))
        cursor = next_cursor
    chars = [ch for left, right, ch in spans if start_px <= (left + right) / 2 <= end_px]
    return "".join(chars)


def _crop_page_edges(
    image: Image.Image,
    lines: list[dict],
    crop_prob: float,
    resolver: FontResolver,
) -> tuple[Image.Image, list[dict]]:
    if random.random() >= crop_prob:
        return image, lines
    w, h = image.size
    left = random.randint(0, max(1, int(w * 0.08))) if random.random() < 0.55 else 0
    top = random.randint(0, max(1, int(h * 0.05))) if random.random() < 0.35 else 0
    right = w - (random.randint(0, max(1, int(w * 0.08))) if random.random() < 0.55 else 0)
    bottom = h - (random.randint(0, max(1, int(h * 0.05))) if random.random() < 0.35 else 0)
    cropped = image.crop((left, top, right, bottom))
    adjusted: list[dict] = []
    for line in lines:
        x1, y1, x2, y2 = line["box"]
        nx1, ny1 = max(0, x1 - left), max(0, y1 - top)
        nx2, ny2 = min(right - left, x2 - left), min(bottom - top, y2 - top)
        if nx2 - nx1 >= 8 and ny2 - ny1 >= 8:
            new_line = dict(line)
            text_x = line.get("origin", [x1, y1])[0]
            font = resolver.get(line["font_family"], line["font_size"], line["bold_like"])
            text_width = int(font.getlength(line["text"]))
            visible_left = max(0, left - text_x)
            visible_right = min(text_width, right - text_x)
            new_line["text"] = _visible_substring(line["text"], font, visible_left, visible_right)
            new_line["box"] = [nx1, ny1, nx2, ny2]
            if new_line["text"]:
                adjusted.append(new_line)
    return cropped, adjusted


def _draw_document(
    page_size: tuple[int, int],
    min_lines: int,
    max_lines: int,
    resolver: FontResolver,
) -> tuple[Image.Image, list[dict]]:
    width, height = page_size
    image = Image.new("L", (width, height), 255)
    draw = ImageDraw.Draw(image)
    family = random.choice(list(FONT_FAMILIES))
    style = random.choices(["footnote", "normal", "large"], weights=[0.18, 0.68, 0.14])[0]
    font_size = _font_size_for_style(style)
    bold = random.random() < 0.22
    font = resolver.get(family, font_size, bold=bold)

    margin_x = random.randint(max(18, width // 28), max(24, width // 10))
    margin_top = random.randint(max(18, height // 32), max(32, height // 10))
    line_gap = random.randint(max(4, font_size // 4), max(8, font_size))
    line_height = font_size + line_gap
    usable_h = height - margin_top - random.randint(24, 90)
    max_fit = max(min_lines, min(max_lines, usable_h // max(1, line_height)))
    n_lines = random.randint(min_lines, max(min_lines, max_fit))

    y = margin_top
    lines: list[dict] = []
    density = random.choice(["dense", "normal", "loose"])
    max_words = {"dense": 13, "normal": 10, "loose": 7}[density]

    for _ in range(n_lines):
        text = _random_line(max_words=max_words)
        text_w, text_h = _text_bbox(draw, text, font)
        max_text_w = width - 2 * margin_x
        while text_w > max_text_w and " " in text:
            text = " ".join(text.split()[:-1])
            text_w, text_h = _text_bbox(draw, text, font)
        x_jitter = random.randint(-8, 12)
        x = max(2, margin_x + x_jitter)
        if random.random() < 0.10:
            x += random.randint(10, 45)
        draw.text((x, y), text, fill=random.randint(0, 45), font=font)
        pad = random.randint(2, 5)
        lines.append(
            {
                "text": text,
                "box": [
                    max(0, x - pad),
                    max(0, y - pad),
                    min(width, x + text_w + pad),
                    min(height, y + text_h + pad),
                ],
                "font_family": family,
                "font_size": font_size,
                "bold_like": bold,
                "origin": [x, y],
            }
        )
        y += line_height + random.randint(-2, 6)
    return image, lines


def _save_overlay(image: Image.Image, lines: list[dict], path: Path) -> None:
    overlay = image.convert("RGB")
    draw = ImageDraw.Draw(overlay)
    for line in lines:
        draw.rectangle(line["box"], outline=(220, 40, 40), width=2)
    overlay.save(path)


def generate_synthetic_dataset(
    output_dir: str | Path,
    num_samples: int,
    page_size: tuple[int, int] = DEFAULT_PAGE_SIZE,
    min_lines: int = 7,
    max_lines: int = 14,
    make_line_crops: bool = True,
    line_crop_truncation_prob: float = 0.20,
    page_crop_prob: float = 0.20,
    degradation: DegradationConfig | None = None,
    preview_count: int = 12,
    seed: int | None = None,
    log_every: int = 100,
) -> dict[str, Path]:
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    output_dir = Path(output_dir)
    pages_dir = output_dir / "pages"
    lines_dir = output_dir / "line_crops"
    previews_dir = output_dir / "previews"
    pages_dir.mkdir(parents=True, exist_ok=True)
    lines_dir.mkdir(parents=True, exist_ok=True)
    previews_dir.mkdir(parents=True, exist_ok=True)

    resolver = FontResolver()
    degradation = degradation or DegradationConfig()
    page_manifest: list[dict] = []
    line_manifest: list[dict] = []
    progress = ProgressLogger("synthetic generation", num_samples, log_every)

    for idx in range(num_samples):
        clean, lines = _draw_document(page_size, min_lines, max_lines, resolver)
        clean, lines = _crop_page_edges(clean, lines, page_crop_prob, resolver)
        degraded = _apply_degradation(clean, degradation)
        page_name = f"page_{idx:06d}.png"
        page_path = pages_dir / page_name
        degraded.save(page_path)

        rel_page = str(page_path.relative_to(output_dir))
        page_manifest.append({"image": rel_page, "boxes": [l["box"] for l in lines], "lines": lines})

        if idx < preview_count:
            _save_overlay(degraded, lines, previews_dir / f"page_{idx:06d}_boxes.png")

        if make_line_crops:
            for line_idx, line in enumerate(lines):
                x1, y1, x2, y2 = line["box"]
                crop = degraded.crop((x1, y1, x2, y2))
                text = line["text"]
                if random.random() < line_crop_truncation_prob and crop.width > 40:
                    cut_left = random.randint(0, max(1, int(crop.width * 0.20)))
                    cut_right = crop.width - random.randint(0, max(1, int(crop.width * 0.20)))
                    if cut_right - cut_left >= 20:
                        font = resolver.get(line["font_family"], line["font_size"], line["bold_like"])
                        text = _visible_substring(text, font, cut_left, cut_right)
                        crop = crop.crop((cut_left, 0, cut_right, crop.height))
                if text:
                    crop_name = f"page_{idx:06d}_line_{line_idx:02d}.png"
                    crop_path = lines_dir / crop_name
                    crop.save(crop_path)
                    line_manifest.append(
                        {
                            "image": str(crop_path.relative_to(output_dir)),
                            "text": text,
                            "source_page": rel_page,
                            "box": line["box"],
                        }
                    )
        step = idx + 1
        if progress.should_log(step):
            avg_lines = sum(len(p["lines"]) for p in page_manifest) / step
            progress.log(
                step,
                f"pages {step} line_crops {len(line_manifest)} avg_lines {avg_lines:.1f} loss n/a accuracy n/a",
            )

    pages_manifest_path = output_dir / "pages_manifest.json"
    lines_manifest_path = output_dir / "lines_manifest.json"
    meta_path = output_dir / "metadata.json"
    pages_manifest_path.write_text(json.dumps({"pages": page_manifest}, indent=2))
    lines_manifest_path.write_text(json.dumps({"samples": line_manifest}, indent=2))
    meta_path.write_text(
        json.dumps(
            {
                "num_pages": num_samples,
                "num_line_crops": len(line_manifest),
                "page_size": page_size,
                "min_lines": min_lines,
                "max_lines": max_lines,
                "degradation": asdict(degradation),
            },
            indent=2,
        )
    )
    return {
        "pages_manifest": pages_manifest_path,
        "lines_manifest": lines_manifest_path,
        "metadata": meta_path,
        "previews": previews_dir,
    }
