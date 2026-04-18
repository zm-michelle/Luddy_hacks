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

PARAGRAPH_SENTENCES = [
    "There are several classic spatial filters for reducing image noise from scanned documents.",
    "The mean filter and the median filter replace each pixel using information from the neighborhood.",
    "This procedure reduces image noise but blurs the image and changes the appearance of small letters.",
    "The main goal was to train a neural network in a supervised way to recover a clean image from a noisy one.",
    "In this particular case, it was much easier to restore a noisy image from a clean one than to clean every image manually.",
    "A new printed document collection has recently been prepared for experiments with degraded office images.",
    "The database contains pages with several typefaces, font sizes, emphasized text, stains, folds, and low contrast.",
    "First of all, most documents do not contain the same amount of text or the same line spacing.",
    "Another important reason was to create restricted tasks commonly used in document analysis and recognition.",
    "The forms also contain a brief set of instructions given to the reader before the document was scanned.",
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
        self._cache: dict[tuple[str, int, bool, bool], ImageFont.FreeTypeFont | ImageFont.ImageFont] = {}

    def _scan_fonts(self) -> list[Path]:
        paths: list[Path] = []
        for root in COMMON_FONT_DIRS:
            if root.exists():
                paths.extend(root.rglob("*.ttf"))
                paths.extend(root.rglob("*.otf"))
                paths.extend(root.rglob("*.ttc"))
        return paths

    def _find_path(self, family: str, bold: bool = False, italic: bool = False) -> Path | None:
        names = FONT_FAMILIES.get(family, FONT_FAMILIES["serif"])
        preferred_styles: list[tuple[bool, bool]] = [(bold, italic)]
        if italic:
            preferred_styles.append((False, True))
        if bold:
            preferred_styles.append((True, False))
        preferred_styles.append((False, False))
        for name in names:
            hints = FONT_FILE_HINTS.get(name, [name.lower()])
            for want_bold, want_italic in preferred_styles:
                for path in self._font_paths:
                    normalized = path.name.lower().replace("-", " ").replace("_", " ")
                    if not any(hint.lower().replace("-", " ").replace("_", " ") in normalized for hint in hints):
                        continue
                    has_bold = "bold" in normalized
                    has_italic = "italic" in normalized or "oblique" in normalized
                    if want_bold and not has_bold:
                        continue
                    if want_italic and not has_italic:
                        continue
                    if not want_bold and bold and has_bold:
                        continue
                    return path
        return None

    def get(
        self,
        family: str,
        size: int,
        bold: bool = False,
        italic: bool = False,
    ) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
        key = (family, size, bold, italic)
        if key in self._cache:
            return self._cache[key]
        path = self._find_path(family, bold=bold, italic=italic)
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


def _paragraph_words() -> list[str]:
    sentences = random.sample(PARAGRAPH_SENTENCES, k=random.randint(4, 7))
    if random.random() < 0.35:
        sentences.append(random.choice(PARAGRAPH_SENTENCES))
    return " ".join(sentences).split()


def _wrap_words_to_lines(
    draw: ImageDraw.ImageDraw,
    font: ImageFont.ImageFont,
    target_width: int,
    n_lines: int,
) -> list[str]:
    words = _paragraph_words()
    lines: list[str] = []
    cursor = 0
    for _ in range(n_lines):
        if cursor >= len(words):
            words.extend(_paragraph_words())
        line_words: list[str] = []
        while cursor < len(words):
            candidate = " ".join(line_words + [words[cursor]])
            if line_words and _text_bbox(draw, candidate, font)[0] > target_width:
                break
            line_words.append(words[cursor])
            cursor += 1
        line = " ".join(line_words)
        if random.random() < 0.18:
            line = "  ".join(line.split(" ", 1)) if " " in line else line
        lines.append(line)
    return lines


def _text_bbox(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> tuple[int, int]:
    box = draw.textbbox((0, 0), text, font=font)
    return box[2] - box[0], box[3] - box[1]


def _add_paper_texture(image: Image.Image, strength: float = 0.55) -> Image.Image:
    arr = np.asarray(image).astype(np.float32)
    h, w = arr.shape
    fine = np.random.normal(0, 4.5 * strength, arr.shape)
    coarse_small = np.random.normal(0, 11 * strength, (max(2, h // 18), max(2, w // 18))).astype(np.float32)
    coarse = cv2.resize(coarse_small, (w, h), interpolation=cv2.INTER_CUBIC)
    arr += fine + coarse
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def _add_fold_or_wrinkle(image: Image.Image, strength: float = 0.45) -> Image.Image:
    arr = np.asarray(image).astype(np.float32)
    h, w = arr.shape
    for _ in range(random.randint(1, 3)):
        if random.random() < 0.55:
            center = random.randint(0, w - 1)
            width = random.uniform(8, 28)
            x = np.arange(w)[None, :]
            shadow = np.exp(-((x - center) ** 2) / (2 * width**2))
            arr -= shadow * random.uniform(10, 28) * strength
        else:
            center = random.randint(0, h - 1)
            width = random.uniform(5, 18)
            y = np.arange(h)[:, None]
            shadow = np.exp(-((y - center) ** 2) / (2 * width**2))
            arr -= shadow * random.uniform(8, 22) * strength
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def _apply_degradation(image: Image.Image, cfg: DegradationConfig) -> Image.Image:
    s = max(0.0, min(1.0, cfg.severity))
    out = image.convert("L")
    out = _add_paper_texture(out, strength=0.35 + s * 0.45)
    if random.random() < 0.40:
        out = _add_fold_or_wrinkle(out, strength=s)
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
            font = resolver.get(
                line["font_family"],
                line["font_size"],
                line["bold_like"],
                line.get("italic_like", False),
            )
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
    render_profile: str = "noisyoffice",
) -> tuple[Image.Image, list[dict]]:
    width, height = page_size
    background = random.randint(232, 247) if render_profile == "noisyoffice" else 255
    image = Image.new("L", (width, height), background)
    draw = ImageDraw.Draw(image)

    if render_profile == "noisyoffice":
        family = random.choices(["serif", "sans", "typewriter"], weights=[0.50, 0.30, 0.20])[0]
        style = random.choices(["footnote", "normal", "large"], weights=[0.22, 0.68, 0.10])[0]
        font_size = {"footnote": random.randint(14, 16), "normal": random.randint(17, 20), "large": random.randint(20, 23)}[style]
        bold = random.random() < 0.18
        italic = family == "serif" and random.random() < 0.65
        font = resolver.get(family, font_size, bold=bold, italic=italic)
        target_min = max(8, min_lines)
        target_max = max(target_min, min(10, max_lines))
        n_lines = random.randint(target_min, target_max)
        line_height = font_size + random.randint(8, 12)
        y = random.randint(-3, 5)
        wrap_width = random.randint(int(width * 0.96), int(width * 1.18))
        source_lines = _wrap_words_to_lines(draw, font, wrap_width, n_lines)
        x_base = random.randint(-22, 26)
    else:
        family = random.choice(list(FONT_FAMILIES))
        style = random.choices(["footnote", "normal", "large"], weights=[0.18, 0.68, 0.14])[0]
        font_size = _font_size_for_style(style)
        bold = random.random() < 0.22
        italic = False
        font = resolver.get(family, font_size, bold=bold)
        margin_x = random.randint(max(18, width // 28), max(24, width // 10))
        margin_top = random.randint(max(18, height // 32), max(32, height // 10))
        line_gap = random.randint(max(4, font_size // 4), max(8, font_size))
        line_height = font_size + line_gap
        usable_h = height - margin_top - random.randint(24, 90)
        max_fit = max(min_lines, min(max_lines, usable_h // max(1, line_height)))
        n_lines = random.randint(min_lines, max(min_lines, max_fit))
        y = margin_top
        source_lines = [_random_line(max_words={"dense": 13, "normal": 10, "loose": 7}[random.choice(["dense", "normal", "loose"])]) for _ in range(n_lines)]
        x_base = margin_x

    lines: list[dict] = []

    for raw_text in source_lines:
        text = raw_text
        text_w, text_h = _text_bbox(draw, text, font)
        if render_profile == "document":
            max_text_w = width - 2 * x_base
            while text_w > max_text_w and " " in text:
                text = " ".join(text.split()[:-1])
                text_w, text_h = _text_bbox(draw, text, font)

        x = x_base + random.randint(-8, 12)
        if render_profile == "document":
            x = max(2, x)
            if random.random() < 0.10:
                x += random.randint(10, 45)

        fill = random.randint(12, 55) if render_profile == "noisyoffice" else random.randint(0, 45)
        draw.text((x, y), text, fill=fill, font=font)
        visible_left = max(0, -x)
        visible_right = min(int(font.getlength(text)), width - x)
        visible_text = _visible_substring(text, font, visible_left, visible_right)
        if not visible_text:
            y += line_height + random.randint(-1, 2)
            continue
        pad = random.randint(1, 3) if render_profile == "noisyoffice" else random.randint(2, 5)
        bx1 = max(0, x - pad)
        by1 = max(0, y - pad)
        bx2 = min(width, x + text_w + pad)
        by2 = min(height, y + text_h + pad)
        if bx2 <= bx1 or by2 <= by1:
            y += line_height + (random.randint(-1, 2) if render_profile == "noisyoffice" else random.randint(-2, 6))
            continue
        lines.append(
            {
                "text": visible_text,
                "box": [bx1, by1, bx2, by2],
                "font_family": family,
                "font_size": font_size,
                "bold_like": bold,
                "italic_like": italic,
                "origin": [max(0, x), y],
            }
        )
        y += line_height + (random.randint(-1, 2) if render_profile == "noisyoffice" else random.randint(-2, 6))
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
    render_profile: str = "noisyoffice",
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
        clean, lines = _draw_document(page_size, min_lines, max_lines, resolver, render_profile)
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
                        font = resolver.get(
                            line["font_family"],
                            line["font_size"],
                            line["bold_like"],
                            line.get("italic_like", False),
                        )
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
                "render_profile": render_profile,
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
