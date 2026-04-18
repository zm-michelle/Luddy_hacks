from __future__ import annotations

import random
from dataclasses import dataclass

import cv2
import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter

from config import DEFAULT_RECOGNIZER_HEIGHT, DEFAULT_RECOGNIZER_WIDTH


def to_grayscale(image: Image.Image) -> Image.Image:
    return image.convert("L")


def pil_to_tensor(image: Image.Image) -> torch.Tensor:
    arr = np.asarray(to_grayscale(image), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)


def normalize_tensor(tensor: torch.Tensor) -> torch.Tensor:
    return (tensor - 0.5) / 0.5


def resize_page(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    return to_grayscale(image).resize(size, Image.BILINEAR)


def resize_mask(mask: Image.Image, size: tuple[int, int]) -> Image.Image:
    return mask.convert("L").resize(size, Image.NEAREST)


def fit_text_image(
    image: Image.Image,
    height: int = DEFAULT_RECOGNIZER_HEIGHT,
    width: int = DEFAULT_RECOGNIZER_WIDTH,
) -> Image.Image:
    image = to_grayscale(image)
    w, h = image.size
    if h <= 0 or w <= 0:
        return Image.new("L", (width, height), 255)

    new_w = max(1, min(width, int(round(w * (height / h)))))
    resized = image.resize((new_w, height), Image.BILINEAR)
    canvas = Image.new("L", (width, height), 255)
    canvas.paste(resized, (0, 0))
    return canvas


def random_affine_mild(image: Image.Image, max_translate: int = 2) -> Image.Image:
    tx = random.randint(-max_translate, max_translate)
    ty = random.randint(-max_translate, max_translate)
    return image.transform(image.size, Image.AFFINE, (1, 0, tx, 0, 1, ty), fillcolor=255)


def add_gaussian_noise(image: Image.Image, sigma: float) -> Image.Image:
    arr = np.asarray(image).astype(np.float32)
    arr += np.random.normal(0, sigma, arr.shape)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def add_salt_pepper(image: Image.Image, amount: float) -> Image.Image:
    arr = np.asarray(image).copy()
    mask = np.random.random(arr.shape)
    arr[mask < amount / 2] = 0
    arr[(mask >= amount / 2) & (mask < amount)] = 255
    return Image.fromarray(arr.astype(np.uint8))


def add_stain(image: Image.Image, opacity: float = 0.18) -> Image.Image:
    arr = np.asarray(image).astype(np.float32)
    h, w = arr.shape
    cx = random.randint(0, w)
    cy = random.randint(0, h)
    rx = random.randint(max(12, w // 12), max(16, w // 3))
    ry = random.randint(max(12, h // 16), max(16, h // 4))
    yy, xx = np.mgrid[:h, :w]
    blob = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2
    mask = np.clip(1 - blob, 0, 1) ** 2
    stain = random.uniform(90, 170)
    arr = arr * (1 - opacity * mask) + stain * (opacity * mask)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def add_uneven_illumination(image: Image.Image, strength: float = 0.18) -> Image.Image:
    arr = np.asarray(image).astype(np.float32)
    h, w = arr.shape
    left = random.uniform(1 - strength, 1 + strength)
    right = random.uniform(1 - strength, 1 + strength)
    top = random.uniform(1 - strength, 1 + strength)
    bottom = random.uniform(1 - strength, 1 + strength)
    x_grad = np.linspace(left, right, w)[None, :]
    y_grad = np.linspace(top, bottom, h)[:, None]
    arr *= (x_grad + y_grad) / 2
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def add_smudge(image: Image.Image, strength: int = 7) -> Image.Image:
    arr = np.asarray(image)
    k = max(3, strength | 1)
    blurred = cv2.GaussianBlur(arr, (k, k), 0)
    mask = np.zeros_like(arr, dtype=np.float32)
    h, w = arr.shape
    for _ in range(random.randint(1, 3)):
        x1, y1 = random.randint(0, w), random.randint(0, h)
        x2 = min(w - 1, max(0, x1 + random.randint(-w // 4, w // 4)))
        y2 = min(h - 1, max(0, y1 + random.randint(-h // 8, h // 8)))
        cv2.line(mask, (x1, y1), (x2, y2), 1.0, random.randint(5, 18))
    mask = cv2.GaussianBlur(mask, (21, 21), 0)[..., None] if mask.ndim == 3 else cv2.GaussianBlur(mask, (21, 21), 0)
    out = arr.astype(np.float32) * (1 - mask) + blurred.astype(np.float32) * mask
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def mild_compression_artifacts(image: Image.Image, quality: int = 45) -> Image.Image:
    arr = np.asarray(image)
    ok, encoded = cv2.imencode(".jpg", arr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return image
    decoded = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)
    return Image.fromarray(decoded)


@dataclass
class RecognitionTransform:
    height: int = DEFAULT_RECOGNIZER_HEIGHT
    width: int = DEFAULT_RECOGNIZER_WIDTH
    augment: bool = False

    def __call__(self, image: Image.Image) -> torch.Tensor:
        image = to_grayscale(image)
        if self.augment:
            if random.random() < 0.25:
                image = random_affine_mild(image)
            if random.random() < 0.35:
                image = ImageEnhance.Contrast(image).enhance(random.uniform(0.65, 1.35))
            if random.random() < 0.25:
                image = image.filter(ImageFilter.GaussianBlur(random.uniform(0.2, 1.0)))
            if random.random() < 0.25:
                image = add_gaussian_noise(image, random.uniform(3, 12))
            if random.random() < 0.15:
                image = add_salt_pepper(image, random.uniform(0.002, 0.015))
        return normalize_tensor(pil_to_tensor(fit_text_image(image, self.height, self.width)))


@dataclass
class DetectionTransform:
    size: tuple[int, int]
    augment: bool = False

    def __call__(self, image: Image.Image, mask: Image.Image) -> tuple[torch.Tensor, torch.Tensor]:
        image = to_grayscale(image)
        if self.augment:
            if random.random() < 0.35:
                image = ImageEnhance.Contrast(image).enhance(random.uniform(0.7, 1.3))
            if random.random() < 0.25:
                image = image.filter(ImageFilter.GaussianBlur(random.uniform(0.2, 0.9)))
            if random.random() < 0.20:
                image = add_gaussian_noise(image, random.uniform(2, 10))
        image = resize_page(image, self.size)
        mask = resize_mask(mask, self.size)
        image_tensor = normalize_tensor(pil_to_tensor(image))
        mask_tensor = (pil_to_tensor(mask) > 0.5).float()
        return image_tensor, mask_tensor
