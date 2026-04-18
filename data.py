from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageDraw
from torch.utils.data import Dataset

from config import DEFAULT_CHARSET, DEFAULT_DETECTOR_SIZE, IMAGE_EXTENSIONS
from decode import encode_text
from transforms import DetectionTransform, RecognitionTransform


def list_image_files(root: str | Path, extensions: tuple[str, ...] = IMAGE_EXTENSIONS) -> list[Path]:
    root = Path(root)
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in extensions)


def _load_manifest(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text())
        if isinstance(data, dict):
            return data.get("samples") or data.get("pages") or data.get("items") or []
        return data
    if path.suffix.lower() == ".csv":
        with path.open(newline="") as f:
            return list(csv.DictReader(f))
    raise ValueError(f"Unsupported manifest format: {path}")


class PrintedLineDataset(Dataset):
    def __init__(
        self,
        manifest_path: str | Path,
        root_dir: str | Path | None = None,
        charset: str = DEFAULT_CHARSET,
        transform: RecognitionTransform | None = None,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.root_dir = Path(root_dir) if root_dir else self.manifest_path.parent
        self.samples = _load_manifest(self.manifest_path)
        self.charset = charset
        self.transform = transform or RecognitionTransform()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = self.samples[idx]
        image_path = Path(item.get("image") or item.get("image_path"))
        if not image_path.is_absolute():
            image_path = self.root_dir / image_path
        text = str(item.get("text") or item.get("transcript") or "")

        image = Image.open(image_path).convert("L")
        target = torch.tensor(encode_text(text, self.charset), dtype=torch.long)
        return {
            "image": self.transform(image),
            "target": target,
            "target_length": torch.tensor(len(target), dtype=torch.long),
            "text": text,
            "path": str(image_path),
        }


def recognition_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    images = torch.stack([item["image"] for item in batch], dim=0)
    targets = [item["target"] for item in batch]
    target_lengths = torch.stack([item["target_length"] for item in batch])
    if targets:
        flat_targets = torch.cat(targets) if sum(len(t) for t in targets) else torch.empty(0, dtype=torch.long)
    else:
        flat_targets = torch.empty(0, dtype=torch.long)
    return {
        "images": images,
        "targets": flat_targets,
        "target_lengths": target_lengths,
        "texts": [item["text"] for item in batch],
        "paths": [item["path"] for item in batch],
    }


class DetectionDataset(Dataset):
    def __init__(
        self,
        manifest_path: str | Path,
        root_dir: str | Path | None = None,
        image_size: tuple[int, int] = DEFAULT_DETECTOR_SIZE,
        transform: DetectionTransform | None = None,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.root_dir = Path(root_dir) if root_dir else self.manifest_path.parent
        self.samples = _load_manifest(self.manifest_path)
        self.transform = transform or DetectionTransform(size=image_size)

    def __len__(self) -> int:
        return len(self.samples)

    def _target_mask(self, size: tuple[int, int], boxes: list[list[int] | tuple[int, int, int, int]]) -> Image.Image:
        mask = Image.new("L", size, 0)
        draw = ImageDraw.Draw(mask)
        for box in boxes:
            x1, y1, x2, y2 = [int(v) for v in box]
            if x2 > x1 and y2 > y1:
                draw.rectangle((x1, y1, x2, y2), fill=255)
        return mask

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = self.samples[idx]
        image_path = Path(item.get("image") or item.get("image_path"))
        if not image_path.is_absolute():
            image_path = self.root_dir / image_path
        boxes = item.get("boxes") or [line["box"] for line in item.get("lines", [])]

        image = Image.open(image_path).convert("L")
        mask = self._target_mask(image.size, boxes)
        image_tensor, mask_tensor = self.transform(image, mask)
        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "boxes": boxes,
            "path": str(image_path),
        }


def detection_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "image": torch.stack([item["image"] for item in batch], dim=0),
        "mask": torch.stack([item["mask"] for item in batch], dim=0),
        "boxes": [item["boxes"] for item in batch],
        "paths": [item["path"] for item in batch],
    }


class OCRFolderDataset(Dataset):
    def __init__(self, input_dir: str | Path, extensions: tuple[str, ...] = IMAGE_EXTENSIONS) -> None:
        self.input_dir = Path(input_dir)
        self.paths = list_image_files(self.input_dir, extensions)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        path = self.paths[idx]
        return {"image": Image.open(path).convert("L"), "path": str(path)}
