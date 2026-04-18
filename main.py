from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from config import (
    DEFAULT_CHARSET,
    DEFAULT_DETECTOR_SIZE,
    DEFAULT_PAGE_SIZE,
    DEFAULT_RECOGNIZER_HEIGHT,
    DEFAULT_RECOGNIZER_WIDTH,
)
from data import DetectionDataset, PrintedLineDataset, detection_collate, recognition_collate
from model import CRNNRecognizer, DBNet
from synthetic_data import DegradationConfig, generate_synthetic_dataset
from transforms import DetectionTransform, RecognitionTransform
from utils import (
    benchmark,
    load_checkpoint,
    run_ocr_folder,
    save_checkpoint,
    set_seed,
    train_detector,
    train_recognizer,
    validate_detector,
    validate_recognizer,
)


def parse_size(value: str) -> tuple[int, int]:
    if "x" not in value.lower():
        raise argparse.ArgumentTypeError("Expected WxH, for example 768x1024.")
    w, h = value.lower().split("x", 1)
    return int(w), int(h)


def device_from_arg(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def build_loader(dataset, args, shuffle: bool, collate_fn=None) -> DataLoader:
    kwargs = {
        "batch_size": args.batch_size,
        "shuffle": shuffle,
        "num_workers": args.num_workers,
        "pin_memory": args.pin_memory,
        "collate_fn": collate_fn,
    }
    if args.num_workers > 0:
        kwargs["prefetch_factor"] = args.prefetch_factor
    return DataLoader(dataset, **kwargs)


def maybe_compile(model: torch.nn.Module, enabled: bool) -> torch.nn.Module:
    if enabled and hasattr(torch, "compile"):
        return torch.compile(model)
    return model


def build_detector(args, device: torch.device) -> DBNet:
    model = DBNet().to(device)
    if args.detector_ckpt:
        load_checkpoint(args.detector_ckpt, model, device=device, strict=False)
    return maybe_compile(model, args.compile_model)


def build_recognizer(args, device: torch.device) -> CRNNRecognizer:
    model = CRNNRecognizer(charset=DEFAULT_CHARSET).to(device)
    if args.recognizer_ckpt:
        load_checkpoint(args.recognizer_ckpt, model, device=device, strict=False)
    return maybe_compile(model, args.compile_model)


def mode_generate(args) -> None:
    cfg = DegradationConfig(
        gaussian_noise=not args.no_gaussian_noise,
        salt_pepper=not args.no_salt_pepper,
        blur=not args.no_blur,
        low_contrast=not args.no_low_contrast,
        uneven_illumination=not args.no_uneven_illumination,
        smudges=not args.no_smudges,
        stains=not args.no_stains,
        faded_print=not args.no_faded_print,
        speckle=not args.no_speckle,
        compression=args.compression,
        severity=args.degradation_severity,
    )
    paths = generate_synthetic_dataset(
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        page_size=args.page_size,
        min_lines=args.min_lines,
        max_lines=args.max_lines,
        make_line_crops=not args.no_line_crops,
        line_crop_truncation_prob=args.line_crop_truncation_prob,
        page_crop_prob=args.page_crop_prob,
        degradation=cfg,
        preview_count=args.preview_count,
        seed=args.seed,
        log_every=args.log_every,
        render_profile=args.render_profile,
    )
    print("synthetic data written:")
    for name, path in paths.items():
        print(f"  {name}: {path}")


def mode_train_detector(args, device: torch.device) -> None:
    transform = DetectionTransform(size=args.detector_size, augment=args.augment)
    train_ds = DetectionDataset(args.detector_manifest, root_dir=args.data_root, image_size=args.detector_size, transform=transform)
    train_loader = build_loader(train_ds, args, shuffle=True, collate_fn=detection_collate)
    val_loader = None
    if args.val_detector_manifest:
        val_ds = DetectionDataset(
            args.val_detector_manifest,
            root_dir=args.data_root,
            image_size=args.detector_size,
            transform=DetectionTransform(size=args.detector_size, augment=False),
        )
        val_loader = build_loader(val_ds, args, shuffle=False, collate_fn=detection_collate)

    model = build_detector(args, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        train_stats = train_detector(model, train_loader, optimizer, device, epoch, args.amp, args.grad_accum_steps, args.log_every)
        print(f"detector epoch {epoch}: train_loss={train_stats['loss']:.4f}")
        if val_loader is not None:
            val_stats = validate_detector(model, val_loader, device, args.amp, args.threshold)
            print(
                "detector val: "
                f"loss={val_stats['loss']:.4f} precision={val_stats['precision']:.3f} "
                f"recall={val_stats['recall']:.3f} f1={val_stats['f1']:.3f}"
            )
        save_checkpoint(output_dir / "detector_last.pt", model, optimizer, epoch)


def mode_train_recognizer(args, device: torch.device) -> None:
    train_ds = PrintedLineDataset(
        args.line_manifest,
        root_dir=args.data_root,
        charset=DEFAULT_CHARSET,
        transform=RecognitionTransform(args.rec_height, args.rec_width, augment=args.augment),
    )
    train_loader = build_loader(train_ds, args, shuffle=True, collate_fn=recognition_collate)
    val_loader = None
    if args.val_line_manifest:
        val_ds = PrintedLineDataset(
            args.val_line_manifest,
            root_dir=args.data_root,
            charset=DEFAULT_CHARSET,
            transform=RecognitionTransform(args.rec_height, args.rec_width, augment=False),
        )
        val_loader = build_loader(val_ds, args, shuffle=False, collate_fn=recognition_collate)

    model = build_recognizer(args, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        train_stats = train_recognizer(model, train_loader, optimizer, device, epoch, args.amp, args.grad_accum_steps, args.log_every)
        print(f"recognizer epoch {epoch}: train_loss={train_stats['loss']:.4f}")
        if val_loader is not None:
            val_stats = validate_recognizer(model, val_loader, device, DEFAULT_CHARSET, args.amp)
            print(
                "recognizer val: "
                f"loss={val_stats['loss']:.4f} cer={val_stats['cer']:.3f} "
                f"wer={val_stats['wer']:.3f} exact={val_stats['exact_match']:.3f}"
            )
        save_checkpoint(output_dir / "recognizer_last.pt", model, optimizer, epoch, {"charset": DEFAULT_CHARSET})


def mode_test_detector(args, device: torch.device) -> None:
    ds = DetectionDataset(
        args.detector_manifest,
        root_dir=args.data_root,
        image_size=args.detector_size,
        transform=DetectionTransform(size=args.detector_size, augment=False),
    )
    loader = build_loader(ds, args, shuffle=False, collate_fn=detection_collate)
    model = build_detector(args, device)
    stats = validate_detector(model, loader, device, args.amp, args.threshold)
    print(json.dumps(stats, indent=2))


def mode_test_recognizer(args, device: torch.device) -> None:
    ds = PrintedLineDataset(
        args.line_manifest,
        root_dir=args.data_root,
        charset=DEFAULT_CHARSET,
        transform=RecognitionTransform(args.rec_height, args.rec_width, augment=False),
    )
    loader = build_loader(ds, args, shuffle=False, collate_fn=recognition_collate)
    model = build_recognizer(args, device)
    stats = validate_recognizer(model, loader, device, DEFAULT_CHARSET, args.amp)
    print(json.dumps(stats, indent=2))


def mode_ocr_folder(args, device: torch.device) -> None:
    recognizer = build_recognizer(args, device)
    detector = None if args.skip_detector else build_detector(args, device)
    results = run_ocr_folder(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        recognizer=recognizer,
        detector=detector,
        device=device,
        skip_detector=args.skip_detector,
        save_crops=args.save_crops,
        save_visualizations=args.save_visualizations,
        detector_size=args.detector_size,
        threshold=args.threshold,
        charset=DEFAULT_CHARSET,
        rec_height=args.rec_height,
        rec_width=args.rec_width,
        log_every=args.log_every,
    )
    print(f"wrote OCR for {len(results)} image(s) to {args.output_dir}")


def mode_benchmark(args, device: torch.device) -> None:
    predictions_path = Path(args.predictions_json) if args.predictions_json else Path(args.output_dir) / "predictions.json"
    if not predictions_path.exists():
        if not args.input_dir:
            raise ValueError("Provide --predictions_json or --input_dir for benchmark mode.")
        mode_ocr_folder(args, device)
    report = benchmark(predictions_path, labels_path=args.labels_manifest, detection_labels_path=args.detector_labels_manifest)
    print(json.dumps(report, indent=2))


def mode_smoke_test(args, device: torch.device) -> None:
    smoke_dir = Path(args.output_dir) / "smoke_synth"
    generate_synthetic_dataset(
        smoke_dir,
        num_samples=2,
        page_size=args.page_size,
        preview_count=2,
        seed=args.seed,
        log_every=1,
        render_profile=args.render_profile,
    )

    detector = DBNet().to(device)
    recognizer = CRNNRecognizer().to(device)
    x_page = torch.randn(1, 1, args.detector_size[1], args.detector_size[0], device=device)
    x_line = torch.randn(1, 1, args.rec_height, args.rec_width, device=device)
    with torch.no_grad():
        det_out = detector(x_page)
        rec_out = recognizer(x_line)
    print(f"detector forward: {tuple(det_out.shape)}")
    print(f"recognizer forward: {tuple(rec_out.shape)}")
    print(f"synthetic previews: {smoke_dir / 'previews'}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Practical OCR for degraded printed document text.")
    parser.add_argument(
        "--mode",
        required=True,
        choices=[
            "generate_synth_data",
            "train_detector",
            "train_recognizer",
            "test_detector",
            "test_recognizer",
            "benchmark",
            "ocr_folder",
            "smoke_test",
        ],
    )
    parser.add_argument("--input_dir", type=str)
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--data_root", type=str)
    parser.add_argument("--detector_manifest", type=str)
    parser.add_argument("--val_detector_manifest", type=str)
    parser.add_argument("--line_manifest", type=str)
    parser.add_argument("--val_line_manifest", type=str)
    parser.add_argument("--labels_manifest", type=str)
    parser.add_argument("--detector_labels_manifest", type=str)
    parser.add_argument("--predictions_json", type=str)
    parser.add_argument("--detector_ckpt", type=str)
    parser.add_argument("--recognizer_ckpt", type=str)

    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--pin_memory", action="store_true")
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--compile_model", action="store_true")
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1337)

    parser.add_argument("--detector_size", type=parse_size, default=DEFAULT_DETECTOR_SIZE)
    parser.add_argument("--rec_height", type=int, default=DEFAULT_RECOGNIZER_HEIGHT)
    parser.add_argument("--rec_width", type=int, default=DEFAULT_RECOGNIZER_WIDTH)
    parser.add_argument("--threshold", type=float, default=0.35)
    parser.add_argument("--skip_detector", action="store_true")
    parser.add_argument("--save_crops", action="store_true")
    parser.add_argument("--save_visualizations", action="store_true")

    parser.add_argument("--num_samples", type=int, default=5000)
    parser.add_argument("--page_size", type=parse_size, default=DEFAULT_PAGE_SIZE)
    parser.add_argument("--min_lines", type=int, default=7)
    parser.add_argument("--max_lines", type=int, default=10)
    parser.add_argument("--preview_count", type=int, default=12)
    parser.add_argument("--no_line_crops", action="store_true")
    parser.add_argument("--line_crop_truncation_prob", type=float, default=0.20)
    parser.add_argument("--page_crop_prob", type=float, default=0.35)
    parser.add_argument("--render_profile", choices=["noisyoffice", "document"], default="noisyoffice")
    parser.add_argument("--degradation_severity", type=float, default=0.7)
    parser.add_argument("--compression", action="store_true")
    parser.add_argument("--no_gaussian_noise", action="store_true")
    parser.add_argument("--no_salt_pepper", action="store_true")
    parser.add_argument("--no_blur", action="store_true")
    parser.add_argument("--no_low_contrast", action="store_true")
    parser.add_argument("--no_uneven_illumination", action="store_true")
    parser.add_argument("--no_smudges", action="store_true")
    parser.add_argument("--no_stains", action="store_true")
    parser.add_argument("--no_faded_print", action="store_true")
    parser.add_argument("--no_speckle", action="store_true")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    set_seed(args.seed)
    device = device_from_arg(args.device)
    print(f"device: {device}")

    if args.mode == "generate_synth_data":
        mode_generate(args)
    elif args.mode == "train_detector":
        if not args.detector_manifest:
            raise ValueError("--detector_manifest is required for train_detector")
        mode_train_detector(args, device)
    elif args.mode == "train_recognizer":
        if not args.line_manifest:
            raise ValueError("--line_manifest is required for train_recognizer")
        mode_train_recognizer(args, device)
    elif args.mode == "test_detector":
        if not args.detector_manifest or not args.detector_ckpt:
            raise ValueError("--detector_manifest and --detector_ckpt are required for test_detector")
        mode_test_detector(args, device)
    elif args.mode == "test_recognizer":
        if not args.line_manifest or not args.recognizer_ckpt:
            raise ValueError("--line_manifest and --recognizer_ckpt are required for test_recognizer")
        mode_test_recognizer(args, device)
    elif args.mode == "ocr_folder":
        if not args.input_dir or not args.recognizer_ckpt:
            raise ValueError("--input_dir and --recognizer_ckpt are required for ocr_folder")
        mode_ocr_folder(args, device)
    elif args.mode == "benchmark":
        mode_benchmark(args, device)
    elif args.mode == "smoke_test":
        mode_smoke_test(args, device)


if __name__ == "__main__":
    main()
