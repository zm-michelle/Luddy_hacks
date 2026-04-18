# Degraded Printed Document OCR

Small PyTorch OCR project for degraded printed document images. The pipeline is:

```text
full page PNG -> lightweight DBNet-style text segmentation -> sorted crops -> CRNN + CTC -> literal OCR text
cropped PNG   -> CRNN + CTC directly
```

There is no spell correction, lexicon correction, or language-model correction in the default path. If a crop visually contains a partial word, the recognizer output is kept literal.

## Files

- `model.py` - compact `DBNet` detector and `CRNNRecognizer`.
- `synthetic_data.py` - realistic office/document synthetic page and line crop generation.
- `data.py` - recognition, detection, and folder datasets.
- `transforms.py` - resize, normalization, OCR-friendly augmentation, degradation helpers.
- `infer.py` - detector postprocessing, crop recognition, folder OCR output.
- `utils.py` - train/validate loops, checkpoints, metrics, benchmark wrapper.
- `main.py` - command-line entry point.
- `config.py`, `decode.py`, `metrics.py` - shared config, CTC decode, CER/WER/detection metrics.

## Install

```bash
pip install -r requirements.txt
```

CPU works. CUDA is used when available with `--device auto`, or explicitly with `--device cuda`.

## Generate Synthetic Data

```bash
python main.py \
  --mode generate_synth_data \
  --output_dir synthetic_docs \
  --num_samples 5000
```

This creates:

- `synthetic_docs/pages/*.png`
- `synthetic_docs/line_crops/*.png`
- `synthetic_docs/pages_manifest.json` for detector training
- `synthetic_docs/lines_manifest.json` for recognizer training
- `synthetic_docs/previews/*_boxes.png` for quick inspection

Synthetic pages default to 7-14 mostly upright printed lines, constrained office-style fonts, PNG output, and configurable degradations.

Useful variants:

```bash
python main.py --mode generate_synth_data --output_dir synthetic_docs_small --num_samples 50 --preview_count 20
python main.py --mode generate_synth_data --output_dir synthetic_clean --num_samples 1000 --no_stains --no_smudges --degradation_severity 0.3
```

## Train Detector

```bash
python main.py \
  --mode train_detector \
  --detector_manifest synthetic_docs/pages_manifest.json \
  --data_root synthetic_docs \
  --output_dir checkpoints \
  --batch_size 4 \
  --epochs 20 \
  --lr 3e-4
```

For GPU:

```bash
python main.py --mode train_detector --detector_manifest synthetic_docs/pages_manifest.json --data_root synthetic_docs --output_dir checkpoints --device cuda --amp --pin_memory --num_workers 4
```

## Train Recognizer

```bash
python main.py \
  --mode train_recognizer \
  --line_manifest synthetic_docs/lines_manifest.json \
  --data_root synthetic_docs \
  --output_dir checkpoints \
  --batch_size 32 \
  --epochs 30 \
  --lr 3e-4
```

The recognizer keeps uppercase, lowercase, punctuation, digits, and spaces. It trains with CTC and greedy CTC decoding.

## Fine-Tune On Degraded Data

Prepare manifests with the same shape as the synthetic manifests:

Recognition manifest:

```json
{
  "samples": [
    {"image": "line_crops/example.png", "text": "Exact visual transcript"}
  ]
}
```

Detection manifest:

```json
{
  "pages": [
    {
      "image": "pages/example.png",
      "boxes": [[10, 20, 400, 48], [10, 58, 390, 86]]
    }
  ]
}
```

Then load the synthetic checkpoint and continue training:

```bash
python main.py --mode train_recognizer --line_manifest target_lines.json --data_root target_data --recognizer_ckpt checkpoints/recognizer_last.pt --output_dir checkpoints_target
python main.py --mode train_detector --detector_manifest target_pages.json --data_root target_data --detector_ckpt checkpoints/detector_last.pt --output_dir checkpoints_target
```

## OCR A Folder

Full-page OCR with detector:

```bash
python main.py \
  --mode ocr_folder \
  --input_dir SimulatedNoisyOffice/simulated_noisy_images_grayscale \
  --output_dir ocr_outputs \
  --detector_ckpt checkpoints/detector_last.pt \
  --recognizer_ckpt checkpoints/recognizer_last.pt \
  --save_visualizations \
  --save_crops
```

Cropped-image OCR without detector:

```bash
python main.py \
  --mode ocr_folder \
  --input_dir SimulatedNoisyOffice/simulated_noisy_images_grayscale \
  --output_dir ocr_outputs_crops \
  --recognizer_ckpt checkpoints/recognizer_last.pt \
  --skip_detector
```

Outputs:

- `texts/<image>.txt`
- `predictions.json`
- optional `crops/<image>/region_*.png`
- optional `visualizations/*_boxes.png` and `*_prob.png`

## Benchmark

If OCR has already been run:

```bash
python main.py \
  --mode benchmark \
  --predictions_json ocr_outputs/predictions.json \
  --labels_manifest labels.json
```

If no labels are provided, benchmark mode still reports the number of predictions and states that GT metrics were not computed.

To run inference and benchmark in one command:

```bash
python main.py \
  --mode benchmark \
  --input_dir SimulatedNoisyOffice/simulated_noisy_images_grayscale \
  --output_dir ocr_outputs \
  --detector_ckpt checkpoints/detector_last.pt \
  --recognizer_ckpt checkpoints/recognizer_last.pt \
  --labels_manifest labels.json
```

## Smoke Test

```bash
python main.py --mode smoke_test --output_dir outputs_smoke
```

This generates two small synthetic pages and checks detector/recognizer forward passes.

## Notes

- PNG is the primary format throughout generation and inference.
- JPG/JPEG/TIF/TIFF can be read for convenience.
- Detector training can start from synthetic annotations or from a checkpoint.
- Use `--skip_detector` whenever inputs are already cropped text images.
- The provided NoisyOffice folder is an enhancement/binarization dataset. It does not include OCR transcripts, so OCR CER/WER require you to provide text labels separately.
