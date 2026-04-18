from __future__ import annotations


def _edit_distance(a: list[str] | str, b: list[str] | str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost))
        prev = cur
    return prev[-1]


def compute_cer(pred: str, target: str) -> float:
    if not target:
        return 0.0 if not pred else 1.0
    return _edit_distance(pred, target) / len(target)


def compute_wer(pred: str, target: str) -> float:
    target_words = target.split()
    if not target_words:
        return 0.0 if not pred.split() else 1.0
    return _edit_distance(pred.split(), target_words) / len(target_words)


def exact_match_ratio(predictions: list[str], targets: list[str]) -> float:
    if not targets:
        return 0.0
    return sum(p == t for p, t in zip(predictions, targets)) / len(targets)


def box_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union else 0.0


def detection_f1(
    pred_boxes: list[tuple[int, int, int, int]],
    gt_boxes: list[tuple[int, int, int, int]],
    iou_threshold: float = 0.5,
) -> dict[str, float]:
    matched_gt: set[int] = set()
    tp = 0

    for pred in pred_boxes:
        best_iou, best_idx = 0.0, -1
        for idx, gt in enumerate(gt_boxes):
            if idx in matched_gt:
                continue
            iou = box_iou(pred, gt)
            if iou > best_iou:
                best_iou, best_idx = iou, idx
        if best_iou >= iou_threshold and best_idx >= 0:
            tp += 1
            matched_gt.add(best_idx)

    fp = len(pred_boxes) - tp
    fn = len(gt_boxes) - tp
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}
