# -*- coding: utf-8 -*-
"""
Local FireDetect evaluator/tuner.

This script intentionally does not modify submission predict.py. It reuses the
same Paddle Inference path, runs the model on a COCO annotation file, compares
predictions with GT boxes, and searches post-process parameters offline.
"""
import argparse
import csv
import importlib.util
import json
import math
import os
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PREDICT_PATH = ROOT / "predict.py"
CLASS_IDS = (1, 2, 3)
CLASS_NAMES = {1: "battery", 2: "board", 3: "fire"}


def load_predict_module():
    spec = importlib.util.spec_from_file_location("submission_predict_runtime", PREDICT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.init_runtime()
    module.paddle.enable_static()
    return module


def load_coco(anno_path):
    with open(anno_path, "r", encoding="utf-8") as f:
        coco = json.load(f)
    images = sorted(coco["images"], key=lambda x: x["file_name"])
    anns_by_image = {img["id"]: [] for img in images}
    for ann in coco["annotations"]:
        if ann.get("iscrowd", 0):
            continue
        cid = int(ann["category_id"])
        if cid not in CLASS_IDS:
            continue
        x, y, w, h = [float(v) for v in ann["bbox"]]
        anns_by_image.setdefault(ann["image_id"], []).append({
            "type": cid,
            "bbox": [x, y, x + w, y + h],
            "area": max(0.0, w) * max(0.0, h),
        })
    return images, anns_by_image


def image_path_for(image_dir, file_name):
    p = Path(image_dir) / file_name
    if p.exists():
        return str(p)
    return str(Path(image_dir) / Path(file_name).name)


def iou_xyxy(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    iw = max(0.0, x2 - x1)
    ih = max(0.0, y2 - y1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else 0.0


def run_inference(model_dir, anno_path, image_dir, raw_path, batch_size):
    predict = load_predict_module()
    pred_config = predict.PredictConfig(str(model_dir))
    detector = predict.Detector(pred_config, str(model_dir))
    images, _ = load_coco(anno_path)

    raw = {"model_dir": str(model_dir), "predictions": {}}
    batch_images, batch_infos, batch_records = [], [], []
    t0 = time.time()

    def flush():
        if not batch_images:
            return
        inputs = predict.create_inputs(batch_images, batch_infos)
        det = detector.predict(inputs)
        boxes = det["boxes"]
        boxes_num = det["boxes_num"]
        start = 0
        for idx, rec in enumerate(batch_records):
            num = int(boxes_num[idx])
            out = []
            for row in boxes[start:start + num]:
                cid = int(row[0]) + 1
                score = float(row[1])
                x1, y1, x2, y2 = [float(v) for v in row[2:6]]
                if cid in CLASS_IDS and math.isfinite(score):
                    x1, x2 = sorted([x1, x2])
                    y1, y2 = sorted([y1, y2])
                    w = x2 - x1
                    h = y2 - y1
                    if w > 0 and h > 0:
                        out.append({
                            "type": cid,
                            "score": score,
                            "bbox": [x1, y1, x2, y2],
                            "area": w * h,
                        })
            raw["predictions"][rec["id"]] = out
            start += num
        batch_images.clear()
        batch_infos.clear()
        batch_records.clear()

    for img in images:
        path = image_path_for(image_dir, img["file_name"])
        if not os.path.exists(path):
            raw["predictions"][Path(img["file_name"]).stem] = []
            continue
        im, info = predict.preprocess(path, detector.preprocess_ops)
        batch_images.append(im)
        batch_infos.append(info)
        batch_records.append({"id": Path(img["file_name"]).stem})
        if len(batch_images) >= batch_size:
            flush()
    flush()

    raw["seconds"] = time.time() - t0
    raw["fps"] = len(images) / raw["seconds"] if raw["seconds"] > 0 else 0.0
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(raw, f, ensure_ascii=False)
    return raw


def load_raw(raw_path):
    with open(raw_path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_thresholds(text):
    vals = [float(v) for v in text.split(",")]
    if len(vals) == 1:
        return {1: vals[0], 2: vals[0], 3: vals[0]}
    if len(vals) != 3:
        raise ValueError("--thresholds expects one value or battery,board,fire")
    return {1: vals[0], 2: vals[1], 3: vals[2]}


def filter_predictions(preds, thresholds, min_area):
    out = []
    for p in preds:
        cid = int(p["type"])
        if p["score"] < thresholds[cid]:
            continue
        if p["area"] < min_area.get(cid, 0.0):
            continue
        out.append(p)
    return sorted(out, key=lambda x: x["score"], reverse=True)


def evaluate(raw, images, anns_by_image, thresholds, min_area, iou_thr=0.5):
    counts = {
        cid: {"tp": 0, "fp": 0, "fn": 0, "gt": 0, "pred": 0}
        for cid in CLASS_IDS
    }
    per_image = []

    for img in images:
        stem = Path(img["file_name"]).stem
        gt_all = anns_by_image.get(img["id"], [])
        pred_all = filter_predictions(
            raw["predictions"].get(stem, []), thresholds, min_area
        )
        img_row = {"image_id": stem, "tp": 0, "fp": 0, "fn": 0}

        for cid in CLASS_IDS:
            gt = [g for g in gt_all if g["type"] == cid]
            pred = [p for p in pred_all if p["type"] == cid]
            counts[cid]["gt"] += len(gt)
            counts[cid]["pred"] += len(pred)
            matched_gt = set()

            for p in pred:
                best_iou = 0.0
                best_idx = None
                for gi, g in enumerate(gt):
                    if gi in matched_gt:
                        continue
                    val = iou_xyxy(p["bbox"], g["bbox"])
                    if val > best_iou:
                        best_iou = val
                        best_idx = gi
                if best_iou >= iou_thr and best_idx is not None:
                    matched_gt.add(best_idx)
                    counts[cid]["tp"] += 1
                    img_row["tp"] += 1
                else:
                    counts[cid]["fp"] += 1
                    img_row["fp"] += 1

            fn = len(gt) - len(matched_gt)
            counts[cid]["fn"] += fn
            img_row["fn"] += fn

        per_image.append(img_row)

    class_metrics = {}
    f1s = []
    for cid in CLASS_IDS:
        tp = counts[cid]["tp"]
        fp = counts[cid]["fp"]
        fn = counts[cid]["fn"]
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1s.append(f1)
        class_metrics[cid] = {
            **counts[cid],
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    return {
        "thresholds": thresholds,
        "min_area": min_area,
        "mean_f1": sum(f1s) / len(f1s),
        "classes": class_metrics,
        "per_image": per_image,
    }


def threshold_values(start, end, step):
    vals = []
    x = start
    while x <= end + 1e-9:
        vals.append(round(x, 4))
        x += step
    return vals


def coordinate_scan(raw, images, anns_by_image, base_thresholds, min_area, start, end, step, rounds):
    best = evaluate(raw, images, anns_by_image, base_thresholds, min_area)
    best_thr = dict(base_thresholds)
    vals = threshold_values(start, end, step)
    history = []
    for _ in range(rounds):
        changed = False
        for cid in CLASS_IDS:
            local_best = best
            local_thr = dict(best_thr)
            for v in vals:
                trial_thr = dict(best_thr)
                trial_thr[cid] = v
                cur = evaluate(raw, images, anns_by_image, trial_thr, min_area)
                if cur["mean_f1"] > local_best["mean_f1"]:
                    local_best = cur
                    local_thr = trial_thr
            if local_best["mean_f1"] > best["mean_f1"]:
                best = local_best
                best_thr = local_thr
                changed = True
                history.append({
                    "class": CLASS_NAMES[cid],
                    "thresholds": dict(best_thr),
                    "mean_f1": best["mean_f1"],
                })
        if not changed:
            break
    return best, history


def write_reports(result, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    clean = {
        "thresholds": result["thresholds"],
        "min_area": result["min_area"],
        "mean_f1": result["mean_f1"],
        "classes": {
            CLASS_NAMES[cid]: result["classes"][cid]
            for cid in CLASS_IDS
        },
    }
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(clean, f, ensure_ascii=False, indent=2)

    rows = sorted(
        result["per_image"],
        key=lambda x: (-(x["fp"] + x["fn"]), -x["fn"], -x["fp"], x["image_id"]),
    )
    with open(out_dir / "per_image_errors.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image_id", "tp", "fp", "fn"])
        writer.writeheader()
        writer.writerows(rows)


def print_summary(result):
    print("thresholds:", {
        CLASS_NAMES[cid]: result["thresholds"][cid] for cid in CLASS_IDS
    })
    print("min_area:", {
        CLASS_NAMES[cid]: result["min_area"].get(cid, 0.0) for cid in CLASS_IDS
    })
    print("mean_f1:", round(result["mean_f1"], 6))
    for cid in CLASS_IDS:
        m = result["classes"][cid]
        print(
            CLASS_NAMES[cid],
            "P", round(m["precision"], 4),
            "R", round(m["recall"], 4),
            "F1", round(m["f1"], 4),
            "TP/FP/FN", f'{m["tp"]}/{m["fp"]}/{m["fn"]}',
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--anno", default=str(ROOT / "PaddleDetection_train" / "dataset" / "firedetect" / "annotations" / "all.json"))
    parser.add_argument("--image_dir", default=str(ROOT / "PaddleDetection_train" / "dataset" / "firedetect" / "images"))
    parser.add_argument("--out_dir", default=str(ROOT / "analysis" / "local_tune"))
    parser.add_argument("--raw", default="")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--thresholds", default="0.38,0.38,0.38")
    parser.add_argument("--min_area", default="0,0,0")
    parser.add_argument("--scan", action="store_true")
    parser.add_argument("--scan_start", type=float, default=0.05)
    parser.add_argument("--scan_end", type=float, default=0.80)
    parser.add_argument("--scan_step", type=float, default=0.01)
    parser.add_argument("--scan_rounds", type=int, default=3)
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    out_dir = Path(args.out_dir)
    raw_path = Path(args.raw) if args.raw else out_dir / "raw_predictions.json"

    if raw_path.exists():
        raw = load_raw(raw_path)
    else:
        raw = run_inference(model_dir, args.anno, args.image_dir, raw_path, args.batch_size)
        print("inference fps:", round(raw.get("fps", 0.0), 3))

    images, anns_by_image = load_coco(args.anno)
    thresholds = parse_thresholds(args.thresholds)
    min_area = parse_thresholds(args.min_area)

    if args.scan:
        result, history = coordinate_scan(
            raw, images, anns_by_image, thresholds, min_area,
            args.scan_start, args.scan_end, args.scan_step, args.scan_rounds,
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "scan_history.json", "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
    else:
        result = evaluate(raw, images, anns_by_image, thresholds, min_area)

    write_reports(result, out_dir)
    print_summary(result)
    print("reports:", out_dir)


if __name__ == "__main__":
    main()
