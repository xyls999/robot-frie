import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path


PD_ROOT = Path(__file__).resolve().parents[1]
DATASET = PD_ROOT / "dataset" / "firedetect"
OUT_DIR = DATASET / "annotations" / "holdout_kfold"

SEED = 20260530
TEMPORAL_GROUPS = 5
HOLDOUT_PER_GROUP = 10
K_FOLDS = 5
FOLD_GROUP_SIZE = 8
CLASS_IDS = (1, 2, 3)
CLASS_NAMES = {1: "battery", 2: "board", 3: "fire"}
FRAME_RE = re.compile(r"frame_(\d+)")


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)


def stem_of(file_name):
    return Path(file_name).stem


def frame_num(file_name):
    match = FRAME_RE.search(file_name)
    if match:
        return int(match.group(1))
    return 10**12


def counter_for_image(image_id, anns_by_image):
    counter = Counter({"images": 1})
    anns = anns_by_image.get(image_id, [])
    if not anns:
        counter["empty"] += 1
    for ann in anns:
        counter[int(ann["category_id"])] += 1
    return counter


def summarize_coco(coco):
    counter = Counter()
    anns_by_image = defaultdict(list)
    for ann in coco["annotations"]:
        cid = int(ann["category_id"])
        if cid in CLASS_IDS:
            counter[cid] += 1
            anns_by_image[ann["image_id"]].append(ann)
    return {
        "images": len(coco["images"]),
        "empty_images": sum(
            1 for img in coco["images"] if not anns_by_image.get(img["id"])
        ),
        "battery": counter[1],
        "board": counter[2],
        "fire": counter[3],
        "boxes": counter[1] + counter[2] + counter[3],
    }


def remap_coco(images, annotations, categories):
    image_id_map = {}
    out_images = []
    out_annotations = []

    for new_id, image in enumerate(images, start=1):
        image_id_map[image["id"]] = new_id
        rec = dict(image)
        rec["id"] = new_id
        out_images.append(rec)

    ann_id = 1
    for ann in annotations:
        if ann["image_id"] not in image_id_map:
            continue
        rec = dict(ann)
        rec["id"] = ann_id
        rec["image_id"] = image_id_map[ann["image_id"]]
        out_annotations.append(rec)
        ann_id += 1

    return {
        "images": out_images,
        "annotations": out_annotations,
        "categories": categories,
    }


def build_coco_from_stems(stems, images_by_stem, anns_by_stem, categories):
    images = [images_by_stem[s] for s in sorted(stems, key=frame_num)]
    annotations = [ann for s in sorted(stems, key=frame_num) for ann in anns_by_stem[s]]
    return remap_coco(images, annotations, categories)


def split_temporal_groups(images):
    ordered = sorted(images, key=lambda img: frame_num(img["file_name"]))
    group_size = len(ordered) // TEMPORAL_GROUPS
    groups = []
    start = 0
    for idx in range(TEMPORAL_GROUPS):
        end = start + group_size
        if idx == TEMPORAL_GROUPS - 1:
            end = len(ordered)
        groups.append(ordered[start:end])
        start = end
    return groups


def distribution_score(counter, target, progress):
    score = 0.0
    for key in ("empty", 1, 2, 3):
        expected = target[key] * progress / HOLDOUT_PER_GROUP
        denom = max(1.0, target[key])
        score += ((counter[key] - expected) / denom) ** 2
    return score


def spread_penalty(candidate, selected):
    if not selected:
        return 0.0
    cand_frame = frame_num(candidate["file_name"])
    distances = [abs(cand_frame - frame_num(img["file_name"])) for img in selected]
    nearest = min(distances)
    if nearest >= 8:
        return 0.0
    return (8 - nearest) * 0.03


def select_holdout_for_group(group, anns_by_image, rng):
    total = Counter()
    per_image = {}
    for img in group:
        cnt = counter_for_image(img["id"], anns_by_image)
        per_image[img["id"]] = cnt
        total.update(cnt)

    target = Counter()
    for key in ("empty", 1, 2, 3):
        target[key] = total[key] * HOLDOUT_PER_GROUP / max(1, len(group))

    selected = []
    selected_counter = Counter()
    candidates = list(group)
    rng.shuffle(candidates)

    for _ in range(HOLDOUT_PER_GROUP):
        best = None
        best_score = None
        for img in candidates:
            if img in selected:
                continue
            trial = Counter(selected_counter)
            trial.update(per_image[img["id"]])
            progress = len(selected) + 1
            score = distribution_score(trial, target, progress)
            score += spread_penalty(img, selected)
            score += rng.random() * 1e-6
            if best_score is None or score < best_score:
                best = img
                best_score = score
        selected.append(best)
        selected_counter.update(per_image[best["id"]])

    return selected


def image_counter(image_ids, anns_by_image):
    counter = Counter()
    for image_id in image_ids:
        counter.update(counter_for_image(image_id, anns_by_image))
    return counter


def build_fold_groups(images, anns_by_image):
    ordered = sorted(images, key=lambda img: frame_num(img["file_name"]))
    chunks = []
    for i in range(0, len(ordered), FOLD_GROUP_SIZE):
        chunk = ordered[i:i + FOLD_GROUP_SIZE]
        ids = [img["id"] for img in chunk]
        chunks.append({"images": chunk, "counter": image_counter(ids, anns_by_image)})
    return chunks


def assign_folds(groups):
    totals = Counter()
    for group in groups:
        totals.update(group["counter"])
    targets = {key: totals[key] / K_FOLDS for key in totals}
    folds = [{"groups": [], "counter": Counter()} for _ in range(K_FOLDS)]

    def class_score(counter):
        score = 0.0
        for key, target in targets.items():
            if key == "images" or target <= 0:
                continue
            score += ((counter[key] - target) / target) ** 2
        return score

    ordered = sorted(
        groups,
        key=lambda g: (
            -(g["counter"][1] + g["counter"][2] + g["counter"][3]),
            -g["counter"]["empty"],
        ),
    )
    for group in ordered:
        best_idx = None
        best_score = None
        for idx, fold in enumerate(folds):
            trial = Counter(fold["counter"])
            trial.update(group["counter"])
            score = fold["counter"]["images"] * 100.0 + class_score(trial)
            score += idx * 1e-6
            if best_score is None or score < best_score:
                best_idx = idx
                best_score = score
        folds[best_idx]["groups"].append(group)
        folds[best_idx]["counter"].update(group["counter"])
    return folds


def main():
    rng = random.Random(SEED)
    all_coco = load_json(DATASET / "annotations" / "all.json")
    categories = all_coco["categories"]

    anns_by_image = defaultdict(list)
    for ann in all_coco["annotations"]:
        anns_by_image[ann["image_id"]].append(ann)

    images_by_stem = {stem_of(img["file_name"]): img for img in all_coco["images"]}
    anns_by_stem = {
        stem_of(img["file_name"]): anns_by_image.get(img["id"], [])
        for img in all_coco["images"]
    }

    temporal_groups = split_temporal_groups(all_coco["images"])
    holdout_images = []
    group_summary = []
    for idx, group in enumerate(temporal_groups):
        selected = select_holdout_for_group(group, anns_by_image, rng)
        holdout_images.extend(selected)
        group_stems = [stem_of(img["file_name"]) for img in group]
        selected_stems = [stem_of(img["file_name"]) for img in selected]
        group_summary.append({
            "group": idx,
            "frame_min": min(group_stems),
            "frame_max": max(group_stems),
            "images": len(group),
            "holdout_images": len(selected),
            "holdout_stems": sorted(selected_stems, key=frame_num),
        })

    holdout_stems = {stem_of(img["file_name"]) for img in holdout_images}
    all_stems = set(images_by_stem)
    train_pool_stems = all_stems - holdout_stems

    holdout_coco = build_coco_from_stems(
        holdout_stems, images_by_stem, anns_by_stem, categories)
    train_pool_coco = build_coco_from_stems(
        train_pool_stems, images_by_stem, anns_by_stem, categories)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    save_json(OUT_DIR / "holdout.json", holdout_coco)
    save_json(OUT_DIR / "train_pool.json", train_pool_coco)

    train_pool_images = [
        images_by_stem[s] for s in sorted(train_pool_stems, key=frame_num)
    ]
    folds = assign_folds(build_fold_groups(train_pool_images, anns_by_image))
    fold_summary = []

    for fold_idx, fold in enumerate(folds):
        val_stems = {
            stem_of(img["file_name"])
            for group in fold["groups"]
            for img in group["images"]
        }
        train_stems = train_pool_stems - val_stems
        fold_train = build_coco_from_stems(
            train_stems, images_by_stem, anns_by_stem, categories)
        fold_val = build_coco_from_stems(
            val_stems, images_by_stem, anns_by_stem, categories)
        train_path = OUT_DIR / f"fold{fold_idx}_train.json"
        val_path = OUT_DIR / f"fold{fold_idx}_val.json"
        save_json(train_path, fold_train)
        save_json(val_path, fold_val)
        fold_summary.append({
            "fold": fold_idx,
            "train": summarize_coco(fold_train),
            "val": summarize_coco(fold_val),
            "train_json": str(train_path.relative_to(DATASET)),
            "val_json": str(val_path.relative_to(DATASET)),
        })

    summary = {
        "seed": SEED,
        "source": "dataset/firedetect/annotations/all.json",
        "note": "Permanent holdout is fixed and must never be used for training, pseudo labels, or threshold tuning. Existing all405-trained models have already seen this holdout and are contaminated for unbiased validation.",
        "temporal_groups": TEMPORAL_GROUPS,
        "holdout_per_group": HOLDOUT_PER_GROUP,
        "folds": K_FOLDS,
        "all": summarize_coco(all_coco),
        "train_pool": summarize_coco(train_pool_coco),
        "holdout": summarize_coco(holdout_coco),
        "temporal_group_summary": group_summary,
        "fold_summary": fold_summary,
    }
    save_json(OUT_DIR / "summary_holdout_kfold.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
