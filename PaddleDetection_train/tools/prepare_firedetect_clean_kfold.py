import json
from collections import Counter, defaultdict
from pathlib import Path


PD_ROOT = Path(__file__).resolve().parents[1]
DATASET = PD_ROOT / "dataset" / "firedetect"
OUT_DIR = DATASET / "annotations" / "kfold"

K = 5
GROUP_SIZE = 8


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)


def stem_of(file_name):
    return Path(file_name).stem


def image_counter(image_ids, anns_by_image):
    counter = Counter()
    for image_id in image_ids:
        anns = anns_by_image.get(image_id, [])
        if not anns:
            counter["empty"] += 1
        for ann in anns:
            counter[ann["category_id"]] += 1
    counter["images"] += len(image_ids)
    return counter


def summarize_coco(coco):
    counter = Counter()
    anns_by_image = defaultdict(list)
    for ann in coco["annotations"]:
        anns_by_image[ann["image_id"]].append(ann)
        counter[ann["category_id"]] += 1
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


def build_groups(coco, anns_by_image):
    images = sorted(coco["images"], key=lambda x: stem_of(x["file_name"]))
    groups = []
    for i in range(0, len(images), GROUP_SIZE):
        chunk = images[i:i + GROUP_SIZE]
        ids = [img["id"] for img in chunk]
        groups.append({"images": chunk, "counter": image_counter(ids, anns_by_image)})
    return groups


def assign_folds(groups):
    totals = Counter()
    for group in groups:
        totals.update(group["counter"])
    targets = {key: totals[key] / K for key in totals}
    folds = [{"groups": [], "counter": Counter()} for _ in range(K)]

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
                best_score = score
                best_idx = idx
        folds[best_idx]["groups"].append(group)
        folds[best_idx]["counter"].update(group["counter"])
    return folds


def main():
    all_coco = load_json(DATASET / "annotations" / "all.json")
    categories = all_coco["categories"]

    anns_by_image = defaultdict(list)
    for ann in all_coco["annotations"]:
        anns_by_image[ann["image_id"]].append(ann)

    images_by_stem = {stem_of(img["file_name"]): img for img in all_coco["images"]}
    anns_by_stem = defaultdict(list)
    for img in all_coco["images"]:
        anns_by_stem[stem_of(img["file_name"])] = anns_by_image.get(img["id"], [])

    folds = assign_folds(build_groups(all_coco, anns_by_image))
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    summary = {
        "k": K,
        "group_size": GROUP_SIZE,
        "source": "dataset/firedetect/annotations/all.json",
        "note": "Clean k-fold only. No generated augmentation images are used.",
        "clean_all": summarize_coco(all_coco),
        "folds": [],
    }

    all_stems = set(images_by_stem)
    for fold_idx, fold in enumerate(folds):
        val_stems = {
            stem_of(img["file_name"])
            for group in fold["groups"]
            for img in group["images"]
        }
        train_stems = all_stems - val_stems

        val_images = [images_by_stem[s] for s in sorted(val_stems)]
        val_annotations = [
            ann for stem in sorted(val_stems) for ann in anns_by_stem[stem]
        ]
        train_images = [images_by_stem[s] for s in sorted(train_stems)]
        train_annotations = [
            ann for stem in sorted(train_stems) for ann in anns_by_stem[stem]
        ]

        train_coco = remap_coco(train_images, train_annotations, categories)
        val_coco = remap_coco(val_images, val_annotations, categories)
        train_path = OUT_DIR / f"fold{fold_idx}_train.json"
        val_path = OUT_DIR / f"fold{fold_idx}_val.json"
        save_json(train_path, train_coco)
        save_json(val_path, val_coco)

        summary["folds"].append({
            "fold": fold_idx,
            "train": summarize_coco(train_coco),
            "val": summarize_coco(val_coco),
            "train_json": str(train_path.relative_to(DATASET)),
            "val_json": str(val_path.relative_to(DATASET)),
        })

    save_json(OUT_DIR / "summary_kfold.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
