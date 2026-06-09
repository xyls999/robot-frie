import json
from collections import Counter, defaultdict
from pathlib import Path


def subset(coco, image_ids):
    image_ids = set(image_ids)
    return {
        "images": [img for img in coco["images"] if img["id"] in image_ids],
        "annotations": [ann for ann in coco["annotations"] if ann["image_id"] in image_ids],
        "categories": coco["categories"],
    }


def frame_num(stem):
    return int(stem.split("_")[1])


def summarize(coco, image_ids, ann_by_image, cat_names, keep_stems=False):
    image_ids = set(image_ids)
    counts = Counter()
    stems = []
    frames = []
    empty = 0
    for img in coco["images"]:
        if img["id"] not in image_ids:
            continue
        stem = Path(img["file_name"]).stem
        stems.append(stem)
        frames.append(frame_num(stem))
        anns = ann_by_image.get(img["id"], [])
        if not anns:
            empty += 1
        for ann in anns:
            counts[ann["category_id"]] += 1
    result = {
        "images": len(image_ids),
        "empty_images": empty,
        "boxes": sum(counts.values()),
        "frame_min": min(frames) if frames else None,
        "frame_max": max(frames) if frames else None,
        "category_counts": {
            cat_names[cid]: counts[cid] for cid in sorted(cat_names)
        },
    }
    if keep_stems:
        result["stems"] = sorted(stems)
    return result


def main():
    root = Path(__file__).resolve().parents[1]
    full_dir = root / "dataset" / "firedetect_full_A_train_0908_restore"
    source_summary = (
        root
        / "dataset"
        / "firedetect"
        / "annotations"
        / "holdout_kfold"
        / "summary_holdout_kfold.json"
    )
    out_dir = full_dir / "annotations" / "jump_holdout_20260605"
    out_dir.mkdir(parents=True, exist_ok=True)

    with (full_dir / "annotations" / "all.json").open("r", encoding="utf-8") as f:
        coco = json.load(f)
    with source_summary.open("r", encoding="utf-8") as f:
        holdout_summary = json.load(f)

    holdout_stems = {
        stem
        for group in holdout_summary["temporal_group_summary"]
        for stem in group["holdout_stems"]
    }
    img_by_stem = {Path(img["file_name"]).stem: img for img in coco["images"]}
    missing = sorted(holdout_stems - set(img_by_stem))
    if missing:
        raise RuntimeError(f"Missing holdout stems in restored dataset: {missing}")

    val_ids = {img_by_stem[stem]["id"] for stem in holdout_stems}
    all_ids = {img["id"] for img in coco["images"]}
    train_ids = all_ids - val_ids

    ann_by_image = defaultdict(list)
    for ann in coco["annotations"]:
        ann_by_image[ann["image_id"]].append(ann)
    cat_names = {cat["id"]: cat["name"] for cat in coco["categories"]}

    for file_name, image_ids in [
        ("train.json", train_ids),
        ("val.json", val_ids),
        ("test.json", val_ids),
    ]:
        with (out_dir / file_name).open("w", encoding="utf-8") as f:
            json.dump(subset(coco, image_ids), f, ensure_ascii=False)

    summary = {
        "source_dataset": "dataset/firedetect_full_A_train_0908_restore/annotations/all.json",
        "source_holdout_stems": "dataset/firedetect/annotations/holdout_kfold/summary_holdout_kfold.json",
        "strategy": "recorded permanent holdout stems: 5 temporal groups x 10 jumped samples",
        "all": summarize(coco, all_ids, ann_by_image, cat_names),
        "train": summarize(coco, train_ids, ann_by_image, cat_names),
        "val": summarize(coco, val_ids, ann_by_image, cat_names, keep_stems=True),
        "temporal_groups": holdout_summary["temporal_group_summary"],
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
