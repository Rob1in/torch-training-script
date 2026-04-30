"""Per-class statistics for Viam JSONL datasets — feasibility check before training.

Reports, per class:
  - total bbox instances
  - images containing the class
  - sequences containing the class
  - images-with-class-but-no-sequence-id
  - train/val sequence counts under a sequence-aware split

Usage:
    python src/dataset_stats.py <dataset_dir> [<dataset_dir> ...]
    python src/dataset_stats.py <dataset_dir> --others
    python src/dataset_stats.py <dataset_dir> --val-split 0.2 --seed 42
"""

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

CLASS_PREFIX = "human_annotated_"


def load_dataset(dataset_dir: Path) -> list[dict]:
    """Read <dataset_dir>/dataset.jsonl and return a list of {binary_data_id, labels, sequence_id}."""
    jsonl_path = dataset_dir / "dataset.jsonl"
    if not jsonl_path.exists():
        raise FileNotFoundError(f"dataset.jsonl not found in {dataset_dir}")

    out = []
    with jsonl_path.open() as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"WARN: {jsonl_path}:{line_num} malformed JSON: {e}", file=sys.stderr)
                continue

            binary_data_id = entry.get("binary_data_id") or entry.get("binaryDataId")
            image_path = entry.get("image_path")

            sequence_id: Optional[str] = None
            for ann in entry.get("classification_annotations") or []:
                label = ann.get("annotation_label", "")
                if label.startswith("sequence_"):
                    sequence_id = label.split("--")[0]
                    break

            labels = [
                bbox.get("annotation_label")
                for bbox in (entry.get("bounding_box_annotations") or [])
                if bbox.get("annotation_label")
            ]

            out.append({
                "binary_data_id": binary_data_id,
                "image_path": image_path,
                "labels": labels,
                "sequence_id": sequence_id,
                "source_dataset": str(dataset_dir),
            })
    return out


def dedupe(images: list[dict]) -> list[dict]:
    """Drop duplicate binary_data_ids; warn if a binary_data_id appears in multiple datasets."""
    seen: dict[str, dict] = {}
    dups = 0
    for img in images:
        bid = img["binary_data_id"]
        if bid is None:
            seen[id(img)] = img
            continue
        if bid in seen:
            dups += 1
            if seen[bid]["source_dataset"] != img["source_dataset"]:
                print(
                    f"WARN: binary_data_id {bid} present in both "
                    f"{seen[bid]['source_dataset']} and {img['source_dataset']} — keeping first.",
                    file=sys.stderr,
                )
            continue
        seen[bid] = img
    if dups:
        print(f"Deduped {dups} duplicate images by binary_data_id.", file=sys.stderr)
    return list(seen.values())


def project_split(
    images: list[dict], val_split: float, seed: int
) -> tuple[set[str], set[str]]:
    """Replicate sequence_aware_split from train.py, but return only the sequence partitions.

    Returns (train_seq_ids, val_seq_ids). Images without sequence_id are implicitly train.
    """
    seq_to_count: dict[str, int] = defaultdict(int)
    no_seq_count = 0
    for img in images:
        sid = img["sequence_id"]
        if sid is None:
            no_seq_count += 1
        else:
            seq_to_count[sid] += 1

    if not seq_to_count:
        return set(), set()

    seq_ids = sorted(seq_to_count.keys())
    rng = random.Random(seed)
    rng.shuffle(seq_ids)

    total_images = sum(seq_to_count.values()) + no_seq_count
    target_val = int(total_images * val_split)

    val_seqs: set[str] = set()
    val_imgs = 0
    split_point = len(seq_ids)
    for i, sid in enumerate(seq_ids):
        if val_imgs >= target_val:
            split_point = i
            break
        val_seqs.add(sid)
        val_imgs += seq_to_count[sid]

    train_seqs = set(seq_ids[split_point:])
    return train_seqs, val_seqs


def compute_class_stats(
    images: list[dict],
    train_seqs: set[str],
    val_seqs: set[str],
) -> dict[str, dict]:
    """Per-label aggregates. Returns {label: {bboxes, images, sequences, no_seq_imgs, train_seqs, val_seqs}}."""
    bboxes: dict[str, int] = defaultdict(int)
    imgs: dict[str, int] = defaultdict(int)
    seqs: dict[str, set[str]] = defaultdict(set)
    no_seq: dict[str, int] = defaultdict(int)

    for img in images:
        seen = set()
        for label in img["labels"]:
            bboxes[label] += 1
            seen.add(label)
        for label in seen:
            imgs[label] += 1
            sid = img["sequence_id"]
            if sid is None:
                no_seq[label] += 1
            else:
                seqs[label].add(sid)

    stats: dict[str, dict] = {}
    for label in bboxes:
        label_seqs = seqs[label]
        stats[label] = {
            "bboxes": bboxes[label],
            "images": imgs[label],
            "sequences": len(label_seqs),
            "no_seq_imgs": no_seq[label],
            "train_seqs": len(label_seqs & train_seqs),
            "val_seqs": len(label_seqs & val_seqs),
        }
    return stats


def print_table(title: str, stats: dict[str, dict]) -> None:
    if not stats:
        print(f"\n{title}: (none)")
        return

    headers = ["class", "bboxes", "images", "sequences", "no_seq_imgs", "train_seqs", "val_seqs"]
    rows = sorted(stats.items(), key=lambda kv: -kv[1]["bboxes"])

    name_w = max(len(headers[0]), max(len(name) for name, _ in rows))
    num_w = {h: max(len(h), max(len(str(s[h])) for _, s in rows)) for h in headers[1:]}

    header_line = f"{headers[0]:<{name_w}}  " + "  ".join(f"{h:>{num_w[h]}}" for h in headers[1:])
    sep = "-" * len(header_line)

    print(f"\n{title}")
    print(sep)
    print(header_line)
    print(sep)
    for name, s in rows:
        row = f"{name:<{name_w}}  " + "  ".join(f"{s[h]:>{num_w[h]}}" for h in headers[1:])
        print(row)
    print(sep)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset_dirs", nargs="+", type=Path, help="One or more directories containing dataset.jsonl")
    parser.add_argument("--others", action="store_true",
                        help=f"Show only labels that DON'T match prefix '{CLASS_PREFIX}' (sanity check).")
    parser.add_argument("--val-split", type=float, default=0.2,
                        help="Fraction of images projected into val split (default: 0.2, matches train.yaml).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Seed for sequence-aware split projection (default: 42, matches train.yaml).")
    args = parser.parse_args()

    all_images: list[dict] = []
    for d in args.dataset_dirs:
        imgs = load_dataset(d)
        print(f"Loaded {len(imgs):>6} images from {d}")
        all_images.extend(imgs)

    images = dedupe(all_images)
    n_with_seq = sum(1 for i in images if i["sequence_id"] is not None)
    n_unique_seqs = len({i["sequence_id"] for i in images if i["sequence_id"] is not None})
    print(
        f"\nTotal: {len(images)} unique images, "
        f"{n_with_seq} with sequence_id ({n_unique_seqs} unique sequences), "
        f"{len(images) - n_with_seq} without."
    )

    train_seqs, val_seqs = project_split(images, args.val_split, args.seed)
    if train_seqs or val_seqs:
        print(
            f"Sequence-aware split projection (val_split={args.val_split}, seed={args.seed}): "
            f"{len(train_seqs)} train sequences, {len(val_seqs)} val sequences."
        )
    else:
        print("No sequence_ids present — train/val projection skipped (all images would land in train).")

    stats = compute_class_stats(images, train_seqs, val_seqs)

    if args.others:
        others = {k: v for k, v in stats.items() if not k.startswith(CLASS_PREFIX)}
        print_table(f"Other bbox labels (not matching prefix '{CLASS_PREFIX}')", others)
    else:
        matched = {k: v for k, v in stats.items() if k.startswith(CLASS_PREFIX)}
        print_table(f"Classes (prefix '{CLASS_PREFIX}')", matched)


if __name__ == "__main__":
    main()
