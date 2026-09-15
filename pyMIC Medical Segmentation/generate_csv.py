"""Generate PyMIC-compatible train/validation/test CSV files for HC18."""

import csv
import random
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
IMAGE_DIR = DATA_DIR / "images"
LABEL_DIR = DATA_DIR / "labels"
CSV_DIR = PROJECT_DIR / "config"

TRAIN_SIZE = 780
VALID_SIZE = 70
RANDOM_SEED = 42


def collect_pairs():
    """Collect image/label paths relative to DATA_DIR."""
    pairs = []
    missing_labels = []

    for image_path in sorted(IMAGE_DIR.glob("*.png")):
        label_path = LABEL_DIR / f"{image_path.stem}_seg.png"
        if not label_path.is_file():
            missing_labels.append(label_path.name)
            continue

        pairs.append(
            [
                image_path.relative_to(DATA_DIR).as_posix(),
                label_path.relative_to(DATA_DIR).as_posix(),
            ]
        )

    if missing_labels:
        preview = ", ".join(missing_labels[:10])
        raise FileNotFoundError(f"Missing {len(missing_labels)} labels: {preview}")

    return pairs


def save_csv(filename, rows):
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    output_path = CSV_DIR / filename
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["image", "label"])
        writer.writerows(rows)


def main():
    pairs = collect_pairs()
    split_at = TRAIN_SIZE + VALID_SIZE
    if len(pairs) < split_at:
        raise ValueError(
            f"Only {len(pairs)} valid pairs found; at least {split_at} are required."
        )

    random.Random(RANDOM_SEED).shuffle(pairs)
    train_rows = pairs[:TRAIN_SIZE]
    valid_rows = pairs[TRAIN_SIZE:split_at]
    test_rows = pairs[split_at:]

    save_csv("train.csv", train_rows)
    save_csv("valid.csv", valid_rows)
    save_csv("test.csv", test_rows)

    print(f"Valid image/label pairs: {len(pairs)}")
    print(
        f"Generated train.csv ({len(train_rows)}), "
        f"valid.csv ({len(valid_rows)}), test.csv ({len(test_rows)})"
    )


if __name__ == "__main__":
    main()
