"""Dataset preparation: download, merge, split BCCD + TXL-PBC datasets."""
import shutil
import random
import logging
import yaml
from pathlib import Path

logger = logging.getLogger(__name__)

CLASS_MAP = {
    # BCCD labels
    "RBC": 0, "WBC": 1, "Platelets": 2,
    # TXL-PBC labels (normalized)
    "red blood cell": 0, "white blood cell": 1, "platelet": 2,
    "rbc": 0, "wbc": 1,
}

SPLITS = {"train": 0.70, "val": 0.20, "test": 0.10}


def prepare_dataset(raw_dir: Path, output_dir: Path, seed: int = 42) -> Path:
    """Merge raw datasets, convert labels, split, and write data.yaml."""
    random.seed(seed)
    all_pairs: list[tuple[Path, Path]] = []

    for dataset_dir in raw_dir.iterdir():
        if not dataset_dir.is_dir():
            continue
        pairs = _collect_pairs(dataset_dir)
        logger.info("Found %d pairs in %s", len(pairs), dataset_dir.name)
        all_pairs.extend(pairs)

    random.shuffle(all_pairs)
    n = len(all_pairs)
    n_train = int(n * SPLITS["train"])
    n_val = int(n * SPLITS["val"])
    split_data = {
        "train": all_pairs[:n_train],
        "val": all_pairs[n_train:n_train + n_val],
        "test": all_pairs[n_train + n_val:],
    }

    for split, pairs in split_data.items():
        img_dir = output_dir / split / "images"
        lbl_dir = output_dir / split / "labels"
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        for img_path, lbl_path in pairs:
            shutil.copy2(img_path, img_dir / img_path.name)
            _convert_label(lbl_path, lbl_dir / lbl_path.name)
        logger.info("Split %s: %d images", split, len(pairs))

    yaml_path = output_dir / "data.yaml"
    _write_yaml(yaml_path, output_dir)
    logger.info("Dataset ready at %s", output_dir)
    return yaml_path


def _collect_pairs(dataset_dir: Path) -> list[tuple[Path, Path]]:
    pairs = []
    for img_path in (dataset_dir / "images").glob("*.[jp][pn]g"):
        lbl_path = dataset_dir / "labels" / (img_path.stem + ".txt")
        if lbl_path.exists():
            pairs.append((img_path, lbl_path))
    return pairs


def _convert_label(src: Path, dst: Path) -> None:
    lines = []
    try:
        with open(src) as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                # Already numeric YOLO format
                if parts[0].isdigit():
                    cls_id = int(parts[0])
                    if cls_id <= 2:
                        lines.append(line.strip())
                else:
                    cls_name = parts[0].lower()
                    cls_id = CLASS_MAP.get(cls_name)
                    if cls_id is not None:
                        lines.append(f"{cls_id} {' '.join(parts[1:])}")
    except Exception as e:
        logger.warning("Label conversion error %s: %s", src, e)
    with open(dst, "w") as f:
        f.write("\n".join(lines))


def _write_yaml(yaml_path: Path, base: Path) -> None:
    data = {
        "path": str(base.resolve()),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "nc": 3,
        "names": ["RBC", "WBC", "Platelet"],
    }
    with open(yaml_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False)
