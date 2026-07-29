"""Image preprocessing pipeline: CLAHE, histogram eq, noise removal, normalization."""
import cv2
import numpy as np
from pathlib import Path


class ImagePreprocessor:
    def __init__(self, target_size: int = 640):
        self.target_size = target_size
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def process(self, image: np.ndarray) -> np.ndarray:
        image = self._resize(image)
        image = self._remove_noise(image)
        image = self._apply_clahe(image)
        return image

    def _resize(self, image: np.ndarray) -> np.ndarray:
        return cv2.resize(image, (self.target_size, self.target_size), interpolation=cv2.INTER_LINEAR)

    def _remove_noise(self, image: np.ndarray) -> np.ndarray:
        return cv2.fastNlMeansDenoisingColored(image, None, 5, 5, 7, 21)

    def _apply_clahe(self, image: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = self._clahe.apply(l)
        return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

    def histogram_equalize(self, image: np.ndarray) -> np.ndarray:
        ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
        return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

    def normalize(self, image: np.ndarray) -> np.ndarray:
        return image.astype(np.float32) / 255.0


class DatasetPreprocessor:
    """Validates labels, removes duplicates, and preprocesses dataset images."""

    def __init__(self, dataset_dir: Path, target_size: int = 640):
        self.dataset_dir = Path(dataset_dir)
        self.preprocessor = ImagePreprocessor(target_size)

    def validate_labels(self, label_path: Path, num_classes: int = 3) -> bool:
        try:
            with open(label_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        return False
                    cls_id = int(parts[0])
                    coords = [float(x) for x in parts[1:]]
                    if cls_id < 0 or cls_id >= num_classes:
                        return False
                    if not all(0.0 <= c <= 1.0 for c in coords):
                        return False
            return True
        except Exception:
            return False

    def remove_duplicates(self, image_paths: list[Path]) -> list[Path]:
        seen_hashes: set[str] = set()
        unique = []
        for p in image_paths:
            img = cv2.imread(str(p))
            if img is None:
                continue
            h = cv2.img_hash.PHash_create().compute(img).tobytes().hex()
            if h not in seen_hashes:
                seen_hashes.add(h)
                unique.append(p)
        return unique

    def process_split(self, split: str, output_dir: Path) -> int:
        src = self.dataset_dir / split / "images"
        dst = output_dir / split / "images"
        dst.mkdir(parents=True, exist_ok=True)
        count = 0
        for img_path in src.glob("*.jpg"):
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            processed = self.preprocessor.process(img)
            cv2.imwrite(str(dst / img_path.name), processed)
            count += 1
        return count
