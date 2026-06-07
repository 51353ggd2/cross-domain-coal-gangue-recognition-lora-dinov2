"""
Analyze dual-domain discrepancy using brightness statistics, average gradient,
and optional feature embeddings.

Outputs:
- brightness_gradient_statistics.csv
- domain_level_summary.csv
"""

from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm


def average_brightness(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(gray.mean())


def average_gradient(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    gradient = np.sqrt(gx ** 2 + gy ** 2)
    return float(gradient.mean())


def collect_statistics(dataset_root):
    dataset_root = Path(dataset_root)
    records = []
    image_paths = sorted(list(dataset_root.rglob("*.jpg")) + list(dataset_root.rglob("*.png")) + list(dataset_root.rglob("*.jpeg")))

    for image_path in tqdm(image_paths, desc="Computing image statistics"):
        image = cv2.imread(str(image_path))
        if image is None:
            continue

        parts = image_path.relative_to(dataset_root).parts
        data_variant = parts[0] if len(parts) > 0 else "unknown"
        label = parts[1] if len(parts) > 1 else "unknown"

        records.append({
            "image_id": image_path.stem,
            "relative_path": str(image_path.relative_to(dataset_root)),
            "data_variant": data_variant,
            "label": label,
            "average_brightness": average_brightness(image),
            "average_gradient": average_gradient(image),
        })

    return pd.DataFrame(records)


if __name__ == "__main__":
    output_dir = Path("../05_Result_Data/Generated_Outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    df = collect_statistics("../01_Dataset/Dataset_To_Be_Added")
    df.to_csv(output_dir / "brightness_gradient_statistics.csv", index=False)

    summary = df.groupby(["data_variant", "label"]).agg(
        count=("image_id", "count"),
        brightness_mean=("average_brightness", "mean"),
        brightness_std=("average_brightness", "std"),
        gradient_mean=("average_gradient", "mean"),
        gradient_std=("average_gradient", "std"),
    ).reset_index()
    summary.to_csv(output_dir / "domain_level_summary.csv", index=False)
