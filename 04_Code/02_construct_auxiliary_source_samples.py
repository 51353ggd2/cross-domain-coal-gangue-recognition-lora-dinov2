"""
Construct recognition-oriented auxiliary source-domain samples.

This script provides a transparent implementation of four operations:
1. illumination compression
2. local illumination perturbation
3. mild blur simulation
4. structural compensation

The goal is not subjective visual enhancement, but source-domain auxiliary sample
construction for target-domain-compatible recognition.
"""

from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm


def illumination_compression(image, gamma=1.45, gain=0.72):
    image_float = image.astype(np.float32) / 255.0
    compressed = gain * np.power(image_float, gamma)
    return np.clip(compressed * 255.0, 0, 255).astype(np.uint8)


def local_illumination_perturbation(image, strength=0.18):
    height, width = image.shape[:2]
    x_grid = np.linspace(-1, 1, width)
    y_grid = np.linspace(-1, 1, height)
    xx, yy = np.meshgrid(x_grid, y_grid)
    mask = 1.0 - strength * (0.5 + 0.5 * np.sin(2.5 * xx + 1.5 * yy))
    perturbed = image.astype(np.float32) * mask[..., None]
    return np.clip(perturbed, 0, 255).astype(np.uint8)


def mild_blur_simulation(image, kernel_size=3):
    if kernel_size <= 1:
        return image
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)


def structural_compensation(image, amount=0.35):
    blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=1.0)
    sharpened = cv2.addWeighted(image, 1.0 + amount, blurred, -amount, 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def construct_auxiliary_sample(image):
    x = illumination_compression(image)
    x = local_illumination_perturbation(x)
    x = mild_blur_simulation(x)
    x = structural_compensation(x)
    return x


def process_directory(input_root, output_root):
    input_root = Path(input_root)
    output_root = Path(output_root)
    image_paths = sorted(list(input_root.rglob("*.jpg")) + list(input_root.rglob("*.png")) + list(input_root.rglob("*.jpeg")))

    for image_path in tqdm(image_paths, desc="Constructing auxiliary samples"):
        image = cv2.imread(str(image_path))
        if image is None:
            continue

        auxiliary = construct_auxiliary_sample(image)
        relative_path = image_path.relative_to(input_root)
        output_path = output_root / relative_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), auxiliary)


if __name__ == "__main__":
    process_directory(
        input_root="../01_Dataset/Dataset_To_Be_Added/Source_Domain_Lab_Raw",
        output_root="../01_Dataset/Dataset_To_Be_Added/Source_Domain_Lab_Enhanced",
    )
