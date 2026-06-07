"""
Prepare object-level coal-gangue classification samples from scene-level images
and normalized bounding-box annotations.

Expected annotation format:
class_id x_center y_center width height

All coordinates are normalized to image width and height.
"""

from pathlib import Path
import cv2
import pandas as pd


CLASS_NAMES = {0: "coal", 1: "gangue"}


def yolo_to_pixel_box(x_center, y_center, width, height, image_width, image_height):
    x_min = int(round((x_center - width / 2.0) * image_width))
    x_max = int(round((x_center + width / 2.0) * image_width))
    y_min = int(round((y_center - height / 2.0) * image_height))
    y_max = int(round((y_center + height / 2.0) * image_height))

    x_min = max(0, min(x_min, image_width - 1))
    x_max = max(0, min(x_max, image_width))
    y_min = max(0, min(y_min, image_height - 1))
    y_max = max(0, min(y_max, image_height))
    return x_min, y_min, x_max, y_max


def crop_objects(image_dir, annotation_dir, output_dir, domain_name):
    image_dir = Path(image_dir)
    annotation_dir = Path(annotation_dir)
    output_dir = Path(output_dir)
    records = []

    image_paths = sorted(list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png")) + list(image_dir.glob("*.jpeg")))

    for image_path in image_paths:
        annotation_path = annotation_dir / f"{image_path.stem}.txt"
        if not annotation_path.exists():
            continue

        image = cv2.imread(str(image_path))
        if image is None:
            continue

        image_height, image_width = image.shape[:2]

        for instance_index, line in enumerate(annotation_path.read_text(encoding="utf-8").splitlines()):
            parts = line.strip().split()
            if len(parts) != 5:
                continue

            class_id = int(float(parts[0]))
            x_center, y_center, width, height = map(float, parts[1:])
            x_min, y_min, x_max, y_max = yolo_to_pixel_box(
                x_center, y_center, width, height, image_width, image_height
            )

            if x_max <= x_min or y_max <= y_min:
                continue

            label = CLASS_NAMES.get(class_id, f"class_{class_id}")
            crop = image[y_min:y_max, x_min:x_max]
            target_dir = output_dir / domain_name / label
            target_dir.mkdir(parents=True, exist_ok=True)

            crop_name = f"{image_path.stem}_{instance_index:04d}_{label}.jpg"
            crop_path = target_dir / crop_name
            cv2.imwrite(str(crop_path), crop)

            records.append({
                "image_id": crop_path.stem,
                "relative_path": str(crop_path),
                "label": label,
                "label_id": class_id,
                "domain": domain_name,
                "source_scene_image": image_path.name,
                "bbox_x_center": x_center,
                "bbox_y_center": y_center,
                "bbox_width": width,
                "bbox_height": height,
            })

    return pd.DataFrame(records)


if __name__ == "__main__":
    dataframe = crop_objects(
        image_dir="../01_Dataset/Original_Scene_Data_To_Be_Added/Laboratory",
        annotation_dir="../01_Dataset/Bounding_Box_Annotations_To_Be_Added/Laboratory",
        output_dir="../01_Dataset/Dataset_To_Be_Added",
        domain_name="Source_Domain_Lab_Raw",
    )
    dataframe.to_csv("../03_Annotations_and_Labels/generated_object_labels.csv", index=False)
