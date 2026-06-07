import os
import csv
import re
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from PIL import ImageFile

from transformers import AutoImageProcessor, Dinov2Model
from peft import PeftModel


ImageFile.LOAD_TRUNCATED_IMAGES = True


                                                           
                        
                                                           

                  
     
          
       
               
                      
MODEL_TYPE = "mlp_layernorm_gelu"

                  
          
       
           
             
DATA_ROOT = r"F:\00_Coal_Gangue_Recognition_Paper\3.30_DINOv2\dataset\过往dataset汇总\现场dataset\test"

                       
                                                  
SAVE_DIR = r"F:\00_Coal_Gangue_Recognition_Paper\3.30_DINOv2\LoRA_DINOv2\yunxing_results\p4"

                              
MODEL_NAME = "facebook/dinov2-base"

             
NUM_CLASSES = 2

                
IMAGE_SIZE = 224

                 
BATCH_SIZE = 4

                  
NUM_WORKERS = 2

         
DEVICE = "cuda"

                                    
                              
CLASS_NAMES = ["coal", "gangue"]

                                   
SAVE_CONFUSION_MATRIX_FIGURE = True


                                                           
           
                                                           

MODEL_PATHS = {
    "linear": {
        "adapter_dir": r"F:\00_Coal_Gangue_Recognition_Paper\3.30_DINOv2\LoRA_DINOv2\权重file\lab_enhance\runs_linear\best_lora_adapter",
        "head_path": r"F:\00_Coal_Gangue_Recognition_Paper\3.30_DINOv2\LoRA_DINOv2\权重file\lab_enhance\runs_linear\best_linear_head.pth",
    },
    "mlp": {
        "adapter_dir": r"F:\00_Coal_Gangue_Recognition_Paper\3.30_DINOv2\LoRA_DINOv2\权重file\4.14+field\runs_mlp\best_lora_adapter",
        "head_path": r"F:\00_Coal_Gangue_Recognition_Paper\3.30_DINOv2\LoRA_DINOv2\权重file\4.14+field\runs_mlp\best_mlp_head.pth",
    },
    "mlp_bn_gelu": {
        "adapter_dir": r"F:\00_Coal_Gangue_Recognition_Paper\3.30_DINOv2\LoRA_DINOv2\权重file\lab_enhance\runs_mlp_bn_gelu\best_lora_adapter",
        "head_path": r"F:\00_Coal_Gangue_Recognition_Paper\3.30_DINOv2\LoRA_DINOv2\权重file\lab_enhance\runs_mlp_bn_gelu\best_mlp_head.pth",
    },
    "mlp_layernorm_gelu": {
        "adapter_dir": r"F:\00_Coal_Gangue_Recognition_Paper\3.30_DINOv2\LoRA_DINOv2\权重file\lab_enhance+field\runs_mlp_layernorm_gelu_20epochs\best_lora_adapter",
        "head_path": r"F:\00_Coal_Gangue_Recognition_Paper\3.30_DINOv2\LoRA_DINOv2\权重file\lab_enhance+field\runs_mlp_layernorm_gelu_20epochs\best_mlp_head.pth",
    },
}


def _safe_name(name: str) -> str:
    """Convert class names into safe csv-column names."""
    name = str(name).strip().lower()
    name = re.sub(r"[^0-9a-zA-Z_]+", "_", name)
    return name.strip("_") or "class"


class DINOv2ImageFolder(datasets.ImageFolder):
    def __init__(self, root, processor, image_size=224):
        super().__init__(root=root)
        self.processor = processor
        self.image_size = image_size

    def __getitem__(self, index):
        path, target = self.samples[index]
        image = self.loader(path).convert("RGB")
        pixel_values = self.processor(
            images=image,
            return_tensors="pt",
            size={"height": self.image_size, "width": self.image_size},
        )["pixel_values"].squeeze(0)
        return pixel_values, target, path


class LoRADINOv2Linear(nn.Module):
    def __init__(self, model_name, adapter_dir, num_classes):
        super().__init__()
        base_model = Dinov2Model.from_pretrained(model_name)
        self.backbone = PeftModel.from_pretrained(base_model, adapter_dir)
        hidden_size = self.backbone.config.hidden_size
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, pixel_values):
        outputs = self.backbone(pixel_values=pixel_values)
        cls_feature = outputs.last_hidden_state[:, 0]
        logits = self.classifier(cls_feature)
        return logits


class LoRADINOv2MLP(nn.Module):
    def __init__(self, model_name, adapter_dir, num_classes, hidden_dim=512, dropout=0.3):
        super().__init__()
        base_model = Dinov2Model.from_pretrained(model_name)
        self.backbone = PeftModel.from_pretrained(base_model, adapter_dir)
        hidden_size = self.backbone.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, pixel_values):
        outputs = self.backbone(pixel_values=pixel_values)
        cls_feature = outputs.last_hidden_state[:, 0]
        logits = self.classifier(cls_feature)
        return logits


class LoRADINOv2MLPBNGELU(nn.Module):
    def __init__(self, model_name, adapter_dir, num_classes, hidden_dim=512, dropout=0.3):
        super().__init__()
        base_model = Dinov2Model.from_pretrained(model_name)
        self.backbone = PeftModel.from_pretrained(base_model, adapter_dir)
        hidden_size = self.backbone.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, pixel_values):
        outputs = self.backbone(pixel_values=pixel_values)
        cls_feature = outputs.last_hidden_state[:, 0]
        logits = self.classifier(cls_feature)
        return logits


class LoRADINOv2MLPLayerNormGELU(nn.Module):
    def __init__(self, model_name, adapter_dir, num_classes, hidden_dim=512, dropout=0.3):
        super().__init__()
        base_model = Dinov2Model.from_pretrained(model_name)
        self.backbone = PeftModel.from_pretrained(base_model, adapter_dir)
        hidden_size = self.backbone.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, pixel_values):
        outputs = self.backbone(pixel_values=pixel_values)
        cls_feature = outputs.last_hidden_state[:, 0]
        logits = self.classifier(cls_feature)
        return logits


def build_model():
    if MODEL_TYPE not in MODEL_PATHS:
        raise ValueError("MODEL_TYPE must be one of: linear / mlp / mlp_bn_gelu / mlp_layernorm_gelu")

    adapter_dir = MODEL_PATHS[MODEL_TYPE]["adapter_dir"]
    head_path = MODEL_PATHS[MODEL_TYPE]["head_path"]

    if MODEL_TYPE == "linear":
        model = LoRADINOv2Linear(MODEL_NAME, adapter_dir, NUM_CLASSES)
    elif MODEL_TYPE == "mlp":
        model = LoRADINOv2MLP(MODEL_NAME, adapter_dir, NUM_CLASSES)
    elif MODEL_TYPE == "mlp_bn_gelu":
        model = LoRADINOv2MLPBNGELU(MODEL_NAME, adapter_dir, NUM_CLASSES)
    elif MODEL_TYPE == "mlp_layernorm_gelu":
        model = LoRADINOv2MLPLayerNormGELU(MODEL_NAME, adapter_dir, NUM_CLASSES)
    else:
        raise ValueError("Unknown MODEL_TYPE")

    checkpoint = torch.load(head_path, map_location="cpu")

    if isinstance(checkpoint, dict) and "classifier_state_dict" in checkpoint:
        state_dict = checkpoint["classifier_state_dict"]
    else:
        state_dict = checkpoint

    model.classifier.load_state_dict(state_dict, strict=False)
    return model


def build_loader(processor):
    dataset = DINOv2ImageFolder(DATA_ROOT, processor, image_size=IMAGE_SIZE)

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    return dataset, loader


def save_predictions_csv(save_path, rows, class_names):
    prob_columns = [f"prob_{_safe_name(c)}" for c in class_names]

    header = [
        "image_path",
        "image_name",
        "true_index",
        "true_label",
        "pred_index",
        "pred_label",
        "pred_confidence",
        *prob_columns,
        "correct",
    ]

    with open(save_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def save_confusion_matrix_csv(save_path, cm, class_names):
    header = ["true\\pred", *class_names]
    with open(save_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for true_name, row in zip(class_names, cm):
            writer.writerow([true_name, *row.tolist()])


def save_classification_report(save_path, report_text):
    with open(save_path, "w", encoding="utf-8-sig") as f:
        f.write(report_text)


def save_summary_txt(save_path, metrics, cm, class_names):
    with open(save_path, "w", encoding="utf-8-sig") as f:
        f.write("Evaluation summary\n")
        f.write("==================\n")
        f.write(f"Model type: {MODEL_TYPE}\n")
        f.write(f"Data root: {DATA_ROOT}\n")
        f.write(f"Class names: {class_names}\n\n")
        for key, value in metrics.items():
            f.write(f"{key}: {value:.6f}\n")

        f.write("\nConfusion matrix\n")
        f.write("================\n")
        f.write("Rows = true labels, columns = predicted labels\n")
        f.write(",".join(["true\\pred", *class_names]) + "\n")
        for true_name, row in zip(class_names, cm):
            f.write(",".join([true_name, *[str(int(x)) for x in row]]) + "\n")


def plot_confusion_matrix(cm, class_names, save_dir, model_type):
    """
    Save a confusion-matrix figure for the paper.
    The upper number in each cell is the count, and the lower number is row-normalized percentage.
    """
    import matplotlib.pyplot as plt

    cm = np.asarray(cm)
    row_sum = cm.sum(axis=1, keepdims=True)
    cm_percent = np.divide(cm, row_sum, out=np.zeros_like(cm, dtype=float), where=row_sum != 0) * 100

    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 11
    plt.rcParams["axes.linewidth"] = 1.0
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42

    fig, ax = plt.subplots(figsize=(5.2, 4.6), dpi=300)
    im = ax.imshow(cm_percent, cmap="Blues", vmin=0, vmax=100)

    ax.set_title("Confusion matrix under the P4 protocol", fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel("Predicted label", fontsize=12)
    ax.set_ylabel("True label", fontsize=12)

    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels([str(c).capitalize() for c in class_names])
    ax.set_yticklabels([str(c).capitalize() for c in class_names])

    for i in range(len(class_names)):
        for j in range(len(class_names)):
            count = cm[i, j]
            percent = cm_percent[i, j]
            text_color = "white" if percent >= 50 else "black"
            ax.text(
                j,
                i,
                f"{count}\n({percent:.1f}%)",
                ha="center",
                va="center",
                color=text_color,
                fontsize=11,
            )

                                   
    ax.set_xticks(np.arange(-0.5, len(class_names), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(class_names), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.5)
    ax.tick_params(which="minor", bottom=False, left=False)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Row-normalized percentage (%)", rotation=270, labelpad=18)

    fig.tight_layout()

    png_path = os.path.join(save_dir, f"confusion_matrix_{model_type}.png")
    pdf_path = os.path.join(save_dir, f"confusion_matrix_{model_type}.pdf")
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    return png_path, pdf_path


@torch.no_grad()
def evaluate(model, loader, device, class_names):
    model.eval()

    all_preds = []
    all_labels = []
    all_rows = []

    for pixel_values, labels, paths in loader:
        pixel_values = pixel_values.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.autocast(device_type="cuda", enabled=(device.type == "cuda")):
            logits = model(pixel_values)

        probs = torch.softmax(logits, dim=1)
        pred_conf, preds = torch.max(probs, dim=1)

        preds_cpu = preds.cpu().numpy().tolist()
        labels_cpu = labels.cpu().numpy().tolist()
        probs_cpu = probs.cpu().numpy().tolist()
        conf_cpu = pred_conf.cpu().numpy().tolist()

        all_preds.extend(preds_cpu)
        all_labels.extend(labels_cpu)

        for path, y_true, y_pred, conf, prob_vec in zip(paths, labels_cpu, preds_cpu, conf_cpu, probs_cpu):
            all_rows.append([
                path,
                os.path.basename(path),
                y_true,
                class_names[y_true],
                y_pred,
                class_names[y_pred],
                f"{conf:.6f}",
                *[f"{p:.6f}" for p in prob_vec],
                int(y_true == y_pred),
            ])

    acc = accuracy_score(all_labels, all_preds)
    precision_macro = precision_score(all_labels, all_preds, average="macro", zero_division=0)
    recall_macro = recall_score(all_labels, all_preds, average="macro", zero_division=0)
    f1_binary = f1_score(all_labels, all_preds, average="binary", zero_division=0)
    f1_macro = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    report = classification_report(
        all_labels,
        all_preds,
        target_names=class_names,
        digits=4,
        zero_division=0,
    )
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(len(class_names))))

    metrics = {
        "accuracy": acc,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_binary": f1_binary,
        "f1_macro": f1_macro,
    }

    return metrics, report, cm, all_rows


def main():
    save_dir = SAVE_DIR.strip()
    os.makedirs(save_dir, exist_ok=True)

    if not os.path.exists(DATA_ROOT):
        print("Error: test dataset directory not found")
        print(DATA_ROOT)
        return

    if DEVICE == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA was selected but is not available. The device will be switched to CPU automatically.")
        device = torch.device("cpu")
    else:
        device = torch.device(DEVICE)

    print("Loading processor...")
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)

    print("Loading dataset...")
    dataset, loader = build_loader(processor)

    print("Class order detected by ImageFolder:", dataset.classes)
    print("Configured CLASS_NAMES:", CLASS_NAMES)

    if dataset.classes != CLASS_NAMES:
        print("Warning: dataset.classes is inconsistent with CLASS_NAMES.")
        print("This evaluation will use the class order detected by ImageFolder:", dataset.classes)

    class_names = dataset.classes

    print("Loading model...")
    model = build_model().to(device)

    print("Model loaded successfully")
    print("Current device:", device)
    print("Model type:", MODEL_TYPE)
    print("Test set size:", len(dataset))

    metrics, report, cm, rows = evaluate(model, loader, device, class_names)

    print("\n================ Final Results ================")
    print("Acc             = %.4f" % metrics["accuracy"])
    print("Precision_macro = %.4f" % metrics["precision_macro"])
    print("Recall_macro    = %.4f" % metrics["recall_macro"])
    print("F1_binary       = %.4f" % metrics["f1_binary"])
    print("F1_macro        = %.4f" % metrics["f1_macro"])

    print("\n========== Classification Report ==========")
    print(report)

    print("========== Confusion Matrix ==========")
    print(cm)

                                                               
                  
                                                               
    predictions_csv = os.path.join(save_dir, f"predictions_{MODEL_TYPE}.csv")
    cm_csv = os.path.join(save_dir, f"confusion_matrix_{MODEL_TYPE}.csv")
    report_txt = os.path.join(save_dir, f"classification_report_{MODEL_TYPE}.txt")
    summary_txt = os.path.join(save_dir, f"summary_{MODEL_TYPE}.txt")

    save_predictions_csv(predictions_csv, rows, class_names)
    save_confusion_matrix_csv(cm_csv, cm, class_names)
    save_classification_report(report_txt, report)
    save_summary_txt(summary_txt, metrics, cm, class_names)

    print("\nPrediction details saved to:")
    print(predictions_csv)

    print("\nconfusion_matrix CSV 已保存到:")
    print(cm_csv)

    print("\nClassification report saved to:")
    print(report_txt)

    print("\nSummary results saved to:")
    print(summary_txt)

    if SAVE_CONFUSION_MATRIX_FIGURE:
        png_path, pdf_path = plot_confusion_matrix(cm, class_names, save_dir, MODEL_TYPE)
        print("\nconfusion_matriximage已保存到:")
        print(png_path)
        print(pdf_path)

    print("==========================================")


if __name__ == "__main__":
    main()
