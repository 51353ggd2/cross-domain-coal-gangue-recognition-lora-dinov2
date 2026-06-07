import os
import csv
import time
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision import datasets

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from tqdm import tqdm
from PIL import ImageFile

from transformers import AutoImageProcessor, Dinov2Model
from peft import LoraConfig, get_peft_model


         
                                                                                                                                                            

ImageFile.LOAD_TRUNCATED_IMAGES = True


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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

        return pixel_values, target


class LoRADINOv2MLP(nn.Module):
    def __init__(
        self,
        model_name: str = "facebook/dinov2-base",
        num_classes: int = 2,
        hidden_dim: int = 512,
        dropout: float = 0.3,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        target_modules=None,
    ):
        super().__init__()

        if target_modules is None:
            target_modules = ["query", "key", "value"]

        base_model = Dinov2Model.from_pretrained(model_name)

        peft_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            target_modules=target_modules,
        )

        self.backbone = get_peft_model(base_model, peft_config)
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


def build_loaders(data_root, processor, batch_size=4, num_workers=2, image_size=224):
    train_set = DINOv2ImageFolder(
        os.path.join(data_root, "train"),
        processor,
        image_size=image_size
    )

    val_set = DINOv2ImageFolder(
        os.path.join(data_root, "val"),
        processor,
        image_size=image_size
    )

    test_set = DINOv2ImageFolder(
        os.path.join(data_root, "test"),
        processor,
        image_size=image_size
    )

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader, train_set.classes


def evaluate(model, loader, device, class_names=None, print_report=False):
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for pixel_values, labels in loader:
            pixel_values = pixel_values.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.autocast(device_type="cuda", enabled=(device == "cuda")):
                logits = model(pixel_values)

            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    acc = accuracy_score(all_labels, all_preds)

    precision_macro = precision_score(
        all_labels,
        all_preds,
        average="macro",
        zero_division=0
    )

    recall_macro = recall_score(
        all_labels,
        all_preds,
        average="macro",
        zero_division=0
    )

    f1_macro = f1_score(
        all_labels,
        all_preds,
        average="macro",
        zero_division=0
    )

    f1_binary = f1_score(
        all_labels,
        all_preds,
        average="binary",
        zero_division=0
    )

    if print_report and class_names is not None:
        print("\n========== Classification Report ==========")
        print(
            classification_report(
                all_labels,
                all_preds,
                target_names=class_names,
                digits=4,
                zero_division=0
            )
        )

        print("========== Confusion Matrix ==========")
        print(confusion_matrix(all_labels, all_preds))

    return {
        "acc": acc,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "f1_binary": f1_binary,
    }


def save_best_checkpoint(model, processor, save_dir, epoch, best_val_f1):
    os.makedirs(save_dir, exist_ok=True)

    adapter_dir = os.path.join(save_dir, "best_lora_adapter")

    model.backbone.save_pretrained(adapter_dir)
    processor.save_pretrained(adapter_dir)

    torch.save(
        {
            "classifier_state_dict": model.classifier.state_dict(),
            "best_val_f1": best_val_f1,
            "epoch": epoch,
        },
        os.path.join(save_dir, "best_mlp_head.pth"),
    )


def load_best_checkpoint(model, save_dir, device):
    ckpt_path = os.path.join(save_dir, "best_mlp_head.pth")

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Best checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device)
    model.classifier.load_state_dict(ckpt["classifier_state_dict"])

    print(
        f"\nLoaded best checkpoint from epoch {ckpt['epoch']} "
        f"with best_val_f1={ckpt['best_val_f1']:.4f}"
    )

    return ckpt


def plot_training_curves(log_path, save_dir):
    """
    Output four-panel training curves:
    (a) Training loss
    (b) Validation precision
    (c) Validation recall
    (d) Validation F1-score
    """
    df = pd.read_csv(log_path)

    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 11
    plt.rcParams["axes.linewidth"] = 1.0

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), dpi=300)

                       
    ax = axes[0, 0]
    ax.plot(df["epoch"], df["train_loss"], linewidth=1.2)
    ax.set_xlabel("epoch")
    ax.set_ylabel("Loss")
    ax.set_title("(a) Training loss")
    ax.grid(alpha=0.25, linestyle="--")

                              
    ax = axes[0, 1]
    ax.plot(df["epoch"], df["val_precision_macro"], linewidth=1.2)
    ax.set_xlabel("epoch")
    ax.set_ylabel("Precision")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("(b) Validation precision")
    ax.grid(alpha=0.25, linestyle="--")

                           
    ax = axes[1, 0]
    ax.plot(df["epoch"], df["val_recall_macro"], linewidth=1.2)
    ax.set_xlabel("epoch")
    ax.set_ylabel("Recall")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("(c) Validation recall")
    ax.grid(alpha=0.25, linestyle="--")

                             
    ax = axes[1, 1]
    ax.plot(df["epoch"], df["val_f1_macro"], linewidth=1.2)
    ax.set_xlabel("epoch")
    ax.set_ylabel("F1-score")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("(d) Validation F1-score")
    ax.grid(alpha=0.25, linestyle="--")

    plt.tight_layout()

    png_path = os.path.join(save_dir, "training_curves.png")
    pdf_path = os.path.join(save_dir, "training_curves.pdf")

    plt.savefig(png_path, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close()

    print(f"\nTraining curves saved to:")
    print(png_path)
    print(pdf_path)


def save_test_summary(save_dir, test_metrics):
    summary_path = os.path.join(save_dir, "test_summary.txt")

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("[TEST RESULTS]\n")
        f.write(f"Accuracy: {test_metrics['acc']:.6f}\n")
        f.write(f"Precision_macro: {test_metrics['precision_macro']:.6f}\n")
        f.write(f"Recall_macro: {test_metrics['recall_macro']:.6f}\n")
        f.write(f"F1_macro: {test_metrics['f1_macro']:.6f}\n")
        f.write(f"F1_binary: {test_metrics['f1_binary']:.6f}\n")

    print(f"\nTest summary saved to:\n{summary_path}")


def main():
                                                               
                           
                                                               
    data_root = r"F:\00_Coal_Gangue_Recognition_Paper\3.30_DINOv2\dataset\4_14+现场dataset"

    save_dir = r"F:\00_Coal_Gangue_Recognition_Paper\3.30_DINOv2\LoRA_DINOv2\权重file\lab_enhance+field\runs_mlp_layernorm_gelu_20epochs"

                                                               
                
                                                               
    model_name = "facebook/dinov2-base"
    num_classes = 2
    image_size = 224

    batch_size = 4
    num_workers = 2

                 
    num_epochs = 20

    learning_rate = 1e-4
    weight_decay = 1e-4

    hidden_dim = 512
    dropout = 0.3

    lora_r = 16
    lora_alpha = 32
    lora_dropout = 0.1

                                                               
             
                                                               
    set_seed(42)
    os.makedirs(save_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("\n========== Environment Check ==========")
    print("Torch version:", torch.__version__)
    print("Torch CUDA:", torch.version.cuda)
    print("CUDA available:", torch.cuda.is_available())
    print("Device:", device)
    print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No CUDA")
    print("=======================================\n")

    if device != "cuda":
        print("[WARNING] CUDA is not available. The model will run on CPU and may be very slow.")

    processor = AutoImageProcessor.from_pretrained(model_name)

    train_loader, val_loader, test_loader, class_names = build_loaders(
        data_root=data_root,
        processor=processor,
        batch_size=batch_size,
        num_workers=num_workers,
        image_size=image_size,
    )

    print("Class names:", class_names)
    print("Train batches:", len(train_loader))
    print("Val batches:", len(val_loader))
    print("Test batches:", len(test_loader))

    model = LoRADINOv2MLP(
        model_name=model_name,
        num_classes=num_classes,
        hidden_dim=hidden_dim,
        dropout=dropout,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=["query", "key", "value"],
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("\n========== Parameter Statistics ==========")
    print("Total params:", total_params)
    print("Trainable params:", trainable_params)
    print("Trainable ratio: {:.4f}%".format(100 * trainable_params / total_params))

    print("\n===== Trainable Parameters =====")
    for name, p in model.named_parameters():
        if p.requires_grad:
            print("TRAINABLE:", name)

    if hasattr(model.backbone, "print_trainable_parameters"):
        print("\n===== PEFT Summary =====")
        model.backbone.print_trainable_parameters()

    optimizer = AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )

    criterion = nn.CrossEntropyLoss()

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(device == "cuda")
    )

                                                               
                  
                                                               
    log_path = os.path.join(save_dir, "training_log.csv")

    log_fields = [
        "epoch",
        "train_loss",
        "val_acc",
        "val_precision_macro",
        "val_recall_macro",
        "val_f1_macro",
        "val_f1_binary",
        "best_val_f1",
        "epoch_time_sec",
        "lr"
    ]

    with open(log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=log_fields)
        writer.writeheader()

    best_val_f1 = -1.0

                                                               
             
                                                               
    print("\n========== Start Training ==========")

    for epoch in range(num_epochs):
        epoch_start_time = time.time()

        model.train()
        running_loss = 0.0

        for pixel_values, labels in tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{num_epochs}"
        ):
            pixel_values = pixel_values.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type="cuda", enabled=(device == "cuda")):
                logits = model(pixel_values)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)

        val_metrics = evaluate(
            model,
            val_loader,
            device,
            class_names=class_names,
            print_report=False
        )

        val_acc = val_metrics["acc"]
        val_precision_macro = val_metrics["precision_macro"]
        val_recall_macro = val_metrics["recall_macro"]
        val_f1_macro = val_metrics["f1_macro"]
        val_f1_binary = val_metrics["f1_binary"]

        current_lr = optimizer.param_groups[0]["lr"]
        epoch_time = time.time() - epoch_start_time

                                          
        if val_f1_macro > best_val_f1:
            best_val_f1 = val_f1_macro
            save_best_checkpoint(
                model,
                processor,
                save_dir,
                epoch + 1,
                best_val_f1
            )

        log_row = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_acc": val_acc,
            "val_precision_macro": val_precision_macro,
            "val_recall_macro": val_recall_macro,
            "val_f1_macro": val_f1_macro,
            "val_f1_binary": val_f1_binary,
            "best_val_f1": best_val_f1,
            "epoch_time_sec": epoch_time,
            "lr": current_lr,
        }

        with open(log_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=log_fields)
            writer.writerow(log_row)

        print(
            f"[Epoch {epoch + 1}/{num_epochs}] "
            f"loss={train_loss:.4f} | "
            f"val_acc={val_acc:.4f} | "
            f"val_precision={val_precision_macro:.4f} | "
            f"val_recall={val_recall_macro:.4f} | "
            f"val_f1_macro={val_f1_macro:.4f} | "
            f"val_f1_binary={val_f1_binary:.4f} | "
            f"best_f1={best_val_f1:.4f} | "
            f"time={epoch_time:.1f}s"
        )

    print("\n========== Training Finished ==========")
    print(f"Training log saved to:\n{log_path}")

                                                               
                 
                                                               
    plot_training_curves(log_path, save_dir)

                                                               
                  
                                                               
    load_best_checkpoint(model, save_dir, device)

    test_metrics = evaluate(
        model,
        test_loader,
        device,
        class_names=class_names,
        print_report=True
    )

    print(
        f"\n[TEST] "
        f"acc={test_metrics['acc']:.4f} | "
        f"precision_macro={test_metrics['precision_macro']:.4f} | "
        f"recall_macro={test_metrics['recall_macro']:.4f} | "
        f"f1_macro={test_metrics['f1_macro']:.4f} | "
        f"f1_binary={test_metrics['f1_binary']:.4f}"
    )

    save_test_summary(save_dir, test_metrics)


if __name__ == "__main__":
    main()