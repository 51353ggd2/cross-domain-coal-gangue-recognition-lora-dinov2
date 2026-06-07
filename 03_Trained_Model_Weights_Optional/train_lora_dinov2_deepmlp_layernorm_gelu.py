import os
import random
import numpy as np

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision import datasets
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
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
            nn.Linear(hidden_size, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.2),

            nn.Linear(256, num_classes),
        )

    def forward(self, pixel_values):
        outputs = self.backbone(pixel_values=pixel_values)
        cls_feature = outputs.last_hidden_state[:, 0]
        logits = self.classifier(cls_feature)
        return logits


def build_loaders(data_root, processor, batch_size=4, num_workers=2, image_size=224):
    train_set = DINOv2ImageFolder(os.path.join(data_root, "train"), processor, image_size=image_size)
    val_set = DINOv2ImageFolder(os.path.join(data_root, "val"), processor, image_size=image_size)
    test_set = DINOv2ImageFolder(os.path.join(data_root, "test"), processor, image_size=image_size)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

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
    f1_binary = f1_score(all_labels, all_preds, average="binary")
    f1_macro = f1_score(all_labels, all_preds, average="macro")

    if print_report and class_names is not None:
        print("\n========== Classification Report ==========")
        print(classification_report(all_labels, all_preds, target_names=class_names, digits=4))

        print("========== Confusion Matrix ==========")
        print(confusion_matrix(all_labels, all_preds))

    return acc, f1_binary, f1_macro


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
    ckpt = torch.load(os.path.join(save_dir, "best_mlp_head.pth"), map_location=device)
    model.classifier.load_state_dict(ckpt["classifier_state_dict"])
    return ckpt


def main():
                                  
    data_root = r"F:\00_Coal_Gangue_Recognition_Paper\3.30_DINOv2\dataset\过往dataset汇总\现场dataset"
    save_dir = r"F:\00_Coal_Gangue_Recognition_Paper\3.30_DINOv2\LoRA_DINOv2\权重file\field\runs_deepmlp_layernorm_gelu"

                         
    model_name = "facebook/dinov2-base"
    num_classes = 2
    image_size = 224
    batch_size = 4
    num_workers = 2
    num_epochs = 10
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

    processor = AutoImageProcessor.from_pretrained(model_name)
    train_loader, val_loader, test_loader, class_names = build_loaders(
        data_root=data_root,
        processor=processor,
        batch_size=batch_size,
        num_workers=num_workers,
        image_size=image_size,
    )

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

    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))

    best_val_f1 = -1.0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for pixel_values, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}"):
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

        val_acc, val_f1_binary, val_f1_macro = evaluate(
            model, val_loader, device, class_names=class_names, print_report=False
        )

        print(
            f"[Epoch {epoch + 1}] "
            f"loss={running_loss / len(train_loader):.4f} | "
            f"val_acc={val_acc:.4f} | "
            f"val_f1_binary={val_f1_binary:.4f} | "
            f"val_f1_macro={val_f1_macro:.4f}"
        )

        if val_f1_binary > best_val_f1:
            best_val_f1 = val_f1_binary
            save_best_checkpoint(model, processor, save_dir, epoch + 1, best_val_f1)

    load_best_checkpoint(model, save_dir, device)
    test_acc, test_f1_binary, test_f1_macro = evaluate(
        model, test_loader, device, class_names=class_names, print_report=True
    )

    print(
        f"[TEST] acc={test_acc:.4f} | "
        f"f1_binary={test_f1_binary:.4f} | "
        f"f1_macro={test_f1_macro:.4f}"
    )


if __name__ == "__main__":
    main()
