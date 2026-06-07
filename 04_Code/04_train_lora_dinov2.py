"""
Reference template for LoRA-DINOv2 coal-gangue classification.

This file is a clean submission-oriented script. It should be adapted to the
final dataset split files before execution.
"""

from pathlib import Path
import argparse
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
from peft import LoraConfig, get_peft_model


class CoalGangueDataset(Dataset):
    def __init__(self, csv_file, dataset_root, processor):
        self.data = pd.read_csv(csv_file)
        self.dataset_root = Path(dataset_root)
        self.processor = processor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]
        image_path = self.dataset_root / row["relative_path"]
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].squeeze(0)
        label = int(row["label_id"])
        return pixel_values, torch.tensor(label, dtype=torch.long)


class DINOv2Classifier(nn.Module):
    def __init__(self, backbone_name, hidden_dim=512, dropout=0.3, num_classes=2):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(backbone_name)
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["query", "key", "value"],
            lora_dropout=0.1,
            bias="none",
        )
        self.backbone = get_peft_model(self.backbone, lora_config)
        feature_dim = self.backbone.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, pixel_values):
        outputs = self.backbone(pixel_values=pixel_values)
        cls_token = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(cls_token)
        return logits


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0

    for pixel_values, labels in loader:
        pixel_values = pixel_values.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(pixel_values)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)

    return total_loss / max(1, len(loader.dataset))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", default="../02_Data_Splits/train_split_template.csv")
    parser.add_argument("--dataset_root", default="..")
    parser.add_argument("--output_dir", default="../05_Result_Data/Generated_Outputs")
    parser.add_argument("--backbone", default="facebook/dinov2-base")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoImageProcessor.from_pretrained(args.backbone)
    dataset = CoalGangueDataset(args.train_csv, args.dataset_root, processor)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    model = DINOv2Classifier(args.backbone).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    history = []
    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(model, loader, optimizer, device)
        history.append({"epoch": epoch, "training_loss": loss})
        print(f"Epoch {epoch:03d} | training_loss={loss:.6f}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(output_dir / "training_history.csv", index=False)
    torch.save(model.state_dict(), output_dir / "lora_dinov2_classifier_state_dict.pt")


if __name__ == "__main__":
    main()
