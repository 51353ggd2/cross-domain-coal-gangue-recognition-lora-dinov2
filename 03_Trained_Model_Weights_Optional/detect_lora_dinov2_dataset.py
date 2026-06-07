import os
import csv
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from PIL import ImageFile

from transformers import AutoImageProcessor, Dinov2Model
from peft import PeftModel


ImageFile.LOAD_TRUNCATED_IMAGES = True


                                                           
                       
                                                           

                  
     
          
       
               
                      
MODEL_TYPE = "linear"

                  
          
       
           
             
DATA_ROOT = r"F:\00_Coal_Gangue_Recognition_Paper\3.30_DINOv2\dataset\过往dataset汇总\现场dataset\test"

                       
SAVE_DIR = r"F:\00_Coal_Gangue_Recognition_Paper\3.30_DINOv2\LoRA_DINOv2\confusion_matrix\linear_csv"

                              
MODEL_NAME = "facebook/dinov2-base"

             
NUM_CLASSES = 2

                
IMAGE_SIZE = 224

                 
BATCH_SIZE = 4

                  
NUM_WORKERS = 2

         
DEVICE = "cuda"

                                    
                              
CLASS_NAMES = ["coal", "gangue"]


                                                           
         
                                                           

MODEL_PATHS = {
    "linear": {
        "adapter_dir": r"F:\00_Coal_Gangue_Recognition_Paper\3.30_DINOv2\LoRA_DINOv2\权重file\field\runs\best_lora_adapter",
        "head_path": r"F:\00_Coal_Gangue_Recognition_Paper\3.30_DINOv2\LoRA_DINOv2\权重file\field\runs\best_linear_head.pth",
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
        "adapter_dir": r"F:\00_Coal_Gangue_Recognition_Paper\3.30_DINOv2\LoRA_DINOv2\权重file\field\runs_mlp_layernorm_gelu\best_lora_adapter",
        "head_path": r"F:\00_Coal_Gangue_Recognition_Paper\3.30_DINOv2\LoRA_DINOv2\权重file\field\runs_mlp_layernorm_gelu\best_mlp_head.pth",
    },
}


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
        pin_memory=True
    )
    return dataset, loader


def save_predictions_csv(save_path, rows):
    with open(save_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "true_label", "pred_label", "correct"])
        writer.writerows(rows)


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

        preds = torch.argmax(logits, dim=1)

        preds_cpu = preds.cpu().numpy().tolist()
        labels_cpu = labels.cpu().numpy().tolist()

        all_preds.extend(preds_cpu)
        all_labels.extend(labels_cpu)

        for path, y_true, y_pred in zip(paths, labels_cpu, preds_cpu):
            all_rows.append([
                path,
                class_names[y_true],
                class_names[y_pred],
                int(y_true == y_pred)
            ])

    acc = accuracy_score(all_labels, all_preds)
    f1_binary = f1_score(all_labels, all_preds, average="binary")
    f1_macro = f1_score(all_labels, all_preds, average="macro")
    report = classification_report(all_labels, all_preds, target_names=class_names, digits=4)
    cm = confusion_matrix(all_labels, all_preds)

    return acc, f1_binary, f1_macro, report, cm, all_rows


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    if not os.path.exists(DATA_ROOT):
        print("Error: test dataset directory not found")
        print(DATA_ROOT)
        return

    device = torch.device(DEVICE if torch.cuda.is_available() or DEVICE == "cpu" else "cpu")

    print("Loading processor...")
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)

    print("Loading dataset...")
    dataset, loader = build_loader(processor)

    print("Class order detected by ImageFolder:", dataset.classes)
    print("Configured CLASS_NAMES:", CLASS_NAMES)

    if dataset.classes != CLASS_NAMES:
        print("Warning: dataset.classes is inconsistent with CLASS_NAMES.")
        print("Suggested CLASS_NAMES:", dataset.classes)

    print("Loading model...")
    model = build_model().to(device)

    print("Model loaded successfully")
    print("Current device:", device)
    print("Model type:", MODEL_TYPE)
    print("Test set size:", len(dataset))

    acc, f1_binary, f1_macro, report, cm, rows = evaluate(model, loader, device, dataset.classes)

    print("\n================ Final Results ================")
    print("Acc          = %.4f" % acc)
    print("F1_binary    = %.4f" % f1_binary)
    print("F1_macro     = %.4f" % f1_macro)

    print("\n========== Classification Report ==========")
    print(report)

    print("========== Confusion Matrix ==========")
    print(cm)

    csv_path = os.path.join(SAVE_DIR, f"predictions_{MODEL_TYPE}.csv")
    save_predictions_csv(csv_path, rows)

    print("\nPrediction details saved to:")
    print(csv_path)
    print("==========================================")


if __name__ == "__main__":
    main()
