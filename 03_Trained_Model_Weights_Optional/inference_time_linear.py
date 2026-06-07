import os
import time
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from transformers import AutoImageProcessor, Dinov2Model
from peft import PeftModel


                                                           
                       
                                                           

                               
LORA_ADAPTER_DIR = r"/3.30_DINOv2/LoRA_DINOv2/权重file/field/runs\best_lora_adapter"

                                     
HEAD_WEIGHT_PATH = r"/3.30_DINOv2/LoRA_DINOv2/权重file/field/runs\best_linear_head.pth"

                   
IMAGE_PATH = r"/3.30_DINOv2/dataset/过往dataset汇总/现场dataset\test\coal\004004_jpg.rf.f95dd1a11bc9fb9e93d69dbb211b630b_2.jpg"

                            
                                                         
FOLDER_PATH = r"/3.30_DINOv2/dataset/过往dataset汇总/现场dataset\test\coal"

                              
MODEL_NAME = "facebook/dinov2-base"

             
NUM_CLASSES = 2

                
IMG_SIZE = 224

           
                       
                             
MODE = "single"

             
REPEAT_TIMES = 100

          
DEVICE = "cuda"


                                                           
         
                                                           

class LoRADINOv2Linear(nn.Module):
    def __init__(self, model_name, lora_adapter_dir, num_classes):
        super().__init__()

        base_model = Dinov2Model.from_pretrained(model_name)
        self.backbone = PeftModel.from_pretrained(base_model, lora_adapter_dir)

        hidden_size = self.backbone.config.hidden_size
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, pixel_values):
        outputs = self.backbone(pixel_values=pixel_values)
        cls_feature = outputs.last_hidden_state[:, 0]
        logits = self.classifier(cls_feature)
        return logits


def load_model():
    device = torch.device(DEVICE if torch.cuda.is_available() or DEVICE == "cpu" else "cpu")

    model = LoRADINOv2Linear(
        model_name=MODEL_NAME,
        lora_adapter_dir=LORA_ADAPTER_DIR,
        num_classes=NUM_CLASSES
    )

    checkpoint = torch.load(HEAD_WEIGHT_PATH, map_location=device)

    if isinstance(checkpoint, dict) and "classifier_state_dict" in checkpoint:
        state_dict = checkpoint["classifier_state_dict"]
    else:
        state_dict = checkpoint

    model.classifier.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()

    return model, device


def get_transform():
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    return transform


def load_one_image(image_path, transform):
    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0)
    return img_tensor


def get_all_images(folder_path):
    image_list = []
    valid_ext = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]

    for name in os.listdir(folder_path):
        ext = os.path.splitext(name)[1].lower()
        if ext in valid_ext:
            image_list.append(os.path.join(folder_path, name))

    image_list.sort()
    return image_list


def cuda_sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()
def test_single_image_time(model, device, transform):
    if not os.path.exists(IMAGE_PATH):
        print("Error: image file not found:")
        print(IMAGE_PATH)
        return

    img_tensor = load_one_image(IMAGE_PATH, transform).to(device)

        
    for _ in range(20):
        _ = model(img_tensor)

    cuda_sync(device)
    start = time.perf_counter()

    for _ in range(REPEAT_TIMES):
        _ = model(img_tensor)

    cuda_sync(device)
    end = time.perf_counter()

    total_time = end - start
    avg_time = total_time / REPEAT_TIMES
    fps = 1.0 / avg_time

    print("======================================")
    print("Test mode: single image")
    print("Image path:", IMAGE_PATH)
    print("Number of repeated runs:", REPEAT_TIMES)
    print("Average inference time per run:%.3f ms" % (avg_time * 1000))
    print("FPS:%.2f" % fps)
    print("======================================")


@torch.no_grad()
def test_folder_time(model, device, transform):
    if not os.path.exists(FOLDER_PATH):
        print("Error: image folder not found:")
        print(FOLDER_PATH)
        return

    image_paths = get_all_images(FOLDER_PATH)

    if len(image_paths) == 0:
        print("Error: no images found in the folder")
        print(FOLDER_PATH)
        return

        
    first_img = load_one_image(image_paths[0], transform).to(device)
    for _ in range(20):
        _ = model(first_img)

    times = []

    for img_path in image_paths:
        img_tensor = load_one_image(img_path, transform).to(device)

        cuda_sync(device)
        start = time.perf_counter()

        _ = model(img_tensor)

        cuda_sync(device)
        end = time.perf_counter()

        one_time = end - start
        times.append(one_time)

    avg_time = sum(times) / len(times)
    fps = 1.0 / avg_time

    print("======================================")
    print("Test mode: image folder")
    print("Folder path:", FOLDER_PATH)
    print("Number of images:", len(image_paths))
    print("Average inference time per image:%.3f ms" % (avg_time * 1000))
    print("Fastest inference time:%.3f ms" % (min(times) * 1000))
    print("Slowest inference time:%.3f ms" % (max(times) * 1000))
    print("FPS:%.2f" % fps)
    print("======================================")


def main():
    print("Loading model...")
    model, device = load_model()
    transform = get_transform()

    print("Model loaded successfully")
    print("Current device:", device)
    print("Model:LoRA-DINOv2 + Linear")

    if MODE == "single":
        test_single_image_time(model, device, transform)
    elif MODE == "folder":
        test_folder_time(model, device, transform)
    else:
        print('Error: MODE must be either "single" or "folder"')


if __name__ == "__main__":
    main()