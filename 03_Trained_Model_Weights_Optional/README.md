# LoRA-DINOv2 Experimental Code

This package contains cleaned English code and evaluation outputs for LoRA-DINOv2-based coal-gangue recognition.

## Main files

- `train_lora_dinov2_linear.py`: LoRA-DINOv2 with a linear classification head.
- `train_lora_dinov2_mlp.py`: LoRA-DINOv2 with an MLP classification head.
- `train_lora_dinov2_mlp_bn_gelu.py`: LoRA-DINOv2 with BatchNorm and GELU.
- `train_lora_dinov2_mlp_layernorm_gelu.py`: LoRA-DINOv2 with LayerNorm and GELU.
- `train_lora_dinov2_deepmlp_layernorm_gelu.py`: LoRA-DINOv2 with a deeper MLP head.
- `train_lora_dinov2_mlp_layernorm_gelu_visualization_pipeline.py`: training and visualization pipeline for the LayerNorm-GELU variant.
- `detect_lora_dinov2_dataset.py`: batch evaluation script for different model variants.
- `inference_time_linear.py`: inference-time measurement script.
- `confusion_matrix/`: prediction outputs, confusion matrix files, and the related evaluation script.

## Dataset structure

The expected dataset structure follows the ImageFolder format:

```text
dataset_root/
├── train/
│   ├── coal/
│   └── gangue/
├── val/
│   ├── coal/
│   └── gangue/
└── test/
    ├── coal/
    └── gangue/
```

## Notes

All Python comments were removed from the source files. Chinese directory names and Chinese path strings were converted to English naming. Local paths are placeholders and should be updated according to the actual experimental environment before running the scripts.
