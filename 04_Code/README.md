# Reference Code

This directory contains clean English-language reference scripts for organizing the supplementary material.

The scripts are designed as academic submission templates. They should be adapted to the final local paths and dataset format before execution.

Recommended workflow:

1. Edit config_template.yaml.
2. Place the dataset in 01_Dataset/Dataset_To_Be_Added.
3. Generate object-level classification crops if scene-level images and bounding boxes are available.
4. Construct auxiliary source-domain samples.
5. Analyze dual-domain discrepancy.
6. Train LoRA-DINOv2 and baseline models.
7. Evaluate the model and export figure source data.
8. Generate publication-ready figures.

The original local project paths should not be released in the supplementary material.
