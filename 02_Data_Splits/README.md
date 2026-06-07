# 02_Data_Splits

This folder provides object-level classification split files for the dual-domain coal-gangue recognition study.

## Source data

The split files were generated from the object-level images placed under:

`01_Dataset/Dataset_To_Be_Added/`

The generated files only reference object-level classification images. Scene-level images under `Original_Scene_Data_To_Be_Added` are not included in these split files.

## Class definition

- `0`: coal
- `1`: gangue

## Data variants

- `Source_Domain_Lab_Raw`: original laboratory-domain object-level samples.
- `Source_Domain_Lab_Enhanced`: recognition-oriented auxiliary source-domain samples.
- `Target_Domain_Field_Raw`: original field-domain object-level samples.

`Source_Domain_Lab_Aligned` is not included because no object-level images were detected for this variant in the provided dataset package.

## Generated protocol files

- `evaluation_protocols.csv`: definition of P1-P4 protocols.
- `data_inventory.csv`: image counts by data variant, class, and base split.
- `all_object_level_samples.csv`: complete object-level sample list.
- `P1_source_in_domain.csv`: source-domain in-domain evaluation.
- `P2_target_in_domain.csv`: target-domain in-domain evaluation.
- `P3_direct_cross_domain.csv`: direct source-to-target transfer evaluation.
- `P4_auxiliary_joint_training.csv`: target-domain evaluation with auxiliary source-domain samples.

## Split rule

For each available object-level data variant and class, a deterministic stratified split was generated:

- training: approximately 70%
- validation: approximately 15%
- testing: approximately 15%

For 30 images per class, this corresponds to:

- 21 training images
- 4 validation images
- 5 testing images

The deterministic shuffle uses a fixed seed to avoid ordering bias from filenames.

## Protocol design

- P1: train, validation, and test sets are all from `Source_Domain_Lab_Raw`.
- P2: train, validation, and test sets are all from `Target_Domain_Field_Raw`.
- P3: training and validation are from `Source_Domain_Lab_Raw`, while testing is from `Target_Domain_Field_Raw`.
- P4: training uses `Target_Domain_Field_Raw` training samples and all `Source_Domain_Lab_Enhanced` auxiliary samples. Validation and testing are only from `Target_Domain_Field_Raw`.

## Notes

The `relative_path` column is written relative to the supplementary material root. The referenced image files should remain in the same directory structure as listed in the CSV files.
