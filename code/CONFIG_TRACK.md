# CUB-200-2011 Keypoint Localization - Hyperparameter Tracking

## Project Overview
- **Task**: Localize 3 keypoints (left_eye, right_eye, beak) on bird images
- **Dataset**: CUB-200-2011 (70/30 train/val split)
- **Backbones**: ResNet50, DenseNet121, YOLOv8
- **Best Result So Far**: ResNet50 with Wing loss, MAE 0.0221, PCK@0.1 98.96%

---

## Current Active Hyperparameters (Final)

| Parameter | Value |
|-----------|-------|
| loss_function | wing |
| learning_rate | 5e-5 |
| weight_decay | 0.0 |
| hidden_dim | 512 |
| dropout | 0.2 |
| visibility_loss_weight | 2.0 |
| augmentation | medium, heavy |
| scheduler | plateau |
| epochs | 50 |
| img_size | 640 |

---

## Best Results - Round 3

| Rank | Model | Config | Avg MAE | PCK@0.1 |
|------|-------|--------|---------|---------|
| 1 | resnet50 | wing_lr5e-05_h512_do0.2_medium | **0.0221** | **98.96%** |
| 2 | resnet50 | wing_lr5e-05_h512_do0.2_light | 0.0244 | 98.16% |
| 3 | densenet | wing_lr1e-04_h512_do0.4_heavy | 0.0261 | 98.53% |
| 4 | densenet | wing_lr5e-05_h512_do0.2_light | 0.0267 | 98.14% |

---

## Optimal Hyperparameters

| Parameter | Optimal Value | Notes |
|-----------|---------------|-------|
| loss_function | wing | Clear winner |
| learning_rate | 5e-5 | Sweet spot |
| weight_decay | 0.0 | No regularization needed |
| hidden_dim | 512 | Best capacity |
| dropout | 0.2 | Lower = better precision |
| augmentation | medium | Best generalization |
| img_size | 640 | Higher res for fine keypoints |
| epochs | 50 | Longer training |

---

## YOLOv8 Configuration

| Parameter | Value |
|-----------|-------|
| imgsz | 640 |
| epochs | 50 |
| patience | 20 |
| lr0 | 0.01 |
| lrf | 0.01 |
| mosaic | 0.5 |
| fliplr | 0.5 |
| flipud | 0.0 |
| degrees | 10 |

---

## Changelog

| Date | Change |
|------|--------|
| 2024-12-22 | Initial document |
| 2024-12-23 | Round 2: Wing loss best (MAE 0.0263) |
| 2024-12-24 | Round 3: Best MAE 0.0221, PCK 98.96% |
| 2024-12-24 | Final: wing, lr=5e-5, do=0.2, medium/heavy aug |
| 2024-12-24 | Updated img_size=640, epochs=50 |
| 2024-12-24 | Reduced grid to 4 configs |