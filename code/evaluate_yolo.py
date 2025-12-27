import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from tqdm import tqdm
import json
import seaborn as sns

from code.config import Config
from dataset import CUBDataset


def plot_metrics_yolo(config, distances, mae_per_part, pck_at_thresholds):
    """Generate all metric plots for YOLO"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # 1. Box plot of distances
    data = [distances[part] for part in config.PART_NAMES if len(distances[part]) > 0]
    labels = [part for part in config.PART_NAMES if len(distances[part]) > 0]

    if len(data) > 0:
        axes[0, 0].boxplot(data, labels=labels)
        axes[0, 0].set_ylabel('Normalized Distance Error')
        axes[0, 0].set_title('YOLOv8 - Error Distribution')
        axes[0, 0].grid(True, alpha=0.3)
    else:
        axes[0, 0].text(0.5, 0.5, 'No data', ha='center', va='center')

    # 2. MAE comparison
    parts = list(mae_per_part.keys())
    mae_totals = [mae_per_part[p]['total'] for p in parts]
    mae_x = [mae_per_part[p]['x'] for p in parts]
    mae_y = [mae_per_part[p]['y'] for p in parts]

    x = np.arange(len(parts))
    width = 0.25
    axes[0, 1].bar(x - width, mae_x, width, label='X error', alpha=0.8)
    axes[0, 1].bar(x, mae_y, width, label='Y error', alpha=0.8)
    axes[0, 1].bar(x + width, mae_totals, width, label='Total error', alpha=0.8)
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(parts)
    axes[0, 1].set_ylabel('MAE (normalized)')
    axes[0, 1].set_title('YOLOv8 - MAE per Part')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3, axis='y')

    # 3. PCK curve
    thresholds = [0.05, 0.1, 0.15, 0.2]
    for part in config.PART_NAMES:
        pck_values = [pck_at_thresholds[part][f'pck@{t}'] for t in thresholds]
        axes[1, 0].plot(thresholds, pck_values, 'o-', label=part, linewidth=2)
    axes[1, 0].set_xlabel('Normalized Distance Threshold')
    axes[1, 0].set_ylabel('PCK (%)')
    axes[1, 0].set_title('YOLOv8 - PCK Curve')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # 4. Success rate vs threshold
    all_dists = []
    for part_dists in distances.values():
        all_dists.extend(part_dists)

    if len(all_dists) > 0:
        all_dists = np.array(all_dists)
        thresh_values = np.linspace(0, 0.3, 50)
        success_rates = [np.mean(all_dists <= t) * 100 for t in thresh_values]

        axes[1, 1].plot(thresh_values, success_rates, 'b-', linewidth=2)
        axes[1, 1].fill_between(thresh_values, success_rates, alpha=0.2)
        axes[1, 1].set_xlabel('Distance Threshold')
        axes[1, 1].set_ylabel('Success Rate (%)')
        axes[1, 1].set_title('YOLOv8 - Success Rate Curve')
        axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(config.REPORTS_DIR / 'yolov8_metrics.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ Created yolov8_metrics.png")


def generate_confusion_matrix_yolo(config, distances, mae_per_part):
    """Generate confusion matrix for YOLO predictions"""
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    # Create a simple error distribution heatmap
    parts = list(mae_per_part.keys())
    error_types = ['X Error', 'Y Error', 'Total Error']

    data = np.array([
        [mae_per_part[p]['x'] for p in parts],
        [mae_per_part[p]['y'] for p in parts],
        [mae_per_part[p]['total'] for p in parts]
    ])

    sns.heatmap(data, annot=True, fmt='.4f', cmap='YlOrRd',
                xticklabels=parts, yticklabels=error_types, ax=ax)
    ax.set_title('YOLOv8 - Error Distribution Heatmap')

    plt.tight_layout()
    plt.savefig(config.REPORTS_DIR / 'yolov8_confusion.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ Created yolov8_confusion.png")


def visualize_single_prediction(config, model, dataset, idx, vis_dir):
    """Visualize a single prediction"""
    batch_data = dataset[idx]
    if len(batch_data) == 4:
        _, targets, visibility, (orig_w, orig_h, img_id) = batch_data
    else:
        _, targets, (orig_w, orig_h, img_id) = batch_data
        visibility = torch.ones(config.NUM_PARTS)

    row = dataset.image_data.iloc[idx]
    img_path = str(config.IMAGES_DIR / row['path'])

    # Run inference
    results = model.predict(img_path, save=False, conf=0.25, verbose=False)

    # Load original image
    original_img = Image.open(img_path).convert('RGB')

    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    colors = ['red', 'blue', 'green']

    # Ground truth image
    gt_img = original_img.copy()
    gt_draw = ImageDraw.Draw(gt_img)

    # Predictions image
    pred_img = original_img.copy()
    pred_draw = ImageDraw.Draw(pred_img)

    # Draw ground truth (only visible keypoints)
    targets_reshaped = targets.view(-1, 2).numpy()
    vis_np = visibility.numpy()

    for j, (color, part_name) in enumerate(zip(colors, config.PART_NAMES)):
        if vis_np[j] == 1.0:
            gt = targets_reshaped[j]
            gt_x = gt[0] * orig_w
            gt_y = gt[1] * orig_h
            radius = max(6, int(min(orig_w, orig_h) * 0.015))
            gt_draw.ellipse(
                [gt_x - radius, gt_y - radius, gt_x + radius, gt_y + radius],
                fill=color, outline='white', width=2
            )

    # Draw predictions
    if len(results) > 0 and results[0].keypoints is not None:
        keypoints = results[0].keypoints.xy[0].cpu().numpy()
        for j, (color, part_name) in enumerate(zip(colors, config.PART_NAMES)):
            if j < len(keypoints):
                pred_x, pred_y = keypoints[j]
                cross_size = max(10, int(min(orig_w, orig_h) * 0.02))
                pred_draw.line(
                    [(pred_x - cross_size, pred_y - cross_size),
                     (pred_x + cross_size, pred_y + cross_size)],
                    fill=color, width=3
                )
                pred_draw.line(
                    [(pred_x - cross_size, pred_y + cross_size),
                     (pred_x + cross_size, pred_y - cross_size)],
                    fill=color, width=3
                )

    ax1.imshow(gt_img)
    ax1.set_title(f'Ground Truth ({int(vis_np.sum())} visible)')
    ax1.axis('off')

    ax2.imshow(pred_img)
    ax2.set_title('YOLOv8 Predictions')
    ax2.axis('off')

    plt.tight_layout()
    plt.savefig(vis_dir / f'sample_{idx}.png', dpi=120, bbox_inches='tight')
    plt.close()


def visualize_predictions_yolo(config, model, dataset, num_samples=6):
    """Generate visualization samples for YOLO"""
    model_vis_dir = config.VIS_DIR / 'yolov8'
    model_vis_dir.mkdir(exist_ok=True)

    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)

    for idx in indices:
        visualize_single_prediction(config, model, dataset, idx, model_vis_dir)

    print(f"  ✓ Generated {len(indices)} visualizations in {model_vis_dir}/")


def evaluate_yolo(config):
    """Evaluate YOLOv8-pose model"""
    from ultralytics import YOLO

    print(f"\nEvaluating YOLOv8...")

    # Check if model exists
    model_path = config.SAVE_DIR / 'yolov8_best.pt'
    if not model_path.exists():
        print(f"Model not found at {model_path}")
        print("Run train_yolo.py first!")
        return None

    # Load model
    model = YOLO(str(model_path))

    # Load validation dataset
    dataset = CUBDataset(config, mode='val')

    print(f"Evaluating on {len(dataset)} validation images...")

    # Track detection stats
    total_images = 0
    images_with_detections = 0
    total_keypoints_detected = 0

    # Collect metrics
    distances = {part: [] for part in config.PART_NAMES}
    mae_per_part = {part: {'x': 0, 'y': 0, 'total': 0, 'count': 0} for part in config.PART_NAMES}

    # Initialize PCK dictionary
    thresholds = [0.05, 0.1, 0.15, 0.2]
    pck_at_thresholds = {part: {f'pck@{t}': 0 for t in thresholds} for part in config.PART_NAMES}

    for idx in tqdm(range(len(dataset))):
        # Handle new 4-tuple format: (image, keypoints, visibility, metadata)
        batch_data = dataset[idx]
        if len(batch_data) == 4:
            _, targets, visibility, (orig_w, orig_h, img_id) = batch_data
        else:
            # Old format compatibility
            _, targets, (orig_w, orig_h, img_id) = batch_data
            visibility = torch.ones(config.NUM_PARTS)

        row = dataset.image_data.iloc[idx]
        img_path = str(config.IMAGES_DIR / row['path'])

        total_images += 1

        # Run inference
        results = model.predict(img_path, save=False, conf=0.25, verbose=False)

        if len(results) > 0 and results[0].keypoints is not None:
            images_with_detections += 1
            keypoints = results[0].keypoints.xy[0].cpu().numpy()
            total_keypoints_detected += len(keypoints)

            # Normalize keypoints
            pred_norm = keypoints.copy()
            pred_norm[:, 0] /= orig_w
            pred_norm[:, 1] /= orig_h

            # Compare with ground truth
            targets_reshaped = targets.view(-1, 2).numpy()
            vis_np = visibility.numpy()

            for j in range(min(len(pred_norm), config.NUM_PARTS)):
                gt = targets_reshaped[j]
                gt_vis = vis_np[j]

                # Only evaluate if ground truth is visible
                if gt_vis == 1.0:
                    pred = pred_norm[j]

                    # Calculate distance
                    dist = np.linalg.norm(pred - gt)
                    distances[config.PART_NAMES[j]].append(dist)

                    # Calculate MAE
                    mae_per_part[config.PART_NAMES[j]]['x'] += abs(pred[0] - gt[0])
                    mae_per_part[config.PART_NAMES[j]]['y'] += abs(pred[1] - gt[1])
                    mae_per_part[config.PART_NAMES[j]]['total'] += dist
                    mae_per_part[config.PART_NAMES[j]]['count'] += 1

    # Print detection stats
    print(f"\nYOLOv8 Detection Stats:")
    print(f"  Total images: {total_images}")
    print(f"  Images with detections: {images_with_detections} ({100 * images_with_detections / total_images:.1f}%)")
    print(f"  Total keypoints detected: {total_keypoints_detected}")
    print(f"  Avg keypoints per image: {total_keypoints_detected / max(images_with_detections, 1):.1f}")

    # Calculate average MAE
    for part in config.PART_NAMES:
        count = mae_per_part[part]['count']
        if count > 0:
            mae_per_part[part]['x'] /= count
            mae_per_part[part]['y'] /= count
            mae_per_part[part]['total'] /= count

    # Calculate PCK
    for part in config.PART_NAMES:
        if len(distances[part]) > 0:
            dists = np.array(distances[part])
            for threshold in thresholds:
                pck = (dists < threshold).mean() * 100
                pck_at_thresholds[part][f'pck@{threshold}'] = pck

    # Calculate aggregate metrics
    avg_mae = np.mean([mae_per_part[p]['total'] for p in config.PART_NAMES if mae_per_part[p]['count'] > 0])
    avg_mae_x = np.mean([mae_per_part[p]['x'] for p in config.PART_NAMES if mae_per_part[p]['count'] > 0])
    avg_mae_y = np.mean([mae_per_part[p]['y'] for p in config.PART_NAMES if mae_per_part[p]['count'] > 0])
    avg_pck_01 = np.mean([pck_at_thresholds[p]['pck@0.1'] for p in config.PART_NAMES])
    beak_mae_x = mae_per_part['beak']['x']
    beak_mae_y = mae_per_part['beak']['y']

    # Generate plots and visualizations
    plot_metrics_yolo(config, distances, mae_per_part, pck_at_thresholds)
    generate_confusion_matrix_yolo(config, distances, mae_per_part)
    visualize_predictions_yolo(config, model, dataset, num_samples=6)

    # Save results
    results = {
        'backbone': 'yolov8',
        'mae_per_part': {k: {kk: float(vv) for kk, vv in v.items() if kk != 'count'}
                         for k, v in mae_per_part.items()},
        'pck_at_thresholds': {k: {kk: float(vv) for kk, vv in v.items()}
                              for k, v in pck_at_thresholds.items()},
        'average_distance': {k: float(np.mean(v)) if len(v) > 0 else 0.0 for k, v in distances.items()},
        'avg_mae': float(avg_mae),
        'avg_mae_x': float(avg_mae_x),
        'avg_mae_y': float(avg_mae_y),
        'avg_pck_01': float(avg_pck_01),
        'beak_mae_x': float(beak_mae_x),
        'beak_mae_y': float(beak_mae_y)
    }

    # Save JSON report
    report_path = config.REPORTS_DIR / 'yolov8_report.json'
    with open(report_path, 'w') as f:
        json.dump(results, f, indent=4)

    # Print summary
    print(f"\n{'=' * 50}")
    print(f"Results for YOLOv8:")
    print(f"{'=' * 50}")

    print("\nMean Absolute Error (normalized):")
    for part in config.PART_NAMES:
        mae = mae_per_part[part]
        print(f"  {part}: x={mae['x']:.4f}, y={mae['y']:.4f}, total={mae['total']:.4f}")

    print("\nPCK (%):")
    for part, pck_dict in pck_at_thresholds.items():
        print(f"  {part}: " + ", ".join([f"{k}: {v:.1f}%" for k, v in pck_dict.items()]))

    print(f"\nResults saved to: {report_path}")

    return results


if __name__ == '__main__':
    config = Config()

    print("\n" + "=" * 60)
    print("CUB-200-2011 YOLOv8-Pose Evaluation")
    print("=" * 60)

    evaluate_yolo(config)