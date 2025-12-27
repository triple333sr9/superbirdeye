import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import DataLoader
from PIL import Image, ImageDraw, ImageFont
from sklearn.metrics import confusion_matrix
import seaborn as sns
import argparse
import json
import sys

sys.path.append('..')
from code.config import Config
from dataset import CUBDataset
from models import PartLocalizer


def evaluate_single_model(config, backbone):
    """Evaluate a single trained model"""
    print(f"\nEvaluating {backbone}...")

    # Check if model exists
    model_path = config.SAVE_DIR / f'{backbone}_best.pth'
    if not model_path.exists():
        print(f"Model {backbone} not found at {model_path}")
        return None

    # Load data
    dataset = CUBDataset(config, mode='val')
    loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    # Load model
    model = PartLocalizer(backbone, config.NUM_PARTS, config).to(config.DEVICE)
    checkpoint = torch.load(model_path, map_location=config.DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Collect predictions
    all_coords_pred = []
    all_coords_gt = []
    all_vis_pred = []
    all_vis_gt = []
    distances = {part: [] for part in config.PART_NAMES}

    with torch.no_grad():
        for images, targets, visibility_gt, _ in loader:
            images = images.to(config.DEVICE)

            coords_pred, vis_pred = model(images)

            all_coords_pred.append(coords_pred.cpu())
            all_coords_gt.append(targets)
            all_vis_pred.append(vis_pred.cpu())
            all_vis_gt.append(visibility_gt)

            # Calculate distances (only for correctly predicted visible keypoints)
            vis_pred_binary = (vis_pred > 0.5).cpu()
            targets_reshaped = targets.view(-1, config.NUM_PARTS, 2)
            visibility_gt_reshaped = visibility_gt.view(-1, config.NUM_PARTS)

            for i in range(coords_pred.shape[0]):
                for j in range(config.NUM_PARTS):
                    gt_vis = visibility_gt_reshaped[i, j].item() == 1.0
                    pred_vis = vis_pred_binary[i, j].item() == 1.0

                    if gt_vis and pred_vis:
                        gt_coord = targets_reshaped[i, j].numpy()
                        pred_coord = coords_pred[i, j].cpu().numpy()
                        dist = np.linalg.norm(pred_coord - gt_coord)
                        distances[config.PART_NAMES[j]].append(dist)

    # Concatenate all predictions
    all_coords_pred = torch.cat(all_coords_pred, dim=0)
    all_coords_gt = torch.cat(all_coords_gt, dim=0)
    all_vis_pred = torch.cat(all_vis_pred, dim=0)
    all_vis_gt = torch.cat(all_vis_gt, dim=0)

    # Visibility metrics
    vis_pred_binary = (all_vis_pred > 0.5).float()
    vis_accuracy = (vis_pred_binary == all_vis_gt).float().mean().item() * 100

    # Per-part visibility accuracy
    vis_acc_per_part = {}
    for j, part_name in enumerate(config.PART_NAMES):
        part_acc = (vis_pred_binary[:, j] == all_vis_gt[:, j]).float().mean().item() * 100
        vis_acc_per_part[part_name] = part_acc

    # Calculate metrics (only for correctly predicted visible keypoints)
    mae_per_part = {}
    pck_at_thresholds = {}
    thresholds = [0.05, 0.1, 0.15, 0.2]

    coords_gt_reshaped = all_coords_gt.view(-1, config.NUM_PARTS, 2)
    vis_gt_reshaped = all_vis_gt.view(-1, config.NUM_PARTS)

    for j, part_name in enumerate(config.PART_NAMES):
        gt_visible = (vis_gt_reshaped[:, j] == 1.0)
        pred_visible = (vis_pred_binary[:, j] == 1.0)
        both_visible = gt_visible & pred_visible

        if both_visible.sum() > 0:
            visible_preds = all_coords_pred[both_visible, j]
            visible_targets = coords_gt_reshaped[both_visible, j]

            # MAE
            mae = torch.abs(visible_preds - visible_targets).mean(dim=0)
            mae_per_part[part_name] = {
                'x': mae[0].item(),
                'y': mae[1].item(),
                'total': torch.norm(mae).item()
            }

            # PCK
            part_dists = torch.norm(visible_preds - visible_targets, dim=1)
            pck_at_thresholds[part_name] = {}
            for threshold in thresholds:
                pck = (part_dists < threshold).float().mean().item() * 100
                pck_at_thresholds[part_name][f'pck@{threshold}'] = pck
        else:
            mae_per_part[part_name] = {'x': 0, 'y': 0, 'total': 0}
            pck_at_thresholds[part_name] = {f'pck@{t}': 0 for t in thresholds}

    # Generate plots and visualizations
    plot_metrics(config, backbone, distances, mae_per_part, pck_at_thresholds, vis_accuracy, vis_acc_per_part)
    generate_confusion_matrix(config, backbone, all_coords_pred, all_coords_gt, all_vis_pred, all_vis_gt)
    visualize_predictions(config, backbone, model, dataset)

    # Save results
    results = {
        'backbone': backbone,
        'mae_per_part': mae_per_part,
        'pck_at_thresholds': pck_at_thresholds,
        'vis_accuracy': vis_accuracy,
        'vis_acc_per_part': vis_acc_per_part,
        'average_distance': {k: np.mean(v) if len(v) > 0 else 0 for k, v in distances.items()},
        'val_loss': checkpoint.get('val_loss', 'N/A')
    }

    report_path = config.REPORTS_DIR / f'{backbone}_report.json'
    with open(report_path, 'w') as f:
        json.dump(results, f, indent=4)

    # Print summary
    print(f"\n{'=' * 50}")
    print(f"Results for {backbone}:")
    print(f"{'=' * 50}")

    print("\nVisibility Prediction:")
    print(f"  Overall Accuracy: {vis_accuracy:.2f}%")
    for part, acc in vis_acc_per_part.items():
        print(f"  {part}: {acc:.2f}%")

    print("\nMean Absolute Error (for correctly detected visible keypoints):")
    for part, mae in mae_per_part.items():
        print(f"  {part}: x={mae['x']:.4f}, y={mae['y']:.4f}, total={mae['total']:.4f}")

    print("\nPCK (%):")
    for part, pck_dict in pck_at_thresholds.items():
        print(f"  {part}: " + ", ".join([f"{k}: {v:.1f}%" for k, v in pck_dict.items()]))

    print(f"\nResults saved to:")
    print(f"  - {report_path}")
    print(f"  - {config.REPORTS_DIR / f'{backbone}_metrics.png'}")
    print(f"  - {config.REPORTS_DIR / f'{backbone}_confusion.png'}")
    print(f"  - Visualizations in {config.VIS_DIR / backbone}/")

    return results


def plot_metrics(config, backbone, distances, mae_per_part, pck_at_thresholds, vis_accuracy, vis_acc_per_part):
    """Generate all metric plots"""

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Box plot of distances
    data = [distances[part] for part in config.PART_NAMES if len(distances[part]) > 0]
    labels = [part for part in config.PART_NAMES if len(distances[part]) > 0]

    if len(data) > 0:
        axes[0, 0].boxplot(data, labels=labels)
        axes[0, 0].set_ylabel('Normalized Distance Error')
        axes[0, 0].set_title(f'{backbone} - Error Distribution\n(Vis Acc: {vis_accuracy:.1f}%)')
        axes[0, 0].grid(True, alpha=0.3)
    else:
        axes[0, 0].text(0.5, 0.5, 'No data', ha='center', va='center')
        axes[0, 0].set_title(f'{backbone} - Error Distribution (No Data)')

    # MAE comparison
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
    axes[0, 1].set_title(f'{backbone} - MAE per Part')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3, axis='y')

    # Visibility accuracy per part
    axes[0, 2].bar(range(len(config.PART_NAMES)),
                   [vis_acc_per_part[p] for p in config.PART_NAMES],
                   color=['red', 'blue', 'green'], alpha=0.7)
    axes[0, 2].set_xticks(range(len(config.PART_NAMES)))
    axes[0, 2].set_xticklabels(config.PART_NAMES, rotation=45)
    axes[0, 2].set_ylabel('Accuracy (%)')
    axes[0, 2].set_title(f'{backbone} - Visibility Prediction Accuracy')
    axes[0, 2].axhline(y=vis_accuracy, color='red', linestyle='--', label=f'Overall: {vis_accuracy:.1f}%')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3, axis='y')

    # PCK curve
    thresholds = [0.05, 0.1, 0.15, 0.2]
    for part in config.PART_NAMES:
        pck_values = [pck_at_thresholds[part][f'pck@{t}'] for t in thresholds]
        axes[1, 0].plot(thresholds, pck_values, 'o-', label=part, linewidth=2)
    axes[1, 0].set_xlabel('Normalized Distance Threshold')
    axes[1, 0].set_ylabel('PCK (%)')
    axes[1, 0].set_title(f'{backbone} - PCK Curve')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Success rate vs threshold
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
        axes[1, 1].set_title(f'{backbone} - Success Rate Curve')
        axes[1, 1].grid(True, alpha=0.3)
    else:
        axes[1, 1].text(0.5, 0.5, 'No data', ha='center', va='center')
        axes[1, 1].set_title(f'{backbone} - Success Rate (No Data)')

    # Summary statistics table
    axes[1, 2].axis('tight')
    axes[1, 2].axis('off')

    summary_data = [
        ['Visibility Accuracy', f'{vis_accuracy:.2f}%'],
        ['Avg MAE (all parts)', f"{np.mean([mae_per_part[p]['total'] for p in config.PART_NAMES]):.4f}"],
        ['Avg PCK@0.1', f"{np.mean([pck_at_thresholds[p]['pck@0.1'] for p in config.PART_NAMES]):.1f}%"],
        ['Samples Evaluated', f"{len(all_dists) if all_dists else 0}"]
    ]

    table = axes[1, 2].table(cellText=summary_data,
                             cellLoc='left',
                             loc='center',
                             bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    for i in range(len(summary_data)):
        table[(i, 0)].set_facecolor('#ecf0f1')
        table[(i, 0)].set_text_props(weight='bold')

    axes[1, 2].set_title('Summary Statistics', fontsize=12, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(config.REPORTS_DIR / f'{backbone}_metrics.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Created metrics plot")


def generate_confusion_matrix(config, backbone, coords_pred, coords_gt, vis_pred, vis_gt, bins=5):
    """Generate confusion matrix"""

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Visibility confusion matrix
    vis_pred_binary = (vis_pred > 0.5).float().view(-1).numpy()
    vis_gt_flat = vis_gt.view(-1).numpy()

    vis_cm = confusion_matrix(vis_gt_flat, vis_pred_binary, labels=[0, 1])

    sns.heatmap(vis_cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Not Visible', 'Visible'],
                yticklabels=['Not Visible', 'Visible'],
                ax=axes[0])
    axes[0].set_xlabel('Predicted Visibility')
    axes[0].set_ylabel('True Visibility')
    axes[0].set_title(f'{backbone} - Visibility Confusion Matrix')

    # Coordinate error heatmap (for visible keypoints only)
    vis_pred_binary_reshaped = (vis_pred > 0.5).view(-1, config.NUM_PARTS)
    vis_gt_reshaped = vis_gt.view(-1, config.NUM_PARTS)
    coords_gt_reshaped = coords_gt.view(-1, config.NUM_PARTS, 2)

    error_data = []
    for j, part_name in enumerate(config.PART_NAMES):
        both_visible = (vis_gt_reshaped[:, j] == 1.0) & (vis_pred_binary_reshaped[:, j] == 1.0)

        if both_visible.sum() > 0:
            preds = coords_pred[both_visible, j].numpy()
            targets = coords_gt_reshaped[both_visible, j].numpy()

            mae_x = np.mean(np.abs(preds[:, 0] - targets[:, 0]))
            mae_y = np.mean(np.abs(preds[:, 1] - targets[:, 1]))
        else:
            mae_x, mae_y = 0, 0

        error_data.append([mae_x, mae_y])

    error_array = np.array(error_data).T

    sns.heatmap(error_array, annot=True, fmt='.4f', cmap='YlOrRd',
                xticklabels=config.PART_NAMES,
                yticklabels=['MAE X', 'MAE Y'],
                ax=axes[1])
    axes[1].set_title(f'{backbone} - Coordinate Error Heatmap')

    plt.tight_layout()
    plt.savefig(config.REPORTS_DIR / f'{backbone}_confusion.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Created confusion matrices")


def visualize_predictions(config, backbone, model, dataset, num_samples=6):
    """Generate visualization samples"""
    model.eval()
    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)

    model_vis_dir = config.VIS_DIR / backbone
    model_vis_dir.mkdir(exist_ok=True)

    colors = ['red', 'blue', 'green']

    for idx in indices:
        image_tensor, targets, visibility_gt, (orig_w, orig_h, img_id) = dataset[idx]
        image_tensor = image_tensor.unsqueeze(0).to(config.DEVICE)

        with torch.no_grad():
            coords_pred, vis_pred = model(image_tensor)
            coords_pred = coords_pred.cpu().squeeze()
            vis_pred = vis_pred.cpu().squeeze()

        row = dataset.image_data.iloc[idx]
        img_path = config.IMAGES_DIR / row['path']
        original_img = Image.open(img_path).convert('RGB')

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Ground truth
        gt_img = original_img.copy()
        gt_draw = ImageDraw.Draw(gt_img)

        # Predictions
        pred_img = original_img.copy()
        pred_draw = ImageDraw.Draw(pred_img)

        targets_reshaped = targets.view(-1, 2).numpy()
        vis_gt_np = visibility_gt.numpy()
        vis_pred_np = vis_pred.numpy()
        vis_pred_binary = (vis_pred_np > 0.5)

        try:
            font = ImageFont.truetype("arial.ttf", 14)
        except:
            font = ImageFont.load_default()

        # Draw keypoints
        for j, (color, part_name) in enumerate(zip(colors, config.PART_NAMES)):
            # Ground truth
            if vis_gt_np[j] == 1.0:
                gt_x = targets_reshaped[j, 0] * orig_w
                gt_y = targets_reshaped[j, 1] * orig_h
                radius = max(6, int(min(orig_w, orig_h) * 0.015))
                gt_draw.ellipse(
                    [gt_x - radius, gt_y - radius, gt_x + radius, gt_y + radius],
                    fill=color, outline='white', width=2
                )

            # Prediction
            if vis_pred_binary[j]:
                pred_x = np.clip(coords_pred[j, 0].item(), 0, 1) * orig_w
                pred_y = np.clip(coords_pred[j, 1].item(), 0, 1) * orig_h

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

                # Add confidence score
                conf_text = f"{vis_pred_np[j]:.2f}"
                pred_draw.text((pred_x + 15, pred_y - 15), conf_text,
                               fill=color, font=font, stroke_width=1, stroke_fill='black')

        ax1.imshow(gt_img)
        ax1.set_title(f'Ground Truth\n({int(vis_gt_np.sum())} visible keypoints)', fontsize=12)
        ax1.axis('off')

        ax2.imshow(pred_img)
        ax2.set_title(f'{backbone} Predictions\n({int(vis_pred_binary.sum())} predicted visible)', fontsize=12)
        ax2.axis('off')

        plt.tight_layout()
        plt.savefig(model_vis_dir / f'sample_{idx}.png', dpi=120, bbox_inches='tight')
        plt.close()

    print(f"✓ Generated {len(indices)} visualizations in {model_vis_dir}/")


def compare_all_models(config):
    """Compare all available trained models"""
    backbones = ['resnet50', 'densenet', 'efficientnet']
    available_models = []

    for backbone in backbones:
        model_path = config.SAVE_DIR / f'{backbone}_best.pth'
        if model_path.exists():
            available_models.append(backbone)
        else:
            print(f"Model {backbone} not found at {model_path}")

    if not available_models:
        print("No trained models found! Train models first using train.py")
        return

    print(f"\nEvaluating {len(available_models)} models: {', '.join(available_models)}")

    all_results = {}
    comparison_data = []

    for backbone in available_models:
        results = evaluate_single_model(config, backbone)
        if results:
            all_results[backbone] = results

            avg_mae = np.mean([results['mae_per_part'][p]['total'] for p in config.PART_NAMES])
            avg_pck_01 = np.mean([results['pck_at_thresholds'][p]['pck@0.1'] for p in config.PART_NAMES])

            comparison_data.append({
                'Model': backbone,
                'Avg MAE': avg_mae,
                'Avg PCK@0.1': avg_pck_01,
                'Vis Accuracy': results['vis_accuracy'],
                'Val Loss': results.get('val_loss', 'N/A')
            })

    if len(available_models) > 1:
        generate_comparison_plots(config, all_results, comparison_data)


def generate_comparison_plots(config, all_results, comparison_data):
    """Generate comparison plots across all models"""
    df = pd.DataFrame(comparison_data)

    csv_path = config.REPORTS_DIR / 'model_comparison.csv'
    df.to_csv(csv_path, index=False)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    models = df['Model'].tolist()

    # MAE comparison
    mae_values = df['Avg MAE'].tolist()
    bars = axes[0, 0].bar(models, mae_values, color=['skyblue', 'lightgreen', 'salmon'])
    axes[0, 0].set_ylabel('Average MAE (lower is better)')
    axes[0, 0].set_title('Model Comparison - MAE')
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    for bar, value in zip(bars, mae_values):
        axes[0, 0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                        f'{value:.4f}', ha='center', va='bottom')

    # PCK comparison
    pck_values = df['Avg PCK@0.1'].tolist()
    bars = axes[0, 1].bar(models, pck_values, color=['skyblue', 'lightgreen', 'salmon'])
    axes[0, 1].set_ylabel('Average PCK@0.1 (%)')
    axes[0, 1].set_title('Model Comparison - PCK@0.1')
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    for bar, value in zip(bars, pck_values):
        axes[0, 1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                        f'{value:.1f}%', ha='center', va='bottom')

    # Visibility accuracy comparison
    vis_values = df['Vis Accuracy'].tolist()
    bars = axes[1, 0].bar(models, vis_values, color=['skyblue', 'lightgreen', 'salmon'])
    axes[1, 0].set_ylabel('Visibility Accuracy (%)')
    axes[1, 0].set_title('Model Comparison - Visibility Prediction')
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    for bar, value in zip(bars, vis_values):
        axes[1, 0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                        f'{value:.1f}%', ha='center', va='bottom')

    # Radar chart
    angles = np.linspace(0, 2 * np.pi, len(config.PART_NAMES), endpoint=False).tolist()
    angles += angles[:1]

    ax_radar = plt.subplot(224, projection='polar')
    ax_radar.set_title('Per-Part MAE Comparison', pad=20)

    for model_name, results in all_results.items():
        mae_values = [results['mae_per_part'][p]['total'] for p in config.PART_NAMES]
        mae_values += mae_values[:1]
        ax_radar.plot(angles, mae_values, 'o-', linewidth=2, label=model_name)
        ax_radar.fill(angles, mae_values, alpha=0.1)

    ax_radar.set_xticks(angles[:-1])
    ax_radar.set_xticklabels(config.PART_NAMES)
    ax_radar.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    ax_radar.grid(True)

    plt.tight_layout()
    plt.savefig(config.REPORTS_DIR / 'model_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\n{'=' * 60}")
    print("MODEL COMPARISON SUMMARY")
    print(f"{'=' * 60}")
    print(f"\n{df.to_string(index=False)}")

    best_mae_idx = df['Avg MAE'].idxmin()
    best_pck_idx = df['Avg PCK@0.1'].idxmax()
    best_vis_idx = df['Vis Accuracy'].idxmax()

    print(f"\n{'=' * 60}")
    print("RECOMMENDATIONS:")
    print(f"{'=' * 60}")
    print(f"• Best for accuracy (lowest MAE): {df.loc[best_mae_idx, 'Model']}")
    print(f"• Best for localization (highest PCK): {df.loc[best_pck_idx, 'Model']}")
    print(f"• Best for visibility detection: {df.loc[best_vis_idx, 'Model']}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate trained models on CUB dataset')
    parser.add_argument('--model', type=str, default='all',
                        choices=['resnet50', 'densenet', 'efficientnet', 'all'],
                        help='Model to evaluate or "all" for all available models')
    parser.add_argument('--samples', type=int, default=6,
                        help='Number of visualization samples to generate')
    args = parser.parse_args()

    config = Config()

    print(f"{'=' * 60}")
    print("CUB-200-2011 Part Localization Evaluation")
    print(f"{'=' * 60}")
    print(f"Device: {config.DEVICE}")
    print(f"Results will be saved to: {config.RESULTS_DIR}")

    if args.model == 'all':
        compare_all_models(config)
    else:
        evaluate_single_model(config, args.model)

    print(f"\n{'=' * 60}")
    print("Evaluation Complete!")
    print(f"{'=' * 60}")