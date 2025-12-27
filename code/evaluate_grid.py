"""Concise evaluation that stores hyperparameters in JSON reports."""
import torch
import numpy as np
import json
import argparse
from torch.utils.data import DataLoader
from code.config import Config
from dataset import CUBDataset
from models import PartLocalizer


def evaluate_model(config, backbone, config_name):
    model_path = config.SAVE_DIR / f'{backbone}_{config_name}_best.pth'
    if not model_path.exists():
        print(f"Model not found: {model_path}")
        return None

    checkpoint = torch.load(model_path, map_location=config.DEVICE, weights_only=False)
    hp = checkpoint.get('hyperparameters', {})
    if hp:
        config.apply_hyperparameters(hp)
    else:
        config.HIDDEN_DIM, config.DROPOUT = 512, 0.4

    dataset = CUBDataset(config, mode='val')
    loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    model = PartLocalizer(backbone, config.NUM_PARTS, config).to(config.DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Collect predictions
    all_preds, all_gt, all_vis_pred, all_vis_gt = [], [], [], []

    with torch.no_grad():
        for images, targets, vis_gt, _ in loader:
            coords, vis = model(images.to(config.DEVICE))
            all_preds.append(coords.cpu())
            all_gt.append(targets.view(-1, config.NUM_PARTS, 2))
            all_vis_pred.append((vis > 0.5).cpu())
            all_vis_gt.append(vis_gt)

    preds = torch.cat(all_preds)
    gt = torch.cat(all_gt)
    vis_pred = torch.cat(all_vis_pred)
    vis_gt = torch.cat(all_vis_gt)

    # Calculate metrics only for correctly detected visible keypoints
    mae_per_part, pck_per_part = {}, {}
    for j, name in enumerate(config.PART_NAMES):
        mask = (vis_gt[:, j] == 1) & (vis_pred[:, j] == 1)
        if mask.sum() > 0:
            p, g = preds[mask, j], gt[mask, j]
            mae = torch.abs(p - g).mean(dim=0)
            mae_per_part[name] = {'x': mae[0].item(), 'y': mae[1].item(),
                                  'total': torch.norm(mae).item()}
            dists = torch.norm(p - g, dim=1)
            pck_per_part[name] = {f'pck@{t}': (dists < t).float().mean().item() * 100
                                  for t in [0.05, 0.1, 0.15, 0.2]}
        else:
            mae_per_part[name] = {'x': 0, 'y': 0, 'total': 0}
            pck_per_part[name] = {f'pck@{t}': 0 for t in [0.05, 0.1, 0.15, 0.2]}

    vis_acc = (vis_pred.float() == vis_gt).float().mean().item() * 100

    # Build results WITH hyperparameters
    results = {
        'model': backbone,
        'config_name': config_name,
        'mae_per_part': mae_per_part,
        'pck_at_thresholds': pck_per_part,
        'vis_accuracy': vis_acc,
        'avg_mae': np.mean([m['total'] for m in mae_per_part.values()]),
        'avg_mae_x': np.mean([m['x'] for m in mae_per_part.values()]),
        'avg_mae_y': np.mean([m['y'] for m in mae_per_part.values()]),
        'avg_pck_01': np.mean([p['pck@0.1'] for p in pck_per_part.values()]),
        'beak_mae_x': mae_per_part['beak']['x'],
        'beak_mae_y': mae_per_part['beak']['y'],
        'training_epochs': checkpoint.get('epoch', 'N/A'),
        # FIX: Include hyperparameters in JSON
        'loss_function': hp.get('loss_function', 'unknown'),
        'learning_rate': hp.get('learning_rate', 0),
        'weight_decay': hp.get('weight_decay', 0),
        'hidden_dim': hp.get('hidden_dim', 512),
        'dropout': hp.get('dropout', 0.4),
        'augmentation': hp.get('augmentation', 'light'),
    }

    report_path = config.GRID_SEARCH_DIR / f'{backbone}_{config_name}_report.json'
    with open(report_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{backbone} | {config_name}")
    print(f"  Vis Acc: {vis_acc:.1f}%, Avg MAE: {results['avg_mae']:.4f}, PCK@0.1: {results['avg_pck_01']:.1f}%")
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--config-name', required=True)
    args = parser.parse_args()
    evaluate_model(Config(), args.model, args.config_name)