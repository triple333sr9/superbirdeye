"""Train SuperAnimal-Bird (ResNet50) on CUB-200-2011. Two configs: medium/heavy aug."""

import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from code.config import Config
from dataset import CUBDataset
from models import PartLocalizer, WingLoss

MODEL_NAME = 'superanimal_bird'


def train(config, aug_level):
    config_name = f"wing_lr5e-05_h512_do0.2_{aug_level}_plateau"
    hp = {'loss_function': 'wing', 'learning_rate': 5e-5, 'weight_decay': 0.0,
          'hidden_dim': 512, 'dropout': 0.2, 'visibility_loss_weight': 2.0,
          'augmentation': aug_level, 'scheduler': 'plateau', 'config_name': config_name}
    config.apply_hyperparameters(hp)

    print(f"\n{'='*60}\nTraining {MODEL_NAME} | {aug_level}\n{'='*60}")

    train_loader = DataLoader(CUBDataset(config, 'train'), config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(CUBDataset(config, 'val'), config.BATCH_SIZE)

    model = PartLocalizer('resnet50', config.NUM_PARTS, config).to(config.DEVICE)
    criterion_coords, criterion_vis = WingLoss(), nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2)

    best_loss, patience = float('inf'), 0

    for epoch in range(config.NUM_EPOCHS):
        # Train
        model.train()
        train_loss, n = 0, 0
        for imgs, tgt, vis_gt, _ in tqdm(train_loader, desc=f'Epoch {epoch+1}'):
            imgs, tgt, vis_gt = imgs.to(config.DEVICE), tgt.to(config.DEVICE), vis_gt.to(config.DEVICE)
            optimizer.zero_grad()
            coords, vis = model(imgs)
            tgt = tgt.view(-1, config.NUM_PARTS, 2)
            mask = vis_gt == 1.0
            loss = criterion_vis(vis, vis_gt) * config.VISIBILITY_LOSS_WEIGHT
            if mask.any():
                loss += criterion_coords(coords.view(-1,2)[mask.view(-1)], tgt.view(-1,2)[mask.view(-1)])
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * imgs.size(0)
            n += imgs.size(0)

        # Validate
        model.eval()
        val_loss, vn = 0, 0
        with torch.no_grad():
            for imgs, tgt, vis_gt, _ in val_loader:
                imgs, tgt, vis_gt = imgs.to(config.DEVICE), tgt.to(config.DEVICE), vis_gt.to(config.DEVICE)
                coords, vis = model(imgs)
                tgt = tgt.view(-1, config.NUM_PARTS, 2)
                mask = vis_gt == 1.0
                loss = criterion_vis(vis, vis_gt) * config.VISIBILITY_LOSS_WEIGHT
                if mask.any():
                    loss += criterion_coords(coords.view(-1,2)[mask.view(-1)], tgt.view(-1,2)[mask.view(-1)])
                val_loss += loss.item() * imgs.size(0)
                vn += imgs.size(0)

        avg_val = val_loss / vn
        scheduler.step(avg_val)
        print(f'  Train: {train_loss/n:.4f} | Val: {avg_val:.4f} | LR: {optimizer.param_groups[0]["lr"]:.0e}')

        if avg_val < best_loss:
            best_loss, patience = avg_val, 0
            torch.save({'model_state_dict': model.state_dict(), 'epoch': epoch,
                        'val_loss': avg_val, 'hyperparameters': hp},
                       config.SAVE_DIR / f'{MODEL_NAME}_{config_name}_best.pth')
        else:
            patience += 1
            if patience >= config.EARLY_STOPPING_PATIENCE:
                print(f'Early stop @ epoch {epoch+1}')
                break

    return config_name


def evaluate(config, config_name):
    ckpt = torch.load(config.SAVE_DIR / f'{MODEL_NAME}_{config_name}_best.pth', map_location=config.DEVICE)
    hp = ckpt.get('hyperparameters', {})
    if hp: config.apply_hyperparameters(hp)

    model = PartLocalizer('resnet50', config.NUM_PARTS, config).to(config.DEVICE)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    loader = DataLoader(CUBDataset(config, 'val'), config.BATCH_SIZE)
    preds, gts, vis_p, vis_g = [], [], [], []

    with torch.no_grad():
        for imgs, tgt, vg, _ in loader:
            c, v = model(imgs.to(config.DEVICE))
            preds.append(c.cpu()); gts.append(tgt.view(-1,3,2))
            vis_p.append((v > 0.5).cpu()); vis_g.append(vg)

    preds, gts = torch.cat(preds), torch.cat(gts)
    vis_p, vis_g = torch.cat(vis_p), torch.cat(vis_g)

    mae, pck = {}, {}
    for j, name in enumerate(config.PART_NAMES):
        m = (vis_g[:,j] == 1) & (vis_p[:,j] == 1)
        if m.sum() > 0:
            p, g = preds[m,j], gts[m,j]
            e = torch.abs(p - g).mean(0)
            mae[name] = {'x': e[0].item(), 'y': e[1].item(), 'total': torch.norm(e).item()}
            d = torch.norm(p - g, dim=1)
            pck[name] = {f'pck@{t}': (d < t).float().mean().item() * 100 for t in [0.05, 0.1, 0.15, 0.2]}
        else:
            mae[name] = {'x': 0, 'y': 0, 'total': 0}
            pck[name] = {f'pck@{t}': 0 for t in [0.05, 0.1, 0.15, 0.2]}

    vis_acc = (vis_p.float() == vis_g).float().mean().item() * 100
    results = {
        'model': MODEL_NAME, 'config_name': config_name,
        'avg_mae': np.mean([m['total'] for m in mae.values()]),
        'avg_mae_x': np.mean([m['x'] for m in mae.values()]),
        'avg_mae_y': np.mean([m['y'] for m in mae.values()]),
        'avg_pck_01': np.mean([p['pck@0.1'] for p in pck.values()]),
        'beak_mae_x': mae['beak']['x'], 'beak_mae_y': mae['beak']['y'],
        'vis_accuracy': vis_acc, 'training_epochs': ckpt.get('epoch', 'N/A'),
        'loss_function': hp.get('loss_function', 'wing'),
        'learning_rate': hp.get('learning_rate', 5e-5),
        'weight_decay': hp.get('weight_decay', 0.0),
        'hidden_dim': hp.get('hidden_dim', 512),
        'dropout': hp.get('dropout', 0.2),
        'augmentation': hp.get('augmentation', 'unknown'),
    }

    with open(config.GRID_SEARCH_DIR / f'{MODEL_NAME}_{config_name}_report.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{MODEL_NAME} | {config_name}: MAE={results['avg_mae']:.4f}, PCK@0.1={results['avg_pck_01']:.1f}%")
    return results


if __name__ == '__main__':
    config = Config()
    config.SAVE_DIR.mkdir(exist_ok=True)
    config.GRID_SEARCH_DIR.mkdir(exist_ok=True)

    print(f"Device: {config.DEVICE}")

    for aug in ['medium', 'heavy']:
        cfg_name = train(config, aug)
        evaluate(config, cfg_name)

    print("\nDone. Run analyze_grid_search.py to include results.")