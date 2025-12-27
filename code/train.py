"""Training script with memory management."""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import sys
import gc

sys.path.append('..')
from code.config import Config
from dataset import CUBDataset
from models import PartLocalizer

import math


class WingLoss(nn.Module):
    def __init__(self, omega=10, epsilon=2):
        super().__init__()
        self.omega = omega
        self.epsilon = epsilon
        self.C = omega - omega * math.log(1 + omega / epsilon)

    def forward(self, pred, target):
        diff = torch.abs(pred - target)
        loss = torch.where(
            diff < self.omega,
            self.omega * torch.log(1 + diff / self.epsilon),
            diff - self.C
        )
        return loss.mean()


def get_loss_function(loss_name):
    if loss_name == 'smoothl1':
        return nn.SmoothL1Loss()
    elif loss_name == 'l1':
        return nn.L1Loss()
    elif loss_name == 'mse':
        return nn.MSELoss()
    elif loss_name == 'wing':
        return WingLoss()
    else:
        raise ValueError(f"Unknown loss function: {loss_name}")


def spatial_prior_loss(coords_pred, visibility_pred):
    vis_mask = visibility_pred > 0.5
    if vis_mask.sum() == 0:
        return torch.tensor(0.0, device=coords_pred.device)
    center = torch.tensor([0.5, 0.5], device=coords_pred.device)
    distances_from_center = torch.norm(coords_pred - center, dim=-1)
    center_penalty = torch.clamp(0.1 - distances_from_center, min=0.0)
    masked_penalty = center_penalty * vis_mask.float()
    return masked_penalty.mean()


def clear_memory():
    """Clear GPU and CPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def train_model(config, backbone, hyperparam_config=None):
    """Train model with specific hyperparameters"""

    if hyperparam_config:
        config.apply_hyperparameters(hyperparam_config)
        config_name = hyperparam_config['config_name']
    else:
        config_name = "default"
        config.LOSS_FUNCTION = 'smoothl1'
        config.LEARNING_RATE = 5e-5
        config.WEIGHT_DECAY = 1e-4
        config.HIDDEN_DIM = 512
        config.DROPOUT = 0.4
        config.VISIBILITY_LOSS_WEIGHT = 3.0
        config.AUG_BRIGHTNESS = 0.2
        config.AUG_CONTRAST = 0.2
        config.AUG_SATURATION = 0.2
        config.AUG_HUE = 0.05
        config.AUG_ROTATION = 5
        config.AUG_FLIP_PROB = 0.3

    print(f"\n{'=' * 70}")
    print(f"Training {backbone} | Config: {config_name}")
    print(f"{'=' * 70}")
    print(f"Loss: {config.LOSS_FUNCTION}, LR: {config.LEARNING_RATE:.0e}, "
          f"WD: {config.WEIGHT_DECAY:.0e}")
    print(f"Hidden: {config.HIDDEN_DIM}, Dropout: {config.DROPOUT}")
    print(f"Image size: {config.IMG_SIZE}, Batch size: {config.BATCH_SIZE}")
    print(f"Epochs: {config.NUM_EPOCHS}")
    print(f"Visibility Loss Weight: {config.VISIBILITY_LOSS_WEIGHT}x")
    print(f"Device: {config.DEVICE}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Clear memory before starting
    clear_memory()

    # Load datasets
    train_dataset = CUBDataset(config, mode='train')
    val_dataset = CUBDataset(config, mode='val')

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE,
                              shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE,
                            shuffle=False, num_workers=0, pin_memory=True)

    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # Create model
    model = PartLocalizer(backbone, config.NUM_PARTS, config).to(config.DEVICE)

    criterion_coords = get_loss_function(config.LOSS_FUNCTION)
    criterion_visibility = nn.BCELoss()

    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE,
                            weight_decay=config.WEIGHT_DECAY)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2, min_lr=1e-7
    )

    best_loss = float('inf')
    patience_counter = 0

    history = {
        'train_loss': [], 'val_loss': [],
        'train_coord_loss': [], 'train_vis_loss': [], 'train_spatial_loss': [],
        'val_coord_loss': [], 'val_vis_loss': [], 'val_spatial_loss': [],
        'train_vis_acc': [], 'val_vis_acc': [], 'learning_rates': []
    }

    for epoch in range(config.NUM_EPOCHS):
        # Training phase
        model.train()
        train_loss = 0
        train_coord_loss_visible = 0
        train_coord_loss_invisible = 0
        train_vis_loss = 0
        train_spatial_loss = 0
        train_vis_correct = 0
        train_vis_total = 0
        train_samples = 0

        pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{config.NUM_EPOCHS}')
        for batch_idx, batch_data in enumerate(pbar):
            if len(batch_data) == 4:
                images, targets, visibility_gt, _ = batch_data
            else:
                images, targets, _ = batch_data
                batch_size = images.size(0)
                visibility_gt = torch.ones(batch_size, config.NUM_PARTS, device=images.device)

            images = images.to(config.DEVICE)
            targets = targets.to(config.DEVICE)
            visibility_gt = visibility_gt.to(config.DEVICE)

            optimizer.zero_grad()

            coords_pred, visibility_pred = model(images)
            targets_reshaped = targets.view(-1, config.NUM_PARTS, 2)

            visible_mask = (visibility_gt == 1.0)
            coord_loss_visible = torch.tensor(0.0, device=config.DEVICE)
            if visible_mask.any():
                coords_pred_flat = coords_pred.view(-1, 2)[visible_mask.view(-1)]
                targets_flat = targets_reshaped.view(-1, 2)[visible_mask.view(-1)]
                loss_x = criterion_coords(coords_pred_flat[:, 0], targets_flat[:, 0]) * config.LOSS_WEIGHT_X
                loss_y = criterion_coords(coords_pred_flat[:, 1], targets_flat[:, 1]) * config.LOSS_WEIGHT_Y
                coord_loss_visible = loss_x + loss_y

            invisible_mask = (visibility_gt == 0.0)
            coord_loss_invisible = torch.tensor(0.0, device=config.DEVICE)
            if invisible_mask.any():
                coords_pred_flat = coords_pred.view(-1, 2)[invisible_mask.view(-1)]
                targets_flat = targets_reshaped.view(-1, 2)[invisible_mask.view(-1)]
                loss_x = criterion_coords(coords_pred_flat[:, 0], targets_flat[:, 0]) * config.LOSS_WEIGHT_X
                loss_y = criterion_coords(coords_pred_flat[:, 1], targets_flat[:, 1]) * config.LOSS_WEIGHT_Y
                coord_loss_invisible = (loss_x + loss_y) * config.LOSS_WEIGHT_INVISIBLE

            coord_loss = coord_loss_visible + coord_loss_invisible
            vis_loss = criterion_visibility(visibility_pred, visibility_gt) * config.VISIBILITY_LOSS_WEIGHT
            spatial_loss = spatial_prior_loss(coords_pred, visibility_pred) * config.SPATIAL_PRIOR_WEIGHT
            total_loss = coord_loss + vis_loss + spatial_loss

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            batch_size = images.size(0)
            train_loss += total_loss.item() * batch_size
            train_coord_loss_visible += coord_loss_visible.item() * batch_size
            train_coord_loss_invisible += coord_loss_invisible.item() * batch_size
            train_vis_loss += vis_loss.item() * batch_size
            train_spatial_loss += spatial_loss.item() * batch_size
            train_samples += batch_size

            vis_pred_binary = (visibility_pred > 0.5).float()
            train_vis_correct += (vis_pred_binary == visibility_gt).sum().item()
            train_vis_total += visibility_gt.numel()

            pbar.set_postfix({
                'loss': total_loss.item(),
                'vis_acc': f'{100.0 * train_vis_correct / train_vis_total:.1f}%'
            })

            # Clear intermediate tensors periodically
            if batch_idx % 50 == 0:
                clear_memory()

        avg_train_loss = train_loss / train_samples
        avg_train_coord_loss = (train_coord_loss_visible + train_coord_loss_invisible) / train_samples
        avg_train_vis_loss = train_vis_loss / train_samples
        avg_train_spatial_loss = train_spatial_loss / train_samples
        avg_train_vis_acc = 100.0 * train_vis_correct / train_vis_total

        # Clear memory before validation
        clear_memory()

        # Validation phase
        model.eval()
        val_loss = 0
        val_coord_loss_visible = 0
        val_coord_loss_invisible = 0
        val_vis_loss = 0
        val_spatial_loss = 0
        val_vis_correct = 0
        val_vis_total = 0
        val_samples = 0

        with torch.no_grad():
            for batch_data in val_loader:
                if len(batch_data) == 4:
                    images, targets, visibility_gt, _ = batch_data
                else:
                    images, targets, _ = batch_data
                    batch_size = images.size(0)
                    visibility_gt = torch.ones(batch_size, config.NUM_PARTS, device=images.device)

                images = images.to(config.DEVICE)
                targets = targets.to(config.DEVICE)
                visibility_gt = visibility_gt.to(config.DEVICE)

                coords_pred, visibility_pred = model(images)
                targets_reshaped = targets.view(-1, config.NUM_PARTS, 2)
                visible_mask = (visibility_gt == 1.0)

                coord_loss_visible = torch.tensor(0.0, device=config.DEVICE)
                if visible_mask.any():
                    coords_pred_flat = coords_pred.view(-1, 2)[visible_mask.view(-1)]
                    targets_flat = targets_reshaped.view(-1, 2)[visible_mask.view(-1)]
                    loss_x = criterion_coords(coords_pred_flat[:, 0], targets_flat[:, 0]) * config.LOSS_WEIGHT_X
                    loss_y = criterion_coords(coords_pred_flat[:, 1], targets_flat[:, 1]) * config.LOSS_WEIGHT_Y
                    coord_loss_visible = loss_x + loss_y

                invisible_mask = (visibility_gt == 0.0)
                coord_loss_invisible = torch.tensor(0.0, device=config.DEVICE)
                if invisible_mask.any():
                    coords_pred_flat = coords_pred.view(-1, 2)[invisible_mask.view(-1)]
                    targets_flat = targets_reshaped.view(-1, 2)[invisible_mask.view(-1)]
                    loss_x = criterion_coords(coords_pred_flat[:, 0], targets_flat[:, 0]) * config.LOSS_WEIGHT_X
                    loss_y = criterion_coords(coords_pred_flat[:, 1], targets_flat[:, 1]) * config.LOSS_WEIGHT_Y
                    coord_loss_invisible = (loss_x + loss_y) * config.LOSS_WEIGHT_INVISIBLE

                coord_loss = coord_loss_visible + coord_loss_invisible
                vis_loss = criterion_visibility(visibility_pred, visibility_gt) * config.VISIBILITY_LOSS_WEIGHT
                spatial_loss = spatial_prior_loss(coords_pred, visibility_pred) * config.SPATIAL_PRIOR_WEIGHT
                total_loss = coord_loss + vis_loss + spatial_loss

                batch_size = images.size(0)
                val_loss += total_loss.item() * batch_size
                val_coord_loss_visible += coord_loss_visible.item() * batch_size
                val_coord_loss_invisible += coord_loss_invisible.item() * batch_size
                val_vis_loss += vis_loss.item() * batch_size
                val_spatial_loss += spatial_loss.item() * batch_size
                val_samples += batch_size

                vis_pred_binary = (visibility_pred > 0.5).float()
                val_vis_correct += (vis_pred_binary == visibility_gt).sum().item()
                val_vis_total += visibility_gt.numel()

        avg_val_loss = val_loss / val_samples
        avg_val_coord_loss = (val_coord_loss_visible + val_coord_loss_invisible) / val_samples
        avg_val_vis_loss = val_vis_loss / val_samples
        avg_val_spatial_loss = val_spatial_loss / val_samples
        avg_val_vis_acc = 100.0 * val_vis_correct / val_vis_total

        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['train_coord_loss'].append(avg_train_coord_loss)
        history['train_vis_loss'].append(avg_train_vis_loss)
        history['train_spatial_loss'].append(avg_train_spatial_loss)
        history['val_coord_loss'].append(avg_val_coord_loss)
        history['val_vis_loss'].append(avg_val_vis_loss)
        history['val_spatial_loss'].append(avg_val_spatial_loss)
        history['train_vis_acc'].append(avg_train_vis_acc)
        history['val_vis_acc'].append(avg_val_vis_acc)
        history['learning_rates'].append(current_lr)

        print(f'\nEpoch {epoch + 1}/{config.NUM_EPOCHS}:')
        print(
            f'  Train Loss: {avg_train_loss:.6f} (Coord: {avg_train_coord_loss:.6f}, Vis: {avg_train_vis_loss:.6f}, Spatial: {avg_train_spatial_loss:.6f})')
        print(
            f'  Val Loss:   {avg_val_loss:.6f} (Coord: {avg_val_coord_loss:.6f}, Vis: {avg_val_vis_loss:.6f}, Spatial: {avg_val_spatial_loss:.6f})')
        print(f'  Train Vis Acc: {avg_train_vis_acc:.2f}% | Val Vis Acc: {avg_val_vis_acc:.2f}%')
        print(f'  LR: {current_lr:.2e}')

        if torch.cuda.is_available():
            print(
                f'  GPU Memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB / {torch.cuda.max_memory_allocated() / 1e9:.2f} GB peak')

        # Save best model
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            patience_counter = 0

            model_name = f'{backbone}_{config_name}_best.pth'
            save_path = config.SAVE_DIR / model_name

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
                'val_coord_loss': avg_val_coord_loss,
                'val_vis_loss': avg_val_vis_loss,
                'val_spatial_loss': avg_val_spatial_loss,
                'val_vis_acc': avg_val_vis_acc,
                'train_loss': avg_train_loss,
                'hyperparameters': hyperparam_config if hyperparam_config else {},
                'history': history
            }, save_path)
            print(f'  [SUCCESS] Saved best model (val_loss: {best_loss:.6f}, vis_acc: {avg_val_vis_acc:.2f}%)')
        else:
            patience_counter += 1
            if patience_counter >= config.EARLY_STOPPING_PATIENCE:
                print(f'\n  Early stopping at epoch {epoch + 1}')
                break

        # Clear memory at end of each epoch
        clear_memory()

    # Final cleanup
    del model, optimizer, scheduler
    clear_memory()

    print(f"\n{'=' * 70}")
    print(f"Training complete: {backbone} | {config_name}")
    print(f"Best val loss: {best_loss:.6f}")
    print(f"{'=' * 70}\n")

    return history, best_loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train part localization model')
    parser.add_argument('--model', type=str, default='resnet50',
                        choices=['resnet50', 'densenet'],
                        help='Backbone model to train')
    parser.add_argument('--config-index', type=int, default=None,
                        help='Hyperparameter config index to use')
    args = parser.parse_args()

    config = Config()

    if args.config_index is not None:
        configs = Config.get_reduced_grid()
        if args.config_index < len(configs):
            hyperparam_config = configs[args.config_index]
        else:
            print(f"Config index {args.config_index} out of range. Using default.")
            hyperparam_config = None
    else:
        hyperparam_config = None

    train_model(config, args.model, hyperparam_config)