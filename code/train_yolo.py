"""Train YOLOv8-pose with optimal hyperparameters."""
import torch
import yaml
from pathlib import Path
import shutil
from tqdm import tqdm
from code.config import Config
from dataset import CUBDataset
from ultralytics import YOLO


def prepare_yolo_dataset(config):
    """Convert CUB dataset to YOLO format"""
    print("Preparing YOLO dataset...")
    yolo_dir = Path("../yolo_data")
    yolo_dir.mkdir(exist_ok=True)

    for split in ['train', 'val']:
        (yolo_dir / split / 'images').mkdir(parents=True, exist_ok=True)
        (yolo_dir / split / 'labels').mkdir(parents=True, exist_ok=True)

    train_dataset = CUBDataset(config, mode='train')
    val_dataset = CUBDataset(config, mode='val')

    for dataset, split in [(train_dataset, 'train'), (val_dataset, 'val')]:
        print(f"Processing {split} split...")
        for idx in tqdm(range(len(dataset))):
            batch_data = dataset[idx]
            if len(batch_data) == 4:
                _, targets, visibility, (orig_w, orig_h, img_id) = batch_data
            else:
                _, targets, (orig_w, orig_h, img_id) = batch_data
                visibility = torch.ones(config.NUM_PARTS)

            row = dataset.image_data.iloc[idx]
            src_img = config.IMAGES_DIR / row['path']
            dst_img = yolo_dir / split / 'images' / f'{img_id}.jpg'
            shutil.copy(src_img, dst_img)

            label_path = yolo_dir / split / 'labels' / f'{img_id}.txt'
            keypoints = targets.view(-1, 2).numpy()
            vis_np = visibility.numpy()

            kpt_str = ""
            for j in range(len(keypoints)):
                x, y = keypoints[j]
                vis = vis_np[j]
                if vis == 1.0:
                    kpt_str += f"{x:.6f} {y:.6f} 2 "
                else:
                    kpt_str += "0 0 0 "

            with open(label_path, 'w') as f:
                f.write(f"0 0.5 0.5 0.8 0.8 {kpt_str.strip()}\n")

    data_yaml = {
        'path': str(yolo_dir.absolute()),
        'train': 'train/images',
        'val': 'val/images',
        'nc': 1,
        'names': ['bird'],
        'kpt_shape': [3, 3]
    }

    yaml_path = yolo_dir / 'data.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(data_yaml, f)

    print(f"Dataset prepared at: {yolo_dir}")
    return str(yaml_path)


def train_yolo(config):
    """Train YOLOv8-pose with optimal hyperparameters"""
    print(f"\n{'='*60}")
    print("Training YOLOv8-pose")
    print(f"{'='*60}")

    data_yaml = prepare_yolo_dataset(config)
    model = YOLO('../yolov8n-pose.pt')

    print(f"Image size: {config.IMG_SIZE}")
    print(f"Epochs: {config.NUM_EPOCHS}")
    print(f"Device: {config.DEVICE}")

    results = model.train(
        data=data_yaml,
        epochs=config.NUM_EPOCHS,
        imgsz=config.IMG_SIZE,
        batch=config.BATCH_SIZE,
        device=0 if torch.cuda.is_available() else 'cpu',
        patience=20,
        lr0=0.01,
        lrf=0.01,
        mosaic=0.5,
        fliplr=0.5,
        flipud=0.0,
        degrees=10,
        save=True,
        project='saved_models',
        name='yolov8',
        exist_ok=True,
        verbose=True
    )

    best_model_path = Path('../saved_models/yolov8/weights/best.pt')
    if best_model_path.exists():
        shutil.copy(best_model_path, config.SAVE_DIR / 'yolov8_best.pt')
        print(f"\nModel saved to: {config.SAVE_DIR / 'yolov8_best.pt'}")

    return results


if __name__ == '__main__':
    config = Config()
    train_yolo(config)