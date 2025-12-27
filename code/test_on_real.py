"""Run images through all trained models and output comparison visualizations."""

import torch
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
from torchvision import transforms
import csv
from code.config import Config
from models import PartLocalizer

config = Config()
COLORS = [(255, 0, 0), (0, 0, 255), (0, 255, 0)]
PART_NAMES = config.PART_NAMES
transform = transforms.Compose([
    transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def draw(image, kps):
    img = image.copy()
    d = ImageDraw.Draw(img)
    w, h = image.size
    r = max(12, int(min(w, h) * 0.03))
    font_size = int(h * 0.05)
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except:
        font = ImageFont.load_default()

    y_offset = 10
    for i, (x, y, conf) in enumerate(kps):
        text = f"{PART_NAMES[i]}: ({x:.0f},{y:.0f}) conf={conf:.2f}"
        d.text((10, y_offset), text, fill=COLORS[i], font=font)
        y_offset += font_size + 5
        if conf > 0.7:
            d.ellipse([x-r, y-r, x+r, y+r], fill=COLORS[i], outline='white', width=2)
    return img


def main():
    Path(config.COMPARISON_INFERENCE_OUTPUT).mkdir(parents=True, exist_ok=True)
    models = []

    for pth in config.SAVE_DIR.glob('*best.pth'):
        backbone = 'resnet50' if 'resnet' in pth.stem.lower() else 'densenet'
        ckpt = torch.load(pth, map_location=config.DEVICE, weights_only=False)
        hp = ckpt.get('hyperparameters', {})
        cfg = Config()
        cfg.HIDDEN_DIM, cfg.DROPOUT = hp.get('hidden_dim', 512), hp.get('dropout', 0.4)
        model = PartLocalizer(backbone, config.NUM_PARTS, cfg).to(config.DEVICE)
        model.load_state_dict(ckpt['model_state_dict'])
        model.eval()
        models.append((model, pth.stem, 'pth'))

    for pt in config.SAVE_DIR.glob('*yolo*.pt'):
        from ultralytics import YOLO
        models.append((YOLO(str(pt)), pt.stem, 'yolo'))

    csv_rows = []

    for img_path in Path(config.INFERENCE_IMAGES_CROPPED).glob('*'):
        print(f"PROCESSING {img_path}")
        if img_path.suffix.lower() not in {'.jpg', '.jpeg', '.png', '.bmp'} or ".__" in img_path.name:
            continue
        original = Image.open(img_path).convert('RGB')
        w, h = original.size
        preds = []

        for model, name, mtype in models:
            if mtype == 'pth':
                with torch.no_grad():
                    coords, vis = model(transform(original).unsqueeze(0).to(config.DEVICE))
                c, v = coords.cpu().squeeze().numpy(), vis.cpu().squeeze().numpy()
                kps = [(c[i, 0] * w, c[i, 1] * h, v[i]) for i in range(config.NUM_PARTS)]
            else:
                res = model.predict(str(img_path), save=False, conf=0.25, verbose=False)
                kps = []
                if res is None:
                    continue
                if res and res[0].keypoints is not None:
                    pts = res[0].keypoints.xy[0].cpu().numpy()
                    confs = res[0].keypoints.conf[0].cpu().numpy() if res[0].keypoints.conf is not None else [1.0]*len(pts)
                    kps = [(pts[i][0], pts[i][1], confs[i]) for i in range(min(len(pts), config.NUM_PARTS))]
                while len(kps) < config.NUM_PARTS:
                    kps.append((0, 0, 0))
            preds.append((kps, name))

            row = {'image': img_path.name, 'model': name}
            for i, (x, y, conf) in enumerate(kps):
                row[f'{PART_NAMES[i]}_x'] = round(x, 1)
                row[f'{PART_NAMES[i]}_y'] = round(y, 1)
                row[f'{PART_NAMES[i]}_conf'] = round(conf, 3)
            csv_rows.append(row)

        fig, axes = plt.subplots(1, len(preds) + 1, figsize=(5 * (len(preds) + 1), 5))
        axes[0].imshow(original); axes[0].set_title('Original'); axes[0].axis('off')
        for i, (kps, name) in enumerate(preds):
            axes[i + 1].imshow(draw(original, kps)); axes[i + 1].set_title(name[:25], fontsize=10); axes[i + 1].axis('off')
        plt.tight_layout()
        plt.savefig(Path(config.COMPARISON_INFERENCE_OUTPUT) / f'{img_path.stem}_comparison.png', dpi=150)
        plt.close()

    with open(Path(config.COMPARISON_INFERENCE_OUTPUT) / 'results.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"Results saved to: {config.COMPARISON_INFERENCE_OUTPUT}")


if __name__ == '__main__':
    main()