"""Dataset with comprehensive keypoint-aware augmentation."""
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageFilter
import pandas as pd
import torchvision.transforms.functional as TF
import numpy as np
import random


class CUBDataset(Dataset):
    def __init__(self, config, mode='train'):
        self.config = config
        self.mode = mode
        self.load_data()
        self.apply_split()

    def load_data(self):
        images_df = pd.read_csv(self.config.IMAGES_LIST, sep=' ', names=['image_id', 'path'])
        parts_df = pd.read_csv(self.config.ANNOTATIONS_FILE, sep=' ',
                               names=['image_id', 'part_id', 'x', 'y', 'visible'])
        parts_df = parts_df[parts_df['part_id'].isin([7, 11, 2])]

        parts_wide = parts_df.pivot(index='image_id', columns='part_id',
                                    values=['x', 'y', 'visible']).reset_index()
        parts_wide.columns = ['image_id'] + [f'{c[0]}{c[1]}' for c in parts_wide.columns[1:]]
        self.image_data = pd.merge(images_df, parts_wide, on='image_id', how='left')

    def apply_split(self):
        np.random.seed(42)
        ids = self.image_data['image_id'].unique()
        n_train = int(0.7 * len(ids))
        shuffled = np.random.permutation(ids)
        split_ids = set(shuffled[:n_train] if self.mode == 'train' else shuffled[n_train:])
        self.image_data = self.image_data[self.image_data['image_id'].isin(split_ids)].reset_index(drop=True)

    def __len__(self):
        return len(self.image_data)

    def __getitem__(self, idx):
        row = self.image_data.iloc[idx]
        img = Image.open(self.config.IMAGES_DIR / row['path']).convert('RGB')
        orig_w, orig_h = img.size

        # Extract keypoints and visibility
        keypoints, visibility = [], []
        for pid in [7, 11, 2]:
            x, y, v = row.get(f'x{pid}', 0), row.get(f'y{pid}', 0), row.get(f'visible{pid}', 0)
            is_vis = pd.notna(v) and v == 1 and pd.notna(x) and x > 0 and y > 0
            if is_vis:
                keypoints.append([x / orig_w, y / orig_h])
                visibility.append(1.0)
            else:
                keypoints.append([0.5, 0.5])
                visibility.append(0.0)

        keypoints = np.array(keypoints, dtype=np.float32)
        visibility = np.array(visibility, dtype=np.float32)

        # Apply augmentation
        if self.mode == 'train':
            img, keypoints, visibility = self._augment(img, keypoints, visibility)

        # Resize and normalize
        img = img.resize((self.config.IMG_SIZE, self.config.IMG_SIZE))
        img = TF.to_tensor(img)
        img = TF.normalize(img, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        return (img, torch.tensor(keypoints.flatten()), torch.tensor(visibility),
                (orig_w, orig_h, row['image_id']))

    def _augment(self, img, kpts, vis):
        """Comprehensive augmentation with keypoint updates."""
        w, h = img.size

        # 1. Horizontal flip
        if random.random() < getattr(self.config, 'AUG_FLIP_PROB', 0.3):
            img = TF.hflip(img)
            kpts[:, 0] = 1.0 - kpts[:, 0]
            kpts[[0, 1]] = kpts[[1, 0]]  # Swap left/right eye
            vis[[0, 1]] = vis[[1, 0]]

        # 2. Rotation
        max_rot = getattr(self.config, 'AUG_ROTATION', 5)
        if max_rot > 0:
            angle = random.uniform(-max_rot, max_rot)
            if abs(angle) > 0.5:
                img = TF.rotate(img, angle, fill=0)
                kpts, vis = self._rotate_keypoints(kpts, vis, angle)

        # 3. Scale
        scale_range = getattr(self.config, 'AUG_SCALE_RANGE', (1.0, 1.0))
        if scale_range != (1.0, 1.0):
            scale = random.uniform(*scale_range)
            new_w, new_h = int(w * scale), int(h * scale)
            img = img.resize((new_w, new_h))

            # Crop or pad to original size
            if scale > 1:
                left = (new_w - w) // 2
                top = (new_h - h) // 2
                img = img.crop((left, top, left + w, top + h))
                # Adjust keypoints
                kpts[:, 0] = (kpts[:, 0] * scale - left / w)
                kpts[:, 1] = (kpts[:, 1] * scale - top / h)
            else:
                # Pad
                pad_left = (w - new_w) // 2
                pad_top = (h - new_h) // 2
                padded = Image.new('RGB', (w, h), (0, 0, 0))
                padded.paste(img, (pad_left, pad_top))
                img = padded
                # Adjust keypoints
                kpts[:, 0] = kpts[:, 0] * scale + pad_left / w
                kpts[:, 1] = kpts[:, 1] * scale + pad_top / h

            kpts, vis = self._validate_keypoints(kpts, vis)

        # 4. Translation
        max_trans = getattr(self.config, 'AUG_TRANSLATE', 0.0)
        if max_trans > 0:
            tx = random.uniform(-max_trans, max_trans)
            ty = random.uniform(-max_trans, max_trans)
            img = TF.affine(img, angle=0, translate=(int(tx * w), int(ty * h)),
                            scale=1.0, shear=0, fill=0)
            kpts[:, 0] += tx
            kpts[:, 1] += ty
            kpts, vis = self._validate_keypoints(kpts, vis)

        # 5. Color jitter (doesn't affect keypoints)
        brightness = getattr(self.config, 'AUG_BRIGHTNESS', 0.1)
        contrast = getattr(self.config, 'AUG_CONTRAST', 0.1)
        saturation = getattr(self.config, 'AUG_SATURATION', 0.1)
        hue = getattr(self.config, 'AUG_HUE', 0.02)

        if brightness > 0:
            img = TF.adjust_brightness(img, 1 + random.uniform(-brightness, brightness))
        if contrast > 0:
            img = TF.adjust_contrast(img, 1 + random.uniform(-contrast, contrast))
        if saturation > 0:
            img = TF.adjust_saturation(img, 1 + random.uniform(-saturation, saturation))
        if hue > 0:
            img = TF.adjust_hue(img, random.uniform(-hue, hue))

        # 6. Gaussian blur
        blur_prob = getattr(self.config, 'AUG_GAUSSIAN_BLUR', 0.0)
        if random.random() < blur_prob:
            radius = random.uniform(0.5, 2.0)
            img = img.filter(ImageFilter.GaussianBlur(radius=radius))

        # 7. Gaussian noise
        noise_std = getattr(self.config, 'AUG_GAUSSIAN_NOISE', 0.0)
        if noise_std > 0:
            img_arr = np.array(img).astype(np.float32) / 255.0
            noise = np.random.normal(0, noise_std, img_arr.shape)
            img_arr = np.clip(img_arr + noise, 0, 1)
            img = Image.fromarray((img_arr * 255).astype(np.uint8))

        # 8. Cutout (random erasing)
        cutout_prob = getattr(self.config, 'AUG_CUTOUT_PROB', 0.0)
        cutout_size = getattr(self.config, 'AUG_CUTOUT_SIZE', 0.1)
        if random.random() < cutout_prob:
            img, kpts, vis = self._apply_cutout(img, kpts, vis, cutout_size)

        return img, kpts, vis

    def _rotate_keypoints(self, kpts, vis, angle):
        """Rotate keypoints around image center."""
        cx, cy = 0.5, 0.5
        rad = np.radians(-angle)
        cos_a, sin_a = np.cos(rad), np.sin(rad)

        for i in range(len(kpts)):
            if vis[i] > 0:
                x, y = kpts[i] - [cx, cy]
                kpts[i] = [cos_a * x - sin_a * y + cx, sin_a * x + cos_a * y + cy]

        return self._validate_keypoints(kpts, vis)

    def _validate_keypoints(self, kpts, vis):
        """Invalidate keypoints that moved out of bounds."""
        for i in range(len(kpts)):
            if vis[i] > 0:
                if not (0.02 <= kpts[i, 0] <= 0.98 and 0.02 <= kpts[i, 1] <= 0.98):
                    kpts[i] = [0.5, 0.5]
                    vis[i] = 0.0
        return kpts, vis

    def _apply_cutout(self, img, kpts, vis, max_size):
        """Apply cutout, invalidating covered keypoints."""
        w, h = img.size
        size = random.uniform(0.05, max_size)
        cx = random.uniform(size / 2, 1 - size / 2)
        cy = random.uniform(size / 2, 1 - size / 2)

        x1, y1 = int((cx - size / 2) * w), int((cy - size / 2) * h)
        x2, y2 = int((cx + size / 2) * w), int((cy + size / 2) * h)

        # Black out region
        img_arr = np.array(img)
        img_arr[y1:y2, x1:x2] = 0
        img = Image.fromarray(img_arr)

        # Invalidate keypoints in cutout region
        for i in range(len(kpts)):
            if vis[i] > 0:
                kx, ky = kpts[i]
                if (cx - size / 2) <= kx <= (cx + size / 2) and \
                        (cy - size / 2) <= ky <= (cy + size / 2):
                    kpts[i] = [0.5, 0.5]
                    vis[i] = 0.0

        return img, kpts, vis