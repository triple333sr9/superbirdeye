"""
Two-Stage Bird Keypoint Detection Pipeline
Stage 1: YOLOv11n (general detection) - detect and crop birds
Stage 2: CUB Pose model - detect keypoints on cropped birds
"""

from pathlib import Path
from PIL import Image, ImageFont
from ultralytics import YOLO
from tqdm import tqdm

from code.config import Config


def get_font(size=16):
    """Get a font, falling back to default if system font not available"""
    try:
        return ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", size)
    except:
        try:
            return ImageFont.truetype("arial.ttf", size)
        except:
            return ImageFont.load_default()


def two_stage_detection(
    source_dir: str,
    output_dir: str = None,
    detection_model_path: str = "yolo11n.pt",
    det_conf_thres: float = 0.5,
    kpt_conf_thres: float = 0.5,
    crop_padding: float = 0.1,
):
    """
    Two-stage bird keypoint detection:
    1. Use YOLOv11n to detect birds and crop them
    2. Use CUB pose model to detect keypoints on cropped birds
    
    Args:
        source_dir: Directory containing images to process
        output_dir: Output directory (default: source_dir/test_result_twostage)
        detection_model_path: Path to YOLOv11n detection model
        pose_model_path: Path to trained CUB pose model
        det_conf_thres: Detection confidence threshold
        kpt_conf_thres: Keypoint confidence threshold
        crop_padding: Padding ratio around detected bird (0.1 = 10% extra on each side)
    """
    source_path = Path(source_dir)
    
    # Set up output directory
    if output_dir is None:
        output_path = source_path / "test_result_twostage"
    else:
        output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load models
    print(f"Loading detection model from {detection_model_path}...")
    det_model = YOLO(detection_model_path)

    # Find images
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
    images = []
    for ext in extensions:
        images.extend(source_path.glob(ext))
    
    # Filter out hidden files (macOS ._ files)
    images = [img for img in images if not img.name.startswith('._')]
    
    print(f"Found {len(images)} images in {source_dir}")
    print(f"Output will be saved to {output_path}")
    
    # Keypoint config
    keypoint_config = {
        0: {'name': 'Beak', 'color': 'green'},
        1: {'name': 'Left Eye', 'color': 'red'},
        2: {'name': 'Right Eye', 'color': 'blue'},
        3: {'name': 'Left Wing', 'color': 'yellow'},
        4: {'name': 'Right Wing', 'color': 'cyan'},
    }
    
    font = get_font(28)
    font_small = get_font(22)
    
    # Header height for legend area
    HEADER_HEIGHT = 260
    
    # Threshold for drawing keypoints (only draw if conf >= this)
    DRAW_THRESHOLD = 0.9
    
    # COCO class ID for bird is 14
    BIRD_CLASS_ID = 14
    
    stats = {
        'total_images': 0,
        'images_with_birds': 0,
        'total_birds_detected': 0,
        'birds_with_keypoints': 0,
        'keypoints_drawn': {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    }
    
    for img_file in tqdm(images, desc="Processing images"):
        stats['total_images'] += 1
        
        # Stage 1: Detect birds using YOLOv11n
        try:
            det_results = det_model.predict(str(img_file), conf=det_conf_thres, verbose=False)
        except Exception as e:
            print(f"Error in detection for {img_file.name}: {e}")
            continue
        
        if not det_results or len(det_results) == 0:
            continue
        
        det_result = det_results[0]
        
        # Filter for birds only (class 14 in COCO)
        if det_result.boxes is None or len(det_result.boxes) == 0:
            continue
        
        boxes = det_result.boxes
        bird_indices = []
        for i, cls in enumerate(boxes.cls):
            if int(cls) == BIRD_CLASS_ID:
                bird_indices.append(i)
        
        if len(bird_indices) == 0:
            continue
        
        stats['images_with_birds'] += 1
        
        # Load original image
        original_img = Image.open(img_file).convert('RGB')
        img_w, img_h = original_img.size
        
        # Process each detected bird
        for bird_idx, box_idx in enumerate(bird_indices):
            stats['total_birds_detected'] += 1
            
            # Get bounding box
            box = boxes.xyxy[box_idx].cpu().numpy()
            x1, y1, x2, y2 = box
            
            # Add padding
            box_w = x2 - x1
            box_h = y2 - y1
            pad_w = box_w * crop_padding
            pad_h = box_h * crop_padding
            
            crop_x1 = max(0, int(x1 - pad_w))
            crop_y1 = max(0, int(y1 - pad_h))
            crop_x2 = min(img_w, int(x2 + pad_w))
            crop_y2 = min(img_h, int(y2 + pad_h))
            
            # Crop the bird
            cropped_img = original_img.crop((crop_x1, crop_y1, crop_x2, crop_y2))
            crop_w, crop_h = cropped_img.size

            # Create canvas with gray header for legend
            canvas_h = crop_h
            canvas = Image.new('RGB', (crop_w, canvas_h), color=(80, 80, 80))  # Gray header
            canvas.paste(cropped_img, (0, 0))  # Paste bird image below header

            # Save canvas with header
            output_name = f"{img_file.stem}_bird{bird_idx}{img_file.suffix}"
            canvas.save(output_path / output_name, quality=95)
    
    # Print summary
    print(f"\n{'=' * 60}")
    print("Two-Stage Detection Complete!")
    print(f"{'=' * 60}")
    print(f"Total images processed: {stats['total_images']}")
    print(f"Images with birds detected: {stats['images_with_birds']}")
    print(f"Total birds detected (cropped): {stats['total_birds_detected']}")
    print(f"Birds with keypoints detected: {stats['birds_with_keypoints']}")

if __name__ == "__main__":
    
    two_stage_detection(
        source_dir=Config.INFERENCE_IMAGES,
        output_dir=Config.INFERENCE_OUTPUT,
        detection_model_path=Config.YOLO_V11,
    )
