"""Config with optimal hyperparameters from grid search results."""
from pathlib import Path
import torch


class Config:
    # Paths
    DATA_DIR = Path(r"/CUB_200_2011")
    IMAGES_DIR = DATA_DIR / "images"
    ANNOTATIONS_FILE = DATA_DIR / "parts/part_locs.txt"
    TRAIN_TEST_SPLIT = DATA_DIR / "train_test_split.txt"
    BBOX_FILE = DATA_DIR / "bounding_boxes.txt"
    IMAGES_LIST = DATA_DIR / "images.txt"

    # Updated parameters
    BATCH_SIZE = 16
    IMG_SIZE = 416
    NUM_EPOCHS = 50
    NUM_PARTS = 3
    PART_NAMES = ['left_eye', 'right_eye', 'beak']
    EARLY_STOPPING_PATIENCE = 10
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Loss weights
    LOSS_WEIGHT_X, LOSS_WEIGHT_Y = 1.5, 1.0
    LOSS_WEIGHT_INVISIBLE = 0.15
    SPATIAL_PRIOR_WEIGHT = 0.3

    # Directories
    RESULTS_DIR = Path("../results")
    VIS_DIR = RESULTS_DIR / "visualizations"
    REPORTS_DIR = RESULTS_DIR / "reports"
    GRID_SEARCH_DIR = RESULTS_DIR / "grid_search"
    SAVE_DIR = Path("../saved_models")
    INFERENCE_IMAGES = Path(r"../inference_images")
    INFERENCE_IMAGES_CROPPED = Path(r"../inference_images_cropped")
    INFERENCE_OUTPUT = Path(r"../inference_output")
    COMPARISON_INFERENCE_OUTPUT = Path(r"../results_comparisons")
    YOLO_V11 = Path(r"/yolo11n.pt")

    for d in [RESULTS_DIR, VIS_DIR, REPORTS_DIR, GRID_SEARCH_DIR, SAVE_DIR]:
        d.mkdir(exist_ok=True)

    # Augmentation presets
    AUGMENTATION_PRESETS = {
        'medium': {
            'flip_prob': 0.5, 'rotation': 10, 'scale_range': (0.9, 1.1),
            'translate': 0.1, 'brightness': 0.2, 'contrast': 0.2,
            'saturation': 0.2, 'hue': 0.05, 'cutout_prob': 0.2,
        },
        'heavy': {
            'flip_prob': 0.5, 'rotation': 15, 'scale_range': (0.8, 1.2),
            'translate': 0.15, 'brightness': 0.3, 'contrast': 0.3,
            'saturation': 0.3, 'hue': 0.1, 'cutout_prob': 0.3,
        },
    }

    @staticmethod
    def get_reduced_grid():
        """Minimal grid with optimal parameters: 4 configs total."""
        configs = []
        for aug in ['medium', 'heavy']:
            configs.append({
                'loss_function': 'wing',
                'learning_rate': 5e-5,
                'weight_decay': 0.0,
                'hidden_dim': 512,
                'dropout': 0.2,
                'visibility_loss_weight': 2.0,
                'augmentation': aug,
                'scheduler': 'plateau',
                'config_name': f"wing_lr5e-05_h512_do0.2_{aug}_plateau"
            })
        print(f"Grid: {len(configs)} configurations")
        return configs

    def apply_hyperparameters(self, hp):
        self.LOSS_FUNCTION = hp['loss_function']
        self.LEARNING_RATE = hp['learning_rate']
        self.WEIGHT_DECAY = hp.get('weight_decay', 0.0)
        self.HIDDEN_DIM = hp['hidden_dim']
        self.DROPOUT = hp['dropout']
        self.VISIBILITY_LOSS_WEIGHT = hp.get('visibility_loss_weight', 2.0)
        self.CONFIG_NAME = hp['config_name']
        self.SCHEDULER = hp.get('scheduler', 'plateau')

        aug = self.AUGMENTATION_PRESETS[hp.get('augmentation', 'medium')]
        self.AUG_FLIP_PROB = aug['flip_prob']
        self.AUG_ROTATION = aug['rotation']
        self.AUG_SCALE_RANGE = aug['scale_range']
        self.AUG_TRANSLATE = aug['translate']
        self.AUG_BRIGHTNESS = aug['brightness']
        self.AUG_CONTRAST = aug['contrast']
        self.AUG_SATURATION = aug['saturation']
        self.AUG_HUE = aug['hue']
        self.AUG_CUTOUT_PROB = aug['cutout_prob']
        return self