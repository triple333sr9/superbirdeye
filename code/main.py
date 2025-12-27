"""Main entry point: trains YOLOv8 then runs grid search."""
import subprocess
import sys


def main():
    print("="*60)
    print("CUB-200-2011 KEYPOINT LOCALIZATION")
    print("="*60)

    # Train YOLOv8 first
    print("\n[1/2] Training YOLOv8...")
    result = subprocess.run([sys.executable, 'train_yolo.py'], capture_output=False)
    if result.returncode == 0:
        print("YOLOv8 training complete")
    else:
        print("YOLOv8 training failed")

    # Run grid search for ResNet50 and DenseNet
    print("\n[2/2] Running grid search...")
    result = subprocess.run([sys.executable, 'grid_search.py'], capture_output=False)
    if result.returncode == 0:
        print("Grid search complete")
    else:
        print("Grid search failed")

    print("\n" + "="*60)
    print("ALL TRAINING COMPLETE")
    print("="*60)


if __name__ == '__main__':
    main()