import os
import torch
from torchvision import transforms
from PIL import Image
import torchvision.models.detection as detection
import torchvision.models as models
import matplotlib.pyplot as plt

# Define paths
models_path = r"C:\Users\jorda\PycharmProjects\CUB_200_2011\outputs\models"
images_path = r"C:\Users\jorda\OneDrive\Desktop\2025-08-12"
output_path = r"C:\Users\jorda\PycharmProjects\CUB_200_2011\outputs\results"

# Image transform
transform = transforms.Compose([transforms.ToTensor()])


# Load model based on filename
def load_model(model_path, model_name):
    state_dict = torch.load(model_path, map_location='cpu', weights_only=True)

    # Select model type based on the filename
    if 'Baseline_CNN' in model_name:
        print(f"Loading CNN model from {model_name}")
        model = models.resnet50(pretrained=False)  # Example: Replace with actual CNN model if necessary
    elif 'Keypoint_Net' in model_name:
        print(f"Loading Keypoint Network model from {model_name}")
        model = models.resnet50(pretrained=False)  # Replace with actual keypoint detection model if needed
    elif 'VAE_Classifier' in model_name:
        print(f"Loading VAE-Classifier model from {model_name}")
        model = models.resnet50(pretrained=False)  # Replace with actual VAE model if necessary
    elif 'Vision_Transformer' in model_name:
        print(f"Loading Vision Transformer model from {model_name}")
        model = models.vision_transformer.vit_b_16(pretrained=False)  # ViT model from torchvision
    elif 'YOLO_Detector' in model_name:
        print(f"Loading YOLO Detector model from {model_name}")
        model = models.detection.fasterrcnn_resnet50_fpn(pretrained=False)  # Use appropriate YOLO model
    else:
        raise ValueError(f"Unknown model type in {model_name}")

    # Load weights and set model to evaluation mode
    model.load_state_dict(state_dict)
    model.eval()
    return model


# Process image
def process_image(image_path, model):
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        return model(image_tensor)


# Run models on images
for model_file in os.listdir(models_path):
    model_path = os.path.join(models_path, model_file)

    # Load model based on file name
    model = load_model(model_path, model_file)

    for image_file in os.listdir(images_path):
        image_path = os.path.join(images_path, image_file)
        result = process_image(image_path, model)

        # Handle model outputs
        if isinstance(model, torch.nn.Module):  # For classifiers or detectors
            if isinstance(model, models.detection.DetectionModel):  # Object detection models like YOLO
                boxes = result[0]['boxes'].cpu().numpy()
                labels = result[0]['labels'].cpu().numpy()
                scores = result[0]['scores'].cpu().numpy()
                # Add code to draw/save bounding boxes

            else:  # For classifiers (ResNet, VAE, etc.)
                predicted_class = result.argmax(1).item()
                # Handle saving prediction

        # Save result image
        plt.imshow(Image.open(image_path))
        plt.title(f'{model_file} Result')
        plt.savefig(f'{output_path}/{model_file}_{image_file}_result.png')
