import os
import torch
from PIL import Image
import torch
import logging
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torchvision import models
from wait.model_utils import initialize_model
from torch import nn


# Load the saved model
def load_model(model_path, num_classes, device):
    """
    Load the saved model for inference.
    """
    # Initialize the model with the same architecture used during training
    model = initialize_model("efficientnet", num_classes, pretrained=False)
    
    # Load the saved state dictionary
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    # Move the model to the appropriate device (CPU or GPU)
    model.to(device)
    
    # Set the model to evaluation mode
    model.eval()
    return model


def preprocess_image(image_path, device):
    """
    Preprocess the input image for inference.
    """
    # Define the same transformations used during training
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to the input size expected by EfficientNet
        transforms.ToTensor(),          # Convert to tensor
        transforms.Normalize(           # Normalize with the same mean and std used during training
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # Load and preprocess the image
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image.to(device)

def infer(model, image_tensor, device):
    """
    Perform inference on a single image.
    """
    # Move the image tensor to the appropriate device
    image_tensor = image_tensor.to(device)
    
    # Perform inference
    with torch.no_grad():  # Disable gradient calculation
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)  # Get the predicted class
    
    return predicted.item()


# Visualize results with continent information
def visualize_results(images, predictions, class_names, continent):
    num_images = len(images)
    num_cols = 4  # Number of columns in the grid
    num_rows = (num_images + num_cols - 1) // num_cols

    plt.figure(figsize=(15, 5 * num_rows))
    plt.suptitle(f"Continent: {continent}", fontsize=16)
    for i, (img, pred) in enumerate(zip(images, predictions)):
        plt.subplot(num_rows, num_cols, i + 1)
        img = img.cpu().permute(1, 2, 0).numpy()  # Convert to HWC and numpy
        img = img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]  # Denormalize
        img = img.clip(0, 1)  # Clip to valid range
        plt.imshow(img)
        plt.title(f"Pred: {class_names[pred]}")
        plt.axis("off")
    plt.tight_layout()
    plt.show()

# Save results with continent information
def save_results(images, predictions, class_names, continent, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for i, (img, pred) in enumerate(zip(images, predictions)):
        img = img.cpu().permute(1, 2, 0).numpy()  # Convert to HWC and numpy
        img = img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]  # Denormalize
        img = img.clip(0, 1)  # Clip to valid range
        plt.imshow(img)
        plt.title(f"Pred: {class_names[pred]}")
        plt.axis("off")
        plt.savefig(os.path.join(output_dir, f"{continent}_{i}.png"), bbox_inches="tight", dpi=300)
        plt.close()

# Main function
def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model_path = "../models/road_surface_classification.pth"
    model = load_model(model_path, num_classes=5, device=device)

    # Define class names
    class_names = ["asphalt", "paving stones", "concrete", "sett", "unpaved"]  # Adjust as needed

    # Define inference folder and output directory
    inference_folder = "../data/inference"
    output_dir = "../results/inference"

    # Iterate through continent subfolders
    for continent in os.listdir(inference_folder):
        continent_folder = os.path.join(inference_folder, continent)
        if not os.path.isdir(continent_folder):
            continue  # Skip non-directory files

        # Load images for inference
        image_paths = [os.path.join(continent_folder, img) for img in os.listdir(continent_folder) if img.endswith((".jpg"))]
        images = [preprocess_image(img_path, device) for img_path in image_paths]
        if not images:
            continue  # Skip if no images in the folder
        images = torch.cat(images)  # Combine into a single batch

        # Run inference
        with torch.no_grad():
            outputs = model(images)
            _, predictions = torch.max(outputs, 1)

        # Visualize results
        visualize_results(images.cpu(), predictions.cpu(), class_names, continent)

        # Save results
        save_results(images.cpu(), predictions.cpu(), class_names, continent, output_dir)

if __name__ == "__main__":
    main()