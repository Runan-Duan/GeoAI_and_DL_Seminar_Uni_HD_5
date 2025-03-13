import os
import torch
from PIL import Image
import logging
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from models.efficient_net_classifier import EfficientNetWithAttention
from models.unet_segmenter import UNetSegmenter
from utils.utils import load_model
from visualization.visualize_predictions import visualize_predictions

def load_model(model_path, num_classes, device, task):
    """
    Load the saved model for inference.
    """
    if task == "classification":
        model = EfficientNetWithAttention(num_classes=num_classes).to(device)
    else:
        model = UNetSegmenter(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def preprocess_image(image_path, device, task, image_size):
    """
    Preprocess the input image for inference.
    """
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image.to(device)

def infer(model, image_tensor, device, task):
    """
    Perform inference on a single image.
    """
    with torch.no_grad():
        outputs = model(image_tensor)
        if task == "classification":
            _, predicted = torch.max(outputs, 1)  # Get the predicted class
            return predicted.item()
        else:
            return outputs  # Return segmentation mask

def visualize_results(images, predictions, class_names, task):
    num_images = len(images)
    num_cols = 4  # Number of columns in the grid
    num_rows = (num_images + num_cols - 1) // num_cols

    plt.figure(figsize=(15, 5 * num_rows))
    for i, (img, pred) in enumerate(zip(images, predictions)):
        plt.subplot(num_rows, num_cols, i + 1)
        img = img.cpu().permute(1, 2, 0).numpy()  # Convert to HWC and numpy
        img = img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]  # Denormalize
        img = img.clip(0, 1)  # Clip to valid range
        if task == "classification":
            plt.title(f"Pred: {class_names[pred]}")
        else:
            plt.title("Segmentation Mask")
        plt.imshow(img)
        plt.axis("off")
    plt.tight_layout()
    plt.show()

def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define task type
    task = "classification"  # Change to "segmentation" for segmentation task

    # Load model
    model_path = r"models/models/best_road_surface_classification.pth" if task == "classification" else "../models/road_segmentation.pth"
    model = load_model(model_path, num_classes=5, device=device, task=task)

    # Define class names (for classification)
    class_names = ["asphalt", "paving stones", "concrete", "sett", "unpaved"]

    # Define inference folder and output directory
    inference_folder = r"data/inference"
    output_dir = r"results/inference"

    print("Preparing images...")
    # Iterate through images
    image_paths = [os.path.join(inference_folder, img) for img in os.listdir(inference_folder) if img.endswith((".jpg"))]
    images = [preprocess_image(img_path, device, task, image_size=512) for img_path in image_paths]
    if not images:
        print("No images")
        return  # Skip if no images in the folder
    images = torch.cat(images)  # Combine into a single batch

    print("Inferencing...")
    # Run inference
    predictions = infer(model, images, device, task)

    # Visualize results
    visualize_results(images.cpu(), predictions.cpu(), class_names, task)

if __name__ == "__main__":
    main()


"""#

import os
import torch
from PIL import Image
import torchvision.transforms as transforms
from models.unet_segmenter import UNetSegmenter
from utils.utils import load_model
import matplotlib.pyplot as plt

def preprocess_image(image_path, device, image_size=(2048, 1024)):
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image.to(device)

def visualize_results(image, mask, prediction):
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.title("Input Image")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(mask, cmap="gray")
    plt.title("Ground Truth Mask")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(prediction, cmap="gray")
    plt.title("Predicted Mask")
    plt.axis("off")

    plt.show()

def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = UNetSegmenter(num_classes=3).to(device)
    model = load_model(model, "../models/road_segmentation.pth", device)

    # Load and preprocess image
    image_path = "../data/inference/high_res_image.jpg"
    image = preprocess_image(image_path, device, image_size=(2048, 1024))

    # Run inference
    with torch.no_grad():
        output = model(image)
        prediction = torch.argmax(output, dim=1).squeeze().cpu().numpy()

    # Load ground truth mask (if available)
    mask_path = "../data/inference/high_res_mask.png"
    mask = Image.open(mask_path).convert("L")
    mask = mask.resize((2048, 1024))

    # Visualize results
    input_image = Image.open(image_path).convert("RGB")
    visualize_results(input_image, mask, prediction)

if __name__ == "__main__":
    main()"""