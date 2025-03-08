import argparse
import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from src.datasets.street_surface_loader import load_streetsurfacevis, SurfaceDataset
from wait.model_utils import initialize_model, load_model
from src.trainers.metrics import accuracy
from src.visualization.visualize_predictions import visualize_predictions
from src.utils.config import load_config
from src.utils.logger import setup_logging
import logging

def evaluate(model, device, test_loader, criterion):
    """
    Evaluate the model on the test dataset.
    """
    model.eval()
    test_loss = 0
    correct = 0
    all_preds, all_targets = [], []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    logging.info(f"Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} "
                 f"({accuracy:.2f}%)")
    return test_loss, accuracy, all_preds, all_targets

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Evaluate StreetSurfaceVis model')
    parser.add_argument('--config', type=str, default="config/street_surface.yaml", help='path to config file')
    parser.add_argument('--model-path', type=str, required=True, help='path to the trained model')
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Set up logging
    setup_logging()

    # Use CUDA if available
    use_cuda = not config['no_cuda'] and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # Load dataset
    _, _, test_images, test_labels = load_streetsurfacevis(config['data_dir'])
    test_dataset = SurfaceDataset(test_images, test_labels, transform=transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]))
    test_loader = DataLoader(test_dataset, batch_size=config['test_batch_size'], shuffle=False)

    # Initialize model
    num_classes = len(set(test_labels))
    model = initialize_model("efficientnet", num_classes=num_classes).to(device)

    # Load trained model
    model = load_model(model, args.model_path, device)

    # Define loss function
    criterion = torch.nn.CrossEntropyLoss()

    # Evaluate model
    test_loss, test_accuracy, all_preds, all_targets = evaluate(model, device, test_loader, criterion)

    # Visualize predictions
    class_names = ["asphalt", "concrete", "paving_stones", "sett", "unpaved"]
    visualize_predictions(test_dataset.images[:5], all_preds[:5], all_targets[:5], class_names)

if __name__ == "__main__":
    main()