import argparse
import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets.street_surface_loader import SurfaceDataset, load_streetsurfacevis
#from datasets.mapillary_loader import MapillaryDataset
from models.efficient_net_classifier import EfficientNetWithAttention
#from models.unet_segmenter import UNetSegmenter
from utils.metrics import iou_score
from utils.config import load_config
from utils.logger import setup_logging
from utils.utils import load_model
from visualization.visualize_predictions import visualize_predictions
import logging

def evaluate_classification(model, device, test_loader, criterion):
    """
    Evaluate the classification model on the test dataset.
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

"""
def evaluate_segmentation(model, device, test_loader, criterion):
    
    # Evaluate the segmentation model on the test dataset.

    model.eval()
    test_loss = 0
    total_iou = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            total_iou += iou_score(output, target)
    test_loss /= len(test_loader.dataset)
    avg_iou = total_iou / len(test_loader)
    logging.info(f"Test set: Average loss: {test_loss:.4f}, IoU: {avg_iou:.4f}")
    return test_loss, avg_iou"""


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Evaluate model')
    parser.add_argument('--config', type=str, required=True, help='path to config file')
    parser.add_argument('--model-path', type=str, required=True, help='path to the trained model')
    parser.add_argument('--task', type=str, choices=['classification', 'segmentation'], required=True, help='task type')
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Set up logging
    setup_logging(log_dir=config['logs_dir'])

    # Use CUDA if available
    use_cuda = not config['no_cuda'] and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # Define test transforms
    test_transforms = transforms.Compose([
        transforms.Resize((config['image_size'], config['image_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=config['mean'], std=config['std'])
    ])

    # Load dataset
    if args.task == "classification":
        _, _, test_images, test_labels = load_streetsurfacevis(config['data_dir'])
        test_dataset = SurfaceDataset(test_images, test_labels, transform=test_transforms)
    else:
        pass #test_dataset = MapillaryDataset(config['data_dir'], transform=test_transforms)

    test_loader = DataLoader(test_dataset, batch_size=config['test_batch_size'], shuffle=False)

    # Initialize model
    if args.task == "classification":
        model = EfficientNetWithAttention(num_classes=config['num_classes']).to(device)
    else:
        pass #model = UNetSegmenter(num_classes=config['num_classes']).to(device)

    # Load trained model
    model = load_model(model, args.model_path, device)

    # Define loss function
    criterion = torch.nn.CrossEntropyLoss()

    # Evaluate model
    if args.task == "classification":
        test_loss, test_accuracy, all_preds, all_targets = evaluate_classification(model, device, test_loader, criterion)
        class_names = ["asphalt", "concrete", "paving_stones", "sett", "unpaved"]  # Update with your class names
        visualize_predictions(test_dataset.images[:5], all_preds[:5], all_targets[:5], class_names)
    else:
        pass #test_loss, test_iou = evaluate_segmentation(model, device, test_loader, criterion)

if __name__ == "__main__":
    main()