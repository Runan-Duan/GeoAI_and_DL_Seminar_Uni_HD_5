import argparse
import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import confusion_matrix, classification_report, jaccard_score
import numpy as np
import logging
import matplotlib.pyplot as plt

from datasets.mapillary_loader import MapillaryDataset
from datasets.street_surface_loader import SurfaceDataset, load_streetsurfacevis
from models.deeplabV3_segmenter import DeepLabV3PlusSegmenter
from models.efficient_net_classifier import EfficientNetWithAttention

from utils.metrics import dice_score
from utils.config import load_config
from utils.logger import setup_logging
from utils.utils import load_model
from visualization.visualize_predictions import visualize_classification_predictions, visualize_segmentation_predictions


def evaluate_classification(model, device, test_loader, criterion):
    """
    Evaluate the model on the test dataset for classification tasks.
    """
    model.eval()
    test_loss = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            with torch.cuda.amp.autocast():  # Mixed precision
                output = model(data)
                test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * np.sum(np.array(all_preds) == np.array(all_targets)) / len(all_targets)
    return test_loss, accuracy, all_preds, all_targets


def evaluate_segmentation(model, device, test_loader):
    """
    Evaluate the model on the test dataset for segmentation tasks.
    """
    model.eval()
    iou_scores = []
    dice_scores = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = (outputs > 0.5).float()  # Threshold at 0.5 for binary segmentation

            # Handle cases where there is no drivable area
            if torch.sum(labels) == 0:  # No drivable area in the ground truth
                if torch.sum(preds) == 0:  # Correct prediction (no drivable area)
                    iou = 1.0  # Perfect IoU
                    dice = 1.0  # Perfect Dice
                else:  # Incorrect prediction (drivable area detected)
                    iou = 0.0  # Zero IoU
                    dice = 0.0  # Zero Dice
            else:  # Drivable area is present in the ground truth
                iou = jaccard_score(labels.cpu().numpy().flatten(), preds.cpu().numpy().flatten())
                dice = dice_score(labels.cpu().numpy(), preds.cpu().numpy())

            iou_scores.append(iou)
            dice_scores.append(dice)

    mean_iou = np.mean(iou_scores)
    mean_dice = np.mean(dice_scores)
    logging.info(f"Mean IoU: {mean_iou:.4f}")
    logging.info(f"Mean Dice: {mean_dice:.4f}")
    return mean_iou, mean_dice



def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Evaluate model')
    parser.add_argument('--config', type=str, required=True, help='path to config file')
    parser.add_argument('--model-type', type=str, choices=['best', 'final'], required=True, help='type of model to evaluate (best or final)')
    parser.add_argument('--task', type=str, choices=['surface_classification', 'segmentation'], required=True, help='task type')
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
    if args.task == "surface_classification":
        _, _, test_images, test_labels = load_streetsurfacevis(config['data_dir'])
        test_dataset = SurfaceDataset(test_images, test_labels, transform=test_transforms)
    elif args.task == "segmentation":
        test_dataset = MapillaryDataset(
            data_dir=config['test_data_dir'],  # Path to testing data
            transform=test_transforms,
            is_testing=True  # No labels for testing
        )
    else:
        raise ValueError(f"Invalid task type: {args.task}")
    
    test_loader = DataLoader(test_dataset, batch_size=config['test_batch_size'], shuffle=False)


    # Initialize model
    if args.task == "surface_classification":
        model = EfficientNetWithAttention(num_classes=config['num_classes']).to(device)
    elif args.task == "segmentation":
        model = DeepLabV3PlusSegmenter().to(device)
    else:
        raise ValueError(f"Invalid task type: {args.task}")
    

    # Load trained model
    if args.model_type == 'best':
        model_path = os.path.join(config['models_dir'], f"best_road_{args.task}.pth")
        results_dir = os.path.join(config['results_dir'], "best_model")
    else:
        model_path = os.path.join(config['models_dir'], f"final_road_{args.task}.pth")
        results_dir = os.path.join(config['results_dir'], "final_model")


    # Create results directory if it doesn't exist
    os.makedirs(results_dir, exist_ok=True)

    model = load_model(model, model_path, device)


    # Define loss function
    if args.task == "surface_classification":
        criterion = torch.nn.CrossEntropyLoss()
    elif args.task == "segmentation":
        criterion = torch.nn.BCELoss()  # Binary Cross-Entropy Loss for segmentation
    else:
        raise ValueError(f"Invalid task type: {args.task}")
    

    # Evaluate model
    if args.task == "surface_classification":
        test_loss, test_accuracy, all_preds, all_targets = evaluate_classification(model, device, test_loader, criterion)
        class_names = ["asphalt", "concrete", "paving_stones", "sett", "unpaved"]  # Update with your class names

        # Compute confusion matrix and classification report
        conf_matrix = confusion_matrix(all_targets, all_preds)
        class_report = classification_report(all_targets, all_preds, target_names=class_names)

        # Save evaluation results
        with open(os.path.join(results_dir, f"evaluation_{args.task}.txt"), "w") as f:
            f.write(f"Test Loss: {test_loss:.4f}\n")
            f.write(f"Test Accuracy: {test_accuracy:.2f}%\n")
            f.write("\nConfusion Matrix:\n")
            f.write(np.array2string(conf_matrix, separator=", "))
            f.write("\n\nClassification Report:\n")
            f.write(class_report)

        # Log evaluation results
        logging.info(f"Test Loss: {test_loss:.4f}")
        logging.info(f"Test Accuracy: {test_accuracy:.2f}%")
        logging.info("Confusion Matrix:\n%s", conf_matrix)
        logging.info("Classification Report:\n%s", class_report)

        # Visualize predictions and save the plot
        visualize_classification_predictions(
            test_dataset.images[:5], all_preds[:5], all_targets[:5], class_names,
            save_path=os.path.join(results_dir, f"predictions_{args.task}.png")
        )
    elif args.task == "segmentation":
        mean_iou, mean_dice = evaluate_segmentation(model, device, test_loader)

        # Save evaluation results
        with open(os.path.join(results_dir, f"evaluation_{args.task}.txt"), "w") as f:
            f.write(f"Mean IoU: {mean_iou:.4f}\n")
            f.write(f"Mean Dice: {mean_dice:.4f}\n")

        # Log evaluation results
        logging.info(f"Mean IoU: {mean_iou:.4f}")
        logging.info(f"Mean Dice: {mean_dice:.4f}")

        # Visualize segmentation predictions
        visualize_segmentation_predictions(
            model, test_loader, device, save_path=os.path.join(results_dir, f"predictions_{args.task}.png")
        )
    else:
        raise ValueError(f"Invalid task type: {args.task}")


if __name__ == "__main__":
    main()