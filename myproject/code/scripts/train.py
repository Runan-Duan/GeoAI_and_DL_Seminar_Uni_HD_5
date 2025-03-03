import argparse
import os
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from src.datasets.street_surface_loader import load_streetsurfacevis, SurfaceDataset
from src.models.model_utils import initialize_model
from src.trainers.train_classifier import train, validate
from src.utils.logger import setup_logging
from src.utils.config import load_config
from src.utils.utils import create_directories, save_model
from src.visualization.plot_loss import plot_loss
from src.visualization.plot_metrics import plot_metrics
import logging

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='StreetSurfaceVis road surface classification')
    parser.add_argument('--config', type=str, default="config/street_surface.yaml", help='path to config file')
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Set random seed
    torch.manual_seed(config['seed'])

    # Use CUDA if available
    use_cuda = not config['no_cuda'] and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # Set up logging
    setup_logging()

    # Create directories
    create_directories([config['processed_dir'], config['models_dir'], config['logs_dir'], config['results_dir']])

    # Define transforms
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load dataset
    train_images, train_labels, test_images, test_labels = load_streetsurfacevis(config['data_dir'])

    # Create datasets
    train_dataset = SurfaceDataset(train_images, train_labels, transform=train_transforms)
    test_dataset = SurfaceDataset(test_images, test_labels, transform=test_transforms)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config['test_batch_size'], shuffle=False)

    # Initialize model
    num_classes = len(set(train_labels + test_labels))
    model = initialize_model("efficientnet", num_classes=num_classes).to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])

    # Training loop
    train_losses, val_losses, val_accuracies = [], [], []
    for epoch in range(1, config['epochs'] + 1):
        train_loss = train(model, device, train_loader, optimizer, criterion, epoch, config['log_interval'])
        val_loss, val_accuracy = validate(model, device, test_loader, criterion)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        logging.info(f"Epoch {epoch} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")

    # Save model
    model_save_path = os.path.join(config['models_dir'], "road_surface_classification.pth")
    save_model(model, model_save_path)

    # Plot loss and metrics
    plot_loss(train_losses, val_losses, save_path=os.path.join(config['results_dir'], "loss_plot.png"))
    plot_metrics(val_accuracies, "Accuracy", save_path=os.path.join(config['results_dir'], "accuracy_plot.png"))

if __name__ == "__main__":
    main()