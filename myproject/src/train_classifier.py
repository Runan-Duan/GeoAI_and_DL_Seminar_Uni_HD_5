import argparse
import os
import logging

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.optim.lr_scheduler import StepLR
from torch.cuda.amp import GradScaler, autocast

from datasets.street_surface_loader import load_streetsurfacevis, SurfaceDataset, split_and_save_data
from models.efficient_net_classifier import EfficientNetWithAttention
from utils.logger import setup_logging
from utils.config import load_config
from utils.utils import create_directories, save_model
from visualization.plot_loss import plot_loss
from visualization.plot_metrics import plot_metrics


def train(model, device, train_loader, optimizer, criterion, epoch, log_interval, scaler, gradient_accumulation_steps):
    model.train()
    running_loss = 0.0
    optimizer.zero_grad()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        with autocast():  # Mixed precision
            output = model(data)
            loss = criterion(output, target) / gradient_accumulation_steps
        scaler.scale(loss).backward()

        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        running_loss += loss.item() * gradient_accumulation_steps
        if batch_idx % log_interval == 0:
            logging.info(f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} "
                         f"({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}")
    return running_loss / len(train_loader)

def validate(model, device, test_loader, criterion):
    model.eval()
    val_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            val_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    val_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    logging.info(f"Validation set: Average loss: {val_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} "
                 f"({accuracy:.2f}%)")
    return val_loss, accuracy

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
    setup_logging(log_dir=config['logs_dir'])

    # Create directories
    create_directories([config['processed_dir'], config['models_dir'], config['logs_dir'], config['results_dir']])

    # Define transforms with data augmentation from config
    train_transforms = transforms.Compose([
        transforms.Resize((config['image_size'], config['image_size'])),
        transforms.RandomHorizontalFlip(config['data_augmentation']['RandomHorizontalFlip']),
        transforms.RandomVerticalFlip(config['data_augmentation']['RandomVerticalFlip']),
        transforms.RandomRotation(config['data_augmentation']['RandomRotation']),
        transforms.ColorJitter(
            brightness=config['data_augmentation']['ColorJitter']['brightness'],
            contrast=config['data_augmentation']['ColorJitter']['contrast'],
            saturation=config['data_augmentation']['ColorJitter']['saturation'],
            hue=config['data_augmentation']['ColorJitter']['hue']
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=config['mean'], std=config['std'])
    ])

    test_transforms = transforms.Compose([
        transforms.Resize((config['image_size'], config['image_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=config['mean'], std=config['std'])
    ])

    # Load dataset
    train_images, train_labels, test_images, test_labels = load_streetsurfacevis(config['data_dir'])
    split_and_save_data(config['data_dir'], config['processed_dir'], val_size=0.2, random_state=42)

    # Create datasets
    train_dataset = SurfaceDataset(train_images, train_labels, transform=train_transforms)
    test_dataset = SurfaceDataset(test_images, test_labels, transform=test_transforms)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config['test_batch_size'], shuffle=False)

    # Initialize model
    num_classes = config['num_classes']
    model = EfficientNetWithAttention(num_classes=num_classes).to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])

    # Learning rate scheduler
    scheduler = StepLR(optimizer, step_size=config['lr_scheduler']['step_size'], gamma=config['lr_scheduler']['gamma'])

    # Early stopping
    best_val_loss = float('inf')
    patience = config['early_stopping']['patience']
    min_delta = config['early_stopping']['min_delta']
    trigger_times = 0

    # Training loop
    train_losses, val_losses, val_accuracies = [], [], []
    for epoch in range(1, config['epochs'] + 1):
        train_loss = train(model, device, train_loader, optimizer, criterion, epoch, config['log_interval'])
        val_loss, val_accuracy = validate(model, device, test_loader, criterion)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        logging.info(f"Epoch {epoch} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")

        # Learning rate scheduler step
        scheduler.step()

        # Early stopping logic
        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            trigger_times = 0
        else:
            trigger_times += 1
            if trigger_times >= patience:
                logging.info("Early stopping triggered!")
                break

    # Save model
    model_save_path = os.path.join(config['models_dir'], "road_surface_classification.pth")
    save_model(model, model_save_path)

    # Plot loss and metrics
    plot_loss(train_losses, val_losses, save_path=os.path.join(config['results_dir'], "loss_plot.png"))
    plot_metrics(val_accuracies, "Accuracy", save_path=os.path.join(config['results_dir'], "accuracy_plot.png"))

if __name__ == "__main__":
    main()