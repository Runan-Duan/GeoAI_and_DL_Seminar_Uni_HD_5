import argparse
import os
import logging

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.cuda.amp import GradScaler, autocast

from datasets.street_surface_loader import MapillaryDataset
from myproject.src.models.deeplabV3_segmenter import DeepLabV3PlusSegmenter  # Updated model import
from utils.logger import setup_logging
from utils.config import load_config
from utils.utils import create_directories, save_model
from visualization.plot_loss import plot_loss


def train(model, device, train_loader, optimizer, criterion, epoch, log_interval, scaler, gradient_accumulation_steps, max_grad_norm):
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
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
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
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            with autocast():  # Mixed precision
                output = model(data)
                val_loss += criterion(output, target).item()
    val_loss /= len(test_loader.dataset)
    logging.info(f"Validation set: Average loss: {val_loss:.4f}")
    return val_loss


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Mapillary Vistas Road Segmentation')
    parser.add_argument('--config', type=str, default="config/road_segmentation.yaml", help='path to config file')
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
    create_directories([config['models_dir'], config['logs_dir'], config['results_dir']])

    # Define transforms
    train_transforms = transforms.Compose([
        transforms.Resize((config['image_size'], config['image_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((config['image_size'], config['image_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
    ])

    # Load training dataset
    train_dataset = MapillaryDataset(
        data_dir=config['train_data_dir'],  # Path to training data
        transform=train_transforms
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=8,
        pin_memory=True
    )

    # Load validation dataset
    val_dataset = MapillaryDataset(
        data_dir=config['val_data_dir'],  # Path to validation data
        transform=val_transforms
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,  # No need to shuffle validation data
        num_workers=8,
        pin_memory=True
    )

    # Initialize model (DeepLabV3+ with EfficientNet-B4)
    model = DeepLabV3PlusSegmenter().to(device)

    # Define loss function (Binary Cross-Entropy Loss)
    criterion = nn.BCELoss()

    # Define optimizer
    optimizer = optim.Adam(model.parameters(), lr=config['lr'], weight_decay=float(config['weight_decay']))

    # Learning rate scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=config['lr_scheduler']['factor'], patience=config['lr_scheduler']['patience'], verbose=True)

    # Mixed precision training
    scaler = GradScaler()

    # Training loop
    train_losses, val_losses = [], []
    best_val_loss = float('inf')  # Track the best validation loss

    for epoch in range(1, config['epochs'] + 1):
        # Train for one epoch
        train_loss = train(model, device, train_loader, optimizer, criterion, epoch, config['log_interval'], scaler, config['gradient_accumulation_steps'], config['max_grad_norm'])
        train_losses.append(train_loss)

        # Validate
        val_loss = validate(model, device, val_loader, criterion)
        val_losses.append(val_loss)

        # Log training and validation loss
        logging.info(f"Epoch {epoch} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # Learning rate scheduler step
        scheduler.step(val_loss)

        # Save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_save_path = os.path.join(config['models_dir'], "best_road_segmentation.pth")
            save_model(model, model_save_path)
            logging.info(f"New best model saved with Val Loss: {val_loss:.4f}")

    # Save final model
    model_save_path = os.path.join(config['models_dir'], "final_road_segmentation.pth")
    save_model(model, model_save_path)
    logging.info("Final model saved.")

    # Plot loss
    plot_loss(train_losses, val_losses, save_path=os.path.join(config['results_dir'], "segmentation_loss_plot.png"))


if __name__ == "__main__":
    main()