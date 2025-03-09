import argparse
import os
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.optim.lr_scheduler import StepLR
from torch.cuda.amp import GradScaler, autocast
from src.datasets.mapillary_loader import MapillaryDataset
from src.models.unet_segmenter import UNetSegmenter
from src.utils.metrics import iou_score
from src.utils.logger import setup_logging
from src.utils.config import load_config
from src.utils.utils import create_directories, save_model
from src.visualization.plot_loss import plot_loss
from src.visualization.plot_metrics import plot_metrics
import logging


def train(model, device, train_loader, optimizer, criterion, epoch, log_interval, scaler, gradient_accumulation_steps):
    model.train()
    running_loss = 0.0
    optimizer.zero_grad()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        with autocast():  # Mixed precision
            output = model(data)
            loss = criterion(output, target) / gradient_accumulation_steps  # Normalize loss
        scaler.scale(loss).backward()  # Scale loss for mixed precision

        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            scaler.step(optimizer)  # Update weights
            scaler.update()
            optimizer.zero_grad()

        running_loss += loss.item() * gradient_accumulation_steps
        if batch_idx % log_interval == 0:
            iou = iou_score(output, target)
            logging.info(f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} "
                         f"({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}, IoU: {iou:.4f}")
    return running_loss / len(train_loader)

def validate(model, device, test_loader, criterion):
    model.eval()
    val_loss = 0
    total_iou = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            val_loss += criterion(output, target).item()
            total_iou += iou_score(output, target)
    val_loss /= len(test_loader.dataset)
    avg_iou = total_iou / len(test_loader)
    logging.info(f"Validation set: Average loss: {val_loss:.4f}, IoU: {avg_iou:.4f}")
    return val_loss, avg_iou


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Mapillary Vistas Road Segmentation')
    parser.add_argument('--config', type=str, default="config/mapillary_segmentation.yaml", help='path to config file')
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

    # Define transforms
    train_transforms = transforms.Compose([
        transforms.Resize((512, 1024)),  # Lower resolution for training
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
        transforms.Resize((512, 1024)),  # Lower resolution for validation
        transforms.ToTensor(),
        transforms.Normalize(mean=config['mean'], std=config['std'])
    ])

    # Load dataset
    train_dataset = MapillaryDataset(config['data_dir'], transform=train_transforms)
    test_dataset = MapillaryDataset(config['data_dir'], transform=test_transforms)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=config['test_batch_size'], shuffle=False, num_workers=4, pin_memory=True)

    # Initialize model
    num_classes = config['num_classes']
    model = UNetSegmenter(num_classes=num_classes).to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])

    # Learning rate scheduler
    scheduler = StepLR(optimizer, step_size=config['lr_scheduler']['step_size'], gamma=config['lr_scheduler']['gamma'])

    # Mixed precision training
    scaler = GradScaler()

    # Early stopping
    best_val_loss = float('inf')
    patience = config['early_stopping']['patience']
    min_delta = config['early_stopping']['min_delta']
    trigger_times = 0

    # Training loop
    train_losses, val_losses, val_ious = [], [], []
    for epoch in range(1, config['epochs'] + 1):
        train_loss = train(model, device, train_loader, optimizer, criterion, epoch, config['log_interval'], scaler, config['gradient_accumulation_steps'])
        val_loss, val_iou = validate(model, device, test_loader, criterion)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_ious.append(val_iou)
        logging.info(f"Epoch {epoch} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val IoU: {val_iou:.4f}")

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
    model_save_path = os.path.join(config['models_dir'], "road_segmentation.pth")
    save_model(model, model_save_path)

    # Plot loss and metrics
    plot_loss(train_losses, val_losses, save_path=os.path.join(config['results_dir'], "loss_plot.png"))
    plot_metrics(val_ious, "IoU", save_path=os.path.join(config['results_dir'], "iou_plot.png"))

if __name__ == "__main__":
    main()
