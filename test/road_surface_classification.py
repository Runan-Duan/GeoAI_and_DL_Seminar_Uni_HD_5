import argparse
import os
import time
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models

from PIL import Image
import logging


# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# Load StreetSurfaceVis dataset
def load_streetsurfacevis(data_dir):
    csv_path = os.path.join(data_dir, "streetSurfaceVis_v1_0.csv")  # annotations file
    try:
        df = pd.read_csv(csv_path)
        train_images, train_labels = [], []
        test_images, test_labels = [], []
        for _, row in df.iterrows():
            image_path = os.path.join(data_dir, "s_1024", row["image_file"])
            if not os.path.exists(image_path):
                logging.warning(f"Image file not found: {image_path}")
                continue
            label = row["surface_type"]

            # test data contains data from 5 cities excluded in the training data
            # use the `train` column to split to avoid data leakage
            if row["train"]:  
                train_images.append(image_path)
                train_labels.append(label)
            else:
                test_images.append(image_path)
                test_labels.append(label)
        return train_images, train_labels, test_images, test_labels
    except Exception as e:
        logging.error(f"Error loading dataset: {e}")
        raise


# Define dataset
class SurfaceDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform
        self.label_to_idx = {label: idx for idx, label in enumerate(set(labels))}

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert("RGB")
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, self.label_to_idx[label]


# Training function
def train(model, device, train_loader, optimizer, criterion, epoch, log_interval):
    model.train()
    running_loss = 0.0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if batch_idx % log_interval == 0:
            logging.info(f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} "
                         f"({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}")
    return running_loss / len(train_loader)


# Validation function
def validate(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    logging.info(f"Validation set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} "
                 f"({accuracy:.2f}%)")
    return test_loss, accuracy


def main():
    parser = argparse.ArgumentParser(description='StreetSurfaceVis road surface classification')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 32)')
    parser.add_argument('--test-batch-size', type=int, default=32, metavar='N',
                        help='input batch size for testing (default: 32)')
    parser.add_argument('--epochs', type=int, default=30, metavar='N',
                        help='number of epochs to train (default: 30)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--data-dir', type=str, default="Streetsurfacevis",
                        help='directory containing the dataset')
    args = parser.parse_args()

    # Set random seed
    torch.manual_seed(args.seed)

    # Use CUDA if available
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # Define transforms
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),  # Data augmentation
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load dataset
    train_images, train_labels, test_images, test_labels = load_streetsurfacevis(args.data_dir)

    # Create datasets
    train_dataset = SurfaceDataset(train_images, train_labels, transform=train_transforms)
    test_dataset = SurfaceDataset(test_images, test_labels, transform=test_transforms)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False)

    # Load pre-trained model
    model = models.efficientnet_b0(pretrained=True)
    num_classes = len(set(train_labels + test_labels))  # Total number of unique classes
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    model = model.to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Training loop
    for epoch in range(1, args.epochs + 1):
        train_loss = train(model, device, train_loader, optimizer, criterion, epoch, args.log_interval)
        val_loss, val_accuracy = validate(model, device, test_loader, criterion)
        logging.info(f"Epoch {epoch} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")

    # Save model
    torch.save(model.state_dict(), "road_surface_classification.pth")
    logging.info("Model saved to road_surface_classification.pth")



if __name__ == "__main__":
    main()
