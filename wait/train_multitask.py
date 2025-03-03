import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from src.models.multitask_model import MultiTaskModel
from src.datasets.mapillary_loader import MapillaryDataset
from src.utils.logger import Logger
import torch.nn.functional as F

def train_multitask_model():
    # Initialize dataset and dataloaders
    train_data = MapillaryDataset(data_csv='data/processed/mapillary_train.csv', 
                                  img_dir='data/mapillary/images', 
                                  label_dir='data/mapillary/labels')
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

    # Initialize the multi-task model
    model = MultiTaskModel(num_classes_classification=5, num_classes_segmentation=21, pretrained=True)  # Adjust num_classes

    # Loss functions
    classification_criterion = torch.nn.CrossEntropyLoss()  # For classification task
    segmentation_criterion = torch.nn.CrossEntropyLoss()   # For segmentation task

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # TensorBoard and logger
    writer = SummaryWriter(log_dir='logs/tensorboard')
    logger = Logger(log_file='logs/train_log.txt')

    # Training loop
    for epoch in range(10):  # Adjust the number of epochs
        model.train()
        running_classification_loss = 0.0
        running_segmentation_loss = 0.0

        for i, (inputs, labels) in enumerate(train_loader):
            optimizer.zero_grad()

            # Forward pass: get both classification and segmentation outputs
            classification_output, segmentation_output = model(inputs)

            # Calculate loss for both tasks
            classification_loss = classification_criterion(classification_output, labels['classification'])
            segmentation_loss = segmentation_criterion(segmentation_output, labels['segmentation'])

            # Total loss (sum or weighted sum of both tasks)
            total_loss = classification_loss + segmentation_loss

            # Backpropagation and optimization
            total_loss.backward()
            optimizer.step()

            running_classification_loss += classification_loss.item()
            running_segmentation_loss += segmentation_loss.item()

            # Log every 10 steps
            if i % 10 == 9:
                logger.log(f"Epoch {epoch+1}, Step {i+1}, Classification Loss: {running_classification_loss/10}, Segmentation Loss: {running_segmentation_loss/10}")
                writer.add_scalar('classification_loss', running_classification_loss/10, epoch * len(train_loader) + i)
                writer.add_scalar('segmentation_loss', running_segmentation_loss/10, epoch * len(train_loader) + i)
                running_classification_loss = 0.0
                running_segmentation_loss = 0.0

        # Save model after every epoch
        torch.save(model.state_dict(), f'models/multitask_model_epoch_{epoch+1}.pth')

if __name__ == "__main__":
    train_multitask_model()



import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models.multitask_model import MultiTaskModel
from utils.metrics import accuracy, iou_score

class MultiTaskTrainer:
    def __init__(self, model, train_loader, val_loader, optimizer, criterion_cls, criterion_seg, device):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion_cls = criterion_cls
        self.criterion_seg = criterion_seg
        self.device = device

    def train_epoch(self):
        self.model.train()
        total_loss = 0.0
        for images, cls_labels, seg_labels in self.train_loader:
            images, cls_labels, seg_labels = images.to(self.device), cls_labels.to(self.device), seg_labels.to(self.device)
            self.optimizer.zero_grad()
            cls_output, seg_output = self.model(images)
            loss_cls = self.criterion_cls(cls_output, cls_labels)
            loss_seg = self.criterion_seg(seg_output, seg_labels)
            loss = loss_cls + loss_seg
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(self.train_loader)

    def validate(self):
        self.model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, cls_labels, seg_labels in self.val_loader:
                images, cls_labels, seg_labels = images.to(self.device), cls_labels.to(self.device), seg_labels.to(self.device)
                cls_output, seg_output = self.model(images)
                loss_cls = self.criterion_cls(cls_output, cls_labels)
                loss_seg = self.criterion_seg(seg_output, seg_labels)
                loss = loss_cls + loss_seg
                val_loss += loss.item()
        return val_loss / len(self.val_loader)

    def train(self, epochs):
        for epoch in range(epochs):
            train_loss = self.train_epoch()
            val_loss = self.validate()
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")