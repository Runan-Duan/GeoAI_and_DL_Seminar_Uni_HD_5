import argparse
import logging
from pathlib import Path
import numpy as np
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision import transforms

from sklearn.metrics import confusion_matrix, classification_report

from models import get_model
from street_surface_loader import StreetSurfaceVis, SURFACE_CLASSES

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image


class RoadSurfaceTrainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if args.cuda else "cpu")
        self._setup_logging()
        self._init_data()
        self._init_model()
        self._init_optimizer()

    def _setup_logging(self):
        Path(self.args.logs_dir).mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_filename = f"{self.args.model_name}_{timestamp}.log"
        log_filepath = Path(self.args.logs_dir) / log_filename
        
        # Clear any existing handlers
        logging.getLogger().handlers = []
        
        # File handler for all logs
        file_handler = logging.FileHandler(log_filepath)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        
        # Stream handler only for WARNING and above
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.WARNING)
        stream_handler.setFormatter(logging.Formatter('%(levelname)s - %(message)s'))
        
        # Configure root logger
        logging.basicConfig(
            level=logging.INFO,
            handlers=[file_handler, stream_handler]
        )
        
        self.logger = logging.getLogger('RoadSurfaceTrainer')

    def _init_data(self):
        dataset = StreetSurfaceVis(
            batch_size=self.args.batch_size,
            data_root=self.args.data_path
        )
        dataset.setup()

        self.train_loader = dataset.train_dataloader()
        self.val_loader = dataset.val_dataloader()
        self.test_loader = dataset.test_dataloader()  # Add this line to load the test data

    def _init_model(self):
        self.model = get_model(self.args.model_name, num_classes=5).to(self.device)
        #class_counts = np.array([3734, 972, 2037, 1363, 1016])
        #class_weights = 1. / class_counts
        #class_weights = class_weights / class_weights.min()    # Resulting weights: [1.0, 3.84, 1.83, 2.74, 3.68]
        class_weights = torch.tensor([1.0, 4.0, 2.0, 3.0, 4.0]).to(self.device)
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)

    def _init_optimizer(self):
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.args.lr,
            weight_decay=0.05
        )
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=self.args.epochs,
            eta_min=1e-6
        )

    def train_epoch(self, epoch):
        self.model.train()
        total_loss, correct, total = 0, 0, 0

        for data, target in self.train_loader:
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()
            _, pred = output.max(1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)

        self.scheduler.step()
        acc = 100. * correct / total
        self.logger.info(f"[Epoch {epoch}] Train Loss: {total_loss:.4f}, Accuracy: {acc:.2f}%")
        return total_loss, acc

    def validate(self, confusion_matrix=True):
        self.model.eval()
        total_loss, correct, total = 0, 0, 0

        class_correct = [0] * 5
        class_total = [0] * 5

        all_targets = []
        all_preds = []

        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                total_loss += loss.item()

                _, pred = output.max(1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)

                # Collect for confusion matrix
                all_targets.extend(target.cpu().numpy())
                all_preds.extend(pred.cpu().numpy())

                for i in range(len(target)):
                    label = target[i].item()
                    class_total[label] += 1
                    if pred[i].item() == label:
                        class_correct[label] += 1

        acc = 100. * correct / total
        self.logger.info(f"Validation Loss: {total_loss:.4f}, Accuracy: {acc:.2f}%")

        for i, cls in enumerate(SURFACE_CLASSES):
            if class_total[i] > 0:
                cls_acc = 100. * class_correct[i] / class_total[i]
                self.logger.info(f"  Class [{cls}] Accuracy: {cls_acc:.2f}% ({class_correct[i]}/{class_total[i]})")
            else:
                self.logger.warning(f"  Class [{cls}] has no samples in validation set.")
        
        if confusion_matrix:
            self.confusion_matrix(all_targets, all_preds)

        return total_loss, acc

    def test(self, confusion_matrix=True):
        self.model.eval()
        total_loss, correct, total = 0, 0, 0

        class_correct = [0] * 5
        class_total = [0] * 5

        all_targets = []
        all_preds = []

        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                total_loss += loss.item()

                _, pred = output.max(1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)

                # Collect for confusion matrix
                all_targets.extend(target.cpu().numpy())
                all_preds.extend(pred.cpu().numpy())

                for i in range(len(target)):
                    label = target[i].item()
                    class_total[label] += 1
                    if pred[i].item() == label:
                        class_correct[label] += 1

        acc = 100. * correct / total
        self.logger.info(f"Test Loss: {total_loss:.4f}, Accuracy: {acc:.2f}%")

        for i, cls in enumerate(SURFACE_CLASSES):
            if class_total[i] > 0:
                cls_acc = 100. * class_correct[i] / class_total[i]
                self.logger.info(f"  Class [{cls}] Accuracy: {cls_acc:.2f}% ({class_correct[i]}/{class_total[i]})")
            else:
                self.logger.warning(f"  Class [{cls}] has no samples in test set.")
        
        if confusion_matrix:
            self.confusion_matrix(all_targets, all_preds)

        return total_loss, acc

    def train(self):
        Path(self.args.models_dir).mkdir(parents=True, exist_ok=True)
        best_acc = 0
        for epoch in range(1, self.args.epochs + 1):
            self.train_epoch(epoch)
            _, val_acc = self.validate()

            if val_acc > best_acc:
                best_acc = val_acc
                model_path = Path(self.args.models_dir) / f'best_model_{self.args.model_name}.pth'
                torch.save(self.model.state_dict(), model_path)
                self.logger.info(f"New best model saved with accuracy: {best_acc:.2f}%")

        # After training, evaluate the model on the test set
        self.test()  # Add this call to evaluate on the test set
        return best_acc

    def confusion_matrix(self, all_targets, all_preds):
        # Log confusion matrix and classification report
        cm = confusion_matrix(all_targets, all_preds)
        self.logger.info(f"\nConfusion Matrix:\n{cm}")

        report = classification_report(all_targets, all_preds, target_names=SURFACE_CLASSES)
        self.logger.info(f"\nClassification Report:\n{report}")
