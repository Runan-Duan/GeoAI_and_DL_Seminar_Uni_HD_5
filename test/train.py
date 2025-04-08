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

from models import get_model
from street_surface_loader import StreetSurfaceVis


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

    def validate(self):
        self.model.eval()
        total_loss, correct, total = 0, 0, 0

        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                total_loss += loss.item()
                _, pred = output.max(1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)

        acc = 100. * correct / total
        self.logger.info(f"Validation Loss: {total_loss:.4f}, Accuracy: {acc:.2f}%")
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
        return best_acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='convnext_small', type=str,
                   choices=['resnet50', 'efficientnet_b4', 'convnext_small', 
                           'vit_b16', 'swin_b', 'convnext_large'],
                   help='Model architecture to use')
    parser.add_argument('--models_dir', default='./models', type=str, help='path to model')
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--epochs', default=120, type=int, help='number of epochs')
    parser.add_argument('--batch_size', default=128, type=int, help='batch size')
    parser.add_argument('--cuda', default=torch.cuda.is_available(), type=bool, help='whether cuda is in use')
    parser.add_argument('--data_path', default='./data/raw/StreetSurfaceVis/s_1024', type=str, help='dataset path')
    parser.add_argument('--logs_dir', default='./logs', type=str, help='logs path')
    args = parser.parse_args()

    print(f"Using Cuda: {torch.cuda.is_available()}")
    trainer = RoadSurfaceTrainer(args)
    trainer.train()


if __name__ == "__main__":
    main()
