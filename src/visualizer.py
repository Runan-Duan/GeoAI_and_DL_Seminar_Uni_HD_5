import argparse
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from models import get_model
from street_surface_loader import StreetSurfaceVis, SURFACE_CLASSES
from sklearn.metrics import confusion_matrix, classification_report

class ModelVisualizer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if args.cuda else "cpu")
        self._init_data()
        self._load_model()
        
    def _init_data(self):
        dataset = StreetSurfaceVis(
            batch_size=self.args.batch_size,
            data_root=self.args.data_path
        )
        dataset.setup()
        self.test_loader = dataset.test_dataloader()
        
    def _load_model(self):
        self.model = get_model(self.args.model_name, num_classes=5).to(self.device)
        model_path = Path(self.args.models_dir) / f'best_model_{self.args.model_name}.pth'
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        class_weights = torch.tensor([1.0, 4.0, 2.0, 3.0, 4.0]).to(self.device)
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        # Verify parameters require grad
        for param in self.model.parameters():
            param.requires_grad = True
        
        self.model.eval()
        print(f"Model {self.args.model_name} loaded with {sum(p.numel() for p in self.model.parameters()):,} parameters")

    def visualize_misclassified(self, num_samples=1, save_dir='misclassified_samples'):
        """Visualize misclassified samples from the test set"""
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        
        # Store misclassified samples
        misclassified_samples = {class_name: [] for class_name in SURFACE_CLASSES}
        
        with torch.no_grad():
            for images, labels in self.test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, preds = torch.max(outputs, 1)
                
                # Find misclassified samples
                misclassified_mask = (preds != labels)
                misclassified_images = images[misclassified_mask]
                misclassified_labels = labels[misclassified_mask]
                misclassified_preds = preds[misclassified_mask]
                
                # Store samples with their true and predicted labels
                for img, true, pred in zip(misclassified_images, misclassified_labels, misclassified_preds):
                    true_class = SURFACE_CLASSES[true.item()]
                    misclassified_samples[true_class].append((img.cpu(), true.item(), pred.item()))
        
        # Visualize samples for each class
        for class_name, samples in misclassified_samples.items():
            if not samples:
                continue
                
            plt.figure(figsize=(15, min(3, num_samples)*5))
            plt.suptitle(f'Misclassified {class_name} samples ({self.args.model_name})', y=1.02)
            
            for i, (img, true, pred) in enumerate(samples[:num_samples]):
                # Denormalize the image
                img = self._denormalize(img)
                
                plt.subplot(1, num_samples, i+1)
                plt.imshow(img)
                plt.title(f'True: {SURFACE_CLASSES[true]} Pred: {SURFACE_CLASSES[pred]}')
                plt.axis('off')
            
            plt.tight_layout()
            plt.savefig(f'{save_dir}/misclassified_{class_name}_{self.args.model_name}.png')
            plt.close()
        
        # Special visualization for concrete-asphalt confusion
        concrete_asphalt_samples = [
            (img, true, pred) for img, true, pred in misclassified_samples['concrete'] 
            if SURFACE_CLASSES[pred] == 'asphalt'
        ]
        
        if concrete_asphalt_samples:
            plt.figure(figsize=(15, min(5, len(concrete_asphalt_samples))*3))
            plt.suptitle(f'Concrete misclassified as Asphalt ({self.args.model_name})', y=1.02)
            
            for i, (img, true, pred) in enumerate(concrete_asphalt_samples[:num_samples]):
                img = self._denormalize(img)
                
                plt.subplot(1, num_samples, i+1)
                plt.imshow(img)
                plt.title(f'True: concrete; Pred: asphalt')
                plt.axis('off')
            
            plt.tight_layout()
            plt.savefig(f'{save_dir}/concrete_asphalt_confusion_{self.args.model_name}.png')
            plt.close()

    def visualize_gradcam(self, num_samples=1, save_dir='gradcam_results', fontsize=15):
        """Visualize Grad-CAM for correctly and incorrectly classified samples"""
        
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        target_layers = self._get_target_layers()
        
        # Create CAM without use_cuda parameter
        cam = GradCAM(
            model=self.model,
            target_layers=target_layers
        )
        
        # visualize concrete and sett samples specifically
        for target_class in ['asphalt', 'concrete', 'paving_stones', 'sett', 'unpaved']:
            class_idx = SURFACE_CLASSES.index(target_class)
            samples_collected = 0
            
            self.model.eval()
            
            for images, labels in self.test_loader:
                if samples_collected >= num_samples:
                    break
                    
                # Filter for our target class
                target_mask = (labels == class_idx)
                if not target_mask.any():
                    continue
                    
                target_images = images[target_mask]
                target_labels = labels[target_mask]
                
                # Move to device
                images_gpu = target_images.to(self.device)
                
                # Get predictions
                with torch.no_grad():
                    outputs = self.model(images_gpu)
                    _, preds = torch.max(outputs, 1)
                
                # Process each sample
                for i in range(len(target_images)):
                    if samples_collected >= num_samples:
                        break
                        
                    img = target_images[i]
                    true_label = target_labels[i].item()
                    pred_label = preds[i].item()
                    
                    # Prepare image for visualization
                    img_np = self._denormalize(img)
                    
                    # Get Grad-CAM - specify target category
                    grayscale_cam = cam(
                        input_tensor=images_gpu[i].unsqueeze(0),
                        targets=[ClassifierOutputTarget(pred_label)]  # Focus on predicted class
                    )
                    
                    visualization = show_cam_on_image(img_np, grayscale_cam[0], use_rgb=True)
                    
                    # Plot results
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
                    ax1.imshow(img_np)
                    ax1.set_title(f'True: {SURFACE_CLASSES[true_label]}\nPred: {SURFACE_CLASSES[pred_label]}', fontsize=fontsize)
                    ax1.axis('off')
                    
                    ax2.imshow(visualization)
                    ax2.set_title(f'{self.args.model_name}\n Focus: {SURFACE_CLASSES[pred_label]}', fontsize=fontsize)
                    ax2.axis('off')
                    
                    plt.tight_layout()
                    plt.savefig(f'{save_dir}/gradcam_{target_class}_{self.args.model_name}_{samples_collected}.png')
                    plt.close()
                    
                    samples_collected += 1

    def _get_target_layers(self):
        """Helper to get target layers for Grad-CAM based on model architecture"""
        model = self.model
        if 'convnext' in self.args.model_name.lower():
            return [model.features[-1]]
        elif 'resnet' in self.args.model_name.lower():
            return [model.layer4[-1]]
        elif 'efficientnet' in self.args.model_name.lower():
            return [model.features[-1]]
        else:
            return [list(model.children())[-2]]  # Fallback for other models

    def _denormalize(self, tensor):
        """Convert normalized tensor back to image"""
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        tensor = tensor * std + mean  # Denormalize
        tensor = tensor.clamp(0, 1).permute(1, 2, 0).numpy()
        return tensor

    def test(self, confusion_matrix=True, visualization=True, run_gradcam=True):
        """Run tests and visualizations"""
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
        print(f"Test Loss: {total_loss:.4f}, Accuracy: {acc:.2f}%")

        for i, cls in enumerate(SURFACE_CLASSES):
            if class_total[i] > 0:
                cls_acc = 100. * class_correct[i] / class_total[i]
                print(f"  Class [{cls}] Accuracy: {cls_acc:.2f}% ({class_correct[i]}/{class_total[i]})")
            else:
                self.logger.warning(f"  Class [{cls}] has no samples in test set.")
        
        if confusion_matrix:
            self.confusion_matrix(all_targets, all_preds)
        if visualization:
            self.visualize_misclassified()
        if run_gradcam:
            self.visualize_gradcam()
        return total_loss, acc
    
    def confusion_matrix(self, all_targets, all_preds):
        # Log confusion matrix and classification report
        cm = confusion_matrix(all_targets, all_preds)
        print(f"\nConfusion Matrix:\n{cm}")

        report = classification_report(all_targets, all_preds, target_names=SURFACE_CLASSES)
        print(f"\nClassification Report:\n{report}")

