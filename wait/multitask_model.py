import torch
import torch.nn as nn
import timm
import segmentation_models_pytorch as smp

class MultiTaskModel(nn.Module):
    def __init__(self, num_classes_classification, num_classes_segmentation, pretrained=True):
        super(MultiTaskModel, self).__init__()

        # Load the EfficientNet backbone
        self.encoder = timm.create_model('efficientnet_b0', pretrained=pretrained, features_only=True)

        # Surface classification head (fully connected)
        in_features = self.encoder.feature_info[-1]['num_chs']
        self.classification_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), 
            nn.Flatten(),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes_classification)
        )

        # Segmentation head (using convolution layers)
        self.segmentation_head = smp.Unet(
            encoder_name="efficientnet-b0",
            encoder_weights="imagenet" if pretrained else None,
            in_channels=3,
            classes=num_classes_segmentation,
        )

    def forward(self, x):
        # Shared backbone (EfficientNet)
        features = self.encoder(x)

        # Surface classification (use features from the last layer of EfficientNet)
        classification_output = self.classification_head(features[-1])

        # Segmentation output
        segmentation_output = self.segmentation_head(x)

        return classification_output, segmentation_output



import torch
import torch.nn as nn
from models.efficient_net_classifier import EfficientNetClassifier
from models.unet_segmentation import UNet

class MultiTaskModel(nn.Module):
    def __init__(self, num_classes, segmentation_classes):
        super(MultiTaskModel, self).__init__()
        self.backbone = EfficientNetClassifier(num_classes=num_classes)
        self.segmentation_head = UNet(n_classes=segmentation_classes)
        
    def forward(self, x):
        classification_output = self.backbone(x)
        segmentation_output = self.segmentation_head(x)
        return classification_output, segmentation_output