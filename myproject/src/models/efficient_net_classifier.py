import torch
import torch.nn as nn
from torchvision import models

class EfficientNetClassifier(nn.Module):
    def __init__(self, num_classes):
        super(EfficientNetClassifier, self).__init__()
        self.model = models.efficientnet_b0(pretrained=True)
        self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, num_classes)

    def forward(self, x):
        return self.model(x)
    

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Aggregate channel information using average and max pooling
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        return self.sigmoid(out)
    

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio, bias=False),
            nn.ReLU(),
            nn.Linear(in_channels // reduction_ratio, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        out = avg_out + max_out
        return out.view(b, c, 1, 1)
    

class EfficientNetWithAttention(nn.Module):
    def __init__(self, num_classes, dropout_prob=0.5):
        super(EfficientNetWithAttention, self).__init__()
        # Load the pre-trained EfficientNet-B0 model
        self.efficientnet = models.efficientnet_b0(pretrained=True)
        
        # Add attention modules
        self.channel_attention = ChannelAttention(in_channels=1280)  # EfficientNet-B0's final feature map has 1280 channels
        self.spatial_attention = SpatialAttention()

        # Add batch normalization
        self.bn = nn.BatchNorm1d(self.efficientnet.classifier[1].in_features)

        # Add dropout for regularization
        self.dropout = nn.Dropout(dropout_prob)

        # Replace the final classification layer
        self.efficientnet.classifier[1] = nn.Linear(self.efficientnet.classifier[1].in_features, num_classes)
        
        # Initialize the final layer weights
        self._initialize_weights(self.efficientnet.classifier[1])

    def _initialize_weights(self, layer):
        if isinstance(layer, nn.Linear):
            nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0)

    def forward(self, x):
        # Extract features from EfficientNet
        x = self.efficientnet.features(x)
        
        # Apply channel attention
        channel_att = self.channel_attention(x)
        x = x * channel_att
        
        # Apply spatial attention
        spatial_att = self.spatial_attention(x)
        x = x * spatial_att
        
        # Global average pooling
        x = self.efficientnet.avgpool(x)
        x = torch.flatten(x, 1)
        
        # Apply batch normalization
        x = self.bn(x)
        
        # Apply dropout
        x = self.dropout(x)
        
        # Classification
        x = self.efficientnet.classifier(x)
        return x