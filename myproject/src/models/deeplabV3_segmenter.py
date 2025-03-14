import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

class DeepLabV3PlusSegmenter(nn.Module):
    def __init__(self):
        super(DeepLabV3PlusSegmenter, self).__init__()
        
        # Use DeepLabV3+ with EfficientNet-B4 as the backbone
        self.model = smp.DeepLabV3Plus(
            encoder_name="efficientnet-b4",  # EfficientNet-B4 backbone
            encoder_weights="imagenet",      # Pretrained on ImageNet
            in_channels=3,                   # Input channels (RGB images)
            classes=1,                       # Output channels (binary segmentation)
            activation="sigmoid",            # Sigmoid activation for binary segmentation
        )
    
    def forward(self, x):
        return self.model(x)
    

"""DeepLabV3+ Architecture:

Uses atrous (dilated) convolutions to capture multi-scale context.

Incorporates an Atrous Spatial Pyramid Pooling (ASPP) module to aggregate features at different scales.

Combines high-level and low-level features for precise boundary detection.

EfficientNet-B4 Backbone:

EfficientNet-B4 is a lightweight and powerful backbone that balances accuracy and computational efficiency.

Pretrained on ImageNet, which provides strong feature extraction capabilities.

Binary Segmentation:

The model outputs a single channel with a sigmoid activation, making it suitable for binary segmentation tasks (e.g., road vs. non-road).
"""