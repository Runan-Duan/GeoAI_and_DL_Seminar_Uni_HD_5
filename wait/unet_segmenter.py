import torch
import torch.nn as nn

class UNetSegmenter(nn.Module):
    def __init__(self, num_classes):
        super(UNetSegmenter, self).__init__()
        # Define your U-Net architecture here
        ...

    def forward(self, x):
        # Forward pass logic
        ...
        return x

def load_segmentation_model(model_path, device):
    model = UNetSegmenter(num_classes=3)  # Adjust num_classes as needed
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model