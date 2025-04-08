import torchvision.models as models
import torch.nn as nn
from torchvision.models import (
    ResNet50_Weights,
    EfficientNet_B4_Weights,
    ConvNeXt_Small_Weights,
    ConvNeXt_Large_Weights,
    ViT_B_16_Weights,
    Swin_B_Weights
)

def get_model(model_name: str, num_classes: int):
    # Common classifier replacement function
    def replace_classifier(model, num_classes):
        if hasattr(model, 'classifier'):
            if isinstance(model.classifier, nn.Sequential):
                in_features = model.classifier[-1].in_features
                model.classifier[-1] = nn.Linear(in_features, num_classes)
            else:
                in_features = model.classifier.in_features
                model.classifier = nn.Linear(in_features, num_classes)
        elif hasattr(model, 'fc'):  # For ResNet
            in_features = model.fc.in_features
            model.fc = nn.Linear(in_features, num_classes)
        elif hasattr(model, 'heads'):  # For ViT/Swin
            in_features = model.heads.head.in_features
            model.heads.head = nn.Linear(in_features, num_classes)
        return model
    
    # Model selection
    if model_name == 'resnet50':
        model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        model = replace_classifier(model, num_classes)
        
    elif model_name == 'efficientnet_b4':
        model = models.efficientnet_b4(weights=EfficientNet_B4_Weights.IMAGENET1K_V1)
        model = replace_classifier(model, num_classes)
        
    elif model_name == 'convnext_small':
        model = models.convnext_small(weights=ConvNeXt_Small_Weights.IMAGENET1K_V1)
        model = replace_classifier(model, num_classes)
        
    elif model_name == 'vit_b16':
        model = models.vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        model = replace_classifier(model, num_classes)
        
    elif model_name == 'swin_b':
        model = models.swin_b(weights=Swin_B_Weights.IMAGENET1K_V1)
        model = replace_classifier(model, num_classes)
        
    elif model_name == 'convnext_large':
        model = models.convnext_large(weights=ConvNeXt_Large_Weights.IMAGENET1K_V1)
        model = replace_classifier(model, num_classes)
        
    else:
        raise ValueError(f"Unknown model: {model_name}. Available: [resnet50, efficientnet_b4, convnext_small, vit_b16, swin_b, convnext_large]")
    
    return model