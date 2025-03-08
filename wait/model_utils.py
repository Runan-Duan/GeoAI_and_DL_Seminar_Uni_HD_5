import torch
import logging

def initialize_model(model_name, num_classes, pretrained=True):
    """
    Initialize a pre-trained model.
    """
    if model_name == "efficientnet":
        model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_efficientnet_b0', pretrained=pretrained)
        model.classifier.fc = torch.nn.Linear(model.classifier.fc.in_features, num_classes)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    logging.info(f"Initialized {model_name} with {num_classes} classes")
    return model