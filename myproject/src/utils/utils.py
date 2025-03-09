import os
import logging
import torch

def create_directories(dirs):
    """
    Create directories if they don't exist.
    """
    for dir in dirs:
        os.makedirs(dir, exist_ok=True)
        logging.info(f"Created directory: {dir}")

def save_model(model, path):
    """
    Save a PyTorch model to the specified path.
    """
    torch.save(model.state_dict(), path)
    logging.info(f"Model saved to {path}")

def load_model(model, path, device):
    """
    Load a PyTorch model from the specified path.
    """
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    logging.info(f"Model loaded from {path}")
    return model