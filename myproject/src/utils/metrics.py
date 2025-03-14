import torch

def accuracy(output, target):
    """
    Compute accuracy for a single batch of predictions and targets
    Different from accuracy in the validation function, which computes accuracy
    over the entire validation dataset
    """
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        correct = (pred == target).float().sum()
        return correct / target.shape[0]
    

def dice_score(y_true, y_pred, smooth=1e-6):
    """
    Compute the Dice Score for binary segmentation.
    """
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    intersection = (y_true * y_pred).sum()
    return (2.0 * intersection + smooth) / (y_true.sum() + y_pred.sum() + smooth)