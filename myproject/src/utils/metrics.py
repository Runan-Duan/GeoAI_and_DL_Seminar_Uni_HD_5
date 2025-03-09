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

def iou_score(output, target):
    """
    Compute Intersection over Union (IoU) for segmentation tasks.
    """
    with torch.no_grad():
        output = torch.argmax(output, dim=1)
        intersection = (output & target).float().sum((1, 2))
        union = (output | target).float().sum((1, 2))
        iou = (intersection + 1e-6) / (union + 1e-6)
        return iou.mean()