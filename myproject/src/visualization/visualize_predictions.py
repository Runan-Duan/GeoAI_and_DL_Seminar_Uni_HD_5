import matplotlib.pyplot as plt
import torch

def visualize_classification_predictions(images, predictions, labels, class_names, save_path, num_images=5):
    """
    Visualize model predictions alongside ground truth labels.
    """
    images = images[:num_images]
    predictions = predictions[:num_images]
    labels = labels[:num_images]

    fig, axes = plt.subplots(1, num_images, figsize=(15, 3))
    for i, (image, pred, label) in enumerate(zip(images, predictions, labels)):
        image = image.permute(1, 2, 0).cpu().numpy()  # Convert to HWC format
        axes[i].imshow(image)
        axes[i].set_title(f"Pred: {class_names[pred]}\nTrue: {class_names[label]}")
        axes[i].axis("off")
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
    

def visualize_segmentation_predictions(model, test_loader, device, save_path, num_samples=5):
    """
    Visualize segmentation predictions for a few samples.
    """
    model.eval()
    images, masks, preds = [], [], []

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            if batch_idx >= num_samples:
                break
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = (output > 0.5).float()  # Threshold at 0.5 for binary segmentation

            images.append(data.cpu())
            masks.append(target.cpu())
            preds.append(pred.cpu())

    # Plot the results
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5 * num_samples))
    for i in range(num_samples):
        axes[i, 0].imshow(images[i].squeeze().permute(1, 2, 0))
        axes[i, 0].set_title("Image")
        axes[i, 1].imshow(masks[i].squeeze(), cmap="gray")
        axes[i, 1].set_title("Ground Truth")
        axes[i, 2].imshow(preds[i].squeeze(), cmap="gray")
        axes[i, 2].set_title("Prediction")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()