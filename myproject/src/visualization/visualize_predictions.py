import matplotlib.pyplot as plt
import torch

def visualize_predictions(images, predictions, labels, class_names, save_path, num_images=5):
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