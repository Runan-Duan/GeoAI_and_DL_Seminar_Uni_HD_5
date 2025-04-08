import matplotlib.pyplot as plt

def plot_loss(train_loss, val_loss, save_path=None):
    """
    Plot training and validation loss over epochs.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(train_loss, label="Train Loss")
    plt.plot(val_loss, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    if save_path:
        plt.savefig(save_path)
    plt.show()