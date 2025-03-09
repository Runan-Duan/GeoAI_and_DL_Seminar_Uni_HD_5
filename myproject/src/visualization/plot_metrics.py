import matplotlib.pyplot as plt

def plot_metrics(metrics, metric_name, save_path=None):
    """
    Plot a metric (e.g., accuracy, IoU) over epochs.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(metrics, label=metric_name)
    plt.xlabel("Epoch")
    plt.ylabel(metric_name)
    plt.title(f"{metric_name} Over Epochs")
    plt.legend()
    if save_path:
        plt.savefig(save_path)
    plt.show()