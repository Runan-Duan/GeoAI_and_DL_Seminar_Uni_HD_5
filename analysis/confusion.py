import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_confusion_matrices(conf_matrices, model_names, classes):
    fig, axes = plt.subplots(1, 3, figsize=(24, 6))
    
    for ax, cm, name in zip(axes, conf_matrices, model_names):
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=classes, yticklabels=classes,
                    cbar=False, ax=ax)
        ax.set_title(f'{name}\nAccuracy: {np.trace(cm)/np.sum(cm):.2%}')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
    
    plt.tight_layout()
    plt.savefig('confusion_matrices_comparison.png')
    plt.show()

# Your confusion matrices
convnext_cm = np.array([
    [721, 17, 2, 2, 5],
    [27, 164, 1, 0, 2],
    [0, 3, 395, 10, 0],
    [1, 0, 3, 269, 0],
    [13, 2, 0, 0, 188]
])

efficientnet_cm = np.array([
    [696, 38, 2, 2, 9],
    [29, 162, 1, 1, 1],
    [2, 3, 396, 7, 0],
    [1, 0, 5, 266, 1],
    [6, 4, 0, 1, 192]
])

resnet_cm = np.array([
    [719, 20, 1, 1, 6],
    [25, 164, 0, 0, 5],
    [4, 5, 389, 10, 0],
    [1, 0, 6, 264, 2],
    [8, 2, 0, 1, 192]
])

class_names = ['asphalt', 'concrete', 'paving_stones', 'sett', 'unpaved']
plot_confusion_matrices([convnext_cm, efficientnet_cm, resnet_cm],
                       ['ConvNeXt', 'EfficientNet', 'ResNet50'],
                       class_names)


def plot_misclassification_patterns(conf_matrices, model_names, class_names):
    concrete_idx = class_names.index('concrete')
    sett_idx = class_names.index('sett')
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
    
    # Concrete misclassification patterns
    concrete_errors = []
    for cm in conf_matrices:
        total = cm[concrete_idx].sum()
        correct = cm[concrete_idx, concrete_idx]
        errors = {class_names[i]: cm[concrete_idx, i] 
                 for i in range(len(class_names)) if i != concrete_idx}
        concrete_errors.append(errors)
    
    width = 0.25
    x = np.arange(len(class_names)-1)
    for i, (errors, name) in enumerate(zip(concrete_errors, model_names)):
        ax1.bar(x + i*width, errors.values(), width, label=name)
    
    ax1.set_title('Concrete Misclassifications')
    ax1.set_xticks(x + width)
    ax1.set_xticklabels([n for n in class_names if n != 'concrete'])
    ax1.legend()
    
    # Sett classification performance
    sett_acc = [cm[sett_idx, sett_idx]/cm[sett_idx].sum() for cm in conf_matrices]
    ax2.bar(model_names, sett_acc, color='green')
    ax2.set_title('Sett Classification Accuracy')
    ax2.set_ylim(0.9, 1.0)
    
    plt.tight_layout()
    plt.savefig('misclassification_patterns.png')
    plt.show()

plot_misclassification_patterns([convnext_cm, efficientnet_cm, resnet_cm],
                              ['ConvNeXt', 'EfficientNet', 'ResNet50'],
                              class_names)


def plot_error_heatmap(conf_matrices, model_names, class_names):
    # Calculate error rates (false negatives)
    error_rates = []
    for cm in conf_matrices:
        rates = []
        for i in range(len(class_names)):
            total = cm[i].sum()
            correct = cm[i,i]
            rates.append((total - correct) / total)
        error_rates.append(rates)
    
    plt.figure(figsize=(10, 6))
    sns.heatmap(error_rates, annot=True, fmt='.2%', cmap='Reds',
                xticklabels=class_names, yticklabels=model_names)
    plt.title('Class-wise Error Rates Across Models')
    plt.xlabel('Class')
    plt.ylabel('Model')
    plt.savefig('error_rates_heatmap.png')
    plt.show()

plot_error_heatmap([convnext_cm, efficientnet_cm, resnet_cm],
                  ['ConvNeXt', 'EfficientNet', 'ResNet50'],
                  class_names)