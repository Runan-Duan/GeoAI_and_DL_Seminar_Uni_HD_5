import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import re

# ---- Directories ----
OUTPUT_DIR = 'output'
METRICS_DIR = os.path.join(OUTPUT_DIR, 'metrics')
CLASS_ACC_DIR = os.path.join(OUTPUT_DIR, 'class_accuracy')
TEST_METRICS_DIR = os.path.join(OUTPUT_DIR, 'test_metrics')
PLOT_DIR = os.path.join(OUTPUT_DIR, 'plots')
os.makedirs(PLOT_DIR, exist_ok=True)

# ---- Load All Model Runs ----
model_results = {}

for file in os.listdir(METRICS_DIR):
    if file.endswith('_metrics.csv'):
        run_id = file.replace('_metrics.csv', '')
        match = re.match(r'(.+?)_\d{8}_\d{6}', run_id)
        if not match:
            continue
        model_name = match.group(1)

        try:
            paths = {
                'metrics': os.path.join(METRICS_DIR, f'{run_id}_metrics.csv'),
                'class_acc': os.path.join(CLASS_ACC_DIR, f'{run_id}_class_accuracy.csv'),
                'test_metrics': os.path.join(TEST_METRICS_DIR, f'{run_id}_test_metrics.csv'),
                'test_class_acc': os.path.join(TEST_METRICS_DIR, f'{run_id}_test_class_accuracy.csv'),
                'test_conf_matrix': os.path.join(TEST_METRICS_DIR, f'{run_id}_test_confusion_matrix.csv'),
            }

            model_results[run_id] = {
                'label': model_name,
                'metrics': pd.read_csv(paths['metrics']),
                'class_acc': pd.read_csv(paths['class_acc'], index_col=0),
                'test_metrics': pd.read_csv(paths['test_metrics']),
                'test_class_acc': pd.read_csv(paths['test_class_acc']),
                'test_conf_matrix': pd.read_csv(paths['test_conf_matrix'], header=None)
            }

        except Exception as e:
            print(f"Failed to load files for {run_id}: {e}")
            continue

# ---- 1) Plot: Validation Accuracy Per Class Across 50 Epochs for Each Model ----
for run_id, data in model_results.items():
    df = data['class_acc']
    plt.figure(figsize=(10, 6))
    for class_name in df.columns:
        plt.plot(df.index, df[class_name], marker='o', label=class_name)

    plt.title(f"Validation Accuracy Per Class Across Epochs - {data['label']}")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend(loc='lower right', bbox_to_anchor=(1, 0))  # Adjusted to bottom-right
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f"{data['label']}_val_accuracy_per_class.png"))
    plt.close()

# ---- 2) Plot: Validation Accuracy Across Three Models for Each Class ----
class_names = ['asphalt', 'concrete', 'paving_stones', 'sett', 'unpaved']
for class_name in class_names:
    plt.figure(figsize=(10, 6))
    for run_id, data in model_results.items():
        plt.plot(data['class_acc'].index, data['class_acc'][class_name], marker='o', label=data['label'])

    plt.title(f"Validation Accuracy for Class {class_name} Across Models")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend(loc='lower right', bbox_to_anchor=(1, 0))  # Adjusted to bottom-right
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f"val_accuracy_class_{class_name}_comparison.png"))
    plt.close()

# ---- 3) Plot: Train and Validation Accuracy for Each Model Over 50 Epochs ----
for run_id, data in model_results.items():
    df = data['metrics']
    plt.figure(figsize=(10, 6))
    plt.plot(df['epoch'], df['train_acc'], label='Train Accuracy', marker='o')
    plt.plot(df['epoch'], df['val_acc'], label='Validation Accuracy', marker='o')
    plt.title(f"Train and Validation Accuracy Over 50 Epochs - {data['label']}")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend(loc='lower right', bbox_to_anchor=(1, 0))  # Adjusted to bottom-right
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f"{data['label']}_train_val_accuracy.png"))
    plt.close()

# ---- 4) Plot: Validation Accuracy Across Three Models Over 50 Epochs ----
plt.figure(figsize=(10, 6))
for run_id, data in model_results.items():
    plt.plot(data['metrics']['epoch'], data['metrics']['val_acc'], marker='o', label=data['label'])

plt.title("Validation Accuracy Across Three Models Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.legend(loc='lower right', bbox_to_anchor=(1, 0))  # Adjusted to bottom-right
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "val_accuracy_comparison.png"))
plt.close()

# ---- 5) Plot: Test Accuracy per Class Across Three Models ----
test_accuracies = {class_name: [] for class_name in ['asphalt', 'concrete', 'paving_stones', 'sett', 'unpaved']}

# Collect test accuracies for each class and model
for run_id, data in model_results.items():
    test_acc_df = data['test_class_acc']
    
    for class_name in test_accuracies:
        # Find the accuracy for the current class and append it to the list
        test_accuracies[class_name].append(test_acc_df.loc[test_acc_df['class'] == class_name, 'accuracy'].values[0])

# Now create the bar plot comparing the test accuracy per class across models
fig, ax = plt.subplots(figsize=(14, 8))

bar_width = 0.25  # Width of each bar
num_classes = len(test_accuracies)
num_models = len(model_results)
index = np.arange(num_classes)  # Positions for class groups

# Define model labels and colors
model_labels = [model_results[run_id]['label'] for run_id in model_results]
colors = plt.cm.tab10.colors[:num_models]  # Different colors for each model

# Create bars for each model, grouped by class
for model_idx in range(num_models):
    # Get accuracies for current model across all classes
    model_acc = [test_accuracies[class_name][model_idx] for class_name in test_accuracies]
    
    # Position bars for this model with appropriate offset
    bar_positions = index + model_idx * bar_width
    ax.bar(bar_positions, model_acc, bar_width, 
           label=model_labels[model_idx], 
           color=colors[model_idx])

# Set the labels and title
ax.set_xlabel("Road Surface Class")
ax.set_ylabel("Test Accuracy (%)")
ax.set_title("Test Accuracy per Class Across Different Models")

# Configure x-axis
ax.set_xticks(index + bar_width * (num_models - 1) / 2)  # Center ticks between model bars
ax.set_xticklabels(['Asphalt', 'Concrete', 'Paving Stones', 'Sett', 'Unpaved'], rotation=0)

# Add grid and legend
ax.grid(True, axis='y', linestyle='--', alpha=0.7)
ax.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')

# Set y-axis limits for better visualization
ax.set_ylim(0, 100)

# Add tight layout and save the plot
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "test_accuracy_per_class_comparison.png"))
plt.close()


# ---- 6) Plot: Confusion Matrix for Each Model ----
for run_id, data in model_results.items():
    cm = data['test_conf_matrix'].values
    labels = ['asphalt', 'concrete', 'paving_stones', 'sett', 'unpaved']

    # Remove the first row (0,1,2,3,4) which should not be part of the confusion matrix
    cm = cm[1:, :]  # Skip the first row which contains 0,1,2,3,4

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.title(f"Confusion Matrix (Test) - {data['label']}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f"{data['label']}_confusion_matrix.png"))
    plt.close()

