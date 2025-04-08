import os
import pandas as pd
import matplotlib.pyplot as plt
from parser import parse_log_file

log_dir = 'logs/'
output_dir = 'output/'
metrics_dir = os.path.join(output_dir, 'metrics')
class_acc_dir = os.path.join(output_dir, 'class_accuracy')
test_metrics_dir = os.path.join(output_dir, 'test_metrics')
os.makedirs(output_dir, exist_ok=True)
os.makedirs(metrics_dir, exist_ok=True)
os.makedirs(class_acc_dir, exist_ok=True)
os.makedirs(test_metrics_dir, exist_ok=True)

model_results = {}

# Process all log files
for filename in os.listdir(log_dir):
    if filename.endswith(".log"):
        model_name = filename.replace(".log", "")
        file_path = os.path.join(log_dir, filename)

        (epoch_metrics, class_accuracy_by_epoch, confusion_matrices,
         test_metrics, test_class_accuracy, test_confusion_matrix,
         test_classification_report) = parse_log_file(file_path)

        # Convert training/validation metrics and per-class accuracy to dataframes
        df_metrics = pd.DataFrame(epoch_metrics)
        df_class_acc = pd.DataFrame.from_dict(class_accuracy_by_epoch, orient='index')

        # Create dataframe for test metrics (single row)
        df_test_metrics = pd.DataFrame([test_metrics])
        df_test_class_acc = pd.DataFrame(list(test_class_accuracy.items()), columns=['class', 'accuracy'])
        
        # Save CSVs
        df_metrics.to_csv(os.path.join(metrics_dir, f"{model_name}_metrics.csv"), index=False)
        df_class_acc.to_csv(os.path.join(class_acc_dir, f"{model_name}_class_accuracy.csv"))
        df_test_metrics.to_csv(os.path.join(test_metrics_dir, f"{model_name}_test_metrics.csv"), index=False)
        df_test_class_acc.to_csv(os.path.join(test_metrics_dir, f"{model_name}_test_class_accuracy.csv"), index=False)
        
        # Save test confusion matrix as CSV if available
        if test_confusion_matrix is not None:
            pd.DataFrame(test_confusion_matrix).to_csv(os.path.join(test_metrics_dir, f"{model_name}_test_confusion_matrix.csv"), index=False)
        
        # Save the test classification report as a text file
        with open(os.path.join(test_metrics_dir, f"{model_name}_test_classification_report.txt"), "w") as f:
            f.write(test_classification_report)
        
        model_results[model_name] = {
            'metrics': df_metrics,
            'class_acc': df_class_acc,
            'test_metrics': df_test_metrics,
            'test_class_acc': df_test_class_acc
        }

# Plot comparison of final validation accuracy
plt.figure(figsize=(10, 6))
for model, data in model_results.items():
    df = data['metrics']
    plt.plot(df['epoch'], df['val_acc'], marker='o', label=model)
plt.title("Validation Accuracy Comparison")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "val_accuracy_comparison.png"))
plt.show()

# Optionally, plot test accuracy as bar plot
plt.figure(figsize=(8, 6))
models = []
test_accs = []
for model, data in model_results.items():
    models.append(model)
    # Extract test accuracy from the df_test_metrics dataframe
    test_accs.append(data['test_metrics'].iloc[0]['test_acc'])
plt.bar(models, test_accs, color='skyblue')
plt.title("Test Accuracy Comparison")
plt.xlabel("Model")
plt.ylabel("Test Accuracy (%)")
plt.ylim(0, 100)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "test_accuracy_comparison.png"))
plt.show()
