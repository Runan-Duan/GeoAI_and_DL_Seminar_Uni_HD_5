import re
from collections import defaultdict

def parse_log_file(file_path):
    epoch_metrics = []
    class_accuracy_by_epoch = defaultdict(dict)
    confusion_matrices = {}
    
    # New test metrics containers
    test_metrics = {}
    test_class_accuracy = {}
    test_confusion_matrix = None
    test_classification_report_lines = []

    in_test_block = False
    in_test_confusion = False
    in_test_report = False

    with open(file_path, 'r') as f:
        lines = f.readlines()

    i = 0
    current_epoch = None
    while i < len(lines):
        line = lines[i].strip()

        # Check for epoch line in the training / validation block.
        if match := re.search(r'\[Epoch (\d+)\]', line):
            current_epoch = int(match.group(1))

        # Training & Validation metrics
        if 'Train Loss' in line:
            train_acc = float(re.search(r'Accuracy: ([\d.]+)%', line).group(1))
        elif 'Validation Loss' in line:
            val_acc = float(re.search(r'Accuracy: ([\d.]+)%', line).group(1))
            epoch_metrics.append({'epoch': current_epoch, 'train_acc': train_acc, 'val_acc': val_acc})
        
        # Validation per-class accuracy
        if not in_test_block and 'Class [' in line:
            # Only pick these if we are not in the test section.
            match = re.search(r'Class \[(.*?)\] Accuracy: ([\d.]+)%', line)
            if match:
                class_name, acc = match.groups()
                class_accuracy_by_epoch[current_epoch][class_name] = float(acc)
        
        # Start of Test block
        if 'Test Loss' in line:
            in_test_block = True
            test_loss_match = re.search(r'Test Loss: ([\d.]+)', line)
            test_acc_match = re.search(r'Accuracy: ([\d.]+)%', line)
            if test_loss_match and test_acc_match:
                test_metrics = {
                    'test_loss': float(test_loss_match.group(1)),
                    'test_acc': float(test_acc_match.group(1))
                }
        
        # Parse test per-class accuracy
        if in_test_block and 'Class [' in line and 'Test Loss' not in line:
            match = re.search(r'Class \[(.*?)\] Accuracy: ([\d.]+)%', line)
            if match:
                class_name, acc = match.groups()
                test_class_accuracy[class_name] = float(acc)
        
        # Test Confusion Matrix block
        if in_test_block and 'Confusion Matrix (Test):' in line:
            test_confusion_matrix = []
            # Assume matrix spans next 5 lines:
            for j in range(i+1, i+6):
                if j < len(lines):
                    row = list(map(int, lines[j].strip().strip('[]').split()))
                    test_confusion_matrix.append(row)
            i += 5  # skip over the matrix rows
        
        # Test Classification Report block
        if in_test_block and 'Classification Report (Test):' in line:
            in_test_report = True
            # Skip header line if present (the next line might be blank or have column titles)
            i += 1
            # Collect all subsequent lines until a blank line or end-of-section:
            while i < len(lines) and lines[i].strip() != "":
                test_classification_report_lines.append(lines[i].rstrip())
                i += 1
            in_test_report = False
            # End of test block can be assumed after classification report
            in_test_block = False

        i += 1

    test_classification_report = "\n".join(test_classification_report_lines)

    return epoch_metrics, class_accuracy_by_epoch, confusion_matrices, test_metrics, test_class_accuracy, test_confusion_matrix, test_classification_report
