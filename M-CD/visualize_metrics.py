import matplotlib.pyplot as plt
import re
import math
from pathlib import Path

def parse_log_file(log_file):
    epochs = []
    train_losses = []
    validation_accs = []
    precisions = []
    recalls = []
    f1_scores = []
    mean_ious = []
    ious = []
    
    def parse_value(value):
        return float(value) if value.lower() != 'nan' else float('nan')

    with open(log_file, 'r') as file:
        for line in file:
            if '**********Test Metrics' in line:
                break
            if 'Epoch:' in line and 'train_loss:' in line:
                epoch = int(re.search(r'Epoch: (\d+)', line).group(1))
                train_loss = float(re.search(r'train_loss: ([\d\.]+)', line).group(1))
                epochs.append(epoch)
                train_losses.append(train_loss)
            elif 'Validation acc' in line:
                validation_acc = parse_value(re.search(r'Validation acc = \([\d\.]+, ([\d\.nNa]+)\)%', line).group(1))
                validation_accs.append(validation_acc)
            elif 'Precision =' in line:
                precision = parse_value(re.search(r'Precision = \([\d\.]+, ([\d\.nNa]+)\)', line).group(1))
                precisions.append(precision)
            elif 'Recall =' in line:
                recall = parse_value(re.search(r'Recall = \([\d\.]+, ([\d\.nNa]+)\)', line).group(1))
                recalls.append(recall)
            elif 'F1 Score =' in line:
                f1_score = parse_value(re.search(r'F1 Score = \([\d\.]+, ([\d\.nNa]+)\)', line).group(1))
                f1_scores.append(f1_score)
            elif 'MeanIoU =' in line:
                mean_iou = parse_value(re.search(r'MeanIoU = ([\d\.nNa]+)', line).group(1))
                mean_ious.append(mean_iou)
            elif 'iou =' in line:
                iou = parse_value(re.search(r'iou = \([\d\.]+, ([\d\.nNa]+)\)', line).group(1))
                ious.append(iou)

    return epochs, train_losses, validation_accs, precisions, recalls, f1_scores, mean_ious, ious

def plot_and_save(epochs, values, metric_name, save_dir):
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, values)
    plt.title(f'{metric_name}')
    plt.xlabel('Epoch')
    plt.ylabel(metric_name)
    plt.tight_layout()
    plt.savefig(save_dir / f'{metric_name}.png')
    plt.close()

def plot_metrics(epochs, train_losses, validation_accs, precisions, recalls, f1_scores, mean_ious, ious, save_dir):
    plot_and_save(epochs, train_losses, 'Training Loss', save_dir)
    plot_and_save(epochs, validation_accs, 'Validation Accuracy', save_dir)
    plot_and_save(epochs, precisions, 'Precision', save_dir)
    plot_and_save(epochs, recalls, 'Recall', save_dir)
    plot_and_save(epochs, f1_scores, 'F1 Score', save_dir)
    plot_and_save(epochs, mean_ious, 'MeanIoU', save_dir)
    plot_and_save(epochs, ious, 'IoU', save_dir)

if __name__ == "__main__":
    log_file_path = Path('./data/results/log_FLOGA_sigma_small_MambaDecoder/2024_09_04_14_26_49/training.log')
    epochs, train_losses, validation_accs, precisions, recalls, f1_scores, mean_ious, ious = parse_log_file(log_file_path)
    save_dir = log_file_path.parent / 'visualizations'
    save_dir.mkdir(exist_ok=True, parents=True)
    plot_metrics(epochs, train_losses, validation_accs, precisions, recalls, f1_scores, mean_ious, ious, save_dir)
