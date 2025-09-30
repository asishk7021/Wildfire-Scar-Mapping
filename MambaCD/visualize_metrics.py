import re
import matplotlib.pyplot as plt
from pathlib import Path
import os
import json 
# Define the path to your text file
dir = Path('/mnt/MambaCD/data/results/sen2/MambaBCD_20240902115819')
os.makedirs(dir / "visualizations", exist_ok=True)
# Initialize lists to store data
epochs = []
train_losses = []

# Regular expression to parse the epoch and loss data
epoch_loss_re = re.compile(r"Epoch is (\d+), overall train loss is ([\d.]+)")

# Read the text file and extract the data
with open(dir / "training_logs.txt", 'r') as file:
    for line in file:
        match = epoch_loss_re.search(line)
        if match:
            epoch = int(match.group(1))
            train_loss = float(match.group(2))
            epochs.append(epoch)
            train_losses.append(train_loss)

# Create a plot using the extracted data
plt.figure()
plt.plot(epochs, train_losses, label='Train Loss')
plt.xlabel('Epoch')
plt.ylabel('Train Loss')
plt.title('Train Loss vs Epochs')
plt.legend()
plt.savefig(dir / 'visualizations/train_loss_vs_epochs.png')
plt.close()

metrics_list = {
    'iters':[],
    'precision': [],
    'recall': [],
    'accuracy': [],
    'f1': [],
    'iou': [],
    'meanIoU': []
}

for filename in os.listdir(dir):
    if filename.startswith("metrics_") and filename.endswith(".json"): 
        # Extract the 'I' value from the filename
        I = int(filename.split('_')[1].split('.')[0])       
        # Read the JSON file
        with open(os.path.join(dir, filename), 'r') as file:
            data = json.load(file)
            metrics_list['iters'].append(I)
            for k in metrics_list.keys():
                if (k == 'iters'):
                    continue
                if (k != 'meanIoU'):
                    metrics_list[k].append(data[k][1])
                else:
                    metrics_list[k].append(data[k])

# Sort the data based on 'iters'
sorted_metrics_list = {k: [] for k in metrics_list.keys() if k != 'iters'}
sorted_indices = sorted(range(len(metrics_list['iters'])), key=lambda i: metrics_list['iters'][i])

for i in sorted_indices:
    for k in sorted_metrics_list.keys():
        sorted_metrics_list[k].append(metrics_list[k][i])


start = min(metrics_list['iters'])
end = max(metrics_list['iters']) + 1
for k in sorted_metrics_list.keys():
    plt.plot(range(start, end, 1), sorted_metrics_list[k])
    plt.xlabel("Iteration Number")
    plt.ylabel(f"{k} per image")
    plt.savefig(Path(f"{dir}/visualizations/validation_{k}.png"))
    plt.close()