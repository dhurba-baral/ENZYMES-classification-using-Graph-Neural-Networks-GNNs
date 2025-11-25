from dataset import ENZYMESDataset
import torch
import numpy as np
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

BATCH_SIZE = 32

#initialize dataset
DATA_ROOT = 'data'
dataset = ENZYMESDataset(root=DATA_ROOT)

num_samples = len(dataset)
print(f"Number of data samples: {num_samples}")

# Analyze first few graphs
sample_graph = dataset[0]
num_features = sample_graph.x.shape[1]
num_classes = len(set([data.y.item() for data in dataset]))
print(f"Number of features per node: {num_features}")
print(f"Number of classes: {num_classes}")

# Detailed dataset information
class_counts = {}
for data in dataset:
    label = data.y.item()
    class_counts[label] = class_counts.get(label, 0) + 1
print(f"   Class distribution: {class_counts}")

# Split dataset
indices = list(range(len(dataset)))
labels = [data.y.item() for data in dataset]

# Train/temp split (60/40)
train_idx, temp_idx = train_test_split(
    indices, test_size=0.2, random_state=42, stratify=labels
)

# Val/Test split (20/20 from remaining 40)
temp_labels = [labels[i] for i in temp_idx]
val_idx, test_idx = train_test_split(
    temp_idx, test_size=0.5, random_state=42, stratify=temp_labels
)

# Create dataloaders
train_dataset = [dataset[i] for i in train_idx]
val_dataset = [dataset[i] for i in val_idx]
test_dataset = [dataset[i] for i in test_idx]

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"\nDataset split completed:")
print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")