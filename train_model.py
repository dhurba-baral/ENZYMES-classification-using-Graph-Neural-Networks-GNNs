import torch
import os
import torch.optim as optim
from models import GCN_1Layer, GCN_2Layer, GAT_Model
from train_functions import train_model
from data_loaders import BATCH_SIZE
from data_loaders import train_loader, val_loader, num_features, num_classes

#reproducibility
torch.manual_seed(42)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nUsing device: {device}")

#make directory to save models, plots
os.makedirs('models', exist_ok=True)
os.makedirs('plots', exist_ok=True)

#hyperparameters
NUM_EPOCHS = 100
LEARNING_RATE = 0.01
HIDDEN_DIM = 64
DROPOUT = 0.5

#loss function
criterion = torch.nn.CrossEntropyLoss()

#path to save model
model1_path = f'models/GCN1_ENZYMES.pth'

#train GCN1 model
model1 = GCN_1Layer(num_features, HIDDEN_DIM, num_classes, DROPOUT)
optimizer = optim.Adam(model1.parameters(), lr=LEARNING_RATE, weight_decay=5e-4)
trained_model, train_losses, val_losses, train_accuracies, val_accuracies, train_f1_scores, val_f1_scores, train_auc_scores, val_auc_scores = train_model(
            model1, train_loader, val_loader, optimizer, criterion,
            device, NUM_EPOCHS, 'GCN1', num_classes
        )
torch.save(trained_model.state_dict(), model_path)
