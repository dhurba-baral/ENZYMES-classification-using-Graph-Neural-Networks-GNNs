import torch
import os
import numpy as np
from torch import optim
from models import GCN_1Layer, GCN_2Layer, GAT_Model
from train_functions import train_model
from data_loaders import train_loader, val_loader, num_features, num_classes
from plot_functions import plot_loss, plot_accuracies

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

#hyperparameters
HIDDEN_DIM = 64
LEARNING_RATE = 0.01
NUM_EPOCHS = 100
DROPOUT = 0.5

#loss function
criterion = torch.nn.CrossEntropyLoss()

#train GCN1 model
model1 = GCN_1Layer(num_features=num_features, hidden_channels=HIDDEN_DIM, num_classes=num_classes, dropout=DROPOUT)
optimizer = optim.Adam(model1.parameters(), lr=LEARNING_RATE, weight_decay=5e-4)

train_losses, val_losses, train_accuracies, val_accuracies, train_f1_scores, val_f1_scores, train_auc_scores, val_auc_scores = train_model(
            model1, train_loader, val_loader, optimizer, criterion,
            device, NUM_EPOCHS, 'GCN1', num_classes
        )
plot_loss(train_losses=train_losses, val_losses=val_losses, model_name='GCN1')
plot_accuracies(train_accuracies=train_accuracies, val_accuracies=val_accuracies, model_name='GCN1')

# #train GCN2 model
# model2 = GCN_2Layer(num_features=num_features, hidden_channels=HIDDEN_DIM, num_classes=num_classes, dropout=DROPOUT)
# optimizer = optim.Adam(model2.parameters(), lr=LEARNING_RATE, weight_decay=5e-4)

# train_losses, val_losses, train_accuracies, val_accuracies, train_f1_scores, val_f1_scores, train_auc_scores, val_auc_scores = train_model(
#             model2, train_loader, val_loader, optimizer, criterion,
#             device, NUM_EPOCHS, 'GCN2', num_classes
#         )
# plot_loss(train_losses=train_losses, val_losses=val_losses, model_name='GCN2')
# plot_accuracies(train_accuracies=train_accuracies, val_accuracies=val_accuracies, model_name='GCN2')

# #train GAT model
# model3 = GAT_Model(num_features=num_features, hidden_channels=HIDDEN_DIM, num_classes=num_classes, heads=4, dropout=DROPOUT)
# optimizer = optim.Adam(model3.parameters(), lr=LEARNING_RATE, weight_decay=5e-4)

# train_losses, val_losses, train_accuracies, val_accuracies, train_f1_scores, val_f1_scores, train_auc_scores, val_auc_scores = train_model(
#             model3, train_loader, val_loader, optimizer, criterion,
#             device, NUM_EPOCHS, 'GAT', num_classes
#         )
# plot_loss(train_losses=train_losses, val_losses=val_losses, model_name='GAT')
# plot_accuracies(train_accuracies=train_accuracies, val_accuracies=val_accuracies, model_name='GAT')