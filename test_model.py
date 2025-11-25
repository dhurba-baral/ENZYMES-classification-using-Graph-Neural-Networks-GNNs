import torch
from test_functions import test_model
from data_loaders import test_loader, num_classes
from plot_functions import plot_roc_curves

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# test GCN 1-Layer model and plot ROC curves
accuracy, f1, auc, labels, probs = test_model(model_path='models/best_GCN1_model.pth', test_loader=test_loader, device=device, model_type='GCN1')
plot_roc_curves(num_classes=num_classes, model_name='GCN1', labels=labels, probs=probs)

# # test GCN 2-Layer model and plot ROC curves
# accuracy, f1, auc, labels, probs = test_model(model_path='models/best_GCN2_model.pth', test_loader=test_loader, device=device, model_type='GCN2')
# plot_roc_curves(num_classes=num_classes, model_name='GCN2', labels=labels, probs=probs)

# # test GAT model and plot ROC curves
# accuracy, f1, auc, labels, probs = test_model(model_path='models/best_GAT_model.pth', test_loader=test_loader, device=device, model_type='GAT')
# plot_roc_curves(num_classes=num_classes, model_name='GAT', labels=labels, probs=probs)