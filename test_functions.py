import torch
from models import GCN_1Layer, GCN_2Layer, GAT_Model
from evaluate import evaluate
from data_loaders import num_features, num_classes

#hidden dimension for models
HIDDEN_DIM = 64

#loss function
criterion = torch.nn.CrossEntropyLoss()

#test the trained model
def test_model(model_path, test_loader, device, model_type):
    # Initialize model
    if model_type == 'GCN1':
        model = GCN_1Layer(num_features=num_features, hidden_channels=HIDDEN_DIM, num_classes=num_classes)
    elif model_type == 'GCN2':
        model = GCN_2Layer(num_features=num_features, hidden_channels=HIDDEN_DIM, num_classes=num_classes)
    elif model_type == 'GAT':
        model = GAT_Model(num_features=num_features, hidden_channels=HIDDEN_DIM, num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Load model weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)

    # Evaluate
    accuracy, f1, auc, val_loss, labels, probs = evaluate(model=model, loader=test_loader, device=device, criterion=criterion)

    print(f"\nTest Results for {model_type}:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"AUC: {auc:.4f}")

    return accuracy, f1, auc, labels, probs