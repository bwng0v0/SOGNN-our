"""
Test averaged model on training data to verify it learned properly
"""
import torch
import numpy as np
from Net import SOGNN
from torch_geometric.data import DataLoader
from datapipe import get_dataset
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

def evaluate_model(model, loader, device='cpu'):
    """
    Evaluate model on given data loader
    """
    model = model.to(device)
    model.eval()

    predictions = []
    labels = []

    with torch.no_grad():
        for data in loader:
            label = data.y.view(-1, 4)  # 4 classes for SEED-IV
            data = data.to(device)
            _, pred = model(data.x, data.edge_index, data.batch)
            pred = pred.detach().cpu().numpy()

            pred = np.squeeze(pred)
            predictions.append(pred)
            labels.append(label)

    predictions = np.vstack(predictions)
    labels = np.vstack(labels)

    # Calculate metrics
    auc = roc_auc_score(labels, predictions, average='macro')
    f1 = f1_score(np.argmax(labels, axis=1), np.argmax(predictions, axis=1), average='macro')
    predictions_class = np.argmax(predictions, axis=-1)
    labels_class = np.argmax(labels, axis=-1)
    acc = accuracy_score(labels_class, predictions_class)

    return auc, acc, f1

# Load the averaged model
model_path = './result/SOGNN_averaged_13folds.pth'
print(f"Loading averaged model from {model_path}...")
checkpoint = torch.load(model_path, weights_only=False)

model = SOGNN()
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print(f"Model info:")
print(f"  - Folds averaged: {checkpoint.get('num_folds', 'unknown')}")
print(f"  - Mean Val Acc: {checkpoint.get('mean_val_acc', 'unknown')}")
print(f"  - Mean Val AUC: {checkpoint.get('mean_val_auc', 'unknown')}")

# Test on fold 0
print("\n" + "="*60)
print("Testing on Fold 0 TRAINING dataset...")
print("="*60)
train_dataset, test_dataset = get_dataset(15, 0)
train_loader = DataLoader(train_dataset, batch_size=16)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

auc, acc, f1 = evaluate_model(model, train_loader, device)

print(f"\nTraining Set Results (Fold 0):")
print(f"  AUC: {auc:.4f}")
print(f"  Accuracy: {acc:.4f}")
print(f"  F1 Score: {f1:.4f}")

print("\n" + "="*60)
print("Testing on Fold 0 TEST dataset...")
print("="*60)
test_loader = DataLoader(test_dataset, batch_size=16)

auc_test, acc_test, f1_test = evaluate_model(model, test_loader, device)

print(f"\nTest Set Results (Fold 0):")
print(f"  AUC: {auc_test:.4f}")
print(f"  Accuracy: {acc_test:.4f}")
print(f"  F1 Score: {f1_test:.4f}")

print("\n" + "="*60)
print("COMPARISON:")
print("="*60)
print(f"Training Acc: {acc:.4f} vs Test Acc: {acc_test:.4f}")
print(f"Training AUC: {auc:.4f} vs Test AUC: {auc_test:.4f}")
print(f"Training F1:  {f1:.4f} vs Test F1:  {f1_test:.4f}")
