"""
Test the best fold model to verify it works correctly
"""
import torch
import numpy as np
from Net import SOGNN
from torch_geometric.data import DataLoader
from datapipe import get_dataset
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

def evaluate_model(model, loader, device='cpu'):
    model = model.to(device)
    model.eval()

    predictions = []
    labels = []

    with torch.no_grad():
        for data in loader:
            label = data.y.view(-1, 4)
            data = data.to(device)
            _, pred = model(data.x, data.edge_index, data.batch)
            pred = pred.detach().cpu().numpy()

            pred = np.squeeze(pred)
            predictions.append(pred)
            labels.append(label)

    predictions = np.vstack(predictions)
    labels = np.vstack(labels)

    auc = roc_auc_score(labels, predictions, average='macro')
    f1 = f1_score(np.argmax(labels, axis=1), np.argmax(predictions, axis=1), average='macro')
    predictions_class = np.argmax(predictions, axis=-1)
    labels_class = np.argmax(labels, axis=-1)
    acc = accuracy_score(labels_class, predictions_class)

    return auc, acc, f1

# Load the best fold model
model_path = './result/SOGNN_best_fold12.pth'
print(f"Loading best fold model from {model_path}...")
checkpoint = torch.load(model_path, weights_only=False)

model = SOGNN()
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print(f"\nModel info:")
print(f"  - Fold: {checkpoint.get('fold', 'unknown')}")
print(f"  - Training Val Acc: {checkpoint.get('val_acc', 'unknown'):.4f}")
print(f"  - Training Val AUC: {checkpoint.get('val_auc', 'unknown'):.4f}")

# Test on fold 12's train and test sets
print("\n" + "="*60)
print("Testing on Fold 12 datasets...")
print("="*60)

train_dataset, test_dataset = get_dataset(15, 12)
train_loader = DataLoader(train_dataset, batch_size=16)
test_loader = DataLoader(test_dataset, batch_size=16)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}\n")

# Test on training set
train_auc, train_acc, train_f1 = evaluate_model(model, train_loader, device)
print(f"Training Set Performance:")
print(f"  AUC: {train_auc:.4f}")
print(f"  Accuracy: {train_acc:.4f}")
print(f"  F1 Score: {train_f1:.4f}")

# Test on test set
test_auc, test_acc, test_f1 = evaluate_model(model, test_loader, device)
print(f"\nTest Set Performance:")
print(f"  AUC: {test_auc:.4f}")
print(f"  Accuracy: {test_acc:.4f}")
print(f"  F1 Score: {test_f1:.4f}")

print("\n" + "="*60)
print("Verification:")
print("="*60)
if test_acc > 0.5:
    print("✓ Model works correctly! Test accuracy is significantly above random (25%)")
else:
    print("✗ Warning: Model performance is poor")
