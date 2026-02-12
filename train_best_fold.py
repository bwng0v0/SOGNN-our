"""
Train and save the best performing fold (Fold 12)
"""
import os
import time
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

from datapipe import build_dataset, get_dataset
from Net import SOGNN

##########################################################
subjects = 15
epochs = 50
classes = 4
Network = SOGNN
best_fold = 12  # Best performing fold from previous training

print('='*60)
print(f'Training Best Fold: {best_fold}')
print('='*60)

def train(model, train_loader, crit, optimizer):
    model.train()
    loss_all = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()

        label = torch.argmax(data.y.view(-1,classes), axis=1)
        label = label.to(device)

        output, _ = model(data.x, data.edge_index, data.batch)

        loss = crit(output, label)
        loss.backward()
        loss_all += data.num_graphs * loss.item()
        optimizer.step()

    return loss_all / len(train_dataset)

def evaluate(model, loader):
    model.eval()

    predictions = []
    labels = []

    with torch.no_grad():
        for data in loader:
            label = data.y.view(-1,classes)
            data = data.to(device)
            _, pred = model(data.x, data.edge_index, data.batch)
            pred = pred.detach().cpu().numpy()

            pred = np.squeeze(pred)
            predictions.append(pred)
            labels.append(label)

    predictions = np.vstack(predictions)
    labels = np.vstack(labels)

    AUC = roc_auc_score(labels, predictions, average='macro')
    f1 = f1_score(np.argmax(labels, axis=1), np.argmax(predictions, axis=1), average='macro')
    predictions = np.argmax(predictions, axis = -1)
    labels = np.argmax(labels, axis = -1)
    acc = accuracy_score(labels, predictions)

    return AUC, acc, f1

# Build dataset if needed
build_dataset(subjects)

# Load fold 12 data
train_dataset, test_dataset = get_dataset(subjects, best_fold)
train_loader = DataLoader(train_dataset, batch_size=16, drop_last=False, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16)

# Setup device and model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')
model = Network().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.00001, weight_decay=0.0001)
crit = torch.nn.CrossEntropyLoss()

# Training loop
print(f'\nTraining Fold {best_fold} for {epochs} epochs...\n')
for epoch in range(epochs):
    t0 = time.time()
    loss = train(model, train_loader, crit, optimizer)
    train_AUC, train_acc, train_f1 = evaluate(model, train_loader)
    val_AUC, val_acc, val_f1 = evaluate(model, test_loader)

    t1 = time.time()
    print('EP{:03d}, Loss:{:.3f}, TrainAUC:{:.3f}, TrainAcc:{:.3f}, ValAUC:{:.3f}, ValAcc:{:.3f}, ValF1:{:.3f}, Time:{:.2f}s'.format(
        epoch+1, loss, train_AUC, train_acc, val_AUC, val_acc, val_f1, (t1-t0)))

    if train_AUC>0.99 and train_acc>0.90:
        print(f'Early stopping at epoch {epoch+1} - Training converged')
        break

print('\n' + '='*60)
print('Final Results:')
print('='*60)
print(f'Fold {best_fold}, Epoch {epoch+1}')
print(f'  Loss: {loss:.4f}')
print(f'  Train AUC: {train_AUC:.4f}, Train Acc: {train_acc:.4f}')
print(f'  Val AUC: {val_AUC:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}')

# Save the best fold model
model_save_path = f'./result/{Network.__name__}_best_fold{best_fold}.pth'
torch.save({
    'model_state_dict': model.state_dict(),
    'fold': best_fold,
    'epoch': epoch+1,
    'train_acc': train_acc,
    'train_auc': train_AUC,
    'val_acc': val_acc,
    'val_auc': val_AUC,
    'val_f1': val_f1,
    'network_name': Network.__name__
}, model_save_path)

print(f'\n*** Model saved to: {model_save_path}')
print(f'    Fold: {best_fold}')
print(f'    Val Acc: {val_acc:.4f}')
print(f'    Val AUC: {val_AUC:.4f}')
print(f'    Val F1: {val_f1:.4f}')
