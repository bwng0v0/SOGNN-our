"""
Self-organized Gragh Neural Network
"""

import os
import time
import numpy as np
import pandas as pd
import scipy.io as sio
import torch 
from torch_geometric.data import DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

from datapipe import build_dataset, get_dataset
from Net import SOGNN

##########################################################
"""
Settings for training
"""
subjects = 15
epochs = 30  # Training for paper experiments (30 epochs ~ 3.5-4 hours for 15 folds)
classes = 4 # Num. of classes (SEED-IV: neutral, sad, fear, happy)
Network = SOGNN

# Train all folds for paper experiments
NUM_FOLDS_TO_TRAIN = 15  # Full training for paper (all folds)

version = 1
print('***'*20)
while(1):
    dfile = './result/{}_LOG_{:.0f}.csv'.format(Network.__name__, version)
    if not os.path.exists(dfile):
        break
    version += 1
print(dfile)
df = pd.DataFrame()
df.to_csv(dfile)   
##########################################################

def train(model, train_loader, crit, optimizer):
    model.train()
    loss_all = 0
    for data in train_loader: 
        data = data.to(device)
        optimizer.zero_grad()
        
        #Multiple Classes classification Loss function
        label = torch.argmax(data.y.view(-1,classes), axis=1)
        label = label.to(device)#, dtype=torch.long) #, dtype=torch.int64)
        
        output, _ = model(data.x, data.edge_index, data.batch)
        
        loss = crit(output, label)
        loss.backward()
        loss_all += data.num_graphs * loss.item()
        optimizer.step() 

    return loss_all / len(train_dataset)

def evaluate(model, loader, save_result=False):
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

    #AUC score estimation 
    AUC = roc_auc_score(labels, predictions, average='macro')
    f1 = f1_score(np.argmax(labels, axis=1), np.argmax(predictions, axis=1), average='macro')
    #Accuracy 
    predictions = np.argmax(predictions, axis = -1)
    labels = np.argmax(labels, axis = -1)
    acc = accuracy_score(labels, predictions)

    return AUC, acc, f1
    
build_dataset(subjects)# Build dataset for each fold

# Create models directory if it doesn't exist
os.makedirs('./models', exist_ok=True)

print('Cross Validation')
result_data = []
all_last_acc = []
all_last_AUC = []

# For averaging model weights across all folds
accumulated_state_dict = None
num_folds_completed = 0

for cv_n in range(0, NUM_FOLDS_TO_TRAIN):
    train_dataset, test_dataset = get_dataset(subjects, cv_n)
    train_loader = DataLoader(train_dataset, batch_size=16, drop_last=False, shuffle=True )
    test_loader = DataLoader(test_dataset, batch_size=16)

    # Use CUDA if available, otherwise use CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    model = Network().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00001, weight_decay=0.0001)
    crit = torch.nn.CrossEntropyLoss() #

    # Track best validation performance for this fold
    best_val_acc = 0.0
    best_val_auc = 0.0
    best_epoch = 0

    epoch_data = []
    for epoch in range(epochs):
        t0 = time.time()
        loss = train(model, train_loader, crit, optimizer)
        train_AUC, train_acc, train_f1 = evaluate(model, train_loader)
        val_AUC, val_acc, val_f1 = evaluate(model, test_loader)

        epoch_data.append([str(cv_n), epoch+1, loss, train_AUC, train_acc, val_AUC, val_acc])
        t1 = time.time()
        print('V{:01d}, EP{:03d}, Loss:{:.3f}, AUC:{:.3f}, Acc:{:.3f}, VAUC:{:.2f}, Vacc:{:.2f}, Time: {:.2f}'.
                  format(cv_n, epoch+1, loss, train_AUC, train_acc, val_AUC, val_acc, (t1-t0)))

        # Save best model for this fold
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_auc = val_AUC
            best_epoch = epoch + 1
            model_save_path = f'./models/{Network.__name__}_fold{cv_n}_best.pth'
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_auc': val_AUC,
                'val_f1': val_f1,
                'train_acc': train_acc,
                'train_auc': train_AUC,
                'fold': cv_n
            }, model_save_path)
            print(f'  → Best model saved (Val Acc: {val_acc:.4f}, Val AUC: {val_AUC:.4f})')

        if train_AUC>0.99 and train_acc>0.90:
            break

    print('Results::::::::::::')
    print('V{:01d}, EP{:03d}, Loss:{:.3f}, AUC:{:.3f}, Acc:{:.3f}, VAUC:{:.2f}, Vacc:{:.2f}, Time: {:.2f}'.
                  format(cv_n, epoch+1, loss, train_AUC, train_acc, val_AUC, val_acc, (t1-t0)))
    print(f'Best model for fold {cv_n}: Epoch {best_epoch}, Val Acc: {best_val_acc:.4f}, Val AUC: {best_val_auc:.4f}')

    result_data.append([str(cv_n), epoch+1, loss, train_AUC, train_acc, val_AUC, val_acc, val_f1])

    # Accumulate model weights for averaging
    current_state_dict = model.state_dict()
    if accumulated_state_dict is None:
        # First fold: initialize with current model
        accumulated_state_dict = {key: value.cpu().clone() for key, value in current_state_dict.items()}
    else:
        # Subsequent folds: add to accumulated weights
        for key in accumulated_state_dict.keys():
            accumulated_state_dict[key] += current_state_dict[key].cpu()
    num_folds_completed += 1
    print(f'Model weights accumulated for fold {cv_n+1}/{subjects}')

    df = pd.DataFrame(data=result_data, columns=['Fold', 'Epoch', 'Loss', 'Tra_AUC', 'Tra_acc', 'Val_AUC', 'Val_acc', 'Val_f1'])

    df.to_csv(dfile)  
    
df = pd.read_csv(dfile)

lastacc = ['Val_acc', df['Val_acc'].mean()]
lastauc = ['Val_AUC', df['Val_AUC'].mean()]
print(lastacc)
print(lastauc)
print('*****************')

# Summary of saved models
print(f'\n{"="*60}')
print(f'SAVED MODELS SUMMARY')
print(f'{"="*60}')
print(f'Total folds trained: {NUM_FOLDS_TO_TRAIN}')
print(f'Models saved in: ./models/')
for i in range(NUM_FOLDS_TO_TRAIN):
    print(f'  - {Network.__name__}_fold{i}_best.pth')
print(f'{"="*60}\n')

# Save averaged model weights
if accumulated_state_dict is not None and num_folds_completed > 0:
    print(f'\nAveraging model weights from {num_folds_completed} folds...')

    # Calculate average by dividing accumulated weights by number of folds
    averaged_state_dict = {key: value / num_folds_completed for key, value in accumulated_state_dict.items()}

    # Create a new model and load averaged weights
    averaged_model = Network()
    averaged_model.load_state_dict(averaged_state_dict)

    # Save the averaged model
    model_save_path = f'./result/{Network.__name__}_averaged_{num_folds_completed}folds.pth'
    torch.save({
        'model_state_dict': averaged_state_dict,
        'num_folds': num_folds_completed,
        'mean_val_acc': df['Val_acc'].mean(),
        'mean_val_auc': df['Val_AUC'].mean(),
        'network_name': Network.__name__
    }, model_save_path)

    print(f'✓ Averaged model saved to: {model_save_path}')
    print(f'  - Folds averaged: {num_folds_completed}')
    print(f'  - Mean Val Acc: {df["Val_acc"].mean():.4f}')
    print(f'  - Mean Val AUC: {df["Val_AUC"].mean():.4f}')
else:
    print('\nWarning: No folds completed, no model saved.')


