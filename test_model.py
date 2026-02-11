"""
Quick test to verify model can be loaded and run
"""
import torch
from Net import SOGNN
from datapipe import get_dataset
from torch_geometric.data import DataLoader

print('Step 1: Loading dataset...')
train_dataset, test_dataset = get_dataset(15, 0)
print(f'  Train dataset size: {len(train_dataset)}')
print(f'  Test dataset size: {len(test_dataset)}')

print('\nStep 2: Creating data loaders...')
train_loader = DataLoader(train_dataset, batch_size=16, drop_last=False, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16)
print(f'  Train batches: {len(train_loader)}')
print(f'  Test batches: {len(test_loader)}')

print('\nStep 3: Creating model...')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'  Using device: {device}')
model = SOGNN()
print('  Model created')

print('\nStep 4: Moving model to device...')
model = model.to(device)
print('  Model moved to device')

print('\nStep 5: Testing forward pass...')
for data in train_loader:
    print(f'  Batch shape: {data.x.shape}')
    data = data.to(device)
    print('  Data moved to device')

    output, pred = model(data.x, data.edge_index, data.batch)
    print(f'  Output shape: {output.shape}')
    print(f'  Prediction shape: {pred.shape}')
    break

print('\nAll tests passed!')
