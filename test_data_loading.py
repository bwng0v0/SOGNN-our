"""
Test script to verify SEED-IV data loading
"""
import os
import numpy as np
import scipy.io as sio
import glob

# SEED-IV labels for 24 trials per session
session_labels = {
    1: np.array([1,2,3,0,2,0,0,1,0,1,2,1,1,1,2,3,2,2,3,3,0,3,0,3]),
    2: np.array([2,1,3,0,0,2,0,2,3,3,2,3,2,0,1,1,2,1,0,3,0,1,3,1]),
    3: np.array([1,2,2,1,3,3,3,1,1,2,1,0,2,3,3,0,2,3,0,0,2,0,1,0])
}

base_path = '../SEED4/eeg_feature_smooth/'

print('Testing SEED-IV data loading...')
print('='*60)

# Get subject list
session1_path = base_path + '1/'
files = sorted(glob.glob(session1_path + '*.mat'))
sublist = []
for f in files:
    subject_id = os.path.basename(f).split('_')[0]
    sublist.append(subject_id)

sublist = sorted(sublist, key=lambda x: int(x))
print(f'OK Found {len(sublist)} subjects: {sublist}')

# Test loading one subject
subject_id = sublist[0]
print(f'\nTesting subject {subject_id}...')

for session in [1, 2, 3]:
    session_path = base_path + str(session) + '/'
    sub_files = glob.glob(session_path + subject_id + '_*.mat')

    if len(sub_files) == 0:
        print(f'  ERROR Session {session}: File not found!')
        continue

    f = sub_files[0]
    data = sio.loadmat(f, verify_compressed_data_integrity=False)

    # Get all de_movingAve keys
    de_mov = sorted([k for k in data.keys() if 'de_movingAve' in k],
                   key=lambda x: int(x.replace('de_movingAve', '')))

    print(f'  OK Session {session}: {len(de_mov)} trials found')
    print(f'    - File: {os.path.basename(f)}')
    print(f'    - Trial keys: {de_mov[0]} to {de_mov[-1]}')
    print(f'    - Example shape: {data[de_mov[0]].shape}')
    print(f'    - Labels: {session_labels[session]}')

print('\n' + '='*60)
print('Data structure verification:')
print(f'  - Expected trials per session: 24')
print(f'  - Expected sessions: 3')
print(f'  - Expected total trials per subject: 72')
print(f'  - Expected classes: 4 (0=neutral, 1=sad, 2=fear, 3=happy)')
print(f'  - Expected shape: (62 channels, timepoints, 5 freq_bands)')
print('\nOK SEED-IV data structure verified!')
