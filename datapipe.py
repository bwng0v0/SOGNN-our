"""

""" 
version = 1

import os 
import numpy as np 
import torch
from torch_geometric.data import Data, InMemoryDataset
from tqdm import tqdm
import scipy.io as sio
import glob  

predictions_dir = './predictions'
if not os.path.exists(predictions_dir):
    os.makedirs(predictions_dir)

subjects = 15 # Num. of subjects used for LOSO
classes = 4 # Num. of classes (SEED-IV: neutral, sad, fear, happy) 

def to_categorical(y, num_classes=None, dtype='float32'): 
    #one-hot encoding
    y = np.array(y, dtype='int16')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0] 
    categorical = np.zeros((n, num_classes), dtype=dtype)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical

class EmotionDataset(InMemoryDataset):
    def __init__(self, stage, root, subjects, sub_i, X=None, Y=None, edge_index=None,
                 transform=None, pre_transform=None):
        self.stage = stage #Train or test
        self.subjects = subjects  
        self.sub_i = sub_i
        self.X = X
        self.Y = Y
        self.edge_index = edge_index
        
        #super(EmotionDataset, self).__init__(root, transform, pre_transform)
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def raw_file_names(self):
        return []
    @property
    def processed_file_names(self):
        return ['./V_{:.0f}_{:s}_CV{:.0f}_{:.0f}.dataset'.format(
                version, self.stage, self.subjects, self.sub_i)]
    def download(self):
        pass
    
    def process(self): 
        data_list = [] 
        # process by samples
        num_samples = np.shape(self.Y)[0]
        for sample_id in tqdm(range(num_samples)): 
            x = self.X[sample_id,:,:]    
            x = torch.FloatTensor(x)
            y = torch.FloatTensor(self.Y[sample_id,:])
            data = Data(x=x, y=y)

            data_list.append(data) 
            
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        
def normalize(data):
    mee=np.mean(data,0)
    data=data-mee
    stdd=np.std(data,0)
    data=data/(stdd+1e-7)
    return data 

def get_data():
    # SEED-IV dataset structure: eeg_feature_smooth/session/subject.mat
    base_path = '../SEED4/eeg_feature_smooth/'

    # SEED-IV labels for 24 trials per session
    session_labels = {
        1: np.array([1,2,3,0,2,0,0,1,0,1,2,1,1,1,2,3,2,2,3,3,0,3,0,3]),
        2: np.array([2,1,3,0,0,2,0,2,3,3,2,3,2,0,1,1,2,1,0,3,0,1,3,1]),
        3: np.array([1,2,2,1,3,3,3,1,1,2,1,0,2,3,3,0,2,3,0,0,2,0,1,0])
    }

    # Get subject list from session 1
    session1_path = base_path + '1/'
    files = sorted(glob.glob(session1_path + '*.mat'))

    sublist = []
    for f in files:
        # Extract subject ID from filename (e.g., "1_20160518.mat" -> "1")
        subject_id = os.path.basename(f).split('_')[0]
        sublist.append(subject_id)

    sublist = sorted(sublist, key=lambda x: int(x))
    print('Total number of subjects: {:.0f}'.format(len(sublist)))
    print(sublist)

    sub_mov = []
    sub_label = []

    for sub_i in range(subjects):
        subject_id = sublist[sub_i]
        mov_data = []
        all_labels = []

        # Load data from all 3 sessions
        for session in [1, 2, 3]:
            session_path = base_path + str(session) + '/'
            # Find the file for this subject in this session
            sub_files = glob.glob(session_path + subject_id + '_*.mat')

            if len(sub_files) == 0:
                print(f'Warning: No file found for subject {subject_id} session {session}')
                continue

            f = sub_files[0]
            print(f)
            data = sio.loadmat(f, verify_compressed_data_integrity=False)

            # Get all de_movingAve keys (1-indexed: de_movingAve1 to de_movingAve24)
            de_mov = sorted([k for k in data.keys() if 'de_movingAve' in k],
                          key=lambda x: int(x.replace('de_movingAve', '')))

            mov_datai = []
            for t in range(24):  # SEED-IV has 24 trials
                key = de_mov[t]
                temp_data = data[key].transpose(0,2,1)  # (62, timepoints, 5) -> (62, 5, timepoints)
                data_length = temp_data.shape[-1]
                mov_i = np.zeros((62, 5, 265))
                mov_i[:,:,:data_length] = temp_data
                mov_i = mov_i.reshape(62, -1)

                mov_datai.append(mov_i)
            mov_datai = np.array(mov_datai)
            mov_data.append(mov_datai)
            all_labels.append(session_labels[session])

        mov_data = np.vstack(mov_data)  # Concatenate all sessions
        mov_data = normalize(mov_data)
        sub_mov.append(mov_data)
        sub_label.append(np.hstack(all_labels))  # Concatenate labels from all sessions

    sub_mov = np.array(sub_mov)
    sub_label = np.array(sub_label)

    return sub_mov, sub_label
    
def build_dataset(subjects):
    load_flag = True
    for sub_i in range(subjects):
        path = './processed/V_{:.0f}_{:s}_CV{:.0f}_{:.0f}.dataset'.format(
                version, 'Train', subjects, sub_i)
        print(path)
        
        if not os.path.exists(path): 
        
            if load_flag:
                mov_coefs, labels = get_data()
                used_coefs = mov_coefs
                load_flag = False
            
            index_list = list(range(subjects))
            del index_list[sub_i]
            test_index = sub_i
            train_index = index_list
            
            print('Building train and test dataset')
            #get train & test
            X = used_coefs[train_index,:].reshape(-1, 62, 265*5)
            Y = labels[train_index,:].reshape(-1)
            testX = used_coefs[test_index,:].reshape(-1, 62, 265*5)
            testY = labels[test_index,:].reshape(-1) 
            #get labels
            _, Y = np.unique(Y, return_inverse=True)
            Y = to_categorical(Y, classes)#
            _, testY = np.unique(testY, return_inverse=True)
            testY = to_categorical(testY, classes)
            
            train_dataset = EmotionDataset('Train', './', subjects, sub_i, X, Y)
            test_dataset = EmotionDataset('Test', './', subjects, sub_i, testX, testY)
            print('Dataset is built.')
            
def get_dataset(subjects, sub_i):
    path = './processed/V_{:.0f}_{:s}_CV{:.0f}_{:.0f}.dataset'.format(
            version, 'Train', subjects, sub_i)
    print(path)
    if not os.path.exists(path): 
        raise IOError('Train dataset is not exist!')
    
    train_dataset = EmotionDataset('Train', './', subjects, sub_i)
    test_dataset = EmotionDataset('Test', './', subjects, sub_i) 

    return train_dataset, test_dataset
