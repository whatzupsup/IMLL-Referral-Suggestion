import pathlib
import random
import h5py
from torch.utils.data import Dataset
import torch
import numpy as np
from tqdm import tqdm

# You should build your custom dataset as below.
class CustomDataset_Test(torch.utils.data.Dataset):
    def __init__(self, root, transform, challenge='test'):

        self.transform = transform
        self.diag_0 = []; self.diag_1 = []; self.diag_2 = []; self.diag_4 = []; self.diag_5 = []; self.diag_6 = []; self.diag_7 = []; self.diag_8 = [];
        self.examples = [];
        self.train_tumor = [];
        self.train_nontumor = [];

        root_h5= pathlib.Path(root) / 'VALIDATION_H5_FINETUNED'
        files_h5 = sorted(list(pathlib.Path(root_h5).iterdir()))

        for i in tqdm(range(len(files_h5))):
            
            diag = h5py.File(files_h5[i], 'r')['diagnosis'][()]
            file_index = int(str(files_h5[i]).split('/')[-1].split('.')[0])

            if diag==0:
                self.diag_0.append(files_h5[i])
            elif diag==1:
                self.diag_1.append(files_h5[i])
            elif diag==2:
                self.diag_2.append(files_h5[i])
            elif diag==4:
                self.diag_4.append(files_h5[i])
            elif diag==5:
                self.diag_5.append(files_h5[i])
            elif diag==6:
                self.diag_6.append(files_h5[i])
            elif diag==7:
                self.diag_7.append(files_h5[i])
            elif diag==8:
                self.diag_8.append(files_h5[i])
                    
        print('DIAG_0: %d, DIAG_1: %d, DIAG_2: %d, DIAG_4: %d, DIAG_5: %d, DIAG_6: %d, DIAG_7: %d, DIAG_8: %d'%((len(self.diag_0), len(self.diag_1), len(self.diag_2), len(self.diag_4), len(self.diag_5), len(self.diag_6), len(self.diag_7), len(self.diag_8))))
         
        for i, input_file in enumerate(self.diag_0):
            self.examples += [str(input_file)]
                
        for i, input_file in enumerate(self.diag_1):
            self.examples += [str(input_file)]
                
        for i, input_file in enumerate(self.diag_2):
            self.examples += [str(input_file)]
        
        for i, input_file in enumerate(self.diag_4):
            self.examples += [str(input_file)]
                
        for i, input_file in enumerate(self.diag_5):
            self.examples += [str(input_file)]
                
        for i, input_file in enumerate(self.diag_6):
            self.examples += [str(input_file)]
                
        for i, input_file in enumerate(self.diag_7):
            self.examples += [str(input_file)]
                
        for i, input_file in enumerate(self.diag_8):
            self.examples += [str(input_file)]
                
        self.examples = sorted(self.examples)
        print('LENGTH SELF.EXAMPLES:', len(self.examples)) #Should be 189+199=388
        
    def __getitem__(self, i):
        
        input_h5 = self.examples[i]
        file_index = int(str(input_h5).split('/')[-1].split('.')[0])
        
        t1ce = np.transpose(h5py.File(input_h5, 'r')['t1ce'][()], (2,1,0)).astype(np.float32)
        t1 = np.transpose(h5py.File(input_h5, 'r')['t1'][()], (2,1,0)).astype(np.float32)
        flair = np.transpose(h5py.File(input_h5, 'r')['flair'][()], (2,1,0)).astype(np.float32)
        dwi = np.transpose(h5py.File(input_h5, 'r')['dwi'][()], (2,1,0)).astype(np.float32)
        adc = np.transpose(h5py.File(input_h5, 'r')['adc'][()], (2,1,0)).astype(np.float32)
        seg = np.transpose(h5py.File(input_h5, 'r')['seg'][()], (2,1,0)).astype(np.float32)
        t1ce_mask = np.transpose(h5py.File(input_h5, 'r')['t1ce_mask'][()], (2,1,0)).astype(np.bool)
        flair_mask = np.transpose(h5py.File(input_h5, 'r')['flair_mask'][()], (2,1,0)).astype(np.bool)
        
        tumor = h5py.File(input_h5, 'r')['tumor'][()].astype(np.float32)
        
        diagnosis = h5py.File(input_h5, 'r')['diagnosis'][()].astype(np.float32)

        if diagnosis==8:
            diagnosis = np.array([[3]]).astype(np.float32)
        
        return self.transform(t1, t1ce, flair, dwi, adc, seg, t1ce_mask, flair_mask, tumor, diagnosis, file_index)
        
    def __len__(self):
        return len(self.examples)
