import torch
from torch.utils.data import dataset
import random
import numpy as np
import logging
import shutil
import time
import h5py
import os
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from Args import Args
from Custom_Test import CustomDataset
from Model import ClassificationModel
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import time
from Utils import mm_normalize

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataTransform:

    def __init__(self, use_seed=True):
        self.use_seed = use_seed

    def __call__(self, t1, t1ce, flair, dwi, adc, seg, t1ce_mask, flair_mask, t_label, d_label, file_index): 

        t1 = mm_normalize(t1, t1ce_mask) #t1-weighted
        t1ce = mm_normalize(t1ce, t1ce_mask) #ce-t1-weighted
        flair = mm_normalize(flair, flair_mask) #flair
        dwi = mm_normalize(dwi, t1ce_mask) #dwi
        adc = mm_normalize(adc, t1ce_mask) #adc
        
        seg1 = (seg == 1).astype(np.float32) # Necrotic portion
        seg2 = (seg == 2).astype(np.float32) # Solid Enhancing Lesion
        seg3 = (seg == 3).astype(np.float32) # Non-Enhancing FLAIR high-intensity Lesion
        
        t_label = np.array([t_label.item()]) # Tumorous/Non-tumorous condition Label
        d_label = np.array([d_label.item()]) # Clinical Referral Label
        
        input_vol = np.concatenate((t1[..., np.newaxis], t1ce[..., np.newaxis], flair[..., np.newaxis], dwi[..., np.newaxis], adc[..., np.newaxis], seg1[..., np.newaxis], seg2[..., np.newaxis], seg3[..., np.newaxis]), axis=3) 
        input_vol = np.transpose(input_vol,(3,0,1,2))
        
        return input_vol, t_label, d_label, file_index

def create_datasets(args):

    val_data = CustomDataset_Test(
        root=args.data_path,
        transform=DataTransform(),
        challenge='test',
    )
    return val_data

def create_data_loaders(args):

    val_data = create_datasets(args)

    val_loader = DataLoader(
        dataset=val_data,
        batch_size=args.batch_size,
        num_workers=8,
        pin_memory=True,
    )
    return val_loader

def run_model(args, model1, model2, model3, model4, model5, data_loader):
    
    model1.eval(); model2.eval(); model3.eval(); model4.eval(); model5.eval() #Applies No Dropout by calling model.eval()
    losses = [];
    accs = [];
    start = time.perf_counter()
    
    prob_matrix = torch.zeros((len(data_loader), 7))

    with torch.no_grad():
        for iter, data in enumerate(tqdm(data_loader)):
            input_vol, t_label, d_label, file_index = data
            input_vol = input_vol.to(args.device)
            t_label = t_label.to(args.device) 
            d_label = d_label.to(args.device)

            output1 = model1(input_vol); output2 = model2(input_vol); output3 = model3(input_vol); output4 = model4(input_vol); output5 = model5(input_vol)
                
            prob_matrix[iter, :] = torch.from_numpy(np.array([torch.sigmoid(output1).item(), torch.sigmoid(output2).item(), torch.sigmoid(output3).item(), torch.sigmoid(output4).item(), torch.sigmoid(output5).item(), (torch.sigmoid(output1).item()+torch.sigmoid(output2).item()+torch.sigmoid(output3).item()+torch.sigmoid(output4).item()+torch.sigmoid(output5).item())/5, t_label.item()]))

        test_auc_avg = roc_auc_score(prob_matrix[:,6], prob_matrix[:,5])
        print('Test AUC: %.4f '%(test_auc_avg))   

def build_model(args):
    
    model = ClassificationModel(
        in_chans=8,
        chans=args.num_chans,
        drop_prob=args.drop_prob
    ).to(args.device)
    
    return model

def load_model(checkpoint_file):
    
    checkpoint = torch.load(checkpoint_file)
    args = checkpoint['args']
    model = build_model(args)
    model.load_state_dict(checkpoint['model'])

    return model

def main(args):
    
    torch.backends.cudnn.benchmark=True
    val_loader = create_data_loaders(args)
    
    model1 = load_model(args.checkpoint1)
    model2 = load_model(args.checkpoint2)
    model3 = load_model(args.checkpoint3)
    model4 = load_model(args.checkpoint4)
    model5 = load_model(args.checkpoint5)

    run_model(args, model1, model2, model3, model4, model5, val_loader)
    
def create_arg_parser():
    parser = Args()
    parser.add_argument('--checkpoint1', type=str,
                        help='Path to an existing checkpoint. Used along with "--resume"')
    parser.add_argument('--checkpoint2', type=str,
                        help='Path to an existing checkpoint. Used along with "--resume"')
    parser.add_argument('--checkpoint3', type=str,
                        help='Path to an existing checkpoint. Used along with "--resume"')
    parser.add_argument('--checkpoint4', type=str,
                        help='Path to an existing checkpoint. Used along with "--resume"')
    parser.add_argument('--checkpoint5', type=str,
                        help='Path to an existing checkpoint. Used along with "--resume"')
    return parser

if __name__ == '__main__':
    args = create_arg_parser().parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
    main(args)
