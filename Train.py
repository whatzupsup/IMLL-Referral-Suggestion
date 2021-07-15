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
from Custom_Train import CustomDataset_Train
from Model import ClassificationModel
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from Utils import mm_normalize

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataTransform:
    def __init__(self, use_seed=True, if_train=True):
        
        self.use_seed = use_seed
        self.if_train = if_train

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

    train_data = CustomDataset_Train(
        root=args.data_path,
        transform=DataTransform(if_train=True),
        challenge='train',
        cv=args.cv
    )
    val_data = CustomDataset_Train(
        root=args.data_path,
        transform=DataTransform(if_train=False),
        challenge='valid',
        cv=args.cv
    )
    return train_data, val_data

def create_data_loaders(args):

    train_data, val_data = create_datasets(args)

    train_loader = DataLoader(
        dataset=train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
    )
    val_loader = DataLoader(
        dataset=val_data,
        batch_size=args.batch_size,
        num_workers=8,
        pin_memory=True,
    )
    
    return train_loader, val_loader

def train_epoch(args, epoch, model, data_loader, optimizer, writer):
    model.train()
    avg_loss = 0.
    start_epoch = start_iter = time.perf_counter()
    global_step = epoch * len(data_loader)
    
    prob_matrix = torch.zeros((len(data_loader), 2))
    bceloss = torch.nn.BCEWithLogitsLoss()
    
    for iter, data in enumerate(tqdm(data_loader)):
        
        input_vol, t_label, d_label, file_index = data
        
        input_vol = input_vol.to(args.device)
        t_label = t_label.to(args.device)
        d_label = d_label.to(args.device)

        randlist = [random.random() for _ in range(3)]
        
        # Qualitative Data Augmentation by Random Flipping
        if randlist[0] > 0.5:
            input_vol = torch.flip(input_vol, [2])
        if randlist[1] > 0.5:
            input_vol = torch.flip(input_vol, [3])
        if randlist[2] > 0.5:
            input_vol = torch.flip(input_vol, [4])
            
        output = model(input_vol)
        
        # Focal Loss
        bce_loss = bceloss(output, t_label)
        pt = torch.exp(-bce_loss)
        alpha = torch.tensor([0.77]).to(args.device) if t_label == 0 else torch.tensor([0.23]).to(args.device)
        
        loss = (alpha*((1-pt)**2)*bce_loss).mean()
        
        prob_matrix[iter, :] = torch.from_numpy(np.array([(torch.sigmoid(output)).item(), t_label.item()]))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_loss = 0.99 * avg_loss + 0.01 * loss.item() if iter > 0 else loss.item()
        writer.add_scalar('TrainLoss', loss.item(), global_step + iter)

        if iter % args.report_interval == 0:
            logging.info(
            f'Epoch = [{epoch:3d}/{args.num_epochs:3d}] '
            f'Iter = [{iter:4d}/{len(data_loader):4d}] '
            f'Loss = {loss.item():.4g} '
            f'Time = {time.perf_counter() - start_iter:.4f}s',
            )
        start_iter = time.perf_counter()
        
    train_auc = roc_auc_score(prob_matrix[:,1], prob_matrix[:,0])
    print('Train AUC: %.4f '%(train_auc))
    
    return avg_loss, time.perf_counter() - start_epoch

def evaluate(args, epoch, model, data_loader, writer):
    
    model.eval()
    losses = [];
    start = time.perf_counter()
    
    prob_matrix = torch.zeros((len(data_loader), 2))
    bceloss = torch.nn.BCEWithLogitsLoss()
    
    with torch.no_grad():
        for iter, data in enumerate(tqdm(data_loader)):
            
            input_vol, t_label, d_label, file_index = data
            input_vol = input_vol.to(args.device)
            t_label = t_label.to(args.device)
            d_label = d_label.to(args.device)
            
            output = model(input_vol)
            
            bce_loss = bceloss(output, t_label)
            pt = torch.exp(-bce_loss)
            alpha = torch.tensor([0.77]).to(args.device) if t_label == 0 else torch.tensor([0.23]).to(args.device)
            loss = (alpha*((1-pt)**2)*bce_loss).mean()

            prob_matrix[iter, :] = torch.from_numpy(np.array([(torch.sigmoid(output)).item(), t_label.item()]))
            
            losses.append(loss.item())
                
        val_auc = roc_auc_score(prob_matrix[:,1], prob_matrix[:,0])
        print('Validation AUC: %.4f '%(val_auc))
    
        writer.add_scalar('Val_Loss', np.mean(losses), epoch)
    return np.mean(losses), time.perf_counter() - start, 

def save_model(args, exp_dir, epoch, model, optimizer, val_loss, is_new_best, best_val_loss):
    torch.save(
        {
            'epoch': epoch,
            'args': args,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'val_loss': val_loss,
            'exp_dir': exp_dir,
            'best_val_loss': best_val_loss
        },
        f=str(exp_dir)+'/'+ 'model_%d.pt'%epoch
    )
    if is_new_best:
        shutil.copyfile(str(exp_dir)+'/'+'model_%d.pt'%epoch, exp_dir / 'best_model.pt')

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

    optimizer = build_optim(args, model.parameters())
    optimizer.load_state_dict(checkpoint['optimizer'])
    
    return checkpoint, model, optimizer

def build_optim(args, params):
    optimizer = torch.optim.Adam(params, args.lr, weight_decay=args.weight_decay)
    return optimizer

def main(args):
    args.exp_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=args.exp_dir / args.sumpath)

    if args.resume:
        checkpoint, model, optimizer = load_model(args.checkpoint)
        args = checkpoint['args']
        best_val_loss = checkpoint['best_val_loss']
        start_epoch = checkpoint['epoch']
        del checkpoint
    else:
        model = build_model(args)
        optimizer = build_optim(args, model.parameters())
        best_val_loss = 1e9
        start_epoch = 0
    logging.info(args)
    logging.info(model)
    
    torch.backends.cudnn.benchmark=True
    train_loader, val_loader = create_data_loaders(args)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_step_size, args.lr_gamma)

    for epoch in range(start_epoch, args.num_epochs):
        train_loss, train_time = train_epoch(args, epoch, model, train_loader, optimizer, writer)
        val_loss, val_time = evaluate(args, epoch, model, val_loader, writer)
        scheduler.step(epoch)

        is_new_best = val_loss < best_val_loss
        best_val_loss = min(best_val_loss, val_loss)
        save_model(args, args.exp_dir, epoch, model, optimizer, val_loss, is_new_best, best_val_loss)
        logging.info(
            f'Epoch = [{epoch:4d}/{args.num_epochs:4d}] TrainLoss = {train_loss:.4g} '
            f'ValLoss = {val_loss:.4g} TrainTime = {train_time:.4f}s DevTime = {val_time:.4f}s',
        )
    writer.close()
    
def create_arg_parser():
    parser = Args()
    return parser

if __name__ == '__main__':
    args = create_arg_parser().parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
    main(args)
