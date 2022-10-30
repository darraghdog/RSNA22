import numpy as np
import sys
import pandas as pd
import re
import json
import os
import torch
import glob
import pylibjpeg
import pydicom
import random
import platform
import ast
from PIL import Image, ImageStat
import cv2
import imagesize
import copy
import collections
from torch.nn import functional as F
from tqdm import tqdm
from torch.utils.data import Dataset
from utils import get_logger, set_pandas_display
from bounding_box import bounding_box as bbfn
import albumentations as A
from albumentations.pytorch import ToTensorV2
from multiprocessing import Pool
from utils import prepare_mask, fetch_filename
import warnings
from pandas.core.common import SettingWithCopyWarning
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)

tqdm.pandas()
set_pandas_display()
logger = get_logger('Dataset', 'INFO')

def batch_to_device(batch, device):
    batch_dict = {key: batch[key].to(device) for key in batch}
    return batch_dict

def cropdim(fname, threshold=0):
    flatImage = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)
    rows = np.where(np.max(flatImage, 0) > threshold)[0]
    cols = np.where(np.max(flatImage, 1) > threshold)[0]
    try:
        return [rows[0], cols[0], rows[-1] + 1, cols[-1] + 1]
    except:
        return [np.nan] * 4

# %timeit cv2.resize(img, (512,512,), interpolation = cv2.INTER_CUBIC)
# batch = [self.__getitem__(i) for i in range(4)]
def create_train_transforms(cfg):
    return A.Compose([
        A.Resize(*cfg.imagesize, interpolation = 1, p=1), # interpolation = cv2.INTER_CUBIC,
        A.VerticalFlip(p=cfg.vflip ),
        A.HorizontalFlip(p=cfg.hflip ),
        A.RandomContrast(limit=cfg.RandomContrast, p=0.75),
        A.ShiftScaleRotate(shift_limit=cfg.shift_limit,
                           scale_limit=cfg.scale_limit,
                           value = 0,
                           rotate_limit=cfg.rotate_limit,
                           p=0.75,
                           border_mode = cv2.BORDER_CONSTANT),
        A.OneOf([
            A.GridDistortion(num_steps=5, distort_limit=0.05, p=1.0),
            A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=1.0)
        ], p=cfg.p_grid_distortion),
        
        A.Cutout(num_holes=cfg.holes,
                 max_h_size=int(cfg.hole_size*cfg.imagesize[0]),
                 max_w_size=int(cfg.hole_size*cfg.imagesize[1]),
                 fill_value=0,
                 always_apply=True, p=0.75),
        A.Normalize(mean=cfg.norm_mean[:],
                    std=cfg.norm_std[:], p=1.0),
        ToTensorV2()
        ],
    additional_targets=dict((f'image{i}', 'image') for i in range(cfg.sequence_len)))


def create_valid_transforms(cfg):
    return A.Compose([
        A.Resize(*cfg.imagesize, p=1), # interpolation = cv2.INTER_CUBIC,
        A.Normalize(mean=cfg.norm_mean[:],
                    std=cfg.norm_std[:], p=1.0),
        ToTensorV2()
        ],
    additional_targets=dict((f'image{i}', 'image') for i in range(cfg.sequence_len)))


def collate_fn(batch):
    # Remove error reads
    batch = [b for b in batch if b is not None]

    cols = 'image label slice_number StudyUID slice_stats'.split()
    batchout = dict((k,torch.stack([b[k] for b in batch])) for k in cols)

    return batchout

tr_collate_fn = collate_fn
val_collate_fn = collate_fn

'''
df = pd.read_csv(cfg.train_df)
mode="train"
class self:
    1
self = CustomDataset(df, cfg, mode = 'train')
batch = [self.__getitem__(i) for i in range(0, 3000, 700)]
batch = tr_collate_fn(batch)
batch = batch_to_device(batch, 'cpu')
'''

'''
        https://arxiv.org/pdf/2010.13336.pdf
        1) Soft tissue window (w1 = 300; c1 = 80); 
        2) Standard bone window (w2 = 1800; c2 = 500); 
        3) Gross bone window (w3 = 650; c3 = 400).
'''

class CustomDataset(Dataset):

    #def __init__(self, df, mode, tokenizer, max_pos, smoothbreaks = False):
    def __init__(self, df, cfg, aug=None, mode="train"):

        self.cfg = cfg
        self.mode = mode
        self.df = df.copy()
        self.bbdf = pd.read_csv(cfg.bbox_df).set_index('StudyInstanceUID')
        if self.mode!='test': 
            self.df = self.df[self.df[self.cfg.target[0]]>-1]
        else:
            self.df = self.df.query('fold == @cfg.fold')
        self.df = self.df[~self.df.StudyInstanceUID.isin(self.cfg.drop_scans)]

        if platform.system() == 'Darwin':
            local_dirs = glob.glob(f'{cfg.data_folder}/*')
            local_dirs = [i.split('/')[-1] for i in local_dirs]
            self.df = self.df[df.StudyInstanceUID.isin(local_dirs)].reset_index(drop = True)
        self.windows = self.cfg.windows #[ (500, 1800), (400, 650), (80, 300)]
        
        if mode=='train':
            self.transform = create_train_transforms(cfg)
        else:
            self.transform = create_valid_transforms(cfg)

        self.df = self.df.reset_index(drop = True)
        self.weights = 1 / (self.df.groupby('StudyInstanceUID')['C1'].transform(len) - cfg.sequence_len)        
        self.df['slice_number_start'] = \
            self.df.groupby('StudyInstanceUID')['slice_number'].transform(max) - (cfg.sequence_len+1)
        
        endidx = self.df['slice_number'] > self.df['slice_number_start']
        lastidx = self.df['slice_number'] == self.df['slice_number_start']
        firstidx = self.df['slice_number'] == 1
        self.df['val_pos'] = (self.df['slice_number'] % (cfg.sequence_len//6)) == 0 
        self.df['trn_pos'] = (self.df['slice_number'] % (12)) == 0 
        #self.weights[endidx] = 0.
        self.df.val_pos.loc[endidx] = False
        self.df.val_pos.loc[lastidx] = True
        self.df.val_pos.loc[firstidx] = True
        self.df.trn_pos.loc[endidx] = False
        self.df.trn_pos.loc[lastidx] = True
        self.df.trn_pos.loc[firstidx] = True
        self.df['slice_number_start'] = self.df['slice_number slice_number_start'.split()].min(1)
        self.val_pos = self.df.val_pos[self.df.val_pos]
        self.trn_pos = self.df.trn_pos[self.df.trn_pos]
        self.df = self.df.set_index('StudyInstanceUID')
        #if mode=='train':
        #    self.sample_weights = self.weights.values
        
        idx = 0

    def __len__(self):
        if self.mode == 'train':
            return len(self.trn_pos)
        return len(self.val_pos)

    def __getitem__(self, idx):
        
        if self.mode == 'train':
            idx = self.trn_pos.index[idx]
        else:
            idx = self.val_pos.index[idx]
        
        samp = self.df.iloc[idx]
        
        
        bb = self.bbdf.loc[samp.name].values
        # Square the bbox
        w,h = bb[2]-bb[0], bb[3]-bb[1]
        w_pad, h_pad = h - min(h,w), w - min(h,w)
        mat_pad = np.array([-w_pad//2, -h_pad//2, w_pad//2, h_pad//2])
        x0,y0,x1,y1 = bb = (mat_pad + bb).clip(0, 512)
        
        start_pos = int(samp.slice_number_start)
        
        dfseq = self.df.loc[samp.name].reset_index().set_index('slice_number')
        dfseq = dfseq.loc[start_pos:].iloc[:self.cfg.sequence_len]
        dfseq = dfseq.iloc[np.random.randint(self.cfg.dicom_steps)::self.cfg.dicom_steps]
        dfseq = dfseq.reset_index().set_index('StudyInstanceUID')

        imgls = []
        zposarr = []
        for idd,slcnum in dfseq.slice_number.items():
            dcmfile = f'{self.cfg.data_folder}/{idd}/{int(slcnum)}.dcm'
            X_orig, zpos = self.read_dicom(dcmfile)
            X_crop = X_orig[y0:y1,x0:x1]
            imgls.append(X_crop)
            zposarr.append(zpos)
        
        img_crops = np.stack(imgls)
        img_crops = torch.from_numpy(img_crops)
        img_crops = F.interpolate(img_crops.unsqueeze(0), 
                                      size = self.cfg.imagesize, 
                                      mode=self.cfg.interpolation_mode).squeeze(0)
        img_crops = [self.windowsfn(m.numpy(), self.windows) for m in img_crops]
        imgls = [cv2.merge(img_crops[i:i + 3]) for i in range(0, len(img_crops), 3)]
            
        
        aug_input = dict((f'image{t}', i) for t,i in enumerate(imgls))
        aug_input['image'] = imgls[0]
        aug_output = self.transform(**aug_input)
        zposarr = np.array(zposarr).astype(float)
        zdirn = (zposarr[1:] - zposarr[:-1]).mean()
        
        slice_stats = torch.arange(start_pos, (start_pos+self.cfg.sequence_len))
        slice_stats = (slice_stats / len(self.df.loc[samp.name])).float()
        out = {'image': torch.stack([aug_output[f'image{t}'] for t,i in enumerate(imgls)])}
        out['label'] = torch.tensor(dfseq[self.cfg.target].values)
        out['StudyUID'] = torch.tensor(int(idd.split('.')[-1]))
        out['slice_number'] = torch.tensor(dfseq.slice_number.values)
        out['slice_stats'] = slice_stats
        
        # flip if they are going in the opposite direction
        if zdirn>0:
            out['label'] = torch.flip(out['label'], [0])
            out['image'] = torch.flip(out['image'], [0])
            out['slice_number'] = torch.flip(out['slice_number'], [0])
            out['slice_stats'] = 1 - torch.flip(out['slice_stats'], [0])
        # print(len(out['image']))        
        if (len(out['image'])!=self.cfg.sequence_len//3):
            return None
        
        return out
    
    def read_dicom(self, dcmfile):
        if 'placeholder' in dcmfile:
            return np.zeros((512,512))-1000
        D = pydicom.dcmread(dcmfile)
        zpos = D.ImagePositionPatient[-1]
        # D.PhotometricInterpretation = 'YBR_FULL'
        m = float(D.RescaleSlope)
        b = float(D.RescaleIntercept)
        D = D.pixel_array.astype('float')*m
        D = D.astype('float')+b
        return D, zpos

    def list_dcm_dir(self, dcmfile):
        dcmfiles = sorted(glob.glob(f'{cfg.data_folder}/{idd}/*'))
        dcmnums = [int(i.split('/')[-1].split('.')[0]) for i in dcmfiles]
        dcmfiles = [x for _, x in sorted(zip(dcmnums, dcmfiles))]
        return dcmfiles
    
    def windowfn(self, img, WL=400, WW=1800):
        upper, lower = WL+WW//2, WL-WW//2
        X = np.clip(img.copy(), lower, upper)
        X = X - np.min(X)
        X = X / np.max(X)
        X = (X*255.0).astype('uint8')
        return X
    
    def windowsfn(self, X, windows):
        return cv2.merge([self.windowfn(X, *w) for t, w in enumerate(windows)])

