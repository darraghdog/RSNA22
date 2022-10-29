import numpy as np
import sys
import pandas as pd
import re
import json
import platform
import os
import torch
import glob
import ast
from PIL import Image, ImageStat
import cv2
import imagesize
import pylibjpeg
import pydicom
import copy
import collections
from tqdm import tqdm
from torch.utils.data import Dataset
from utils import get_logger, set_pandas_display
import albumentations as A
from albumentations.pytorch import ToTensorV2
from multiprocessing import Pool
from utils import prepare_mask, fetch_filename
from bounding_box import bounding_box as bbfn

tqdm.pandas()
set_pandas_display()
logger = get_logger('Dataset', 'INFO')

def batch_to_device(batch, device):
    batch_dict = {key: batch[key].to(device) for key in batch}
    return batch_dict

def statsfn(fname):
    return cv2.imread(fname, cv2.IMREAD_UNCHANGED).mean()

def statsfn2(fname):
    return cv2.imread(fname, cv2.IMREAD_UNCHANGED).max()

def statsfn3(fname):
    img = cv2.imread(fname, cv2.IMREAD_UNCHANGED)
    return np.count_nonzero(img) / img.size

def cropdim(fname, threshold=0):
    flatImage = cv2.imread(fname, cv2.IMREAD_UNCHANGED)
    rows = np.where(np.max(flatImage, 0) > threshold)[0]
    cols = np.where(np.max(flatImage, 1) > threshold)[0]
    return [rows[0], cols[0], rows[-1] + 1, cols[-1] + 1]


def create_train_transforms(cfg):
    return A.Compose([
        A.Resize(*cfg.imagesize, interpolation = 1, p=1), # interpolation = cv2.INTER_CUBIC,
        A.VerticalFlip(p=cfg.vflip ),
        A.HorizontalFlip(p=cfg.hflip ),
        A.Transpose(p=cfg.transpose ),
        A.RandomContrast(limit=cfg.RandomContrast, p=0.75),
        A.ShiftScaleRotate(shift_limit=cfg.shift_limit,
                           scale_limit=cfg.scale_limit,
                           value = 0,
                           rotate_limit=cfg.rotate_limit,
                           p=0.75,
                           border_mode = cv2.BORDER_CONSTANT),
        
        A.Cutout(num_holes=cfg.holes,
                 max_h_size=int(cfg.hole_size*cfg.imagesize[0]),
                 max_w_size=int(cfg.hole_size*cfg.imagesize[1]),
                 fill_value=0,
                 always_apply=True, p=0.75),
        A.Normalize(mean=cfg.norm_mean[:],
                    std=cfg.norm_std[:], p=1.0),
        ToTensorV2()
        ], bbox_params=A.BboxParams(format='albumentations', \
                                    label_fields=['class_labels']))


def create_valid_transforms(cfg):
    return A.Compose([
        A.Resize(*cfg.imagesize, p=1), # interpolation = cv2.INTER_CUBIC,
        A.Normalize(mean=cfg.norm_mean[:],
                    std=cfg.norm_std[:], p=1.0),
        ToTensorV2()
        ], bbox_params=A.BboxParams(format='albumentations', \
                                    label_fields=['class_labels']))


def collate_fn(batch):
    # Remove error reads
    batch = [b for b in batch if b is not None]

    batchout = {'image' : torch.stack([b['image'] for b in batch]),
                'labels' : torch.stack([b['labels'] for b in batch]),
                'StudyUID' : torch.stack([b['StudyUID'] for b in batch]),
                'slice_numbers' : torch.stack([b['slice_numbers'] for b in batch]),
                }
    
    return batchout

tr_collate_fn = collate_fn
val_collate_fn = collate_fn

'''
df = pd.read_csv(cfg.train_df)
mode="train"
class self:
    1
self = CustomDataset(df, cfg, mode = 'train')
batch = [self.__getitem__(i) for i in tqdm(range(0, 1000, 50))]
batch = tr_collate_fn(batch)
batch = batch_to_device(batch, 'cpu')
'''

class CustomDataset(Dataset):
    
    #def __init__(self, df, mode, tokenizer, max_pos, smoothbreaks = False):
    def __init__(self, df, cfg, aug=None, mode="train"):

        self.cfg = cfg
        self.mode = mode
        self.df = df.copy()
        if 'has_box' not in self.df.columns:
            keycols = 'StudyInstanceUID slice_number fold'.split()
            self.df = self.df[keycols]
            self.df['x0 y0'.split()] = 0
            self.df['x1 y1'.split()] = 511
            self.df['has_box'] = 1

        self.df = self.df[~self.df.StudyInstanceUID.isin(cfg.drop_scans)].reset_index(drop = True)
        
        # Slight offset in dummy bbox to make albumentations work
        self.df.loc[self.df.has_box==0, 'x0 y0'.split()]-=2
        self.df.loc[self.df.has_box==0, 'x1 y1'.split()]+=2
        
        if platform.system() == 'Darwin':
            local_dirs = glob.glob(f'{cfg.data_folder}/*')
            local_dirs = [i.split('/')[-1] for i in local_dirs]
            self.df = self.df[df.StudyInstanceUID.isin(local_dirs)].reset_index(drop = True)
        self.windows = self.cfg.windows #[ (500, 1800), (400, 650), (80, 300)]
        
        if mode=='train':
            self.transform = create_train_transforms(cfg)
        else:
            self.transform = create_valid_transforms(cfg)

        self.df = self.df.sort_values(['StudyInstanceUID', 'slice_number'])
        
        # Always take three slice, so cannot take the last two
        #slice_number_start = \
        #    self.df.groupby('StudyInstanceUID')['slice_number'].transform(max) - 2
        
        self.df['start_pos'] = self.df.groupby('StudyInstanceUID')['slice_number'] \
                                .apply(lambda x: (len(x)-2) * [True] + 2*[False]).explode().tolist()
        #self.df['start_pos'] = self.df.slice_number <= slice_number_start 
        if self.mode!='train':
            # In validation/test have each slice in one image only
            val_start_pos = \
            self.df.groupby('StudyInstanceUID')['slice_number'].transform(lambda x: ([1,0,0] * len(x))[:len(x)])
            self.df['start_pos'].loc[val_start_pos==0] = False
        self.ids = self.df[self.df['start_pos']].index.tolist()
        idx = 20
        
    def __len__(self):
        return len(self.ids)
    
    def __getitem__(self, idx):
        ''' 
        slice_number = self.df.loc[self.ids[idx]].slice_number
        idd = self.df.loc[self.ids[idx]].StudyInstanceUID
        dfseq = self.df.set_index('StudyInstanceUID').loc[idd]
        dfseq = dfseq.set_index('slice_number').loc[slice_number:].iloc[:3]
        '''

        dfseq = self.df.loc[self.ids[idx]:][:3]
        dfseq = dfseq.set_index('slice_number')
        idd = dfseq.StudyInstanceUID.iloc[0]

        imgls = []
        for slcnum in dfseq.index:
            dcmfile = f'{self.cfg.data_folder}/{idd}/{int(slcnum)}.dcm'
            if self.cfg.load_jpg:
                jpgfile = dcmfile.replace('_images/', '_images_jpg') + '.jpg'
                img = cv2.imread(jpgfile, cv2.IMREAD_GRAYSCALE)
            else:
                X, zpos = self.read_dicom(dcmfile)
                img = self.windowsfn(X, self.windows)
            imgls.append(img)
        # if len(imgls) != 3:  print(dfseq)
        assert len(imgls) == 3
        img = cv2.merge(imgls)
        
        if self.mode == 'train':
            if np.random.random():
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        bbox = dfseq['x0 y0'.split()].min().tolist() + \
                    dfseq['x1 y1'.split()].max().tolist()
        bbox = np.array([bbox]) / 512
        
        
        uid = torch.tensor(int(idd.split('.')[-1]))
        slice_numbers = torch.tensor(dfseq.index.tolist())

        class_labels = ['has_box'] 
        try:
            transformed = self.transform(image=img, bboxes=bbox, class_labels=class_labels)
            mat = transformed['image']
            bboxes = transformed['bboxes'][0]
        except:
            # print(transformed['bboxes'])
            return None
        labels = torch.tensor(list(bboxes) + [dfseq.has_box.max()])
        if self.mode == 'test':
            mat = mat.half()
            labels = labels.half()

        outd = {'image' : mat , 
                'StudyUID': uid,
                'slice_numbers': slice_numbers, 
                'labels' : labels}
        
        return outd
    
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

