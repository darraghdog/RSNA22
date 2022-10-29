# pip install scikit-multilearn
import os
os.chdir('..')
import sys
import glob
import torch 
import platform
import nibabel as nib
from tqdm import tqdm
import re
from PIL import Image
import pydicom
import cv2
import pandas as pd
import random
import numpy as np 
from utils import set_pandas_display
from utils import read_dicom, windowsfn
from os.path import abspath, dirname
set_pandas_display()

# Load the training studies
trndf = pd.read_csv('datamount/train_folded_v01.csv')
'''
Write out a file with all slices
'''
trnalldf = glob.glob('datamount/train_images/**/*.dcm', recursive = True)
trnalldf = pd.Series(trnalldf).str.split('/', expand = True).iloc[:,-2:]
trnalldf.columns = keys = 'StudyInstanceUID slice_number'.split()
trnalldf['slice_number'] = trnalldf.slice_number.str.split('.').str[0].astype(int)
trnalldf['fold'] = trndf.set_index('StudyInstanceUID').loc[trnalldf.StudyInstanceUID].fold.values
trnalldf = trnalldf.sort_values(keys).set_index('StudyInstanceUID')

trnalldf.to_csv('datamount/train_all_slices_v01.csv.gz')

'''
For the labelled segmentations, obtain the z-axis direction from the dicoms
'''
# Get the keys of the labelled segmentations
keycols = 'StudyInstanceUID slice_number'.split()
segls = glob.glob('datamount/segmentations/*.nii')
segls = set([i.split('/')[-1].replace('.nii', '') for i in segls])

# Get the names of the associated dicoms for these studies
dcmfiles = glob.glob('datamount/train_images/*/*.*')
dcmfiles = pd.Series(dcmfiles).str.split('/', expand=True).iloc[:, -2:]
dcmfiles.columns = keycols
dcmfiles.slice_number = dcmfiles.slice_number.str.replace('.dcm', '').astype(int)
dcmfiles = dcmfiles[dcmfiles.StudyInstanceUID.isin(segls)].reset_index(drop = True)

# Load the dicoms and obtain the z-axis position
dposns = []
for t,row in tqdm(dcmfiles.iterrows(), total = len(dcmfiles)):
    dnm = f'datamount/train_images/{row.StudyInstanceUID}/{row.slice_number}.dcm'
    D = pydicom.dcmread(dnm)
    dposns.append(D.ImagePositionPatient)

# Calculate Z-axis direction for each study
dcmfiles['xpos ypos zpos'.split()] = np.array(dposns)
trnsdf = trndf.merge(dcmfiles, on = 'StudyInstanceUID')
trnsdf = trnsdf.sort_values('StudyInstanceUID fold slice_number'.split()).reset_index(drop = True)
zdirndf = trnsdf.groupby('StudyInstanceUID')['zpos'].apply(lambda x: x.values)
zdirndf = zdirndf.apply(lambda x: (x[1:]-x[:-1]).mean())

'''
Load the segmentations, align z-axis direction and find a study level 
bounding box based on outer segmentation position
'''

def bboxfn(image, threshold=0):
    if len(image.shape) == 3:
        flatImage = np.max(image, 2)
    else:
        flatImage = image
    if flatImage.max() == 0:
        ycenter = flatImage.shape[0]//2
        xcenter = flatImage.shape[1]//2
        bbox = (xcenter, ycenter, xcenter, ycenter, 0)
        return bbox
        
    rows = np.where(np.max(flatImage, 0) > threshold)[0]
    cols = np.where(np.max(flatImage, 1) > threshold)[0]
    bbox = (rows[0], cols[0], rows[-1], cols[-1], 1)
    return bbox

cols = 'x0 y0 x1 y1 has_box'.split()
colsdim = 'height width'.split()
trnsdf[cols  + colsdim] = -1

segls = glob.glob('datamount/segmentations/*.nii')

for segnm in tqdm(segls):
    #print(segnm)
    seg = nib.load(segnm).get_fdata()
    segid = segnm.split('/')[-1].replace('.nii', '')
    if zdirndf[segid]<0:
        seg = seg[:, ::-1, ::-1].transpose(2, 1, 0)
    else:
        seg = seg[:, ::-1].transpose(2, 1, 0)
    '''
    The provided segmentation labels have values of 1 to 7 for C1 to C7 (seven cervical 
        vertebrae) and 8 to 19 for T1 to T12 (twelve thoracic vertebrae are located in 
        the center of your upper and middle back), and 0 for everything else.
    '''
    # Remove T type = horacic vertebrae
    seg[seg>7] = 0.
    bboxmat = np.array([bboxfn(s) for s in seg])
    idx = trnsdf.StudyInstanceUID==segid
    trnsdf.loc[idx, cols] = bboxmat
    trnsdf.loc[idx, colsdim] = bboxmat[:,2:4] - bboxmat[:,:2]

trnsdf = trnsdf[keycols + cols + colsdim  + ['fold'] ]

trnsdf.to_csv('datamount/train_bbox_v01.csv.gz', index = False)


