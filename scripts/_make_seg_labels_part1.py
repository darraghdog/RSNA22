# pip install scikit-multilearn
import os
import sys
import glob
import platform
import nibabel as nib
from tqdm import tqdm
import re
from PIL import Image
import pydicom
import numpy as np 
import pandas as pd

def one_hot(m, ncols):
    x = np.zeros(ncols, dtype = int)
    i,c = np.unique(m, return_counts=True)
    x[i.astype(int)] = c
    return x

trndf = pd.read_csv('datamount/train_folded_v01.csv')

'''
Get slice names for each study, as well as study z-axis direction
'''
dcmfiles = glob.glob('datamount/train_images/*/*.*')
dcmfiles = pd.Series(dcmfiles).str.split('/', expand=True).iloc[:, -2:]
dcmfiles.columns = 'StudyInstanceUID slice_number'.split()
dcmfiles.slice_number = dcmfiles.slice_number.str.replace('.dcm', '').astype(int)
dposns = []
for t,row in tqdm(dcmfiles.iterrows()):
    dnm = f'datamount/train_images/{row.StudyInstanceUID}/{row.slice_number}.dcm'
    D = pydicom.dcmread(dnm)
    dposns.append(D.ImagePositionPatient)
dcmfiles['xpos ypos zpos'.split()] = np.array(dposns)
trnsdf = trndf.merge(dcmfiles, on = 'StudyInstanceUID')

cols = 'StudyInstanceUID fold slice_number'.split()
trnsdf = trnsdf.sort_values(cols).reset_index(drop = True)
zdirndf = trnsdf.groupby('StudyInstanceUID')['zpos'].apply(lambda x: x.values)
zdirndf = zdirndf.apply(lambda x: (x[1:]-x[:-1]).mean())

'''
Create vertebrae label - sum of vertebrae segmentation pixels divided
by the max number pixels of that vertebrae seen in any slice
'''

cols = ['BG'] + [f'C{i}' for i in range(1,8)] + [f'T{i}' for i in range(1, 13)]
trnsdf[cols] = -1
segls = glob.glob('datamount/segmentations/*.nii')

for segnm in tqdm(segls):
    seg = nib.load(segnm).get_fdata()
    segid = segnm.split('/')[-1].replace('.nii', '')
    if zdirndf[segid]<0:
        seg = seg[:, ::-1, ::-1].transpose(2, 1, 0)
    else:
        seg = seg[:, ::-1].transpose(2, 1, 0)
    pixct = np.stack([one_hot(m,len(cols)) for m in seg])
    idx = trnsdf.StudyInstanceUID==segid
    trnsdf.loc[idx, cols] = pixct

ccols = [f'C{i}' for i in range(1,8)]
trnsdf[ccols] = trnsdf[ccols]/trnsdf.groupby('StudyInstanceUID')[ccols].transform(max)
trnsdf.loc[trnsdf.query('T1==-1').index, ccols] = -1

trnsdf.to_csv('datamount/train_image_level_v03.csv.gz', index = False)
