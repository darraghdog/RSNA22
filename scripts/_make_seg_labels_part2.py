# pip install scikit-multilearn
import os
import sys
import glob
import platform
import nibabel as nib
from tqdm import tqdm
import pydicom
import numpy as np 
import pandas as pd


cols_pred = [f'C{i}_pred' for i in range(1,8)]
cols_frac = [f'C{i}_frac' for i in range(1,8)]
cols_c = [f'C{i}' for i in range(1,8)]
key_cols = 'StudyInstanceUID slice_number'.split()
trndf = pd.read_csv('datamount/train_folded_v01.csv')
trnldf = pd.read_csv('datamount/train_image_level_v03.csv.gz')

'''
Load up vertebrae predictions which we want to aggregate
'''
vprednms = sorted(glob.glob(f'weights/cfg_dh_seg_02G_test/fold*/*.pth'))
vprednms += sorted(glob.glob(f'weights/cfg_dh_seg_04A_test//fold*/*.pth'))
vprednms += sorted(glob.glob(f'weights/cfg_dh_seg_04F_test//fold*/*.pth'))
val_datas = [torch.load(v, map_location=torch.device('cpu')) for v in vprednms]
slnums = torch.cat([v['slice_number'] for v in val_datas])
stuids = torch.cat([v['StudyUID'].unsqueeze(1).repeat(1, len(slnums[0])) for v in val_datas])
preds = torch.cat([v['logits'] for v in val_datas])
dd = pd.DataFrame({'StudyInstanceUID': [f'1.2.826.0.1.3680043.{i}' for i in stuids.flatten().numpy()], 
                  'slice_number': slnums.flatten().numpy(), })
dd[cols_pred] = preds.view(-1, 7).numpy()
# We have multiple predictions for each slice, as it is repeated in different windows
# We average these results
dd = dd.groupby(key_cols)[cols_pred].mean().reset_index()

dd[cols_c+['zpos']] = dd.merge(trnldf, on = key_cols, how = 'left', sort=False)[cols_c+['zpos']] 
dd[cols_pred] = dd[cols_pred].clip(0, 1)

'''
Create fracture labels by multiplying the study level vertebrae 
fracture by the slice level vertebrae prediction
'''

dd['fold'] = trndf.set_index('StudyInstanceUID').loc[dd.StudyInstanceUID].fold.values
dd[cols_frac] = dd[cols_pred].values * \
                trndf.set_index('StudyInstanceUID').loc[dd.StudyInstanceUID][cols_c].values


# Fix up and write out
dd = dd.drop(cols_c, 1)
dd = dd.drop(['zpos'], 1)

args = {'index' : False, 'float_format': '%.3f'}
dd.to_csv('datamount/train_image_pseu_v04_crop.csv.gz', **args)