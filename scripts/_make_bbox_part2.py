# pip install scikit-multilearn
import os
# os.chdir('..')
import sys
import glob
import torch 
import platform
import pandas as pd
import numpy as np 
from utils import set_pandas_display
set_pandas_display()

# Load the training studies
trndf = pd.read_csv('datamount/train_folded_v01.csv')

# Get the mapping of each study to fold
mapdf = trndf['StudyInstanceUID fold'.split()].drop_duplicates()
mapdf['uid'] = mapdf.StudyInstanceUID.str.split('.').str[-1].astype(int)
mapdf = mapdf.groupby('fold')['uid'].apply(list)

# Load the bounding box predictions
INDIR = 'weights/cfg_loc_dh_01B_test/'
predls = []
for fold in range(5):
    for fn in glob.glob(f'{INDIR}/fold{fold}/*'):
        print(f'Loading ... {fn}')
        test_data = torch.load(fn, map_location=torch.device('cpu'))
        tmpdf = pd.DataFrame(test_data['preds'].numpy().clip(0, 1), 
                              columns = 'x0 y0 x1 y1 has_bbox'.split())
        tmpdf['uid'] = test_data['StudyUID'].numpy()
        tmpdf['slice_numbers'] = test_data['slice_numbers'].numpy()[:,1]
        tmpdf['slice_numbers_all'] = test_data['slice_numbers'].numpy().tolist()
        # Filter on the fold for that prediction
        tmpdf = tmpdf[tmpdf.uid.isin(mapdf.loc[fold])]
        predls.append(tmpdf)
preddf = pd.concat(predls).reset_index(drop = True)
print(f'All bboxes shape : {preddf.shape}')

# Find the smallest box which covers all C1-C7 vertebrae in each study. 
bbpreddf = pd.concat([ \
    preddf.query('has_bbox > 0.5').groupby('uid')['x0 y0'.split()].apply(min),
    preddf.query('has_bbox > 0.5').groupby('uid')['x1 y1'.split()].apply(max)], 1)

'''
Aggregated bounding box to find where the vertebrae start and end
This will be used to filter out slices in the last stage of training
'''
def window_range(g, thresh = 0.1, window = 5, min_periods=3, center=True):
    bbseqprobas = g.has_bbox.rolling(window, min_periods=min_periods, center=center).mean()
    if bbseqprobas.max() <= thresh:
        sn_from, sn_to = g.slice_numbers.iloc[[0,-1]]
        return [sn_from, sn_to]
    bb_range = np.where(bbseqprobas > thresh)[0]
    sn_from = g.slice_numbers.iloc[bb_range[0]]
    sn_to = g.slice_numbers.iloc[bb_range[-1]]
    return [sn_from, sn_to]

wrls = preddf.groupby('uid').apply(window_range)
wrdf = pd.DataFrame(wrls.tolist(), index = wrls.index, columns = 'slnum_from slnum_to'.split())
bbpreddf['slnum_from slnum_to'.split()] = wrdf.loc[bbpreddf.index]
bbpreddf['slnum_max'] = preddf.groupby('uid')['slice_numbers'].max().loc[bbpreddf.index]

bbpreddf.index = [f'1.2.826.0.1.3680043.{i}'  for i in bbpreddf.index]
bbpreddf.index = bbpreddf.index.rename('StudyInstanceUID')

bbpreddf.iloc[:,:2] = np.floor((bbpreddf.iloc[:,:2] * 512)).astype(int).clip(0, 512)
bbpreddf.iloc[:,2:4] = np.ceil((bbpreddf.iloc[:,2:4] * 512)).astype(int).clip(0, 512)

# Show a histogram of the ratio of slices we would keep per study
((bbpreddf.slnum_to - bbpreddf.slnum_from) / bbpreddf.slnum_max).hist(bins = 50 )

bbpreddf.to_csv('datamount/train_bbox_pred_v02.csv.gz')

