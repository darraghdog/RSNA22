import copy
import pandas as pd
import torch
from torch.nn import functional as F
from tqdm import tqdm
import numpy as np
import itertools
from operator import itemgetter

'''
val_df = df.copy()
'''

def post_process_pipeline(cfg, val_data, val_df):
    pass
    '''
    preds = val_data['preds'].float().cpu().numpy()
    imgs = val_df['image'].tolist()
    
    colnames = val_df.filter(regex = 'min|max').columns.tolist()
    pred_df = pd.DataFrame(preds, 
                           index = imgs, 
                           columns = colnames)
            
    return pred_df
    '''
