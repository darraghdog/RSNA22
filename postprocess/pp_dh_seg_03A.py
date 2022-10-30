import pandas as pd
import torch
from torch.nn import functional as F
from tqdm import tqdm
from utils import get_logger, set_pandas_display
set_pandas_display()

def post_process_pipeline(cfg, val_data, val_df):
    
    key_cols = 'StudyInstanceUID slice_number'.split()
    cols_pred = [i+'_pred' for i in cfg.target]
    pp_out = val_df.reset_index()[key_cols + cfg.target].copy()
    return pp_out
    
    '''
    slnums = val_data['slice_number'].cpu()
    stuids = val_data['StudyUID'].unsqueeze(1).repeat(1, len(slnums[0])).cpu()
    preds = val_data['logits'].cpu()
    dd = pd.DataFrame({'StudyInstanceUID': [f'1.2.826.0.1.3680043.{i}' for i in stuids.flatten().numpy()], 
                      'slice_number': slnums.flatten().numpy(), })
    dd[cols_pred] = preds.view(-1, 7).numpy()
    dd = dd.groupby(key_cols)[cols_pred].mean().reset_index()
    
    pp_out[cols_pred] = pp_out.merge(dd, on = key_cols, \
                                 how = 'left', sort=False)[cols_pred]
    # print(pp_out.head(100))
    return pp_out
    '''
