import numpy as np
from sklearn.metrics import f1_score
import torch
import scipy as sp
from sklearn.metrics import log_loss, roc_auc_score
import torch.nn.functional as F

def calc_metric(cfg, pp_out, val_df, pre="val"):

    pred_df = pp_out
    if cfg.loss=='auc':
        y_act = pred_df['fracture'].values
        y_pred = pred_df.preds.values
        score = roc_auc_score(y_act, y_pred)
    
    if cfg.loss=='mse':
        cols_pred = [i+'_pred' for i in cfg.target]
        criterion = torch.nn.MSELoss(reduction='none')
        if pp_out.isna().sum().sum()>0:
            idx =  pp_out.isna().sum(1)>0
            print(f'Dropping na : {idx.mean():0.4f}')
            pp_out = pp_out[~idx]
            
        act = torch.from_numpy(pp_out[cfg.target].values)
        pred = torch.from_numpy(pp_out[cols_pred].clip(0,1).values)
        #print(act[:20])
        #print(pred[:20])
        loss = criterion(pred, act)#.mean()
        loss[act!=0] *= 2 # cfg.positive_weight
        score = loss.mean()

    if hasattr(cfg, "neptune_run"):
        cfg.neptune_run[f"{pre}/score/"].log(score, step=cfg.curr_step)
        print(f"{pre} score: {score:.6}")
    
    return score
