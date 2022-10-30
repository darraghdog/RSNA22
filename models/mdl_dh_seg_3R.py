from torch import nn
import glob
import torch
import math
from torch.nn import functional as F
from torch.nn.parameter import Parameter
import numpy as np
import torch.utils.checkpoint
from torch.nn.functional import pad
import timm
import torchvision
import pdb
import warnings
from typing import Optional
from torch.nn.utils.rnn import pad_sequence

from transformers.models.fsmt.modeling_fsmt import SinusoidalPositionalEmbedding

'''
class self:
    1
'''

class Net(nn.Module):
    def __init__(self, cfg):
        super(Net, self).__init__()

        self.cfg = cfg

        if self.cfg.offline_inference:
            pretrained=False
        else:
            pretrained=True
        self.backbone = timm.create_model(cfg.backbone, 
                                          pretrained=pretrained, 
                                          in_chans= len(cfg.windows)*3, 
                                          num_classes=len(self.cfg.target_seg))
        try:
            hidden_size = self.backbone.fc.in_features
            self.backbone.fc = torch.nn.Identity()
        except:
            hidden_size = self.backbone.classifier.in_features
            self.backbone.classifier = torch.nn.Identity()
        
        self.posemb = SinusoidalPositionalEmbedding(202, 16, 201)
        self.poschemb = SinusoidalPositionalEmbedding(4, 16, 3)
        
        rnn_dim = hidden_size + self.posemb.embedding_dim # + self.poschemb.embedding_dim
        self.rnn = nn.LSTM(rnn_dim,
                           rnn_dim,
                           batch_first=True, 
                           num_layers=cfg.rnn_num_layers,
                           dropout=cfg.rnn_dropout,
                           bidirectional=True)
        
        self.head_seg = nn.Linear(rnn_dim * 2 ,len(cfg.target_seg))
        self.head_frac = nn.Linear(rnn_dim * 2 ,len(cfg.target_frac))
        self.head_frac_all = nn.Linear(rnn_dim * 2 ,1)
        self.dropout = nn.Dropout(cfg.head_dropout)
        
        self.criterion = torch.nn.MSELoss(reduction='none')

    def forward(self, batch):
        
        bsize, seqlen, hh,ww,cc  = batch['image'].shape
        
        slice_pos = (batch['slice_stats'] * (self.posemb.padding_idx - 1))
        slice_pos = slice_pos.round().long()
        posemb = self.posemb(slice_pos[:,1::3])#.view(bsize, -1)
        
        '''
        # Channel of the slice
        chlmat = torch.tensor([[[0,1,2]]]).repeat(bsize, seqlen, 1)
        chlmat = chlmat.view(bsize, -1).to(posemb.device).long()
        poschemb = self.poschemb(chlmat)
        '''
        
        y_seg = batch['label_seg'].float()[:,1::3]
        y_frac = batch['label_frac'].float()[:,1::3]
        y_frac_all = y_frac.max(-1)[0]
        x = batch['image']
        
        x = x.view(bsize * seqlen, hh,ww,cc )
        embs = self.backbone(x) 
        embs = embs.view(bsize, seqlen, -1)
        '''
        # Expand for 3 channels
        embs = embs.repeat_interleave(3,1)
        embs = torch.cat((embs, posemb, poschemb), -1)
        '''
        embs = torch.cat((embs, posemb), -1)
        
        logits = self.rnn(embs)[0]
        
        logits_seg = self.head_seg(self.dropout(logits))
        logits_frac = self.head_frac(self.dropout(logits))
        logits_frac_all = self.head_frac_all(self.dropout(logits))
        logits_frac_all = logits_frac_all.squeeze(-1)
        
        if not self.cfg.offline_inference:
            # print(logits.dtype, y.dtype) 
            loss_seg = self.criterion(logits_seg, y_seg)#.mean()
            loss_seg[y_seg!=0] *= self.cfg.positive_weight
            loss_seg = loss_seg.mean()
            loss_frac = self.criterion(logits_frac, y_frac)#.mean()
            loss_frac[y_frac!=0] *= self.cfg.positive_weight
            loss_frac = loss_frac.mean()
            loss_frac_all = self.criterion(logits_frac_all, y_frac_all)#.mean()
            loss_frac_all[y_frac_all!=0] *= self.cfg.positive_weight
            loss_frac_all = loss_frac_all.mean()
            
            loss = loss_seg + loss_frac + loss_frac_all
            
        else:
            loss = -1.
        
        output = {
            'logits_seg': logits_seg,
            'logits_frac': logits_frac,
            'logits_frac_all': logits_frac_all,
            'loss' : loss,
            'loss_seg' : loss_seg,
            'loss_frac' : loss_frac,
            'loss_frac_all' : loss_frac_all,
            'slice_number': batch['slice_number'], 
            'StudyUID': batch['StudyUID'], 
            }
        return output
