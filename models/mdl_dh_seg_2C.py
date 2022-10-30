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
                                          num_classes=len(self.cfg.target))
        hidden_size = self.backbone.fc.in_features
        self.backbone.fc = torch.nn.Identity()
        
        self.posemb = SinusoidalPositionalEmbedding(202, 16, 201)
        self.poschemb = SinusoidalPositionalEmbedding(4, 16, 3)
        
        rnn_dim = hidden_size + self.posemb.embedding_dim + self.poschemb.embedding_dim
        self.rnn = nn.LSTM(rnn_dim,
                           rnn_dim,
                           batch_first=True, 
                           num_layers=cfg.rnn_num_layers,
                           dropout=cfg.rnn_dropout,
                           bidirectional=True)
        
        self.head = nn.Linear(rnn_dim * 2 ,len(cfg.target))
        #self.head1 = nn.Linear((hidden_size + self.posemb.embedding_dim) * 2,len(cfg.target))
        #self.head2 = nn.Linear((hidden_size + self.posemb.embedding_dim) * 2,len(cfg.target))
        #self.head3 = nn.Linear((hidden_size + self.posemb.embedding_dim) * 2,len(cfg.target))
        self.dropout = nn.Dropout(cfg.head_dropout)
        
        if self.cfg.loss == 'bce':
            self.criterion = torch.nn.BCEWithLogitsLoss() 
        elif self.cfg.loss == 'mse':
            self.criterion = torch.nn.MSELoss(reduction='none')

    def forward(self, batch):
        
        bsize, seqlen, _,_,_  = batch['image'].shape
        
        slice_pos = (batch['slice_stats'] * (self.posemb.padding_idx - 1))
        slice_pos = slice_pos.round().long()
        posemb = self.posemb(slice_pos)#.view(bsize, -1)
        
        # Channel of the slice
        chlmat = torch.tensor([[[0,1,2]]]).repeat(bsize, seqlen, 1)
        chlmat = chlmat.view(bsize, -1).to(posemb.device).long()
        poschemb = self.poschemb(chlmat)
        
        y = batch['label'].float()
        x = batch['image']
        
        embs = [self.backbone(x) for x in batch['image']]
        embs = torch.stack(embs)
        
        # Expand for 3 channels
        embs = embs.repeat_interleave(3,1)
        embs = torch.cat((embs, posemb, poschemb), -1)
        
        logits = self.rnn(embs)[0]
        logits = self.dropout(logits)
        logits = self.head(logits)
        
        if not self.cfg.offline_inference:
            # print(logits.dtype, y.dtype) 
            loss = self.criterion(logits, y)#.mean()
            loss[y!=0] *= self.cfg.positive_weight
            loss = loss.mean()
            
        else:
            loss = -1.
        
        output = {
            'logits': logits,
            'loss' : loss,
            'slice_number': batch['slice_number'], 
            'StudyUID': batch['StudyUID'], 
            }
        return output
