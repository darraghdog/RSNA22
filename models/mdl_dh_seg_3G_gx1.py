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

class Attention(nn.Module):
    def __init__(self, feature_dim, step_dim, bias=True, **kwargs):
        super(Attention, self).__init__(**kwargs)
        
        self.supports_masking = True

        self.bias = bias
        self.feature_dim = feature_dim
        self.step_dim = step_dim
        self.features_dim = 0
        
        weight = torch.zeros(feature_dim, 1)
        nn.init.xavier_uniform_(weight)
        self.weight = nn.Parameter(weight)
        
        if bias:
            self.b = nn.Parameter(torch.zeros(step_dim))
        
    def forward(self, x, mask=None):
        feature_dim = self.feature_dim
        step_dim = self.step_dim

        eij = torch.mm(
            x.contiguous().view(-1, feature_dim), 
            self.weight
        ).view(-1, step_dim)
        
        if self.bias:
            eij = eij + self.b
            
        eij = torch.tanh(eij)
        a = torch.exp(eij)
        
        if mask is not None:
            a = a * mask

        a = a / torch.sum(a, 1, keepdim=True) + 1e-10

        weighted_input = x * torch.unsqueeze(a, -1)
        return torch.sum(weighted_input, 1)

class SpatialDropout(nn.Dropout2d):
    def forward(self, x):
        x = x.unsqueeze(2)    # (N, T, 1, K)
        x = x.permute(0, 3, 2, 1)  # (N, K, 1, T)
        x = super(SpatialDropout, self).forward(x)  # (N, K, 1, T), some features are masked
        x = x.permute(0, 3, 2, 1)  # (N, T, 1, K)
        x = x.squeeze(2)  # (N, T, K)
        return x

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
        hidden_size = self.backbone.fc.in_features
        self.backbone.fc = torch.nn.Identity()
        

        for param in self.backbone.parameters():
            param.requires_grad = False

        self.posemb = SinusoidalPositionalEmbedding(202, 16, 201)
        self.poschemb = SinusoidalPositionalEmbedding(4, 16, 3)
        
        self.rnn_dim = rnn_dim = hidden_size # + self.posemb.embedding_dim + self.poschemb.embedding_dim
        self.rnn = nn.LSTM(rnn_dim,
                           rnn_dim,
                           batch_first=True, 
                           num_layers=cfg.rnn_num_layers,
                           dropout=cfg.rnn_dropout,
                           bidirectional=True)
        self.attention = Attention(self.rnn_dim*2, self.cfg.rnn_seq_len)
        
        self.head = nn.Linear(rnn_dim * 4 ,1 + len(cfg.target_seg))
        self.dropout = nn.Dropout(cfg.head_dropout)
        
        self.loss_fn = nn.BCEWithLogitsLoss(reduction='none')
        self.weights_plus = torch.nn.Parameter(torch.tensor(cfg.loss_weights['+']), 
                                               requires_grad = False)
        self.weights_neg  = torch.nn.Parameter(torch.tensor(cfg.loss_weights['-']), 
                                               requires_grad = False)

    def forward(self, batch):
        
        y = batch['label']
        embls = []
        with torch.no_grad():
            for m in torch.split(batch['image'], self.cfg.cnn_chunk_size):
                if m.shape[0]==1:
                    embls.append(self.backbone(m.repeat(2,1,1,1))[:1])
                else:
                    embls.append( self.backbone(m))
                    
        embs = torch.cat(embls)
        seqlen = 1 + batch['imageidx'].max().item()
        embs = [embs[batch['imageidx']==i] for i in range(seqlen)]
        if self.cfg.interpolation_mode == 'linear':
            embs = [F.interpolate(e.permute(1,0).unsqueeze(0),
                              size = (self.cfg.rnn_seq_len),
                              mode=self.cfg.interpolation_mode)[0].permute(1,0) \
                                if len(e)> self.cfg.rnn_seq_len else e for e in embs]
        else:
            embs = [F.interpolate(e.unsqueeze(0).unsqueeze(0), 
                              size = (self.cfg.rnn_seq_len, self.rnn_dim), 
                              mode=self.cfg.interpolation_mode)[0,0] \
                                if len(e)> self.cfg.rnn_seq_len else e for e in embs]

        # pad first seq to desired length
        embs[0] = nn.ConstantPad1d((0, 0, 0, self.cfg.rnn_seq_len - embs[0].shape[0]), 0)(embs[0])

        embs = pad_sequence(embs).permute(1,0,2)
        mask = (embs.abs().sum(-1)>0.).long()
        #print(embs.shape)

        h_lstm1, _ = self.rnn(embs)
        max_pool, _ = torch.max(h_lstm1, 1)
        #print(h_lstm1.shape)
        att_pool = self.attention(h_lstm1, mask)
        logits = torch.cat((max_pool, att_pool), 1)
        logits = self.dropout(logits)
        logits = self.head(logits)
        
        if not self.cfg.offline_inference:
            loss = self.loss_fn(logits, y.to(logits.dtype))
            weights = y * self.weights_plus + (1 - y) * self.weights_neg
            loss = (loss * weights).sum(axis=1)
            w_sum = weights.sum(axis=1)
            loss = torch.div(loss, w_sum)
            loss = loss.mean()
            
        else:
            loss = -1.
        
        output = {
            'logits': logits,
            'loss' : loss,
            }
        return output
