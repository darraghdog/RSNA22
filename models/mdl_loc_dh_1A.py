# https://github.com/GuanshuoXu/RSNA-STR-Pulmonary-Embolism-Detection/blob/main/trainall/lung_localization/splitall/train0.py
from torch import nn
import torch
import math
from torch.nn import functional as F
from torch.nn.parameter import Parameter
import numpy as np
from transformers import AutoModel, AutoConfig
import torch.utils.checkpoint
from torch.nn.functional import pad
# import segmentation_models_pytorch as smp
# from efficientnet_pytorch import EfficientNet
import timm

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
                                          num_classes=len(self.cfg.targets[:4]))
        self.bboxfc = self.backbone.classifier
        self.backbone.classifier = torch.nn.Identity()
        self.fc = torch.nn.Linear(in_features=self.bboxfc.in_features, 
                                      out_features=1, 
                                      bias=True)
        self.criterion = torch.nn.L1Loss()
        self.criterionbce = nn.BCEWithLogitsLoss( )
        
    def forward(self, batch):
        
        labels_bb = batch['labels'][:,:4]
        labels_cls = batch['labels'][:,-1:].clip(0.05, 0.95)
        x = batch['image']
        x = self.backbone(x)
        logits_cls = self.fc(x)
        logits_bb = self.bboxfc(x)
        
        out = torch.cat((logits_bb, torch.sigmoid(logits_cls)), 1)
        
        if not self.cfg.offline_inference:
            idx = labels_cls.flatten()>0.5
            loss_bb = self.criterion(logits_bb[idx],labels_bb[idx])
            loss_cls = self.criterionbce(logits_cls,labels_cls)
            loss = self.cfg.loss_weights[0] * loss_bb + self.cfg.loss_weights[1] * loss_cls
            if loss_bb!=loss_bb:
                print(logits_bb[idx])
                print(labels_bb[idx])
        else:
            loss = -1
            loss_bb = -1
            loss_cls = -1
        
        return {'preds': out, 
                'loss' : loss, 
                'StudyUID': batch['StudyUID'], 
                'slice_numbers': batch['slice_numbers'], 
                'loss_bbox' : loss_bb, 
                'loss_cls' : loss_cls}
