"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""
from thop import profile
from thop import clever_format
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import copy
import random 
import numpy as np 
from typing import List 

from ...core import register


__all__ = ['RTDETR', ]


@register()
class RTDETR(nn.Module):
    __inject__ = ['backbone', 'encoder', 'decoder', ]

    def __init__(self, \
        backbone: nn.Module, 
        encoder: nn.Module, 
        decoder: nn.Module, 
    ):
        super().__init__()
        self.backbone = backbone
        self.decoder = decoder
        self.encoder = encoder
        
    def forward(self, x, targets=None, test=False):
        B, T, C, H, W = x.size()
        
        x = x.reshape(B*T, C, H, W)
        memory = []
        x = self.backbone(x)
        '''
        try:
          macs, params = profile(self.encoder, inputs=(copy.deepcopy(x), ), verbose = False)
          macs, params = macs/(B*T), params/(B*T)
          print(f"tracking macs = {macs/1e9}G")
          print(f"tracking params = {params/1e6}M")
        except:
          pass
        '''
        x = self.encoder(x) 
        if test:
            memory += copy.deepcopy(x)
        x, head_query, eye_query = self.decoder(x, B, T, targets, test=test)
        if test:
            return x, memory, head_query, eye_query
        else:
            return x
    
    def deploy(self, ):
        self.eval()
        for m in self.modules():
            if hasattr(m, 'convert_to_deploy'):
                m.convert_to_deploy()
        return self 
