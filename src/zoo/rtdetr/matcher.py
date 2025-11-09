"""
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
Modules to compute the matching cost and solve the corresponding LSAP.

Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np
from scipy.optimize import linear_sum_assignment
from typing import Dict 

from .box_ops import box_cxcywh_to_xyxy, generalized_box_iou, box_iou

from ...core import register


@register()
class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    __share__ = ['use_focal_loss', ]

    def __init__(self, weight_dict, use_focal_loss=False, alpha=0.25, gamma=2.0):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = weight_dict['cost_class']
        self.cost_bbox = weight_dict['cost_bbox']
        self.cost_giou = weight_dict['cost_giou']

        self.use_focal_loss = use_focal_loss
        self.alpha = alpha
        self.gamma = gamma

        assert self.cost_class != 0 or self.cost_bbox != 0 or self.cost_giou != 0, "all costs cant be 0"
        
    @torch.no_grad()
    def matcher_infer(self, target, src, return_frame = False):
       N1, N2, T = src.size(1), target.size(1), target.size(0)

       matrix = torch.zeros(N1, N2).to(src.device)
       weight = torch.zeros(N1, N2).to(src.device)
       matrix_all = []
       for i in range(T):
          src_bbox, tgt_bbox = src[i,:,:4], target[i,:,:4]
          cost_giou = box_iou(box_cxcywh_to_xyxy(src_bbox), box_cxcywh_to_xyxy(tgt_bbox))[0]
 
          for k in range(N2):
             if torch.sum(tgt_bbox[k]) > 1e-06:
                weight[:,k] += 1
     
          matrix += cost_giou
          matrix_all.append(cost_giou.transpose(0, 1))
          
       matrix = matrix / (weight + 1e-8)
       matrix = matrix.transpose(0, 1)
       
       if return_frame:
         return matrix.detach().cpu().numpy(), matrix_all
       return matrix.detach().cpu().numpy()
       
    @torch.no_grad()   
    def forward_encoder(self, outputs: Dict[str, torch.Tensor], targets):
        bs, num_queries = outputs["pred_logits"].shape[:2]
        device = outputs["pred_logits"].device
        # We flatten to compute the cost matrices in a batch
        if self.use_focal_loss:
            out_prob = F.sigmoid(outputs["pred_logits"].flatten(0, 1))
        else:
            out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]

        out_bbox = outputs["pred_head_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]

        # Also concat the target labels and boxes
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["head_boxes"] for v in targets])

        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class].
        # The 1 is a constant that doesn't change the matching, it can be ommitted.
        tgt_ids = torch.zeros_like(tgt_ids).to(device)
        if self.use_focal_loss:
            out_prob = out_prob[:, tgt_ids]
            neg_cost_class = (1 - self.alpha) * (out_prob ** self.gamma) * (-(1 - out_prob + 1e-8).log())
            pos_cost_class = self.alpha * ((1 - out_prob) ** self.gamma) * (-(out_prob + 1e-8).log())
            cost_class = pos_cost_class - neg_cost_class        
        else:
            cost_class = -out_prob[:, tgt_ids]

        # Compute the L1 cost between boxes
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

        # Compute the giou cost betwen boxes
        #print(out_bbox.size(), tgt_bbox.size())
        cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))
        
        # Final cost matrix
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        C = C.view(bs, num_queries, -1).cpu()

        sizes = [len(v["head_boxes"]) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        indices = [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]

        return {'indices': indices}
        
    @torch.no_grad()
    def forward(self, outputs: Dict[str, torch.Tensor], targets):
        
        B, T = len(targets), targets[0]['head_bbox'].size(1)
        _, N1, _ = outputs['pred_head_boxes'].size()
        out_bbox = outputs['pred_head_boxes'].reshape(B, T, N1, -1)
        out_logits = outputs["pred_logits"].reshape(B, T, N1, -1)
        device = outputs["pred_logits"].device
        indices = []
        mats = []
        for i in range(B):
          N2 = targets[i]['head_bbox'].size(0)
          head_bbox = targets[i]['head_bbox'].transpose(0, 1)
          C = torch.zeros(N1, N2).to(device)
          weight = 0
          for j in range(T):
            pred_bbox = out_bbox[i][j] ###N1 4
            pred_logits = out_logits[i][j]
            tgt_bbox = head_bbox[j]### N2 4
          
            mask = []
            for bbox in tgt_bbox:
                if torch.sum(bbox) > 0:
                  mask.append(1)
                else:
                  mask.append(0)
                  
            mask = torch.tensor(mask).reshape(1, N2).to(device)
            weight += mask
            cost_bbox = torch.cdist(pred_bbox, tgt_bbox, p=1) ### N1 N2, loss
            cost_bbox = cost_bbox * mask
            
            cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(pred_bbox), box_cxcywh_to_xyxy(tgt_bbox))
            cost_giou = cost_giou * mask   
         
            out_prob = F.sigmoid(pred_logits)
            cost_class = -out_prob[:, 0].reshape(N1, 1)
            cost_class = cost_class.repeat(1, N2) * mask
       
            C += self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
          
          C /= weight
          src_id, tgt_id = linear_sum_assignment(C.cpu().detach().numpy())
          
          indice = []
          indice.append(torch.tensor(src_id, dtype=torch.int64).to(device))
          indice.append(torch.tensor(tgt_id, dtype=torch.int64).to(device))
          indices += [indice] * T
          
          mat = torch.zeros(N1, N2).to(device)
          for src, tgt in zip(src_id, tgt_id):
            mat[src, tgt] = 1
            
          mats.append(mat)  
        return {'indices': indices, 'mat': mats}
        