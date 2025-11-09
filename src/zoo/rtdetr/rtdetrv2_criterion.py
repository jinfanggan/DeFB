"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""
import json
import torch 
import torch.distributed as dist
import torch.nn as nn 
import torch.distributed
import torch.nn.functional as F 
import torchvision
import copy
from .box_ops import box_cxcywh_to_xyxy, box_iou, generalized_box_iou
from ...misc.dist_utils import get_world_size, is_dist_available_and_initialized
from ...core import register

class focal_loss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, num_classes = 2, size_average=True):
     
        super(focal_loss,self).__init__()
        self.size_average = size_average
        if isinstance(alpha,list):
            assert len(alpha)==num_classes  
            self.alpha = torch.Tensor(alpha)
        else:
            assert alpha<1  
            self.alpha = torch.zeros(num_classes)
            self.alpha[0] += alpha
            self.alpha[1:] += (1-alpha) 

        self.gamma = gamma

    def forward(self, preds, labels):
        """
        focal_loss损失计算
        :param preds:   预测类别. size:[B,N,C] or [B,C]    分别对应与检测与分类任务, B批次, N检测框数, C类别数
        :param labels:  实际类别. size:[B,N] or [B]        [B*N个标签(假设框中有目标)]，[B个标签]
        :return:
        """
                
        #固定类别维度，其余合并(总检测框数或总批次数)，preds.size(-1)是最后一个维度
        preds = preds.view(-1,preds.size(-1))
        self.alpha = self.alpha.to(preds.device)
        
        #使用log_softmax解决溢出问题，方便交叉熵计算而不用考虑值域
        preds_logsoft = F.log_softmax(preds, dim=1) 
        
     	#log_softmax是softmax+log运算，那再exp就算回去了变成softmax
        preds_softmax = torch.exp(preds_logsoft)    
   
        # 这部分实现nll_loss ( crossentropy = log_softmax + nll)
        preds_softmax = preds_softmax.gather(1,labels.view(-1,1)) 
        preds_logsoft = preds_logsoft.gather(1,labels.view(-1,1))

        self.alpha = self.alpha.gather(0,labels.view(-1))

        # torch.pow((1-preds_softmax), self.gamma) 为focal loss中 (1-pt)**γ
        
        #torch.mul 矩阵对应位置相乘，大小一致
        loss = -torch.mul(torch.pow((1-preds_softmax), self.gamma), preds_logsoft) 
    
        #torch.t()求转置
        loss = torch.mul(self.alpha, loss.t())
        #print(loss.size()) [1,5]
        
        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
       
        return loss

@register()
class RTDETRCriterionv2(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    __share__ = ['num_classes', ]
    __inject__ = ['matcher', ]

    def __init__(self, \
        matcher, 
        weight_dict, 
        losses, 
        alpha=0.2, 
        gamma=2.0, 
        num_classes=1, 
        action_classes=2,
        boxes_weight_format=None,
        share_matched_indices=False):
        """Create the criterion.
        Parameters:
            matcher: module able to compute a matching between targets and proposals
            num_classes: number of object categories, omitting the special no-object category
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            boxes_weight_format: format for boxes weight (iou, )
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.action_classes = action_classes
        self.weight_dict = weight_dict
        self.losses = losses 
        #self.classifier_loss = focal_loss()
        self.classifier_loss = nn.CrossEntropyLoss(ignore_index=-1)
        self.boxes_weight_format = boxes_weight_format
        self.share_matched_indices = share_matched_indices
        self.alpha = alpha
        self.gamma = gamma

    def loss_labels_focal(self, outputs, targets, indices, num_boxes):
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        target = F.one_hot(target_classes, num_classes=self.num_classes+1)[..., :-1]
        loss = torchvision.ops.sigmoid_focal_loss(src_logits, target, self.alpha, self.gamma, reduction='none')
        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes

        return {'loss_focal': loss}
    
    def loss_blink(self, outputs, targets, indices, num_boxes, values=None):
        idx = self._get_src_permutation_idx(indices)
        
        if values is None:
            src_boxes = outputs['pred_head_boxes'][idx]
            target_boxes = torch.cat([t['head_boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
            ious, _ = box_iou(box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes))
            ious = torch.diag(ious).detach()
        else:
            ious = values 
        src_logits = outputs['pred_blinks']
        target_classes_o = torch.cat([t["blink_gt"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.action_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        target = F.one_hot(target_classes, num_classes=self.action_classes + 1)[..., :-1]

        target_score_o = torch.zeros_like(target_classes, dtype=src_logits.dtype)
        target_score_o[idx] = ious.to(target_score_o.dtype)
        target_score = target_score_o.unsqueeze(-1) * target

        pred_score = F.sigmoid(src_logits).detach()
        weight = self.alpha * pred_score.pow(self.gamma) * (1 - target) + target_score

        loss = F.binary_cross_entropy_with_logits(src_logits, target_score, weight=weight, reduction='none')
        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes
        return loss
        
    def loss_labels_vfl(self, outputs, targets, indices, num_boxes, values=None):
        idx = self._get_src_permutation_idx(indices)
        
        if values is None:
            src_boxes = outputs['pred_head_boxes'][idx]
            target_boxes = torch.cat([t['head_boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
            ious, _ = box_iou(box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes))
            ious = torch.diag(ious).detach()
        else:
            ious = values 
        src_logits = outputs['pred_logits']
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        target = F.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :-1]

        target_score_o = torch.zeros_like(target_classes, dtype=src_logits.dtype)
        target_score_o[idx] = ious.to(target_score_o.dtype)
        target_score = target_score_o.unsqueeze(-1) * target

        pred_score = F.sigmoid(src_logits).detach()
        weight = self.alpha * pred_score.pow(self.gamma) * (1 - target) + target_score

        loss = F.binary_cross_entropy_with_logits(src_logits, target_score, weight=weight, reduction='none')
        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes
        return {'loss_vfl': loss}

    def loss_boxes(self, outputs, targets, indices, num_boxes, boxes_weight=None):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        losses = {}

        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_head_boxes'][idx]
        device = src_boxes.device
        target_boxes = torch.cat([t['head_boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        
        losses['loss_bbox_head'] = torch.tensor(0.0).to(device)   
        losses['loss_giou_head'] = torch.tensor(0.0).to(device)   

        mask_head = []
        mask_eye = []
        for t, (_, i) in zip(targets, indices):
          for tgt_id in i:
            if torch.sum(t['head_boxes'][tgt_id][2:]) > 0:
              mask_head.append(1)
            else:
              mask_head.append(0)
            if torch.sum(t['eye_boxes'][tgt_id][2:]) > 0:
              mask_eye.append(1)
            else:
              mask_eye.append(0)
              
        mask_head = torch.tensor(mask_head, dtype=torch.bool).to(device)      
        mask_eye = torch.tensor(mask_eye, dtype=torch.bool).to(device)

        
        if num_boxes > 0:
          src_boxes, target_boxes = src_boxes[mask_head], target_boxes[mask_head]
          
          loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
          losses['loss_bbox_head'] += loss_bbox.sum() / num_boxes
          loss_giou = 1 - torch.diag(generalized_box_iou(\
              box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes)))
         
          loss_giou = loss_giou if boxes_weight is None else loss_giou * boxes_weight
  
          losses['loss_giou_head'] += loss_giou.sum() / num_boxes
          
        
        if num_boxes > 0 and 'pred_eye_boxes' in outputs.keys():
        
          losses['loss_bbox_eye'] = torch.tensor(0.0).to(device)   
          losses['loss_giou_eye'] = torch.tensor(0.0).to(device)   
          idx = self._get_src_permutation_idx(indices)
          src_boxes = outputs['pred_eye_boxes'][idx]
          target_boxes = torch.cat([t['eye_boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
  
          src_boxes, target_boxes = src_boxes[mask_eye], target_boxes[mask_eye]
          loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
          losses['loss_bbox_eye'] += loss_bbox.sum() / num_boxes
          
          loss_giou = 1 - torch.diag(generalized_box_iou(\
              box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes)))
         
          loss_giou = loss_giou if boxes_weight is None else loss_giou * boxes_weight
  
          losses['loss_giou_eye'] += loss_giou.sum() / num_boxes
        
        if 'pred_blinks' in outputs.keys():
          #losses['loss_blinks'] = self.loss_blink(outputs, targets, indices, num_boxes)
          losses['loss_blinks'] = torch.tensor(0.0).to(device) 
          idx = self._get_src_permutation_idx(indices)
          src_blink = outputs['pred_blinks'][idx]
          target_blink = torch.cat([t['blink_gt'][i] for t, (_, i) in zip(targets, indices)], dim=0)
          
          losses['loss_blinks'] += self.classifier_loss(src_blink.reshape(-1, 2), target_blink.reshape(-1))
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'boxes': self.loss_boxes,
            'focal': self.loss_labels_focal,
            'vfl': self.loss_labels_vfl,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def transform_targets(self, targets):
        targets_loss = []
        num_boxes = 0
        for i in range(len(targets)):
            head_bbox = targets[i]['head_bbox']
            eye_bbox = targets[i]['eye_bbox']
            blink_gt = targets[i]['blink_gt']
            T = head_bbox.size(1)
            N = head_bbox.size(0)
            device = eye_bbox.device
            for j in range(T):
                target_frame = {'head_boxes': [], 'eye_boxes':[],'labels':[], 'blink_gt':[]}
                for k in range(N):
                    target_frame['head_boxes'].append(head_bbox[k, j])
                    target_frame['eye_boxes'].append(eye_bbox[k, j])
                    target_frame['blink_gt'].append(blink_gt[k, j])
                    if torch.sum(head_bbox[k, j]) > 0:
                        target_frame['labels'].append(0)
                        num_boxes += 1
                    else:
                        target_frame['labels'].append(1)

                target_frame['head_boxes'] = torch.stack(target_frame['head_boxes'], dim=0)
                target_frame['eye_boxes'] = torch.stack(target_frame['eye_boxes'], dim=0)
                target_frame['blink_gt'] = torch.stack(target_frame['blink_gt'], dim=0)
                target_frame['labels'] = torch.tensor(target_frame['labels']).to(device, dtype=torch.long)
                targets_loss.append(target_frame)
                
        return targets_loss, num_boxes
    
    def forward(self, outputs, targets, **kwargs):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if 'aux' not in k}
        match_results = []
        
        targets_for_match = copy.deepcopy(targets)
        targets, num_boxes = self.transform_targets(targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        
        if is_dist_available_and_initialized():
            torch.distributed.all_reduce(num_boxes)
  
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()
        
        # Retrieve the matching between the outputs of the last layer and the targets
        matched = self.matcher(outputs_without_aux, targets_for_match)
        indices = matched['indices']
        match_results.append(matched['mat'])
        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            meta = self.get_loss_meta_info(loss, outputs, targets, indices)            
            l_dict = self.get_loss(loss, outputs, targets, indices, num_boxes, **meta)
            l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
            losses.update(l_dict)

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                if not self.share_matched_indices:
                    matched = self.matcher(aux_outputs, targets_for_match)
                    indices = matched['indices']
                    match_results.append(matched['mat'])
                    
                for loss in self.losses:
                    meta = self.get_loss_meta_info(loss, aux_outputs, targets, indices)
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **meta)
                    l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                    l_dict = {k + f'_aux_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        # In case of cdn auxiliary losses. For rtdetr
        if 'dn_aux_outputs' in outputs:
            assert 'dn_meta' in outputs, ''
            indices = self.get_cdn_matched_indices(outputs['dn_meta'], targets)
            dn_num_boxes = num_boxes * outputs['dn_meta']['dn_num_group']
            for i, aux_outputs in enumerate(outputs['dn_aux_outputs']):
                for loss in self.losses:
                    meta = self.get_loss_meta_info(loss, aux_outputs, targets, indices)
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, dn_num_boxes, **meta)
                    l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                    l_dict = {k + f'_dn_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        # In case of encoder auxiliary losses. For rtdetr v2
        if 'enc_aux_outputs' in outputs:
            assert 'enc_meta' in outputs, ''
          
            for i, aux_outputs in enumerate(outputs['enc_aux_outputs']):
                indices_encoder = self.matcher.forward_encoder(aux_outputs, targets)['indices']
                for loss in self.losses:
                    meta = self.get_loss_meta_info(loss, aux_outputs, targets, indices_encoder)
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices_encoder, num_boxes, **meta)
                    l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                    l_dict = {k + f'_enc_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)
        
        #self.match_results_IS(match_results)
        return losses
    
    def match_results_IS(self, match_results):
        IS = 0
        for i in range(len(match_results) - 1):
          pre_macth = match_results[i]
          cur_match = match_results[i + 1]

          for j in range(len(pre_macth)):
            pre_macth_single = pre_macth[j]
            cur_match_single = cur_match[j]
            
            IS += torch.sum(torch.abs(pre_macth_single - cur_match_single))/len(pre_macth)
        
        if is_dist_available_and_initialized():
            torch.distributed.all_reduce(IS)
        
        IS = torch.clamp(IS / get_world_size(), min=0)
        
        if dist.get_rank() == 0:
          with open('match_stable.json', 'r') as f:
            results = json.load(f)
          
          results.append(IS.item())
          
          with open('match_stable.json', 'w') as f:
            json.dump(results, f)
             
        
    def get_loss_meta_info(self, loss, outputs, targets, indices):
        if self.boxes_weight_format is None:
            return {}

        src_boxes = outputs['pred_boxes'][self._get_src_permutation_idx(indices)]
        target_boxes = torch.cat([t['boxes'][j] for t, (_, j) in zip(targets, indices)], dim=0)

        if self.boxes_weight_format == 'iou':
            iou, _ = box_iou(box_cxcywh_to_xyxy(src_boxes.detach()), box_cxcywh_to_xyxy(target_boxes))
            iou = torch.diag(iou)
        elif self.boxes_weight_format == 'giou':
            iou = torch.diag(generalized_box_iou(\
                box_cxcywh_to_xyxy(src_boxes.detach()), box_cxcywh_to_xyxy(target_boxes)))
        else:
            raise AttributeError()

        if loss in ('boxes', ):
            meta = {'boxes_weight': iou}
        elif loss in ('vfl', ):
            meta = {'values': iou}
        else:
            meta = {}

        return meta

    @staticmethod
    def get_cdn_matched_indices(dn_meta, targets):
        """get_cdn_matched_indices
        """
        dn_positive_idx, dn_num_group = dn_meta["dn_positive_idx"], dn_meta["dn_num_group"]
        num_gts = [len(t['labels']) for t in targets]
        device = targets[0]['labels'].device
      
        dn_match_indices = []
        for i, num_gt in enumerate(num_gts):
            if num_gt > 0:
                gt_idx = torch.arange(num_gt, dtype=torch.int64, device=device)
                gt_idx = gt_idx.tile(dn_num_group)
                assert len(dn_positive_idx[i]) == len(gt_idx)
                dn_match_indices.append((dn_positive_idx[i], gt_idx))
            else:
                dn_match_indices.append((torch.zeros(0, dtype=torch.int64, device=device), \
                    torch.zeros(0, dtype=torch.int64,  device=device)))
        
        return dn_match_indices
