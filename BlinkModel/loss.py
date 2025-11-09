import random
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F

        
class BlinkLoss(nn.Module):
    def __init__(self, blink_length=4, alpha=0.75, gamma=2.0):
        super(BlinkLoss, self).__init__()
        self.blink_length = blink_length
        self.relu = nn.ReLU()
        self.alpha = alpha
        self.gamma = gamma
        self.smooth_l1_loss = nn.SmoothL1Loss()
        self.classifier_loss = nn.CrossEntropyLoss(ignore_index=-1)
        #self.classifier_loss = self.focal_loss
        # 初始化损失记录和计数
        self.fully_supervised_loss_total = 0.0
      
        self.fully_supervised_samples = 0

    def focal_loss(self, inputs, targets):
        # 确保 inputs 和 targets 的形状匹配
        T, C = inputs.shape
        inputs = inputs.view(-1, C)
        targets = targets.view(-1)

        # 只计算标签不为 -1 的部分
        valid_indices = targets != -1
        inputs = inputs[valid_indices]
        targets = targets[valid_indices]

        # 将 targets 转为 one-hot 编码以匹配 inputs 的形状
        targets_one_hot = F.one_hot(targets, num_classes=inputs.size(-1)).float()

        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets_one_hot, reduction='none')
        pt = torch.exp(-BCE_loss)  # pt is the probability that the label is correctly classified
        focal_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        return focal_loss.mean()

    def forward(self, blink_score, labels, supervision_types, epoch):
        B = len(blink_score)
        total_loss = 0.0
        w = [1,1,1,1,1]
        for i in range(B):
            supervision_type = supervision_types
            blink_score_i = blink_score[i]
            labels_i = labels[i] if labels is not None else None
            T = blink_score_i.size(0)
          
            if supervision_type == 'fully_supervised':
                # 完全监督的损失
                loss_type = 1
                loss_blink = self.classifier_loss(blink_score_i, labels_i)
                loss = loss_blink 
                self.fully_supervised_loss_total += loss.item()
                self.fully_supervised_samples += 1
                
            total_loss += loss * w[loss_type]

        return total_loss / B


    

    def reset_loss(self):
        self.fully_supervised_loss_total = 0.0
        self.fully_supervised_samples = 0
      
    def print_loss_summary(self):
        if self.fully_supervised_samples > 0:
            print(f"Fully Supervised Loss: {self.fully_supervised_loss_total / self.fully_supervised_samples}")
        

