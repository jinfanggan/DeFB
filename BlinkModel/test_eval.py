import os
import h5py
import torch
import numpy as np
from tqdm import tqdm
from datetime import datetime
from scipy.optimize import linear_sum_assignment

def calculate_iou(pred_interval, gt_interval):
    intersection = max(0, min(pred_interval[1], gt_interval[1]) - max(pred_interval[0], gt_interval[0]))
    union = max(pred_interval[1], gt_interval[1]) - min(pred_interval[0], gt_interval[0])
    return intersection / union if union > 0 else 0

class Eval:
    def __init__(self, path, sample_points, window_size=64, stride=56, batch_size=64, is_load_memory = True):
        # 保持文件打开状态
        self.h5_file = h5py.File(path, 'r')
        self.is_load_memory = is_load_memory
        self.sample_points = sample_points
        if self.is_load_memory:
          sample_num = self.h5_file['blink_features'].shape[0]
          self.test_features = []
          self.head_feature = []
          self.eye_feature = []
          
          for sample_id in range(sample_num):
              #print(self.h5_file['head_query'][sample_id].shape[0])
              total_length = self.h5_file['blink_features'][sample_id].shape[0] // (self.sample_points * 256)
             
              blink_features =  torch.tensor(self.h5_file['blink_features'][sample_id], dtype=torch.float32).reshape(total_length, self.sample_points * 256)
              head_query = torch.tensor(self.h5_file['head_query'][sample_id], dtype=torch.float32).reshape(total_length, 256)
              eye_query = torch.tensor(self.h5_file['eye_query'][sample_id], dtype=torch.float32).reshape(total_length, 256)
        
              
              self.test_features.append(blink_features)
              self.head_feature.append(head_query)
              self.eye_feature.append(eye_query)
          
        else:
          self.test_features = self.h5_file['blink_features']
          self.head_feature = self.h5_file['head_query']
          self.eye_feature = self.h5_file['eye_query']
        
        self.test_gt = self.h5_file['blink_gt']
        self.window_size = window_size
        self.stride = stride
        self.batch_size = batch_size

    def evaluate_model_from_h5(self, model, device, iou_thresholds=[0.5,0.6,0.7,0.8,0.9]):
        model.eval()
        # 初始化 TP、FP、FN 的总计数
        all_tp = {iou_thresh: 0 for iou_thresh in iou_thresholds}
        all_fp = {iou_thresh: 0 for iou_thresh in iou_thresholds}
        all_fn = {iou_thresh: 0 for iou_thresh in iou_thresholds}

        with torch.no_grad():
            for i in tqdm(range(self.test_gt.shape[0])):
                # 获取单个样本的特征和标签
                if self.is_load_memory:
                  features = self.test_features[i]
                  total_length = features.shape[0]
                  gt_blink_intervals = self.test_gt[i].reshape(-1, 3)
                  head_query_all = self.head_feature[i]
                  eye_query_all = self.eye_feature[i]
                else:
                  total_length = self.test_features[i].shape[0] // (self.sample_points * 256)
                  features = self.test_features[i].reshape(total_length, self.sample_points * 256)
                  gt_blink_intervals = self.test_gt[i].reshape(-1, 3)
                  head_query_all = self.head_feature[i].reshape(total_length, -1)
                  eye_query_all = self.eye_feature[i].reshape(total_length, -1)

                # 滑窗推理，汇总所有得分
                scores = []
                batch_windows = []
                batch_head_windows = []
                batch_eye_windows = []

                for start_idx in range(0, total_length - self.window_size + 1, self.stride):
                    end_idx = start_idx + self.window_size
                    window_features = features[start_idx:end_idx]
                    head_features = head_query_all[start_idx:end_idx]
                    eye_features = eye_query_all[start_idx:end_idx]

                    batch_windows.append(window_features)
                    batch_head_windows.append(head_features)
                    batch_eye_windows.append(eye_features)

                    # 如果批量大小已达到设置的值，则进行推理
                    if len(batch_windows) == self.batch_size:
                        batch_windows_tensor = torch.stack(batch_windows).to(device)
                        batch_head_tensor = torch.stack(batch_head_windows).to(device)
                        batch_eye_tensor = torch.stack(batch_eye_windows).to(device)

                        batch_scores = model(batch_windows_tensor, batch_head_tensor, batch_eye_tensor)
                        batch_scores = torch.softmax(batch_scores, dim=-1)[..., -1]  # 获取眨眼类别的得分

                        for b_idx, score in enumerate(batch_scores):
                            scores.append((start_idx - (self.batch_size - b_idx - 1) * self.stride, score.cpu().numpy()))

                        batch_windows = []
                        batch_head_windows = []
                        batch_eye_windows = []
                        
                # 如果有剩余的窗口，进行推理
                if len(batch_windows) > 0:
                    batch_windows_tensor = torch.stack(batch_windows).to(device)
                    batch_head_tensor = torch.stack(batch_head_windows).to(device)
                    batch_eye_tensor = torch.stack(batch_eye_windows).to(device)

                    batch_scores = model(batch_windows_tensor, batch_head_tensor, batch_eye_tensor)
                    batch_scores = torch.softmax(batch_scores, dim=-1)[..., -1]  # 获取眨眼类别的得分

                    for b_idx, score in enumerate(batch_scores):
                        scores.append((start_idx - (len(batch_windows) - b_idx - 1) * self.stride, score.cpu().numpy()))

                # 汇总所有窗口得分到全局分数数组
                full_scores = np.zeros(total_length, dtype=np.float32)
                count_scores = np.zeros(total_length, dtype=np.int32)

                for start_idx, score in scores:
                    full_scores[start_idx:start_idx + len(score)] += score
                    count_scores[start_idx:start_idx + len(score)] += 1

                # 平均化重叠区域的得分
                valid_indices = count_scores > 0
                full_scores[valid_indices] /= count_scores[valid_indices]

                # 获取预测的眨眼区间
                pred_blink_intervals = []
                current_start = -1
                for t in range(total_length):
                    if full_scores[t] > 0.4:  # 0.5 是置信度阈值，可以调整
                        if current_start == -1:
                            current_start = t
                    else:
                        if current_start != -1:
                            if t - 1 - current_start >= 2 and t - 1 - current_start <= 15 :
                                pred_blink_intervals.append((current_start, t - 1))
                                current_start = -1
                if current_start != -1:
                    pred_blink_intervals.append((current_start, total_length - 1))

                # 计算每个阈值下的 TP, FP, FN
                iou = np.zeros((len(pred_blink_intervals), len(gt_blink_intervals)))
                for i, pred in enumerate(pred_blink_intervals):
                    for j, gt in enumerate(gt_blink_intervals):
                        iou[i, j] = calculate_iou(pred, gt)
                       
                for iou_thresh in iou_thresholds:
                    tp, fp, fn = 0, 0, 0
                    matched = []

                    # 计算预测区间与 GT 区间的 IOU
                    for i, pred in enumerate(pred_blink_intervals):
                        best_iou = 0
                        best_gt_idx = -1
                        for j, gt in enumerate(gt_blink_intervals):
                            if j not in matched:
                                if iou[i, j] > best_iou:
                                    best_iou = iou[i, j]
                                    best_gt_idx = j

                        if best_iou >= iou_thresh:
                            tp += 1
                            matched.append(best_gt_idx)
                        else:
                            fp += 1

                    fn += len(gt_blink_intervals) - len(matched)
                    
                    # 更新 TP, FP, FN 的总计数
                    all_tp[iou_thresh] += tp
                    all_fp[iou_thresh] += fp
                    all_fn[iou_thresh] += fn

        # 计算每个 IOU 阈值下的精确率、召回率，并计算 AP
        precisions = {}
        recalls = {}
        avg_precisions = {}

        for iou_thresh in iou_thresholds:
            tp = all_tp[iou_thresh]
            fp = all_fp[iou_thresh]
            fn = all_fn[iou_thresh]

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0

            precisions[iou_thresh] = precision
            recalls[iou_thresh] = recall

            # 使用精确率和召回率计算 AP
            if precision + recall > 0:
                avg_precisions[f'blink-ap@{iou_thresh}'] = 2 * (precision * recall) / (precision + recall)
            else:
                avg_precisions[f'blink-ap@{iou_thresh}'] = 0

            print(f"IOU Threshold {iou_thresh}: Precision = {precision}, Recall = {recall}")

        return avg_precisions




