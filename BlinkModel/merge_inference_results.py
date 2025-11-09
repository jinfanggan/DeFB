import os
import h5py
import torch
import json
import numpy as np
from tqdm import tqdm
import argparse
from datetime import datetime
from model import BlinkTransformerDecoder
import importlib.util
import sys

def calculate_iou(pred_interval, gt_interval):
    intersection = max(0, min(pred_interval[1], gt_interval[1]) - max(pred_interval[0], gt_interval[0]))
    union = max(pred_interval[1], gt_interval[1]) - min(pred_interval[0], gt_interval[0])
    return intersection / union if union > 0 else 0

class Eval:
    def __init__(self, path, sample_points, window_size=64, stride=56, batch_size=64, is_load_memory = True):
        # 保持文件打开状态
        self.h5_file = h5py.File(path, 'r')
        self.is_load_memory = is_load_memory
        self.vid = self.h5_file['video_id']
        self.person_id = self.h5_file['person_id']
        self.sample_points = sample_points
        if self.is_load_memory:
          sample_num = self.h5_file['blink_features'].shape[0]
          self.test_features = []
          self.head_feature = []
          self.eye_feature = []
          
          for sample_id in range(sample_num):
              total_length = self.h5_file['blink_features'][sample_id].shape[0] // (self.sample_points* 256)
              
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
        
        self.window_size = window_size
        self.stride = stride
        self.batch_size = batch_size

    def evaluate_model_from_h5(self, model, device):
        model.eval()
        results = []

        with torch.no_grad():
            for i in tqdm(range(len(self.test_features))):
                # 获取单个样本的特征和标签
                if self.is_load_memory:
                  features = self.test_features[i]
                  total_length = features.shape[0]
                  head_query_all = self.head_feature[i]
                  eye_query_all = self.eye_feature[i]
                else:
                  total_length = self.test_features[i].shape[0] // (self.sample_points * 256)
                  features = self.test_features[i].reshape(total_length, self.sample_points * 256)
                  head_query_all = self.head_feature[i].reshape(total_length, -1)
                  eye_query_all = self.eye_feature[i].reshape(total_length, -1)

                if total_length < self.window_size:
                    padding_length = self.window_size - total_length
                    padding_feature = torch.zeros(padding_length, self.sample_points * 256)
                    padding_query = torch.zeros(padding_length, 256)
                    features = torch.cat([features, padding_feature], dim=0)
                    head_query_all = torch.cat([head_query_all, padding_query], dim=0)
                    eye_query_all = torch.cat([eye_query_all, padding_query], dim=0)

                # 滑窗推理，汇总所有得分
                scores = []
                batch_windows = []
                batch_head_windows = []
                batch_eye_windows = []

                for start_idx in range(0, max(total_length - self.window_size + 1, 1), self.stride):
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
                
                full_scores = np.zeros(total_length, dtype=np.float32)
                count_scores = np.zeros(total_length, dtype=np.int32)

                if total_length < self.window_size:
                    score= scores[0][1][:total_length]
                    full_scores += score
                    count_scores += 1
                else:
                    for start_idx, score in scores:
                        full_scores[start_idx:start_idx + len(score)] += score
                        count_scores[start_idx:start_idx + len(score)] += 1

                # 平均化重叠区域的得分
                valid_indices = count_scores > 0
                full_scores[valid_indices] /= count_scores[valid_indices]

                # 存储预测结果到 results 列表中
                result = {
                    "video_id": self.vid[i],
                    'person_id':self.person_id[i],
                    "full_scores": full_scores.tolist()  # 将 numpy 数组转换为 list 以存储为 JSON
                }
                
                results.append(result)

        return results

    def close(self):
        if self.h5_file:
            self.h5_file.close()

def merge_results(write_path, prediction_results, output_path):
    # 读取原始推理结果文件
    try:
        with open(write_path, 'r') as f:
            original_results = json.load(f)
    except FileNotFoundError:
        print(f"Error: {write_path} not found.")
        return

    # 创建一个字典来快速查找预测结果
    prediction_dict = {
        (prediction['video_id'].decode('utf-8'), prediction['person_id'].decode('utf-8')): prediction
        for prediction in prediction_results
    }

    # 合并结果
    for instance in original_results:
        video_id = str(instance['video_id'])
        person_id = str(instance['instance_id'])
        # 检查是否有对应的预测结果
        if (video_id, person_id) in prediction_dict:
            # 更新原始结果中对应的眨眼区间和分数
            prediction = prediction_dict[(video_id, person_id)]
            instance['blink_scores'] = prediction['full_scores']
        else:
            # 如果没有对应的推理结果，填充全 0 的分数
            bbox_len = len(instance['bboxes'])  # 通过 'bboxes' 键来获取时间长度
            instance['blink_scores'] = [0.0] * bbox_len  # 填充全 0 的分数

    # 将合并后的结果写入新的 JSON 文件
    #os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(original_results, f)
    
    print(f"Updated results saved to {output_path}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge prediction results into existing JSON file.")
    parser.add_argument('--read_path', type=str, default="/data/data4/zengwenzheng/detrs-blink/results/test_results/pred.json",
                        help="Path to the original JSON file generated by test_instblink_plus_eye_q_only.py.")
    parser.add_argument('--h5_file_path', type=str, default="/data/data4/zengwenzheng/detrs-blink/BinkDetectionDataset/omni_blink_test_dataset_new.h5",
                        help="Path to the h5 file used for evaluation.")
    parser.add_argument('--output_path', type=str, default="/data/data4/zengwenzheng/detrs-blink/results/test_results/pred_score.json",
                        help="Path to save the updated JSON file.")
    parser.add_argument('--config', type=str, required=True, help='Path to the config file')
    args = parser.parse_args()

    config_path = args.config
    spec = importlib.util.spec_from_file_location("config", args.config)
    config_module = importlib.util.module_from_spec(spec)
    sys.modules["config"] = config_module
    spec.loader.exec_module(config_module)
    config = config_module.Config()

    # 加载模型
    device = "cuda"
    model = BlinkTransformerDecoder(map_size = config.sample_point, infer_len=config.window_size, roi_feature_encoder = config.roi_feature_encoder)
    print(config.model_path)
    model.load_state_dict(torch.load(config.model_path, map_location=torch.device('cpu')))
    total_params = sum(p.numel() for p in model.parameters())
    print('total_params:',total_params)
    model.to(device)
    model.eval()

    # 加载数据集并进行推理
    evaler = Eval(path=args.h5_file_path, sample_points = config.sample_point, window_size=config.window_size, stride=config.stride_sample)
    prediction_results = evaler.evaluate_model_from_h5(model, device)
    evaler.close()

    # 合并结果并保存
    merge_results(args.read_path, prediction_results, args.output_path)

