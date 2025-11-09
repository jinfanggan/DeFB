import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os
import random
from tqdm import tqdm

class OmniBlinkMixedDataset(Dataset):
    def __init__(self, dataset_dir, sample_points, window_size, stride, seed=42, load_to_memory=True):
        """
        dataset_dir: 包含各个子数据集的目录路径，例如:
            {
                'fully_supervised': 'path/to/fully_supervised_blink_dataset_dir/',
                'count_annotated': 'path/to/count_annotated_blink_dataset_dir/',
                'unsupervised': 'path/to/unsupervised_blink_dataset_dir/',
                'point_supervision': 'path/to/point_supervision_blink_dataset_dir/',
                'sliding_window_supervision': 'path/to/sliding_window_supervision_blink_dataset_dir/'
            }
        load_to_memory: 是否将数据加载到内存中，并使用 float16 精度
        """
        random.seed(seed)
        np.random.seed(seed)
        self.dataset_dir = dataset_dir
        self.window_size = window_size
        self.stride = stride
        self.load_to_memory = load_to_memory
        self.h5_files = {}
        self.sample_points = sample_points
        # 打开所有 H5 文件
        self.open_h5_files()

        # 存储所有滑窗的索引 (dataset_name, file_path, sample_idx, start_idx, end_idx)
        self.sliding_windows = []
        self.precompute_sliding_windows()
        print(len(self.sliding_windows))
        
    def open_h5_files(self):
        # 遍历 dataset_dir，打开所有目录下的 H5 文件
        for dataset_name, dir_path in tqdm(self.dataset_dir.items()):
            h5_files = [os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.endswith('.h5')]
            self.h5_files[dataset_name] = []
            for file_path in tqdm(h5_files):
                h5_file = h5py.File(file_path, 'r')
                if self.load_to_memory:
                    # 将数据加载到内存，并转换为 float16 精度
                    dataset = {
                        'blink_features': [],
                        'head_query': [],
                        'eye_query': [],
                        'blink_gt': [],
                        'mask':[]
                    }
                    
                    for sample_id in range(h5_file['blink_features'].shape[0]):
                        total_length = h5_file['blink_features'][sample_id].shape[0] // (self.sample_points * 256)
        
                        blink_features =  torch.tensor(h5_file['blink_features'][sample_id], dtype=torch.float32).reshape(total_length, self.sample_points * 256)
                        head_query = torch.tensor(h5_file['head_query'][sample_id], dtype=torch.float32).reshape(total_length, 256)
                        eye_query = torch.tensor(h5_file['eye_query'][sample_id], dtype=torch.float32).reshape(total_length, 256)
                        blink_gt = torch.tensor(h5_file['blink_gt'][sample_id], dtype=torch.long)
                        mask = torch.tensor(h5_file['mask'][sample_id], dtype=torch.bool)
                        dataset['blink_features'].append(blink_features)
                        dataset['head_query'].append(head_query)
                        dataset['eye_query'].append(eye_query)
                        dataset['blink_gt'].append(blink_gt)
                        dataset['mask'].append(mask)
                        
                    self.h5_files[dataset_name].append(dataset)
                    h5_file.close()
                  
                else:
                    self.h5_files[dataset_name].append(h5_file)

    def precompute_sliding_windows(self):
        # 遍历所有 H5 文件，计算滑窗索引
        for dataset_name, h5_files in self.h5_files.items():
            for h5_file in h5_files:
                if isinstance(h5_file, dict):  # 数据已加载到内存
                    blink_features = h5_file['blink_features']
                    num_samples = len(blink_features)
                else:  # 数据未加载到内存，直接从文件读取
                    blink_features = h5_file['blink_features']
                    num_samples = blink_features.shape[0]
                    
                for i in range(num_samples):
                    total_length = blink_features[i].shape[0] 
                    if total_length < self.window_size:
                        continue
                    num_windows = (total_length - self.window_size) // self.stride + 2
                    for j in range(num_windows):
                        if j != num_windows - 1:
                          start_idx = j * self.stride
                          end_idx = start_idx + self.window_size
                          self.sliding_windows.append((dataset_name, h5_file, i, start_idx, end_idx))
                        else:
                          start_idx = total_length - self.window_size
                          end_idx = total_length
                          self.sliding_windows.append((dataset_name, h5_file, i, start_idx, end_idx))

    def __len__(self):
        return len(self.sliding_windows)

    def __getitem__(self, idx):
        # 获取滑窗索引
        dataset_name, h5_file, sample_idx, start_idx, end_idx = self.sliding_windows[idx]
        
        if self.load_to_memory:
            blink_features_all = h5_file['blink_features'][sample_idx]
            head_query_all = h5_file['head_query'][sample_idx]
            eye_query_all = h5_file['eye_query'][sample_idx]
            blink_gt_flat = h5_file['blink_gt'][sample_idx]
            mask = h5_file['mask'][sample_idx]
        else:
            blink_features_flat = h5_file['blink_features'][sample_idx]
            total_length = blink_features_flat.shape[0] // (self.sample_points * 256)
            blink_features_all = blink_features_flat.reshape(total_length, self.sample_points * 256)
            head_query_all = h5_file['head_query'][sample_idx]
            head_query_all = head_query_all.reshape(total_length, 256)
            eye_query_all = h5_file['eye_query'][sample_idx]
            eye_query_all = eye_query_all.reshape(total_length, 256)
            blink_gt_flat = h5_file['blink_gt'][sample_idx]
            mask = h5_file['mask'][sample_idx]
            
        #print(head_query_all[0].shape, len(head_query_all))
        blink_features = blink_features_all[start_idx:end_idx]
        head_query = head_query_all[start_idx:end_idx]
        eye_query = eye_query_all[start_idx:end_idx]
        gt_mask = mask[start_idx:end_idx]
        
        # 根据监督类型获取标签
        if dataset_name == 'fully_supervised':
            blink_gt = blink_gt_flat.reshape(-1, 3)
            label = torch.zeros(self.window_size).long()
            for i in range(self.window_size):
                for blink in blink_gt:
                    if start_idx + i >= blink[0] and start_idx + i <= blink[1]:
                        label[i] = 1
            label *= gt_mask

        elif dataset_name == 'count_annotated':
            blink_gt = blink_gt_flat.reshape(-1, 3)
            blink_count = 0
            for blink in blink_gt:
              if blink[0] > end_idx or blink[1] < start_idx:
                continue
              elif start_idx <= blink[0] < end_idx and start_idx <= blink[1] < end_idx:
                blink_count += 1
           
            label = torch.tensor([blink_count], dtype=torch.float)

        elif dataset_name == 'point_supervision':
            blink_gt = blink_gt_flat  # 这里不是区间，是直接随机选的点
            label = torch.zeros(self.window_size).long()
            for blink_point in blink_gt:
                blink_point = int(blink_point)
                if start_idx <= blink_point < end_idx:
                    label[blink_point - start_idx] = 1

        elif dataset_name == 'catagory_supervision':
            blink_gt = blink_gt_flat.reshape(-1, 3)
            # 检查滑窗内是否有眨眼
            has_blink = any(start_idx <= blink[0] < end_idx and start_idx <= blink[1] < end_idx for blink in blink_gt)
            label = torch.tensor([1 if has_blink else 0], dtype=torch.long)

        elif dataset_name == 'unsupervised':
            label = None

        return blink_features, head_query, eye_query, label, dataset_name



# 自定义 collate_fn 函数，直接收集为列表
def custom_collate_fn(batch):
    features, head_query, eye_query, labels, dataset_names = zip(*batch)
    features = torch.stack(features, dim=0)
    head_query = torch.stack(head_query, dim=0)
    eye_query = torch.stack(eye_query, dim=0)
    labels = list(labels)
    dataset_names = list(dataset_names)
    return features, head_query, eye_query, labels, dataset_names

# 使用 DataLoader
if __name__ == "__main__":
    dataset_dir = {
        'fully_supervised': 'path/to/fully_supervised_blink_dataset_dir/',
        'count_annotated': 'path/to/count_annotated_blink_dataset_dir/',
        'unsupervised': 'path/to/unsupervised_blink_dataset_dir/',
        'point_supervision': 'path/to/point_supervision_blink_dataset_dir/',
        'sliding_window_supervision': 'path/to/sliding_window_supervision_blink_dataset_dir/'
    }
    window_size = 16
    stride = 8
    batch_size = 32

    dataset = OmniBlinkMixedDataset(dataset_dir, window_size, stride)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)

    for i, (features, labels, dataset_names) in enumerate(data_loader):
        print(f"Batch {i+1}")
        print(f"Features: {features.shape}, Labels: {len(labels)}, Dataset types: {dataset_names}")
        if i == 4:
            break
