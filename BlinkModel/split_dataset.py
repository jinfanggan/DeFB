import h5py
import numpy as np
import random
import os
import importlib.util
import sys
import argparse
def split_dataset(config, h5_file_path, split_ratios, seed=42, frame_sampling_rate=1, chunk_size=50):
    # 设置随机种子
    random.seed(seed)
    np.random.seed(seed)

    # 打开 HDF5 文件
    with h5py.File(h5_file_path, 'r') as f:
        total_samples = f['blink_gt'].shape[0]

        # 获取所有的 indices，并随机打乱
        all_indices = list(range(total_samples))
        random.shuffle(all_indices)

        # 计算每种监督类型的样本数量
        split_indices = {}
        current_idx = 0
        for i, (label_type, ratio) in enumerate(split_ratios.items()):
            # 如果是最后一个监督类型，直接分配剩余的样本数，避免舍入误差
            if i == len(split_ratios) - 1:
                split_indices[label_type] = all_indices[current_idx:]
            else:
                num_samples = int(total_samples * ratio)
                split_indices[label_type] = all_indices[current_idx:current_idx + num_samples]
                current_idx += num_samples

        # 定义输出目录
        output_base_dir = f'split_blink_datasets/{config.config_name}'
        os.makedirs(output_base_dir, exist_ok=True)

        # 创建标签生成函数
        def generate_labels(label_type, blink_gt_data):
            if label_type == 'point_supervision':
                blink_gt_data = blink_gt_data.reshape(-1, 3)
                # 在眨眼区间中随机选择一帧作为标签
                return [np.random.choice([start, end]) for start, end, _ in blink_gt_data]
            else:
                return blink_gt_data  # 对于其他监督类型，直接返回原始标签

        # 保存数据到各个小文件中
        def save_to_sub_files(indices, dataset_name):
            num_files = len(indices) // chunk_size + 1
            for file_idx in range(num_files):
                start_idx = file_idx * chunk_size
                end_idx = min(start_idx + chunk_size, len(indices))
                if start_idx >= end_idx:
                    break

                # 创建新的小 H5 文件
                output_file_path = os.path.join(output_base_dir, dataset_name, f'{dataset_name}_part_{file_idx + 1}.h5')
                with h5py.File(output_file_path, 'w') as out_f:
                    for key in f.keys():
                        dataset_shape = list(f[key].shape)
                        dataset_shape[0] = end_idx - start_idx

                        out_f.create_dataset(
                            f'{key}', shape=(end_idx - start_idx, *f[key].shape[1:]),
                            dtype=f[key].dtype,
                            chunks=(1, *f[key].shape[1:]),
                            compression='gzip', compression_opts=4
                        )

                        # 保存数据到当前小文件
                        for idx, data_idx in enumerate(indices[start_idx:end_idx]):
                            data = f[key][data_idx]

                            if key == 'blink_gt':
                                # 生成指定的监督标签
                                data = generate_labels(dataset_name, data)

                            out_f[f'{key}'][idx] = data

        # 为每种监督类型保存数据
        for label_type, indices in split_indices.items():
            os.makedirs(os.path.join(output_base_dir, label_type), exist_ok=True)
            save_to_sub_files(indices, label_type)

if __name__ == '__main__':
    # 通过命令行参数加载配置文件
    parser = argparse.ArgumentParser(description="Train Blink Model")
    parser.add_argument('--config', type=str, required=True, help='Path to the config file')
    args = parser.parse_args()
    
    config_path = args.config
    spec = importlib.util.spec_from_file_location("config", args.config)
    config_module = importlib.util.module_from_spec(spec)
    sys.modules["config"] = config_module
    spec.loader.exec_module(config_module)
    config = config_module.Config()

    # 示例调用
    split_ratios = config.split_ratio
    h5_path = config.h5_path
    print('start_split_dataset')
    
    split_dataset(config, h5_path, split_ratios)



