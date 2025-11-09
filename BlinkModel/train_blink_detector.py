import os
import torch
import h5py
import numpy as np
from datetime import datetime
from dataset import OmniBlinkMixedDataset, custom_collate_fn
from model import BlinkTransformerDecoder
from test_eval import Eval
from loss import BlinkLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
import importlib.util
import sys
import argparse
import json
import random

def train_one_epoch(model, optimizer, data_loader, loss_fn, device, epoch, name):
    model.train()
    blink_loss_path = os.path.join('work_dirs', name, 'blink_loss.json')
    with open(blink_loss_path, 'r') as f:
      blink_loss = json.load(f)
      
    for (features, head_query, eye_query, labels, supervision_types) in tqdm(data_loader):
        features = features.to(device)
        head_query = head_query.to(device)
        eye_query = eye_query.to(device)
        T = features.size(1)
        labels = [label.to(device) if label is not None else None for label in labels]

        scores = model(features, head_query, eye_query)
        loss_supervised = loss_fn(scores, labels, 'fully_supervised', epoch)
        
        # 计算总损失并优化
        total_loss = loss_supervised 
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        blink_loss.append(loss_supervised.item())
        
    loss_fn.print_loss_summary()
    loss_fn.reset_loss()
    with open(blink_loss_path, 'w') as f:
      json.dump(blink_loss, f)
      

def init_seeds(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

if __name__ == '__main__':
    # 创建时间戳目录
    # 通过命令行参数加载配置文件
    parser = argparse.ArgumentParser(description="Train Blink Model")
    parser.add_argument('--config', type=str, required=True, help='Path to the config file')
    args = parser.parse_args()
    init_seeds()
    
    config_path = args.config
    spec = importlib.util.spec_from_file_location("config", args.config)
    config_module = importlib.util.module_from_spec(spec)
    sys.modules["config"] = config_module
    spec.loader.exec_module(config_module)
    config = config_module.Config()

    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = os.path.join("work_dirs", config.config_name)
    os.makedirs(save_dir, exist_ok=True)

    # 保存配置到文件
    config_copy_path = os.path.join(save_dir, "config.py")
    with open(config_copy_path, 'w') as f:
        f.write(open(config_path).read())

    # 初始化模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BlinkTransformerDecoder(map_size = config.sample_point, infer_len=config.window_size, roi_feature_encoder = config.roi_feature_encoder).to(device)
    
    if config.load_from is not None:
      ckpt = torch.load(config.load_from, map_location='cpu')
      model_state_dict = model.state_dict()
  
      skipped_keys = []
  
      for key, value in ckpt.items():
          if key in model_state_dict:
              if model_state_dict[key].shape != value.shape:
                  skipped_keys.append(key)
              else:
                  model_state_dict[key] = value
          else:
              skipped_keys.append(key)
  
      # 加载匹配的参数
      model.load_state_dict(model_state_dict, strict=False)
  
      # 打印总结信息
      if skipped_keys:
          print(f"Skipped {len(skipped_keys)} keys due to mismatch or missing in the model.")
      else:
          print("All keys matched successfully.")

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    loss_fn = BlinkLoss()
    evaler = Eval(path=config.test_set, sample_points = config.sample_point, window_size=config.window_size, stride=config.stride_sample)
    evaluation_results = []
    best_ap = 0
    
    blink_loss_path = os.path.join('work_dirs', config.config_name, 'blink_loss.json')
    with open(blink_loss_path, 'w') as f:
      json.dump([], f)
      
    for pipeline in config.pipelines:
        print(f"start {pipeline['stage']} stage!")
        train_set = dict()
        for subset in pipeline['set']:
            if subset in config.train_set.keys():
                train_set[subset] = config.train_set[subset]
            # 加载数据集
        dataset = OmniBlinkMixedDataset(train_set, config.sample_point, config.window_size, config.stride_sample)
        data_loader = DataLoader(dataset, num_workers=16, batch_size=config.batch_size, shuffle=True, collate_fn=custom_collate_fn)
        for epoch in range(pipeline['epoch']):
            print(f"{pipeline['stage']} Epoch {epoch + 1}")
            train_one_epoch(model, optimizer, data_loader, loss_fn, device, epoch, config.config_name)

            if epoch % pipeline['test_gap'] == pipeline['test_gap'] - 1:
                eval_results = evaler.evaluate_model_from_h5(model, device)
                ap = sum(eval_results.values())/5
                torch.save(model.state_dict(), f'work_dirs/{config.config_name}/{epoch}.pth')
                if pipeline['stage'] == 'train' and best_ap <= ap:
                    best_ap = ap
                    torch.save(model.state_dict(), config.model_path)
                    
                evaluation_results.append({"epoch": f"{pipeline['stage']} epoch_{epoch + 1}", "results": eval_results})
                print(config.config_name, ap)

                eval_path = os.path.join(save_dir, "evaluation_results.json")
                with open(eval_path, 'w') as f:
                    json.dump(evaluation_results, f)
            
        if pipeline['test_gap'] > epoch:
            torch.save(model.state_dict(), config.model_path)
