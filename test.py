import os 
import sys 
import h5py
import json
import numpy as np
import copy
import cv2
from thop import profile
from thop import clever_format
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from PIL import Image
import argparse
from src.core.workspace import create
from src.data.transforms._transforms import ConvertPILImage
from src.misc import dist_utils
from src.core import YAMLConfig, yaml_utils
from src.solver import TASKS
from src.zoo.rtdetr.matcher import HungarianMatcher
from BlinkModel.model import BlinkTransformerDecoder
import importlib.util
import sys
import torch
from tqdm import tqdm
import torch.nn.functional as F 
import math
from scipy.optimize import linear_sum_assignment
from torchvision.ops import nms, roi_align, roi_pool
from thop import profile
import time
from torch import Tensor
from concurrent.futures import ThreadPoolExecutor, as_completed
def box_cxcywh_to_xyxy(x: Tensor) -> Tensor:
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)
    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--track_config', help='Config file', default = "configs/rtdetrv2/detrs-blink_len=30.yml")
    parser.add_argument('--blink_config', help='Config file', default = "configs/BlinkModule/full_v1.py")
    parser.add_argument('--checkpoint',help='Checkpoint file', default = "/data/data4/zengwenzheng/detrs-blink/output/rtdetrv2_r50vd_6x_coco_len=30/checkpoint0000.pth")
    parser.add_argument(
        '--json',
        default="/data/data4/zengwenzheng/data/dataset_building/mpeblink2_1/annotations/test.json",
        help='Path to mpeblink test json file')   
    parser.add_argument(
        '--root', default="/data/data4/zengwenzheng/data/dataset_building/mpeblink2_1/test_rawframes/", help='Path to image file') 
    parser.add_argument(
        '--mode', default="test", help='Path to image file') 
    parser.add_argument(
        '--output', default="mpeblink_v2", help='Path to image file') 
        
    args = parser.parse_args()
    return args
  
def load_img(img_names, root, max_workers=4):
    T = len(img_names)
    t1 = time.time()
    input_tensor = np.zeros((T, 3, 360, 640), dtype=np.float32)

    def process_image(name, frame_id):
        img_path = os.path.join(root, name)
        img = cv2.imread(img_path)  # 读取图片
        img = np.transpose(img, (2, 0, 1))  # 转换维度为 (C, H, W)
        return frame_id, img

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_image, name, frame_id): frame_id
            for frame_id, name in enumerate(img_names)
        }

        for future in as_completed(futures):
            frame_id, img = future.result()
            input_tensor[frame_id] = img

    # 转换为 PyTorch 张量并归一化
    input_tensor /= 255
    input_tensor = torch.tensor(input_tensor)
    t2 = time.time()
    #print((t2 - t1)/T)
    return input_tensor

    
def bbox_nms(det_bboxes, det_eye_bboxes, det_blinks_eye, nms_threshold=0.2):
    sorted_order = torch.mean(det_bboxes[:,:,-1], dim=-1).sort(descending=True)[1]
    det_blinks_eye = det_blinks_eye[sorted_order]
    det_eye_bboxes= det_eye_bboxes[sorted_order]
    det_bboxes = det_bboxes[sorted_order]
  
    mat_nms = matcher.matcher_infer(det_bboxes.permute(1,0,2), det_bboxes.permute(1,0,2))

    num_samples = det_bboxes.size(0)
    preserved_list = []
    suppressed = torch.zeros(num_samples, dtype=torch.bool)
    for i in range(num_samples):
        if not suppressed[i]:
            preserved_list.append(i)
            suppressed |= (mat_nms[i] > nms_threshold)
            suppressed[i] = False
    
    det_blinks_eye = det_blinks_eye[preserved_list]
    det_eye_bboxes = det_eye_bboxes[preserved_list]
    det_bboxes = det_bboxes[preserved_list]
    
    return det_bboxes, det_eye_bboxes, det_blinks_eye
    
def get_output(track_model, blink_model, input_tensor, window_size, person_threshold = 0.5):
    stride = window_size // 2
    sample_point = (5, 4)
    device = input_tensor.device
    output, memory, head_query, eye_query = track_model(input_tensor, test=True)
    
    head_bbox = output['pred_head_boxes']
    eye_bbox = output['pred_eye_boxes']
    out_logits = output["pred_logits"]
    T, N, _ = head_bbox.size()
    out_logits = out_logits.reshape(T, N)
    out_logits = F.sigmoid(out_logits)
    
    out_logits = out_logits.transpose(0, 1) ### N, T
    score, indice = torch.topk(out_logits, k=10, dim=-1)

    cls_score = torch.mean(score, dim=-1)
    save_index = cls_score > person_threshold
    
    if torch.sum(save_index) == 0:
      max_score = torch.topk(cls_score, k=1, dim=-1)[1]
      save_index[max_score] = True
      
    cls_score = cls_score[save_index].reshape(1, -1, 1).repeat(T, 1, 1)

    head_bbox = head_bbox.transpose(0, 1)
    head_bbox = head_bbox[save_index].transpose(0, 1)

    eye_bbox = eye_bbox.transpose(0, 1)
    eye_bbox = eye_bbox[save_index].transpose(0, 1) ###T N
    T, N, _ = eye_bbox.size()

    bbox_all = []
    eye_roi = box_cxcywh_to_xyxy(eye_bbox.reshape(T*N, -1))
    eye_roi = eye_roi.reshape(T, N, -1)
    for i in range(T):
        bbox_all.append(eye_roi[i])
    
    det_blinks_eye = []
    for i in range(len(memory[-3:])):
        #print(memory[i].size())
        bbox_single_scale = []
        T, C, H, W = memory[i].size()
        whwh = torch.tensor([[W, H, W, H]]).to(device)
        for bbox in bbox_all:
          bbox_single_scale.append(bbox * whwh)
        feature_map = roi_align(memory[i], bbox_single_scale, sample_point).permute(0, 2, 3, 1) ### BN D H W
        T_N, h, W, d = feature_map.size()
        feature_map = feature_map.reshape(T_N, h*W, d)
        det_blinks_eye.append(feature_map)
        
    det_blinks_eye = torch.cat(det_blinks_eye, dim=1)
    det_blinks_eye = det_blinks_eye.reshape(T, N, -1).transpose(0, 1) ###N, T, D
    det_head_query = head_query.transpose(0, 1)[save_index]
    det_eye_query = eye_query.transpose(0, 1)[save_index]

   
    pred_blinks = torch.zeros(N, T, 2).to(device)
    pred_counts = torch.zeros(T).to(device)
   
    input_window_features = []
    input_head_features = []
    input_eye_features = []
    infer_pos = []
    
    infer_win_num = max((T - window_size)//stride + 1, 1)
    
    if T < window_size:
      padding_len = window_size - T
      N, _, query_dim = det_head_query.size()
      padding_query = torch.zeros(N, padding_len, query_dim).to(device)
      det_head_query = torch.cat([det_head_query, padding_query], dim=1)
      det_eye_query = torch.cat([det_eye_query, padding_query], dim=1)
      
      sample_dim = det_blinks_eye.size(-1)
      padding_sample = torch.zeros(N, padding_len, sample_dim).to(device)
      det_blinks_eye = torch.cat([det_blinks_eye, padding_sample], dim=1)
      
      pred_results = blink_model(det_blinks_eye, det_head_query, det_eye_query)[:, : -padding_len]
      pred_blinks = torch.softmax(pred_results, dim=-1)[:, :, 1]
      pred_blinks = pred_blinks.reshape(N, T, 1).transpose(0, 1)

    else:
      input_window_features = []
      input_head_features = []
      input_eye_features = []
      infer_pos = []
      for i in range(0, infer_win_num):
        if i != infer_win_num - 1:
          start_idx = i * stride
        else:
          start_idx = T - window_size
        end_idx = start_idx + window_size
        pred_counts[start_idx:end_idx] += 1
        infer_pos.append((start_idx, end_idx))
        input_window_features.append(det_blinks_eye[:, start_idx:end_idx])
        input_head_features.append(det_head_query[:, start_idx:end_idx])
        input_eye_features.append(det_eye_query[:, start_idx:end_idx])
      
      input_window_features = torch.cat(input_window_features, dim=0)
      input_head_features = torch.cat(input_head_features, dim=0)
      input_eye_features = torch.cat(input_eye_features, dim=0)
     
      pred_results = blink_model(input_window_features, input_head_features, input_eye_features)
      
      for b_id, (start_idx, end_idx) in enumerate(infer_pos):
        pred_blinks[:, start_idx:end_idx] += pred_results[b_id * N: b_id * N + N]
        
      pred_blinks = torch.softmax(pred_blinks, dim=-1)[:, :, 1]
      pred_blinks /= pred_counts.reshape(1, T)
      pred_blinks = pred_blinks.reshape(N, T, 1).transpose(0, 1)
      
    head_bbox = torch.cat([head_bbox, cls_score], dim=-1)
    eye_bbox = torch.cat([eye_bbox, cls_score], dim=-1)
    
    '''
    macs, params = profile(track_model, inputs=(input_tensor, ), verbose = False)
    macs = macs/(T)
    print(f"track macs = {macs/1e9}G")
    print(f"track params = {params/1e6}M")
    
    macs, params = profile(blink_model, inputs=(input_window_features, input_head_features, input_eye_features), verbose = False)
    macs = macs/(N*T)
    print(f"blink macs = {macs/1e9}G")
    print(f"blink params = {params/1e6}M")
    '''
    return (head_bbox, None), eye_bbox, pred_blinks

def main(track_model, blink_model, matcher, window_size, device, json_path, root, mode='val', output='mpeblink_v2'):
   
    total_params = sum(p.numel() for p in track_model.parameters()) + sum(p.numel() for p in blink_model.parameters())
    print('total_params:',total_params)
    anno = json.load(open(json_path))
    whwh = torch.tensor([640, 360, 640, 360, 1])
    results = []
    clip_len = 42 # define the video clip length for a single forward propagation
    stride = 14 # define the stride
    
    #clip_len = 64
    #stride = 24
    
    iou_threshold = 0.2
    nms_threshold = 0.5
    val_num = 0
    person_threshold = 0.5

    for video in tqdm(anno['videos']):

        imgs = video['file_names']
      
        video_det_bboxes = []
        video_det_eye_bboxes = []

        video_det_blinks_eye = []

        datas, threads = [], []
        video_length = len(imgs)
        imgs = load_img(imgs, root)
   
        anno_gt = []

        for instance_gt in anno['annotations']:
            if instance_gt['video_id'] == video['id']:
                anno_gt.append(instance_gt)

        if video_length <= clip_len:   
            clip_num = 1
        else:
            clip_num = math.ceil((video_length-clip_len)/stride) + 1
            
        imgs = imgs.to(device)
        for clip_index in range(0, clip_num):
            if clip_index!=clip_num-1:  # Determine if it is the last clip
                cur_clip = imgs[clip_index*stride:clip_index*stride + clip_len]
                clip_overlap = clip_len - stride
            else:   # If it is the last clip, take the last clip_num frame backwards
                cur_clip = imgs[-clip_len:]
                if (video_length-clip_len)%stride:
                    clip_overlap = clip_len - (video_length-clip_len)%stride
                else:
                    clip_overlap = clip_len - stride
            input_img = cur_clip[None]
            with torch.no_grad():
                t1 = time.time()
                (det_bboxes, det_labels), det_eye_bboxes, det_blinks_eye = get_output(track_model, blink_model, input_img, window_size, person_threshold)
                t2 = time.time()
                #print(t2 - t1)
       
            # Perform inter-clip matching
            if clip_index!=0:
                    
                previous_det_bboxes_for_match = video_det_bboxes[:,-clip_overlap:,:]
                
                det_bboxes = det_bboxes.permute(1,0,2)
                det_eye_bboxes = det_eye_bboxes.permute(1,0,2)
                det_blinks_eye = det_blinks_eye.permute(1,0,2)
               
                det_bboxes, det_eye_bboxes, det_blinks_eye = bbox_nms(det_bboxes, det_eye_bboxes, det_blinks_eye, nms_threshold)

                previous_person_num = previous_det_bboxes_for_match.size(0)

                # Next, perform pre-padding foe the upcoming clip, length=clip_len-clip_overlap bbox:[0,0,0,0], blink:[0]
                next_padding_bboxes = torch.zeros([previous_person_num,clip_len-clip_overlap,5]).to(video_det_bboxes.device) 
                video_det_bboxes = torch.cat((video_det_bboxes, next_padding_bboxes),1)
                video_det_eye_bboxes = torch.cat((video_det_eye_bboxes, next_padding_bboxes),1)
                
                next_padding_blinks = torch.zeros([previous_person_num,clip_len-clip_overlap,1]).to(video_det_blinks_eye.device)
                video_det_blinks_eye = torch.cat((video_det_blinks_eye, next_padding_blinks),1)
                
                # perform matching
                previous_det_bboxes_for_iou = previous_det_bboxes_for_match.permute(1,0,2)
                det_boxes_for_iou = det_bboxes.permute(1,0,2)[:clip_overlap,:,:]
                mat = matcher.matcher_infer(previous_det_bboxes_for_iou, det_boxes_for_iou)
              
                det_assigned = torch.zeros(det_bboxes.shape[0])
                for i in range(0, min(mat.shape)): 
                    tar = np.unravel_index(mat.argmax(), mat.shape)
                    if mat[tar[0], tar[1]] < iou_threshold: 
                    
                        new_person_bboxes = det_bboxes[tar[1], -(clip_len):, :].unsqueeze(0)  # 锟斤拷锟叫猴拷锟芥看锟斤拷锟角诧拷锟斤拷维锟饺对碉拷
                        new_person_eye_bboxes = det_eye_bboxes[tar[1], -(clip_len):, :].unsqueeze(0)  # 锟斤拷锟叫猴拷锟芥看锟斤拷锟角诧拷锟斤拷维锟饺对碉拷
                        new_person_blinks_eye = det_blinks_eye[tar[1], -(clip_len):, :].unsqueeze(0)
                       
                        new_person_pre_bboxes = torch.zeros([1,video_det_bboxes.size(1)-(clip_len),5]).to(video_det_bboxes.device)
                        new_person_pre_blinks = torch.zeros([1,video_det_blinks_eye.size(1)-(clip_len),1]).to(video_det_blinks_eye.device)
                        
                        new_person_bboxes = torch.cat((new_person_pre_bboxes, new_person_bboxes), 1)
                        new_person_eye_bboxes = torch.cat((new_person_pre_bboxes, new_person_eye_bboxes), 1)
                        new_person_blinks_eye = torch.cat((new_person_pre_blinks, new_person_blinks_eye), 1)

                        video_det_bboxes = torch.cat((video_det_bboxes, new_person_bboxes), 0)
                        video_det_eye_bboxes = torch.cat((video_det_eye_bboxes, new_person_eye_bboxes), 0)
                        video_det_blinks_eye = torch.cat((video_det_blinks_eye, new_person_blinks_eye), 0)
                       


                        mat[tar[0],:] = -10000
                        mat[:,tar[1]] = -10000
                        det_assigned[tar[1]] = 1    # 锟斤拷锟斤拷index = tar[1]锟斤拷锟铰硷拷锟斤拷锟斤拷丫锟斤拷锟斤拷锟斤拷锟?
                    else: #说锟斤拷锟斤拷前锟斤拷匹锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷值要锟斤拷锟?
                        mat[tar[0], :] = -10000
                        mat[:, tar[1]] = -10000
                        # 锟斤拷锟斤拷选锟斤拷直锟接帮拷前锟斤拷det_bboxes锟斤拷展锟斤拷锟铰碉拷18帧None锟斤拷锟斤拷为锟斤拷前det_box[tar[1]]锟斤拷值
                        video_det_bboxes[tar[0], -(clip_len-clip_overlap):, :] = det_bboxes[tar[1], -(clip_len-clip_overlap):, :] # 锟斤拷锟斤拷锟叫匡拷锟杰伙拷锟斤拷锟斤拷锟斤拷
                        video_det_eye_bboxes[tar[0], -(clip_len-clip_overlap):, :] = det_eye_bboxes[tar[1], -(clip_len-clip_overlap):, :] # 锟斤拷锟斤拷锟叫匡拷锟杰伙拷锟斤拷锟斤拷锟斤拷
                        video_det_blinks_eye[tar[0], -(clip_len-clip_overlap):, :] = det_blinks_eye[tar[1], -(clip_len-clip_overlap):, :]
                        
                        video_det_bboxes[tar[0], -clip_len:-(clip_len-clip_overlap), :] = (video_det_bboxes[tar[0], -clip_len:-(clip_len-clip_overlap), :] + det_bboxes[tar[1], -clip_len:-(clip_len-clip_overlap), :])/2

                        video_det_eye_bboxes[tar[0], -clip_len:-(clip_len-clip_overlap), :] = (video_det_eye_bboxes[tar[0], -clip_len:-(clip_len-clip_overlap), :] + det_eye_bboxes[tar[1], -clip_len:-(clip_len-clip_overlap), :])/2
                        
                        
                        video_det_blinks_eye[tar[0], -clip_len:-(clip_len-clip_overlap), :] = (video_det_blinks_eye[tar[0], -clip_len:-(clip_len-clip_overlap), :] + det_blinks_eye[tar[1], -clip_len:-(clip_len-clip_overlap), :])/2

                       
                        det_assigned[tar[1]] = 1    # Mark the new prediction result for index = tar[1] has been processed

                for index in range(0, det_assigned.shape[0]):
                    if det_assigned[index] == 0: # This new prediction result has not been processed yet and is a new id

                        new_person_bboxes = det_bboxes[index, -(clip_len):, :].unsqueeze(0)  # 锟斤拷锟叫猴拷锟芥看锟斤拷锟角诧拷锟斤拷维锟饺对碉拷
                        new_person_eye_bboxes = det_eye_bboxes[index, -(clip_len):, :].unsqueeze(0)  # 锟斤拷锟叫猴拷锟芥看锟斤拷锟角诧拷锟斤拷维锟饺对碉拷
                        new_person_blinks_eye = det_blinks_eye[index, -(clip_len):, :].unsqueeze(0)
                        
                        new_person_pre_bboxes = torch.zeros([1,video_det_bboxes.size(1)-(clip_len),5]).to(video_det_bboxes.device)
                        new_person_pre_blinks = torch.zeros([1,video_det_blinks_eye.size(1)-(clip_len),1]).to(video_det_blinks_eye.device)
                        
                        new_person_bboxes = torch.cat((new_person_pre_bboxes, new_person_bboxes), 1)
                        new_person_eye_bboxes = torch.cat((new_person_pre_bboxes, new_person_eye_bboxes), 1)
                        new_person_blinks_eye = torch.cat((new_person_pre_blinks, new_person_blinks_eye), 1)
                        
                        video_det_bboxes = torch.cat((video_det_bboxes, new_person_bboxes), 0)
                        video_det_eye_bboxes = torch.cat((video_det_eye_bboxes, new_person_eye_bboxes), 0)
                        video_det_blinks_eye = torch.cat((video_det_blinks_eye, new_person_blinks_eye), 0)
                        
                        det_assigned[index] = 1    # Mark the new prediction result for index = tar[1] has been processed

            else: # for the first video_cilp
                det_bboxes = det_bboxes.permute(1,0,2)
                det_eye_bboxes = det_eye_bboxes.permute(1,0,2)
                det_blinks_eye = det_blinks_eye.permute(1,0,2)
                
                det_bboxes, det_eye_bboxes, det_blinks_eye = bbox_nms(det_bboxes, det_eye_bboxes, det_blinks_eye, nms_threshold)
                
                video_det_blinks_eye = det_blinks_eye
                video_det_eye_bboxes = det_eye_bboxes
                video_det_bboxes = det_bboxes # 锟斤拷锟揭伙拷锟揭拷锟斤拷锟斤拷锟斤拷为前锟斤拷锟斤拷片锟斤拷锟角伙拷锟斤拷锟斤拷锟侥ｏ拷锟斤拷锟斤拷锟饺憋拷锟侥憋拷
                
        
        N, T, _ = video_det_bboxes.size()
        det_bboxes =  video_det_bboxes.permute(1,0,2)
        det_eye_bboxes =  video_det_eye_bboxes.permute(1,0,2)
        det_blinks_eye = video_det_blinks_eye.permute(1,0,2)
       
        whwh = whwh.to(det_bboxes.device) 
        det_bboxes = det_bboxes * whwh
        det_eye_bboxes = det_eye_bboxes * whwh
        
        for inst_ind in range(det_bboxes.size(1)):  # 锟斤拷锟斤拷锟斤拷取锟斤拷top10锟斤拷query锟斤拷息
          objs = dict(
              video_id=video['id'],
              score=det_bboxes[:, inst_ind, -1][torch.where(det_bboxes[:, inst_ind, -1]>0)].mean().item(),  # 锟斤拷锟斤拷锟脚度碉拷锟斤拷0锟斤拷去锟斤拷
              category_id=1,
              bboxes=[],
              instance_id = val_num,
              eye_bboxes=[],
              blink_scores=det_blinks_eye[:, inst_ind, -1].detach().cpu().numpy().tolist(),
              blink_gt=[],
              score_per_img=[]
              )  
              
          for sub_ind in range(det_bboxes.size(0)):   # for the prediction results of each frame
              m = det_bboxes[
                  sub_ind, inst_ind,
                  :-1].detach().cpu().numpy().tolist()
              if (m[0] + m[1] + m[2] + m[3]) == 0:
                  m = [0, 0, 0, 0]
              else:
                  m = [m[0] - m[2]/2, m[1] - m[3]/2, m[2], m[3]]

              objs['bboxes'].append(m)

              m = det_eye_bboxes[
                  sub_ind, inst_ind,
                  :-1].detach().cpu().numpy().tolist()
              if (m[0] + m[1] + m[2] + m[3]) == 0:
                  m = [0, 0, 0, 0]
              else:
                  m = [m[0] - m[2]/2, m[1] - m[3]/2, m[2], m[3]]

              objs['eye_bboxes'].append(m)
              objs['score_per_img'].append(det_bboxes[sub_ind,inst_ind,-1].item())
          results.append(objs) 
    
    if mode == 'test':
      os.makedirs('results/test_results',exist_ok=True)
      write_path = os.path.join('results/test_results', f'{output}.json')
      
      json.dump(results, open(write_path, 'w'))
      print('Done')
      print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
       
if __name__ == '__main__':
  args = parse_args()
  cfg = YAMLConfig(args.track_config)
  device = 'cuda'
  track_model = cfg.model
  ckpt_load = {}
  
  ckpt = torch.load(args.checkpoint, map_location=torch.device('cpu'))['model']

  for k, v in ckpt.items():
    ckpt_load[k] = v
 
  track_model.load_state_dict(ckpt_load,strict=True)
  track_model.to(device)
  track_model.eval()
  
  spec = importlib.util.spec_from_file_location("config", args.blink_config)
  config_module = importlib.util.module_from_spec(spec)
  sys.modules["config"] = config_module
  spec.loader.exec_module(config_module)
  config = config_module.Config()
    
  blink_model = BlinkTransformerDecoder(map_size = config.sample_point, infer_len=config.window_size, roi_feature_encoder = config.roi_feature_encoder)
  blink_model.load_state_dict(torch.load(config.model_path, map_location=torch.device('cpu')))
  blink_model.to(device)
  blink_model.eval()
    
  weight_dict = {'cost_class': 0, 'cost_bbox': 0, 'cost_giou': 1}
  matcher = HungarianMatcher(weight_dict)
  main(track_model, blink_model, matcher, config.window_size, device, args.json, args.root, args.mode, args.output)