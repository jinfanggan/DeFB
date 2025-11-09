import os 
import sys 
import h5py
import json
import cv2
import numpy as np
import copy
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from PIL import Image
import argparse
from src.core.workspace import create
from src.data.transforms._transforms import ConvertPILImage
from src.misc import dist_utils
from src.core import YAMLConfig, yaml_utils
from src.solver import TASKS
from src.zoo.rtdetr.matcher import HungarianMatcher
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
    parser.add_argument('--config', help='Config file', default = "configs/rtdetrv2/detrs-blink_len=30.yml")
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

def save_results_to_h5(val_results, h5_file_path):
    with h5py.File(h5_file_path, 'a') as f:
        if 'blink_features' not in f:
            dt = h5py.vlen_dtype(np.dtype('float32'))
            f.create_dataset('blink_features', shape=(0,), maxshape=(None,), dtype=dt)
            f.create_dataset('eye_query', shape=(0,), maxshape=(None,), dtype=dt)
            f.create_dataset('head_query', shape=(0,), maxshape=(None,), dtype=dt)
            f.create_dataset('blink_gt', shape=(0,), maxshape=(None,), dtype=dt)
            dt = h5py.vlen_dtype(np.dtype('bool'))
            f.create_dataset('mask', shape=(0,), maxshape=(None,), dtype=dt)
            
            dt = h5py.special_dtype(vlen=str)
            f.create_dataset('person_id', shape=(0,), maxshape=(None,), dtype=dt)
            f.create_dataset('video_id', shape=(0,), maxshape=(None,), dtype=dt)
            
        current_size = f['person_id'].shape[0]
        new_size = current_size + len(val_results)

        f['blink_features'].resize(new_size, axis=0)
        f['head_query'].resize(new_size, axis=0)
        f['eye_query'].resize(new_size, axis=0)
        f['blink_gt'].resize(new_size, axis=0)
        f['person_id'].resize(new_size, axis=0)
        f['video_id'].resize(new_size, axis=0)
        f['mask'].resize(new_size, axis=0)
     
        for i, result in enumerate(val_results):
            index = current_size + i
           
            blink_features_flat = np.array(result['blink_features'], dtype='float16').flatten() 
            head_query = np.array(result['head_query'], dtype='float16').flatten()
            eye_query = np.array(result['eye_query'], dtype='float16').flatten()
            blink_gt_flat = np.array(result['blink_gt'], dtype='float16').flatten()
            mask = np.array(result['mask'], dtype='bool').flatten()
           
            f['blink_features'][index] = blink_features_flat
            f['head_query'][index] = head_query
            f['eye_query'][index] = eye_query
            f['blink_gt'][index] = blink_gt_flat
            f['mask'][index] = mask
            f['person_id'][index] = str(result['instance_id'])
            f['video_id'][index] = str(result['video_id'])

    
def load_img(img_names, root, max_workers=4):
    T = len(img_names)
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

    return input_tensor

def bbox_nms(det_bboxes, det_eye_bboxes, det_blinks_eye, det_head_query, det_eye_query, nms_threshold=0.2):
    sorted_order = torch.mean(det_bboxes[:,:,-1], dim=-1).sort(descending=True)[1]
    det_blinks_eye = det_blinks_eye[sorted_order]
    det_eye_bboxes= det_eye_bboxes[sorted_order]
    det_head_query = det_head_query[sorted_order]
    det_eye_query = det_eye_query[sorted_order]
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
    det_head_query = det_head_query[preserved_list]
    det_eye_query = det_eye_query[preserved_list]
    
    return det_bboxes, det_eye_bboxes, det_blinks_eye, det_head_query, det_eye_query
    
def get_output(model, input_tensor, person_threshold = 0.5):
    device = input_tensor.device
    sample_point = (5, 4)
    #t1 = time.time()
    output, memory, head_query, eye_query = model(input_tensor, test=True)
    #t2 = time.time()
    #print(t2 - t1)
    head_bbox = output['pred_head_boxes']
    eye_bbox = output['pred_eye_boxes']
    out_logits = output["pred_logits"]
  
    T, N, _ = head_bbox.size()
    out_logits = out_logits.reshape(T, N)
    out_logits = F.sigmoid(out_logits)
    invalid_index = out_logits < 0
    head_bbox[invalid_index] = 0
    eye_bbox[invalid_index] = 0
    
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
    for i in range(len(memory)):
        bbox_single_scale = []
        T, C, H, W = memory[i].size()
        whwh = torch.tensor([[W, H, W, H]]).to(eye_roi.device)
        for bbox in bbox_all:
          bbox_single_scale.append(bbox * whwh)
        feature_map = roi_align(memory[i], bbox_single_scale, sample_point).permute(0, 2, 3, 1) ### BN D H W
        T_N, h, W, d = feature_map.size()
        feature_map = feature_map.reshape(T_N, h*W, d)
        det_blinks_eye.append(feature_map)
        
    det_blinks_eye = torch.cat(det_blinks_eye, dim=1)
    det_blinks_eye = det_blinks_eye.reshape(T, N, -1)
    head_bbox = torch.cat([head_bbox, cls_score], dim=-1)
    eye_bbox = torch.cat([eye_bbox, cls_score], dim=-1)
    
    det_head_query = head_query.transpose(0, 1)[save_index].transpose(0, 1)
    det_eye_query = eye_query.transpose(0, 1)[save_index].transpose(0, 1)
    
    return (head_bbox, None), eye_bbox, det_blinks_eye, det_head_query, det_eye_query

def main(model, matcher, data_processor, device, json_path, root, mode='val', output='mpeblink_v2'):
   
    total_params = sum(p.numel() for p in model.parameters())
    model.eval()
    print('total_params:',total_params)
    anno = json.load(open(json_path))
    whwh = torch.tensor([640, 360, 640, 360, 1])
    results = []
    clip_len = 42 # define the video clip length for a single forward propagation
    stride = 14 # define the stride
    sample_points = 4 * 5 * 3
    query_dim = 256
    sample_dim = 256
    
    h5_file_path = f"BinkDetectionDataset/{output}"
    os.makedirs(h5_file_path, exist_ok=True)
    h5_file_name = f"BinkDetectionDataset/{output}/{mode}_dataset.h5"
    if os.path.exists(h5_file_name):
      os.remove(h5_file_name)
    
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

            #input_img = collect(cur_clip, root, data_processor).to(device)
            
            input_img = cur_clip[None].to(device)
            with torch.no_grad():
                (det_bboxes, det_labels), det_eye_bboxes, det_blinks_eye, det_head_query, det_eye_query = get_output(model, input_img, person_threshold)
               
       
            # Perform inter-clip matching
            if clip_index!=0:
                    
                previous_det_bboxes_for_match = video_det_bboxes[:,-clip_overlap:,:]
                
                det_bboxes = det_bboxes.permute(1,0,2)
                det_eye_bboxes = det_eye_bboxes.permute(1,0,2)
                det_blinks_eye = det_blinks_eye.permute(1,0,2)
                det_head_query = det_head_query.permute(1,0,2)
                det_eye_query = det_eye_query.permute(1,0,2)
                
                det_bboxes, det_eye_bboxes, det_blinks_eye, det_head_query, det_eye_query = bbox_nms(det_bboxes, det_eye_bboxes, det_blinks_eye, det_head_query, \
                                                                                    det_eye_query, nms_threshold)

                previous_person_num = previous_det_bboxes_for_match.size(0)

                # Next, perform pre-padding foe the upcoming clip, length=clip_len-clip_overlap bbox:[0,0,0,0], blink:[0]
                next_padding_bboxes = torch.zeros([previous_person_num,clip_len-clip_overlap,5]).to(video_det_bboxes.device) 
                video_det_bboxes = torch.cat((video_det_bboxes, next_padding_bboxes),1)
                video_det_eye_bboxes = torch.cat((video_det_eye_bboxes, next_padding_bboxes),1)
                
                next_padding_blinks = torch.zeros([previous_person_num,clip_len-clip_overlap,sample_points*sample_dim]).to(video_det_blinks_eye.device)
                video_det_blinks_eye = torch.cat((video_det_blinks_eye, next_padding_blinks),1)
                
                next_padding_querys = torch.zeros([previous_person_num,clip_len-clip_overlap,sample_dim]).to(video_det_blinks_eye.device)
                video_det_head_query = torch.cat((video_det_head_query, next_padding_querys),1)
                video_det_eye_query = torch.cat((video_det_eye_query, next_padding_querys),1)
                
                # perform matching
                previous_det_bboxes_for_iou = previous_det_bboxes_for_match.permute(1,0,2)
                det_boxes_for_iou = det_bboxes.permute(1,0,2)[:clip_overlap,:,:]
                mat = matcher.matcher_infer(previous_det_bboxes_for_iou, det_boxes_for_iou)
              
                det_assigned = torch.zeros(det_bboxes.shape[0])
                for i in range(0, min(mat.shape)): # 锟斤拷锟狡ワ拷锟絚ur_person_num锟轿ｏ拷pre锟斤拷锟窖撅拷锟斤拷none锟剿ｏ拷锟斤拷锟斤拷影锟届，同时锟斤拷id也锟斤拷锟斤拷锟斤拷iou锟斤拷锟斤拷锟斤拷驯锟秸硷拷荻锟斤拷锟斤拷锟斤拷殖锟斤拷锟?

                    tar = np.unravel_index(mat.argmax(), mat.shape)
                    if mat[tar[0], tar[1]] < iou_threshold:  # 锟斤拷时锟斤拷锟斤拷锟斤拷值锟剿ｏ拷说锟斤拷锟斤拷锟斤拷锟斤拷锟铰碉拷锟斤拷锟斤拷锟斤拷锟接碉拷注锟斤拷锟斤拷锟?
                        # 1.锟铰斤拷一锟斤拷id,锟斤拷video_det_bboxes.size(1)帧锟斤拷None bbox,label,blink
                        # 2.然锟襟，帮拷锟斤拷锟斤拷锟絛im=0锟斤拷video_det_bboxes concat
                        new_person_bboxes = det_bboxes[tar[1], -(clip_len):, :].unsqueeze(0)  # 锟斤拷锟叫猴拷锟芥看锟斤拷锟角诧拷锟斤拷维锟饺对碉拷
                        new_person_eye_bboxes = det_eye_bboxes[tar[1], -(clip_len):, :].unsqueeze(0)  # 锟斤拷锟叫猴拷锟芥看锟斤拷锟角诧拷锟斤拷维锟饺对碉拷
                        new_person_blinks_eye = det_blinks_eye[tar[1], -(clip_len):, :].unsqueeze(0)
                        new_person_head_query = det_head_query[tar[1], -(clip_len):, :].unsqueeze(0)
                        new_person_eye_query = det_eye_query[tar[1], -(clip_len):, :].unsqueeze(0)
                        
                        new_person_pre_bboxes = torch.zeros([1,video_det_bboxes.size(1)-(clip_len),5]).to(video_det_bboxes.device)
                        new_person_pre_blinks = torch.zeros([1,video_det_blinks_eye.size(1)-(clip_len),sample_points*sample_dim]).to(video_det_blinks_eye.device)
                        new_person_pre_querys = torch.zeros([1,video_det_blinks_eye.size(1)-(clip_len),sample_dim]).to(video_det_blinks_eye.device)
                        
                        new_person_bboxes = torch.cat((new_person_pre_bboxes, new_person_bboxes), 1)
                        new_person_eye_bboxes = torch.cat((new_person_pre_bboxes, new_person_eye_bboxes), 1)
                        new_person_blinks_eye = torch.cat((new_person_pre_blinks, new_person_blinks_eye), 1)
                        new_person_head_query = torch.cat((new_person_pre_querys, new_person_head_query), 1)
                        new_person_eye_query = torch.cat((new_person_pre_querys, new_person_eye_query), 1)

                        video_det_bboxes = torch.cat((video_det_bboxes, new_person_bboxes), 0)
                        video_det_eye_bboxes = torch.cat((video_det_eye_bboxes, new_person_eye_bboxes), 0)
                        video_det_blinks_eye = torch.cat((video_det_blinks_eye, new_person_blinks_eye), 0)
                        video_det_head_query = torch.cat((video_det_head_query, new_person_head_query), 0)
                        video_det_eye_query = torch.cat((video_det_eye_query, new_person_eye_query), 0)


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
                        video_det_head_query[tar[0], -(clip_len-clip_overlap):, :] = det_head_query[tar[1], -(clip_len-clip_overlap):, :]
                        video_det_eye_query[tar[0], -(clip_len-clip_overlap):, :] = det_eye_query[tar[1], -(clip_len-clip_overlap):, :]
                       
                        video_det_bboxes[tar[0], -clip_len:-(clip_len-clip_overlap), :] = (video_det_bboxes[tar[0], -clip_len:-(clip_len-clip_overlap), :] + det_bboxes[tar[1], -clip_len:-(clip_len-clip_overlap), :])/2

                        video_det_eye_bboxes[tar[0], -clip_len:-(clip_len-clip_overlap), :] = (video_det_eye_bboxes[tar[0], -clip_len:-(clip_len-clip_overlap), :] + det_eye_bboxes[tar[1], -clip_len:-(clip_len-clip_overlap), :])/2
                        
                        
                        video_det_blinks_eye[tar[0], -clip_len:-(clip_len-clip_overlap), :] = (video_det_blinks_eye[tar[0], -clip_len:-(clip_len-clip_overlap), :] + det_blinks_eye[tar[1], -clip_len:-(clip_len-clip_overlap), :])/2

                        video_det_head_query[tar[0], -clip_len:-(clip_len-clip_overlap), :] = (video_det_head_query[tar[0], -clip_len:-(clip_len-clip_overlap), :] + det_head_query[tar[1], -clip_len:-(clip_len-clip_overlap), :])/2
                        
                        
                        video_det_eye_query[tar[0], -clip_len:-(clip_len-clip_overlap), :] = (video_det_eye_query[tar[0], -clip_len:-(clip_len-clip_overlap), :] + det_eye_query[tar[1], -clip_len:-(clip_len-clip_overlap), :])/2
                       
                        det_assigned[tar[1]] = 1    # Mark the new prediction result for index = tar[1] has been processed

                for index in range(0, det_assigned.shape[0]):
                    if det_assigned[index] == 0: # This new prediction result has not been processed yet and is a new id

                        new_person_bboxes = det_bboxes[index, -(clip_len):, :].unsqueeze(0)  # 锟斤拷锟叫猴拷锟芥看锟斤拷锟角诧拷锟斤拷维锟饺对碉拷
                        new_person_eye_bboxes = det_eye_bboxes[index, -(clip_len):, :].unsqueeze(0)  # 锟斤拷锟叫猴拷锟芥看锟斤拷锟角诧拷锟斤拷维锟饺对碉拷
                        new_person_blinks_eye = det_blinks_eye[index, -(clip_len):, :].unsqueeze(0)
                        new_person_head_query = det_head_query[index, -(clip_len):, :].unsqueeze(0)
                        new_person_eye_query = det_eye_query[index, -(clip_len):, :].unsqueeze(0)
                        
                        
                        new_person_pre_bboxes = torch.zeros([1,video_det_bboxes.size(1)-(clip_len),5]).to(video_det_bboxes.device)
                        new_person_pre_blinks = torch.zeros([1,video_det_blinks_eye.size(1)-(clip_len),sample_points*sample_dim]).to(video_det_blinks_eye.device)
                        new_person_pre_querys = torch.zeros([1,video_det_blinks_eye.size(1)-(clip_len),sample_dim]).to(video_det_blinks_eye.device)
                        
                        new_person_bboxes = torch.cat((new_person_pre_bboxes, new_person_bboxes), 1)
                        new_person_eye_bboxes = torch.cat((new_person_pre_bboxes, new_person_eye_bboxes), 1)
                        new_person_blinks_eye = torch.cat((new_person_pre_blinks, new_person_blinks_eye), 1)
                        new_person_head_query = torch.cat((new_person_pre_querys, new_person_head_query), 1)
                        new_person_eye_query = torch.cat((new_person_pre_querys, new_person_eye_query), 1)
                        
                        
                        video_det_bboxes = torch.cat((video_det_bboxes, new_person_bboxes), 0)
                        video_det_eye_bboxes = torch.cat((video_det_eye_bboxes, new_person_eye_bboxes), 0)
                        video_det_blinks_eye = torch.cat((video_det_blinks_eye, new_person_blinks_eye), 0)
                        video_det_head_query = torch.cat((video_det_head_query, new_person_head_query), 0)
                        video_det_eye_query = torch.cat((video_det_eye_query, new_person_eye_query), 0)
                        
                        det_assigned[index] = 1    # Mark the new prediction result for index = tar[1] has been processed

                    
            else: # for the first video_cilp
                det_bboxes = det_bboxes.permute(1,0,2)
                det_eye_bboxes = det_eye_bboxes.permute(1,0,2)
                det_blinks_eye = det_blinks_eye.permute(1,0,2)
                det_head_query = det_head_query.permute(1,0,2)
                det_eye_query = det_eye_query.permute(1,0,2)
                
                det_bboxes, det_eye_bboxes, det_blinks_eye, det_head_query, det_eye_query = bbox_nms(det_bboxes, det_eye_bboxes, det_blinks_eye, det_head_query, \
                                                                                                                        det_eye_query, nms_threshold)
                
                video_det_blinks_eye = det_blinks_eye
                video_det_head_query = det_head_query
                video_det_eye_query = det_eye_query
                video_det_eye_bboxes = det_eye_bboxes
                video_det_bboxes = det_bboxes # 锟斤拷锟揭伙拷锟揭拷锟斤拷锟斤拷锟斤拷为前锟斤拷锟斤拷片锟斤拷锟角伙拷锟斤拷锟斤拷锟侥ｏ拷锟斤拷锟斤拷锟饺憋拷锟侥憋拷
                
        
        N, T, _ = video_det_bboxes.size()
        det_bboxes =  video_det_bboxes.permute(1,0,2)
        det_eye_bboxes =  video_det_eye_bboxes.permute(1,0,2)
        det_blinks_eye = video_det_blinks_eye.permute(1,0,2)
        det_eye_query = video_det_eye_query.permute(1,0,2)
        det_head_query = video_det_head_query.permute(1,0,2)
        
        whwh = whwh.to(det_bboxes.device) 
        det_bboxes = det_bboxes * whwh
        det_eye_bboxes = det_eye_bboxes * whwh
        
        val_results = []
        if mode == 'train' or mode == 'val':

          gt_bboxes = []
          val_sign = 0
          for gt in anno_gt:
              gt_bbox = gt['bboxes']
              for index, bbox in enumerate(gt_bbox):
                  val_sign = 1
                  if bbox is not None:
                      gt_bbox[index] = [(bbox[0] + bbox[2]/2), (bbox[1] + bbox[3]/2),\
                                         bbox[2], bbox[3], 1]
                  else:
                      gt_bbox[index] = [0, 0, 0, 0, 0]
              gt_bboxes.append(gt_bbox)
              
          if val_sign == 0:
            continue
          gt_bboxes = torch.tensor(gt_bboxes).permute(1, 0, 2).to(det_bboxes.device)
          gt_bboxes = gt_bboxes 
         
          mat, frames_iou = matcher.matcher_infer(gt_bboxes, det_bboxes, return_frame = True)
         
          rows, cols = linear_sum_assignment(1 - mat)
        #   print(mat)
          for (row, col) in zip(rows, cols):
              #if mat[row, col] < 0.5:
              #  continue
              
              inst_ind = col
              objs = dict(
                  video_id=video['id'],
                  instance_id = anno['annotations'][row]['id'],
                  bboxes=[],
                  eye_bboxes=[],
                  score=det_bboxes[:, inst_ind, -1][torch.where(det_bboxes[:, inst_ind, -1]>0)].mean().item(),
                  category_id=1,
                  blink_features=[],
                  score_per_img=[],
                  eye_query = [],
                  head_query = [],
                  mask = []
              )
              try:
                objs['blink_gt'] = anno_gt[row]['blinks']
              except:
                blink_binary = anno_gt[row]['blinks_binary']
                blink_gt = []
                start, end = 0, 0
                sign = 0
                
                for i in range(len(blink_binary)):
                  if blink_binary[i] > 0:
                    if sign == 0:
                      start = i
                    sign = 1
                    
                  else:
                    if sign > 0:
                      end = i
                      sign = 0
                      blink_gt.append([start, end - 1, 0])
                
                if sign == 1:
                  blink_gt.append([start, len(blink_binary) - 1, 0])
                
                objs['blink_gt'] = blink_gt
                
              for sub_ind in range(det_bboxes.size(0)):  # for the prediction results of each frame
                  m = det_bboxes[
                      sub_ind, inst_ind,
                      :-1].detach().cpu().numpy().tolist()
                  if (m[0] + m[1] + m[2] + m[3]) == 0:
                      m = [0, 0, 0, 0]
                  else:
                      m = [m[0] - m[2]/2, m[1] - m[3]/2, m[2], m[3]]
  
                  objs['bboxes'].append(m)
                  mask_frame = frames_iou[sub_ind][row, col] > 0.5
                  mask_frame = mask_frame.cpu().detach().numpy().tolist()
                  objs['mask'].append(mask_frame)
                  m = det_eye_bboxes[
                      sub_ind, inst_ind,
                      :-1].detach().cpu().numpy().tolist()
                  if (m[0] + m[1] + m[2] + m[3]) == 0:
                      m = [0, 0, 0, 0]
                  else:
                      m = [m[0] - m[2]/2, m[1] - m[3]/2, m[2], m[3]]
  
                  objs['eye_bboxes'].append(m)
                  objs['blink_features'].append(det_blinks_eye[sub_ind, inst_ind].cpu().detach().numpy().tolist())
                  objs['score_per_img'].append(det_bboxes[sub_ind,inst_ind,-1].item())
                  objs['eye_query'].append(det_eye_query[sub_ind, inst_ind].cpu().detach().numpy().tolist())
                  objs['head_query'].append(det_head_query[sub_ind, inst_ind].cpu().detach().numpy().tolist())
                  
              val_results.append(copy.deepcopy(objs))
              objs.pop('blink_features')
              results.append(objs)
              val_num += 1
        
        else:
            for inst_ind in range(det_bboxes.size(1)):  # 锟斤拷锟斤拷锟斤拷取锟斤拷top10锟斤拷query锟斤拷息
              objs = dict(
                  video_id=video['id'],
                  score=det_bboxes[:, inst_ind, -1][torch.where(det_bboxes[:, inst_ind, -1]>0)].mean().item(),  # 锟斤拷锟斤拷锟脚度碉拷锟斤拷0锟斤拷去锟斤拷
                  category_id=1,
                  bboxes=[],
                  instance_id = val_num,
                  blink_features=[],
                  eye_bboxes=[],
                  blink_gt=[],
                  score_per_img=[],
                  eye_query = [],
                  head_query = [],
                  mask = []
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
                  objs['blink_features'].append(det_blinks_eye[sub_ind, inst_ind].cpu().detach().numpy().tolist())
                  objs['score_per_img'].append(det_bboxes[sub_ind,inst_ind,-1].item())
                  objs['eye_query'].append(det_eye_query[sub_ind, inst_ind].cpu().detach().numpy().tolist())
                  objs['head_query'].append(det_head_query[sub_ind, inst_ind].cpu().detach().numpy().tolist())
                  
              val_results.append(copy.deepcopy(objs)) 
              objs.pop('blink_features')
              objs.pop('eye_query')
              objs.pop('head_query')
              results.append(objs) 
              val_num += 1


        if len(val_results) != 0:
            save_results_to_h5(val_results, h5_file_name)
        print(val_num)
    
    if mode == 'test':
      os.makedirs('results/test_results',exist_ok=True)
      write_path = os.path.join('results/test_results', f'{output}.json')
      
      json.dump(results, open(write_path, 'w'))
      print('Done')
      print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
       
if __name__ == '__main__':
  args = parse_args()
  cfg = YAMLConfig(args.config)
  device = 'cuda'
  model = cfg.model
  ckpt_load = {}
  
  ckpt = torch.load(args.checkpoint, map_location=torch.device('cpu'))['model']

  for k, v in ckpt.items():
    ckpt_load[k] = v
 
  model.load_state_dict(ckpt_load,strict=True)
  model.to(device)
  weight_dict = {'cost_class': 0, 'cost_bbox': 0, 'cost_giou': 1}
  matcher = HungarianMatcher(weight_dict)
  data_processor = ConvertPILImage()
  main(model, matcher, data_processor, device, args.json, args.root, args.mode, args.output)