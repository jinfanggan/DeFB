import os.path as osp
import random
from collections import defaultdict
import numpy as np
import torch
from PIL import Image 
import cv2
import os
from .mpeblink_api import MPEblink
from torch.utils.data import Dataset
from ...core import register
from .._misc import convert_to_tv_tensor

@register()
class MPEblinkV2Dataset(Dataset):
    CLASSES = ('person_face')
    __inject__ = ['transforms', ]
    def __init__(self,
                 ann_file="/data/data4/zengwenzheng/data/dataset_building/mpeblink2_1/annotations/train.json",
                 img_fold = "/data/data4/zengwenzheng/data/dataset_building/mpeblink2_1/train_rawframes/",
                 transforms=None,
                 clip_length=12,
                 classes=None,
                 stride = 1,
                 infer_length = 1,
                 frame_interval = 2,
                 data_root=None,
                 img_prefix='',
                 seg_prefix=None,
                 with_eye_bbox=True,
                 proposal_file=None,
                 test_mode=False,
                 filter_empty_gt=True):
                 
        self.ann_file = ann_file
        self.frame_interval = frame_interval
        self.img_fold = img_fold
        self.stride = stride
        self.infer_length = infer_length
        self.clip_length = clip_length
        self.global_step = 0
        self.data_root = data_root
        self.img_prefix = img_prefix
        self.seg_prefix = seg_prefix
        self.with_eye_bbox = with_eye_bbox
        self.proposal_file = proposal_file
        self.test_mode = test_mode
        self.filter_empty_gt = filter_empty_gt
        self.epoch = 0
        # join paths if data_root is specified
        if self.data_root is not None:
            if not osp.isabs(self.ann_file):
                self.ann_file = osp.join(self.data_root, self.ann_file)
            if not (self.img_prefix is None or osp.isabs(self.img_prefix)):
                self.img_prefix = osp.join(self.data_root, self.img_prefix)
            if not (self.seg_prefix is None or osp.isabs(self.seg_prefix)):
                self.seg_prefix = osp.join(self.data_root, self.seg_prefix)
            if not (self.proposal_file is None
                    or osp.isabs(self.proposal_file)):
                self.proposal_file = osp.join(self.data_root,
                                              self.proposal_file)
        # load annotations (and proposals)
        self.data_infos, _ = self.load_annotations(
            self.ann_file)  # 返回的是(videoId,video frame)这样的list，数量为总帧数，用于getitem的index

        if self.proposal_file is not None:  # 没进这个
            self.proposals = self.load_proposals(self.proposal_file)
        else:
            self.proposals = None
            
        self.sample_idx = []
        # filter images too small
        if not test_mode:
            valid_inds = self._filter_imgs()  # 在dataset中删掉一些帧，这些帧中，没有一个instance，只要这帧中有一个instance就不会被删
            self.data_infos = [self.data_infos[i] for i in valid_inds]
            
            for i in range(len(self.data_infos)):
              vid, frame_id = self.data_infos[i]
              if i % self.stride == 0 and vid < 423: ###人少的间隔采样
                self.sample_idx.append(self.data_infos[i])
              elif i % (self.stride // 8 + 1) == 0 and vid >= 423: ###人多采样
                self.sample_idx.append(self.data_infos[i])

        # set group flag for the sampler
        if not self.test_mode:
            self._set_group_flag()
        self.data_infos_found = set(self.data_infos)
        print(f'origin__num = {len(self.sample_idx)}')
        # processing pipeline
        self.pipeline = transforms
        
    def set_epoch(self, epoch):
        self.epoch = epoch
    
    def set_clip_length(self, clip_length):
        self.clip_length = clip_length
       
    def load_annotations(self, ann_file):
        self.mpeblink = MPEblink(ann_file)  # coco api来读其gt标注文件的
        self.cat_ids = self.mpeblink.getCatIds()  # 就是类别1-40的数字组成的list
        self.cat2label = {cat_id: i for i, cat_id in
                          enumerate(self.cat_ids)}  # 变成一个字典{1:0,2:1,3:2 ..... 40:39}代表类别和标签的映射关系，实际上就是1-40变为0-39
        vid_ids = self.mpeblink.getVidIds()  # 一个列表1-2238，是video的数量
        print(f'total_vid:{len(vid_ids)}')
        vid_infos = []
        for i in vid_ids:
            info = self.mpeblink.loadVids([i])[0]  # 加载当前video id的video的基本信息，就是data['video']的len=9的dict
            info['filenames'] = info['file_names']  # 这为啥复制一个差不多的东西出来
            vid_infos.append(info)
        self.vid_infos = vid_infos  # 各个video的一些基本信息

        img_ids = []
        img_ids_select = []
        for idx, vid_info in enumerate(self.vid_infos):
            for frame_id in range(len(vid_info['filenames'])):
                img_ids.append((idx, frame_id))

        return img_ids, img_ids_select

    def _set_group_flag(self):
        """Set flag according to image aspect ratio.

        Images with aspect ratio greater than 1 will be set as group 1,
        otherwise group 0.
        """
        self.flag = np.zeros(len(self), dtype=np.uint8)
        for i, (vid, frame_id) in enumerate(self.sample_idx):
            video_info = self.vid_infos[vid]
            if video_info['width'] / video_info['height'] > 1:
                self.flag[i] = 1

    def _filter_imgs(self, min_size=32):
        """Filter images too small or without ground truths."""
        valid_inds = []
        ids_with_ann = []

        if self.filter_empty_gt:
            for i, (vid, frame_id) in enumerate(self.data_infos):
                vid_id = self.vid_infos[vid]['id']
                ann_ids = self.mpeblink.getAnnIds(
                    vidIds=[vid_id])  # 返回Video id=1的annotation的id,这个video里有几个instance就会返回几个id
                ann_info = self.mpeblink.loadAnns(ann_ids)  # 根据返回的id取出它们的annotation信息
                anns = [
                    ann['bboxes'][frame_id] for ann in ann_info
                    if ann['bboxes'][frame_id] is not None
                ]
                if anns:
                    ids_with_ann.append(
                        1)  # 最终，ids_with_ann有完整帧数个元素(61845)如果某一帧没有instance,则该元素值为0，如果这一帧有的instance有，有的instance的bbox是none,则元素值也是1，最终，sum()=61341
                else:
                    ids_with_ann.append(0)
        for i, (vid, frame_id) in enumerate(self.data_infos):  # 下面这个循环是进一步看有没有图像分辨率小于32的，实际没有
            if self.filter_empty_gt and not ids_with_ann[i]:
                continue
            if min(self.vid_infos[vid]['width'],
                   self.vid_infos[vid]['height']) >= min_size:
                valid_inds.append(i)
        return valid_inds  # 最终返回了61341个元素，对应了data_infos中的index，目前，同时存在bbox和bbox=none的帧，仍然在valid_inds中

    def get_img_info(self, idx):
        vid, frame_id = self.data_infos[idx]
        vid_info = self.vid_infos[vid]
        img_info = dict(
            file_name=vid_info['file_names'][frame_id],
            filename=vid_info['filenames'][frame_id],
            width=vid_info['width'],
            height=vid_info['height'],
            frame_id=frame_id)
        return img_info

    def get_ann_info(self, idx):
        vid, frame_id = self.data_infos[idx]
        vid_id = self.vid_infos[vid]['id']  # 其实就是vid+1吧
        ann_ids = self.mpeblink.getAnnIds(vidIds=[vid_id])
        ann_info = self.mpeblink.loadAnns(ann_ids)
        return self._parse_ann_info(ann_info, frame_id)

    def get_cat_ids(self, idx):
        vid, frame_id = self.data_infos[idx]
        vid_id = self.vid_infos[vid]['id']
        ann_ids = self.mpeblink.getAnnIds(vidIds=[vid_id])
        ann_info = self.mpeblink.loadAnns(ann_ids)
        return [ann['category_id'] for ann in ann_info]

    def Manhattan(self, p1, p2):
        x1 = p1[0]
        x2 = p2[0]
        y1 = p1[1]
        y2 = p2[1]
        result = abs(x1 - x2) + abs(y1 - y2)
        return result

    def get_eye_region(self, pos_left, pos_right):
        h = self.Manhattan(pos_left, pos_right)
        return h

    def get_min_max_position(self, coordinates, indices):
        x_values = [coordinates[i][0] for i in indices]
        y_values = [coordinates[i][1] for i in indices]
        min_x = min(x_values)
        max_x = max(x_values)
        min_y = min(y_values)
        max_y = max(y_values)
        return min_x, max_x, min_y, max_y

    def _parse_ann_info(self, ann_info, frame_id):
        """Parse bbox and mask annotation.

        Args:
            ann_info (list[dict]): Annotation info of an image.
            with_mask (bool): Whether to parse mask annotations.

        Returns:
            dict: A dict containing the following keys: bboxes, bboxes_ignore,
                labels, masks, seg_map. "masks" are raw annotations and not
                decoded into binary masks.
        """
        gt_bboxes = []
        gt_labels = []
        gt_ids = []
        gt_bboxes_ignore = []
        gt_blinks = []
        gt_eye_bboxes = []

        for i, ann in enumerate(ann_info):  # 遍历每一个instance
            bbox = ann['bboxes'][frame_id]  # 提取这个instance的[frame_id]那一帧的bbox
       
            # area = ann['areas'][frame_id]
            if bbox is None:  # 这里要注意，如果这一帧的某个instance gt是none,那么这个instance(这一帧)不会写在gt中
                continue
            
            x1, y1, w, h = bbox
            bbox = [max(min(x1,x1 + w), 0), max(min(y1, y1 + h), 0), min(max(x1,x1 + w), 640), min(max(y1, y1 + h), 360)]

            if self.with_eye_bbox:
                if len(ann['landmark'][frame_id]) == 98:
                  selected_landmark_index = [33, 38, 46, 50, 53, 60, 65, 66, 67, 72, 73, 74, 75]
                else:
                  selected_landmark_index = [17, 21, 22, 26, 36, 40, 41, 45, 46, 47, 29]
                
                min_x, max_x, min_y, max_y = self.get_min_max_position(ann['landmark'][frame_id],
                                                                       selected_landmark_index)
                                                                       
                if max(min_x, max_x, min_y, max_y) > 1000 or min(min_x, max_x, min_y, max_y) < -1000:
                  eye_bbox = [0,0,0,0]
                else:
                  padding_ratio = 0.1
                  w, h = max_x - min_x, max_y - min_y
                  eye_bbox = [max(min_x - padding_ratio * w,0) , max(min_y - padding_ratio * h,0), \
                                  min(max_x + padding_ratio * w,640), min(max_y + padding_ratio * h,360)]  # 这个bbox换成了局部眼部区域
           
            ### 以上为v2版本新加入的eye_bbox

            if ann.get('iscrowd', False):
                gt_bboxes_ignore.append(bbox)
            else:
                gt_bboxes.append(bbox)
                gt_ids.append(ann['id'] -1)  # youtube instance id start from 1. 把从1开始变为从0开始算id
                gt_labels.append(self.cat2label[ann['category_id']])  # 这个好像也是类别从1开始变为从0开始，也就是json中的类别减1
                # gt_masks.append(self.youtube.annToMask(ann, frame_id)) # 应该是转化为一个0-1的mask 对于眨眼，先把mask去掉
                gt_blinks.append(ann['blinks_binary'][frame_id])
                gt_eye_bboxes.append(eye_bbox)
                
        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)  # 就是从list转为numpy array
            gt_labels = np.array(gt_labels, dtype=np.int64)
            gt_blinks = np.array(gt_blinks, dtype=np.int64)
            gt_eye_bboxes = np.array(gt_eye_bboxes, dtype=np.float32)
        else:  # 这个循环我估计没进过吧，因为前面滤过一次valid_ins了
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        if self.with_eye_bbox:
            ann = dict(
                bboxes=gt_bboxes,
                labels=gt_labels,
                blinks=gt_blinks,
                eye_bboxes=gt_eye_bboxes,
                bboxes_ignore=gt_bboxes_ignore,
                ids=gt_ids)
        else:
            ann = dict(
                bboxes=gt_bboxes,
                labels=gt_labels,
                blinks=gt_blinks,
                bboxes_ignore=gt_bboxes_ignore,
                ids=gt_ids)

        return ann

    def pre_pipeline(self, results):
        results['img_prefix'] = self.img_prefix
        results['seg_prefix'] = self.seg_prefix
        results['proposal_file'] = self.proposal_file
        results['bbox_fields'] = []
        results['mask_fields'] = []
        results['seg_fields'] = []

    def __getitem__(self, idx):
        if self.test_mode:
            raise NotImplementedError
        while True:
            data = self.prepare_train_clip(idx)
            return data

    def prepare_train_clip(self, idx):
        vid, frame_id = self.sample_idx[idx]
        vid_info = self.vid_infos[vid]
   
        sample_range = range(len(vid_info['filenames']))
        valid_idxs = []
        for i in sample_range:
            valid_idx = (vid, i)
            if valid_idx in self.data_infos_found:
                valid_idxs.append(valid_idx)
                
        vid_length = len(valid_idxs)
        assert len(valid_idxs) > 0
        # 上面几行是根据valid_idx来确定当前视频内有哪些帧可以被采样选取
        frame_interval = self.frame_interval  # 现在是2帧一采样
        # print(self.clip_length)
        min_index = max(0,  frame_id - frame_interval * self.clip_length // 2)
        max_index = min(vid_length,  frame_id + frame_interval * self.clip_length // 2)
        if max_index <= min_index:
          max_index, min_index = vid_length, 0
          
        if self.clip_length % 2 == 0:
            index_pre = [(vid, frame_id - frame_interval * i) for i in range(1, self.clip_length // 2) if
                         (frame_id - frame_interval * i) >= valid_idxs[0][1] and \
                         (vid, frame_id - frame_interval * i) in valid_idxs]
            pre_res = [(vid, valid_idxs[random.randint(min_index, max_index - 1)][1]) for i in range(0, self.clip_length // 2 - len(index_pre) - 1)]  # 补第一帧可用帧补剩下的
            index_pre = index_pre + pre_res
            index_post = [(vid, frame_id + frame_interval * i) for i in range(1, self.clip_length // 2 + 1) if
                          (frame_id + frame_interval * i) <= valid_idxs[-1][1] and \
                          (vid, frame_id + frame_interval * i) in valid_idxs]
            post_res = [(vid, valid_idxs[random.randint(min_index, max_index - 1)][1]) for i in range(0, self.clip_length // 2 - len(index_post))]  # 用最后一可用帧补
            index_post += post_res

            index_except_center = index_pre + [(vid, frame_id)] + index_post
            valid_idxs = [self.data_infos.index(_) for _ in index_except_center]
            valid_idxs.sort()

        else:
            index_pre = [(vid, frame_id - frame_interval * i) for i in range(1, self.clip_length // 2 + 1) if
                         (frame_id - frame_interval * i) >= valid_idxs[0][1] and \
                         (vid, frame_id - frame_interval * i) in valid_idxs]
            pre_res = [(vid, valid_idxs[random.randint(min_index, max_index - 1)][1]) for i in range(0, self.clip_length // 2 - len(index_pre))]  # 补第一帧可用帧补剩下的
            index_pre = index_pre + pre_res
            index_post = [(vid, frame_id + frame_interval * i) for i in range(1, self.clip_length // 2 + 1) if
                          (frame_id + frame_interval * i) <= valid_idxs[-1][1] and \
                          (vid, frame_id + frame_interval * i) in valid_idxs]
            post_res = [(vid, valid_idxs[random.randint(min_index, max_index - 1)][1]) for i in range(0, self.clip_length // 2 - len(index_post))]  # 用最后一可用帧补
            index_post += post_res

            index_except_center = index_pre + [(vid, frame_id)] + index_post
            valid_idxs = [self.data_infos.index(_) for _ in index_except_center]
            valid_idxs.sort()
     
        clip = []

        for _ in valid_idxs:
            clip.append(self.prepare_train_img(_))

        collect_data = []
        sample_num = self.clip_length // self.infer_length
    
        for i in range(sample_num):
            data = {'image':[], 'ann':{}, 'head_bbox':[], 'eye_bbox':[], 'blinks':[]}
            start_index = i * self.infer_length
            end_index = start_index + self.infer_length
            sample_clip = clip[start_index:end_index]
            
            for t, (image, ann) in enumerate(sample_clip):
                data['image'].append(image)
                for p, person_id in enumerate(ann['idx']):
                    if str(person_id) not in data['ann'].keys():
                        data['ann'][str(person_id)] = {}
                        data['ann'][str(person_id)]['head_bbox'] = torch.zeros(len(sample_clip), 4)
                        data['ann'][str(person_id)]['eye_bbox'] = torch.zeros(len(sample_clip), 4)
                        data['ann'][str(person_id)]['blink_gt'] = torch.zeros(len(sample_clip), dtype = torch.long)
            
                    data['ann'][str(person_id)]['head_bbox'][t] = ann['head_boxes'][p]
                    data['ann'][str(person_id)]['eye_bbox'][t] = ann['eye_boxes'][p]
                    data['ann'][str(person_id)]['blink_gt'][t] = ann['blink_gt'][p]

            data['image'] = torch.stack(data['image'])
            data['head_bbox'] = torch.stack([v['head_bbox'] for k, v in data['ann'].items()])
            data['eye_bbox'] = torch.stack([v['eye_bbox'] for k, v in data['ann'].items()])
            data['blink_gt'] = torch.stack([v['blink_gt'] for k, v in data['ann'].items()])
            
            collect_data.append(data)

        return collect_data

    def prepare_test_clip(self, idx):
        raise NotImplementedError

    def prepare_train_img(self, idx):
        img_info = self.get_img_info(idx)  # 获取一些信息，比较重要的是图像的路径，其实该函数内部可以通过vid获取整个视频的信息list，只不过这里取出了frame_id那一帧的信息
        ann_info = self.get_ann_info(idx)
        results = dict(img_info=img_info, ann_info=ann_info)
        img_path = os.path.join(self.img_fold, results['img_info']['file_name'])
        image = Image.open(img_path)
      
        coco_result = {}
        coco_result['image_id'] = results['img_info']['frame_id']
        coco_result['idx'] = results['ann_info']['ids']
  
        coco_result['bboxes'] = convert_to_tv_tensor(np.concatenate((results['ann_info']['bboxes'],results['ann_info']['eye_bboxes']),axis=0) , key='boxes', \
                    spatial_size=image.size[::-1])

        image, coco_result, _ = self.pipeline(image, coco_result, self.epoch)
        
        num_bbox = coco_result['bboxes'].size(0)
    
        coco_result['head_boxes'] = coco_result['bboxes'][:num_bbox//2]
        coco_result['eye_boxes'] = coco_result['bboxes'][num_bbox//2:]
        coco_result['blink_gt'] = ann_info['blinks']
        
        return (image, coco_result)

    def __len__(self):
        return len(self.sample_idx)

if __name__ == '__main__':
  dataset = MPEblinkV2Dataset()
  print(dataset.__getitem__(0))