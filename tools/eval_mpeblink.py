import os 
import sys 
from argparse import ArgumentParser
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from src.data.dataset.mpeblink_api import MPEblink
#from src.data.dataset.mpeblink_eval_eye_api import MPEblinkEval
from src.data.dataset.mpeblink_eval_api import MPEblinkEval
import random

def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        '--gt_json', default="/data/data4/zengwenzheng/data/dataset_building/mpeblink_cvpr2023/annotations/test.json", help='Path to annotation json file')
    
    parser.add_argument(
        '--pred_json',default="/data/data4/zengwenzheng/code/cvpr2023_extension/other_methods/blink_eyelid/blink_eyelid_code/blink_eyelid_result.json", help='Path to pred json file')
    parser.add_argument(
        '--root', default="/data/data4/zengwenzheng/data/dataset_building/mpeblink2_1/test_rawframes/", help='Path to image file') 
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
   
    args = parser.parse_args()
    return args

def main(args):
    mpeblink = MPEblink(args.gt_json)
    mpeblink_dets = mpeblink.loadRes(args.pred_json)
    random.seed(0)
    for ann in mpeblink_dets.anns:
        if 'score' in mpeblink_dets.anns[ann]:
            break
    
        mpeblink_dets.anns[ann]['score'] = 1.0

    vid_ids = mpeblink.getVidIds()
    for res_type in ['bbox']:
        iou_type = res_type
        mpeblink_eval = MPEblinkEval(mpeblink, mpeblink_dets, iou_type, args.pred_json)
        mpeblink_eval.params.vidIds = vid_ids
        mpeblink_eval.evaluate()
        mpeblink_eval.accumulate()
        mpeblink_eval.action_ap()
        mpeblink_eval.summarize()

if __name__ == '__main__':
    args = parse_args()
    main(args)