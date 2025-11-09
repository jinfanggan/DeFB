trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM EXIT
ulimit -n 65536
set -e

CUDA_VISIBLE_DEVICES=1,2 torchrun --master_port=9909 --nproc_per_node=2 tools/train.py  --use-amp --seed=0 -c "configs/rtdetrv2/detrs-blink_len=10_mpeblinkv1.yml" -t "rtdetrv2_r50vd_6x_coco_ema.pth"

CUDA_VISIBLE_DEVICES=1,2 torchrun --master_port=9909 --nproc_per_node=2 tools/train.py  --use-amp --seed=0 -c "configs/rtdetrv2/detrs-blink_len=30_mpeblinkv1.yml" -t "output/rtdetrv2_r50vd_6x_coco_len=10_mpeblinkv1/last.pth"

CUDA_VISIBLE_DEVICES=1 python infer_trainset.py --config "configs/rtdetrv2/detrs-blink_len=30_mpeblinkv1.yml" --output 'mpeblink_v1' --checkpoint "output/rtdetrv2_r50vd_6x_coco_len=30_mpeblinkv1/checkpoint0000.pth" --json "/data/data1/ganjinfang/ICCV-2025/mpeblink_cvpr2023/annotations/val.json" --root "/data/data1/ganjinfang/ICCV-2025/mpeblink_cvpr2023/val_rawframes" --mode 'val' &

CUDA_VISIBLE_DEVICES=1 python infer_trainset.py --config "configs/rtdetrv2/detrs-blink_len=30_mpeblinkv1.yml" --output 'mpeblink_v1' --checkpoint "output/rtdetrv2_r50vd_6x_coco_len=10_mpeblinkv1/checkpoint0000.pth" --json "/data/data1/ganjinfang/ICCV-2025/mpeblink_cvpr2023/annotations/train.json" --root "/data/data1/ganjinfang/ICCV-2025/mpeblink_cvpr2023/train_rawframes" --mode 'train' 


python BlinkModel/split_dataset.py --config "configs/BlinkModule/full_v1.py"

CUDA_VISIBLE_DEVICES=1 python BlinkModel/train_blink_detector.py --config "configs/BlinkModule/full_v1.py"

CUDA_VISIBLE_DEVICES=1 python test.py --track_config "configs/rtdetrv2/detrs-blink_len=30_mpeblinkv1.yml" --blink_config "configs/BlinkModule/full_v1.py" --output 'mpeblink_v1' --checkpoint "output/rtdetrv2_r50vd_6x_coco_len=30_mpeblinkv1/checkpoint0000.pth" --json "/data/data1/ganjinfang/ICCV-2025/mpeblink_cvpr2023/annotations/test.json" --root "/data/data1/ganjinfang/ICCV-2025/mpeblink_cvpr2023/test_rawframes" --mode 'test'

python tools/instblink_plus_result_convertor_args.py --json "results/test_results/mpeblink_v1.json" --output "results/blink_converted_results/mpeblink_v1.json" --threshold 0.07

python tools/eval_mpeblink.py --gt_json "/data/data1/ganjinfang/ICCV-2025/mpeblink_cvpr2023/annotations/test.json" --pred_json "results/blink_converted_results/mpeblink_v1.json"

