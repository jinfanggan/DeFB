class Config:
    def __init__(self):
        self.config_name = 'full_v1'
        self.train_set = {'fully_supervised': 'split_blink_datasets/full_v1/fully_supervised/'}
        self.split_ratio = {'fully_supervised': 1}
        self.pipelines = [{'stage': 'train', 'set': ['fully_supervised'], 'epoch': 40, 'test_gap': 1}]
        self.model_path = 'work_dirs/full_v1/final_ckpt.pth'
        self.h5_path = "BinkDetectionDataset/mpeblink_v1/train_dataset.h5"
        self.test_set = "BinkDetectionDataset/mpeblink_v1/val_dataset.h5"
        self.window_size =16
        self.stride_sample = 8
        self.batch_size = 192
        self.feature_dim = 256
        self.sample_point = 60
        self.num_heads = 8
        self.num_layers = 3
        self.learning_rate = 0.0001
        self.load_from = None
        self.roi_feature_encoder = True