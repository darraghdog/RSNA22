import os
import sys
import platform
PATH = './'
'''
PATH = '/mount/rsna2022'
if platform.system()=='Darwin':
    PATH = '/Users/dhanley/Documents/rsna2022'
    PATH = '/Users/dhanley/Documents/RSNA22'
    os.chdir(f'{PATH}')
    sys.path.append("configs")
    sys.path.append("models")
    sys.path.append("data")
    sys.path.append("postprocess")
'''
from default_config import basic_cfg
import pandas as pd

cfg = basic_cfg
cfg.debug = True

# paths
if platform.system()!='Darwin':
    cfg.name = os.path.basename(__file__).split(".")[0]
    cfg.output_dir = f"{PATH}/weights/{os.path.basename(__file__).split('.')[0]}"

cfg.data_dir = f"{PATH}/datamount/"
cfg.data_folder = cfg.data_dir + "train_images/"
# cfg.train_df = f'{cfg.data_dir}/train_bbox_v01.csv.gz'
# cfg.test_df = cfg.train_df = f'{cfg.data_dir}/train_image_pseu_v04.csv.gz' # f'{cfg.data_dir}/train_bbox_v01.csv.gz'
cfg.test_df = cfg.train_df = 'datamount/train_all_slices_v01.csv.gz'

# stages
cfg.test = True
# cfg.test_data_folder = cfg.data_dir + "test/"
cfg.test_data_folder = cfg.data_folder # cfg.data_dir + "test/"
cfg.val = False
cfg.train = False
#cfg.test = False
#cfg.test_data_folder = cfg.data_dir + "test/"
#cfg.train = True
cfg.filter_val = False
cfg.train_val =  False
cfg.eval_epochs = 1
cfg.local_rank = 0
cfg.create_submission = False

#logging
cfg.neptune_project = "light/kaggle-rsna2022"
cfg.neptune_connection_mode = "async"
cfg.tags = "debug"

#model
cfg.model = "mdl_loc_dh_1A"
cfg.backbone = 'efficientnet_b1'
cfg.save_weights_only = True

# DATASET
cfg.dataset = "ds_loc_dh_1A"
if platform.system() == 'Darwin':
    cfg.debugcount = -1# 500
cfg.imagesize = (512,512)
cfg.windows = [(400, 1800)] # [ (500, 1800), (400, 650), (80, 300)]
cfg.drop_scans = '1.2.826.0.1.3680043.20574 1.2.826.0.1.3680043.29952'.split()
cfg.targets = 'x0 y0 x1 y1 has_box'.split()
cfg.load_jpg = False

# Augmentations
#cfg.p_grid_distortion = 0.5
cfg.shift_limit = 0.2
cfg.scale_limit = 0.2
cfg.rotate_limit = 20
cfg.RandomContrast = 0.2
cfg.hflip = 0.5
cfg.transpose = 0.
cfg.vflip = 0.
cfg.holes = 0
cfg.hole_size = 0.1
cfg.norm_mean = [0.22363983, 0.18190407, 0.2523437]
cfg.norm_std = [0.32451536, 0.2956294,  0.31335256]

# OPTIMIZATION & SCHEDULE
cfg.fold = 0
cfg.epochs = 1
cfg.warmup = (cfg.epochs) * 0.1

cfg.lr = 0.0002
cfg.optimizer = "AdamW"
#cfg.weight_decay = 0.00001
#cfg.clip_grad = 5
cfg.batch_size = 32
cfg.batch_size_val = 256
cfg.mixed_precision = True
cfg.pin_memory = False
cfg.grad_accumulation = 1
cfg.num_workers = 16
cfg.loss_weights = [0.8, 0.2]

# EVAL
cfg.calc_metric = False # True
cfg.post_process_pipeline = "pp_loc_dh_01A"
# augs & tta

cfg.train_aug = None
cfg.val_aug = None

#Saving
cfg.save_weights_only = True
cfg.save_only_last_ckpt = True
cfg.save_checkpoint = False
cfg.train_aug = None
cfg.val_aug = None
cfg.pretrained_weights_strict = False
cfg.pop_weights = ["posemb.weight", 'poschemb.weight']

