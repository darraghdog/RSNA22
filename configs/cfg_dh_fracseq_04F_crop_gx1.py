import os
import sys
import platform

PATH='./'
'''
PATH = '/mount/rsna2022'
if platform.system()=='Darwin':
    PATH = '/Users/dhanley/Documents/rsna2022'
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
cfg.train_df = f'{cfg.data_dir}/train_image_pseu_v04_crop.csv.gz'
cfg.bbox_df = f'{cfg.data_dir}/train_bbox_pred_v02.csv.gz'
cfg.meta_df = f'{cfg.data_dir}/train_folded_v01.csv'

cfg.target_seg = [f'C{i}_pred' for i in range(1,8)]
cfg.target_frac = [f'C{i}_frac' for i in range(1,8)]
cfg.target = [f'C{i}' for i in range(1,8)]

# stages
cfg.test = False
cfg.test_data_folder = cfg.data_dir + "test/"
cfg.train = True
cfg.train_val =  False
cfg.eval_epochs = 1
cfg.filter_val = False

#logging
cfg.neptune_project = "light/kaggle-rsna2022"
cfg.neptune_connection_mode = "async"
cfg.tags = "debug"

#model
cfg.model = "mdl_dh_seg_3G_gx1" # "mdl_dh_seg_2C"
cfg.backbone = 'resnest50d'  #'efficientnet-b2'
cfg.save_weights_only = True
# cfg.stride = 2
# cfg.interpolation_mode = 'bicubic'
cfg.rnn_num_layers = 1
#cfg.rnn_dropout=0.1
#cfg.head_dropout = 0.1
cfg.rnn_dropout=0.
cfg.head_dropout = 0.0
cfg.rnn_seq_len = 192
cfg.interpolation_mode = 'bilinear'

# DATASET
cfg.dataset = "ds_dh_fracseg_3I_crop"
if platform.system() == 'Darwin':
    cfg.debugcount = -1# 500
cfg.imagesize = (512,512)
cfg.windows = [(400, 1800)] # [ (500, 1800), (400, 650), (80, 300)]
cfg.drop_scans = '1.2.826.0.1.3680043.20574 1.2.826.0.1.3680043.29952'.split()
cfg.sequence_len = 32 * 3
cfg.dicom_steps = 1
cfg.random_sampler_frac = 0.1
cfg.cnn_chunk_size = 64
cfg.interpolate_chunk_size = 32 * 3
cfg.norm_on_cuda = False
cfg.loss_weights = {
        '-' : [7, 1, 1, 1, 1, 1, 1, 1],
        '+' : [14, 2, 2, 2, 2, 2, 2, 2],
        }
cfg.pretrained_weights_strict = False
cfg.pop_weights = 'posemb poschemb rnn.weight_ih_l0 rnn.weight_hh_l0 rnn.bias_ih_l0 rnn.bias_hh_l0'.split()
# cfg.load_jpg = True
cfg.load_jpg = False

# Augmentations
cfg.p_grid_distortion = 0.5
cfg.shift_limit = 0.2
cfg.scale_limit = 0.2
cfg.rotate_limit = 20
cfg.RandomContrast = 0.2
cfg.hflip = 0.5
cfg.vflip = 0.
cfg.holes = 8
cfg.hole_size = 0.1
cfg.norm_mean = [0.22363983, 0.18190407, 0.2523437]
cfg.norm_std = [0.32451536, 0.2956294,  0.31335256]

# OPTIMIZATION & SCHEDULE
cfg.fold = 0
cfg.epochs = 3

cfg.lr = 0.001
cfg.weight_decay = 5.0e-4
cfg.optimizer = "AdamW" # "staggered_lr_optimizer"#"AdamW"
cfg.warmup = (cfg.epochs) * 0.1
cfg.batch_size = 8
cfg.mixed_precision = True
cfg.pin_memory = False
cfg.grad_accumulation = 1.
cfg.num_workers = 16
cfg.loss='mse'
cfg.positive_weight = 2
# cfg.eval_steps = 5000 # 00

# EVAL
cfg.calc_metric = False # True
# cfg.metric = 'auc'
cfg.post_process_pipeline = "pp_dh_seg_03A"
# augs & tta

#Saving
cfg.save_weights_only = True
cfg.save_only_last_ckpt = True
cfg.save_checkpoint = True
cfg.train_aug = None
cfg.val_aug = None
