import os
import sys
import platform
PATH = './'
'''
PATH = '/mount/rsna2022'
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
cfg.name = os.path.basename(__file__).split(".")[0]
cfg.output_dir = f"{PATH}/weights/{os.path.basename(__file__).split('.')[0]}"

cfg.data_dir = f"{PATH}/datamount/"
cfg.data_folder = cfg.data_dir + "train_images/"
# cfg.train_df = f'{cfg.data_dir}/train_folded_v01.csv'
# cfg.train_df = f'{cfg.data_dir}/train_image_level_v03.csv.gz'
cfg.test_df = cfg.train_df = f'{cfg.data_dir}/train_image_level_v03.csv.gz'

'''
dd = pd.read_csv(cfg.train_df)
dd = dd.query('C1>-1').reset_index(drop=True)
dd = dd.sort_values('StudyInstanceUID fold'.split())
dd.filter(like='C')[:2000].rolling(5).mean().plot(figsize = (20,2))
'''
cfg.target = [f'C{i}' for i in range(1,8)]

# stages
cfg.test = True
# cfg.test_data_folder = cfg.data_dir + "test/"
cfg.test_data_folder = cfg.data_folder # cfg.data_dir + "test/"
cfg.val = False
cfg.train = False
cfg.train_val =  False
cfg.eval_epochs = 1
cfg.filter_val = False
cfg.local_rank = 0
cfg.create_submission = False

#logging
cfg.neptune_project = "light/kaggle-rsna2022"
# cfg.neptune_project = "darragh/uwm"
cfg.neptune_connection_mode = "async"
cfg.tags = "debug"

#model
cfg.model = "mdl_dh_seg_2C"
cfg.backbone = 'resnest50d'  #'efficientnet-b2'
cfg.save_weights_only = True
# cfg.stride = 2
# cfg.interpolation_mode = 'bicubic'
cfg.rnn_num_layers = 1
cfg.rnn_dropout=0.1
cfg.head_dropout = 0.1

# cfg.ema = True
# cfg.ema_start = 15
# cfg.ema_decay = 0.997

# DATASET
cfg.dataset = "ds_dh_seg_2D"
if platform.system() == 'Darwin':
    cfg.debugcount = -1# 500
cfg.imagesize = (512,512)
#cfg.channels = 'large_bowel small_bowel stomach'
#cfg.threshold = 0.5
#cfg.drop_cases = 'case7_day0 case81_day30'.split()
cfg.windows = [(400, 1800)] # [ (500, 1800), (400, 650), (80, 300)]
cfg.drop_scans = '1.2.826.0.1.3680043.20574 1.2.826.0.1.3680043.29952'.split()
cfg.sequence_len = 32 * 3
cfg.dicom_steps = 1
cfg.random_sampler_frac = 0.1

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
cfg.epochs = 1
#cfg.weighted_random_sampler = True
#cfg.positive_weight = 12
#cfg.random_sampler_frac = 0.4

cfg.lr = 0.0003
cfg.weight_decay = 5.0e-4
cfg.optimizer = "AdamW" # "staggered_lr_optimizer"#"AdamW"
cfg.warmup = (cfg.epochs) * 0.1
cfg.batch_size = 2
cfg.mixed_precision = True
cfg.pin_memory = False
cfg.grad_accumulation = 8.
cfg.num_workers = 16
cfg.loss='mse'
cfg.positive_weight = 2
cfg.eval_steps = 350# 00

# EVAL
cfg.calc_metric = True
# cfg.metric = 'auc'
cfg.post_process_pipeline = "pp_dh_seg_02A"
# augs & tta

#Saving
cfg.save_weights_only = True
cfg.save_only_last_ckpt = True
cfg.save_checkpoint = False
cfg.train_aug = None
cfg.val_aug = None
cfg.pretrained_weights_strict = False
cfg.pop_weights = ["posemb.weight", 'poschemb.weight']
