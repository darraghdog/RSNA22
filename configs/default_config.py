from types import SimpleNamespace
from copy import deepcopy

cfg = SimpleNamespace(**{})

# stages
cfg.train = True
cfg.val = True
cfg.test = True
cfg.train_val = True

# dataset
cfg.dataset = "base_ds"
cfg.n_classes = 1
cfg.batch_size = 32
cfg.val_df = None
cfg.test_df = None
cfg.batch_size_val = None
cfg.normalization = None
cfg.train_aug = False
cfg.val_aug = False
cfg.test_augs = False
cfg.suffix = ".jpg"

cfg.data_sample = -1
cfg.dropout_tokens = 0.0

# img model
cfg.backbone = "tf_efficientnet_b0_ns"
cfg.pretrained = True
cfg.pretrained_weights = None
cfg.pop_weights = None
cfg.pretrained_weights_strict = True
cfg.pool = "avg"
cfg.in_chans = 3
cfg.gem_p_trainable = False
cfg.offline_inference = False
cfg.eval_steps = 0
cfg.eval_epochs = 1
cfg.eval_train_epochs = 5
cfg.drop_path_rate = None
cfg.drop_rate = None
cfg.dropout = 0.0
cfg.warmup = 0
cfg.label_smoothing = 0

# training
cfg.fold = 0
cfg.val_fold = -1
cfg.lr = 1e-4
cfg.schedule = "cosine"
cfg.weight_decay = 0
cfg.optimizer = "Adam"  # "Adam", "fused_Adam", "SGD", "fused_SGD"
cfg.epochs = 10
cfg.seed = -1
cfg.resume_training = False
cfg.simple_eval = False
cfg.do_test = True
cfg.do_seg = False
cfg.eval_ddp = True
cfg.clip_grad = 0
cfg.debug = False
cfg.save_val_data = True
cfg.gradient_checkpointing = False
cfg.awp = False
cfg.awp_per_step = False
cfg.pseudo_df = None

# eval
cfg.calc_metric = True
cfg.calc_metric_epochs = 1
cfg.calc_metric2 = False
cfg.post_process_pipeline = "base_pp"

# ressources
cfg.find_unused_parameters = False
cfg.mixed_precision = True
cfg.grad_accumulation = 1
cfg.syncbn = False
cfg.gpu = 0
cfg.dp = False
cfg.num_workers = 4
cfg.drop_last = True
cfg.save_checkpoint = True
cfg.save_only_last_ckpt = False
cfg.save_weights_only = False
cfg.save_first_batch = False

# logging,
cfg.neptune_project = None
cfg.neptune_connection_mode = "async"
cfg.tags = None
cfg.save_first_batch = False
cfg.save_first_batch_preds = False
cfg.sgd_nesterov = True
cfg.sgd_momentum = 0.9
cfg.clip_mode = "norm"
cfg.data_sample = -1
cfg.track_grad_norm = True
cfg.grad_norm_type = 2. # use from "torch._six import inf" to use inf norm
cfg.norm_eps = 1e-4

cfg.model_channel_norm = False

cfg.mixup = 0

cfg.mix_beta = 0.5
cfg.mixadd = False

cfg.loss = "bce"

cfg.tta = []

cfg.s3_bucket_name = ""
cfg.s3_access_key = ""
cfg.s3_secret_key = ""

basic_cfg = cfg
