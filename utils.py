from transformers import get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup
from transformers import AdamW, Adafactor
from torch.optim import AdamW as AdamW_torch
import random
import os
import numpy as np
import pandas as pd
from torch.utils.data import Sampler, RandomSampler, SequentialSampler, DataLoader
import torch
from torch.utils.data import RandomSampler, SequentialSampler, DataLoader, WeightedRandomSampler
from torch import nn, optim
from boto3.session import Session
import boto3

# from torch.cuda.amp import GradScaler, autocast
# from torch.nn.parallel import DistributedDataParallel as NativeDDP
import importlib
import math
import neptune.new as neptune
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import logging
import pickle
import glob
import pydicom
import pylibjpeg
from torch.utils.data import Dataset, DataLoader
import cv2
import json
from pathlib import Path
from torch._six import inf
def calc_grad_norm(parameters,norm_type=2.):
    
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == inf:
        norms = [p.grad.detach().abs().max().to(device) for p in parameters]
        total_norm = norms[0] if len(norms) == 1 else torch.max(torch.stack(norms))
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)
    if torch.logical_or(total_norm.isnan(), total_norm.isinf()):
        total_norm = None
        
    return total_norm
        
class OrderedDistributedSampler(Sampler):
    def __init__(self, dataset, num_replicas=None, rank=None):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas

        print("TOTAL SIZE", self.total_size)

    def __iter__(self):
        indices = list(range(len(self.dataset)))

        # add extra samples to make it evenly divisible
        indices += indices[: (self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[
            self.rank * self.num_samples : self.rank * self.num_samples + self.num_samples
        ]
        print(
            "SAMPLES",
            self.rank * self.num_samples,
            self.rank * self.num_samples + self.num_samples,
        )
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples


def sync_across_gpus(t, world_size):
    torch.distributed.barrier()
    gather_t_tensor = [torch.ones_like(t) for _ in range(world_size)]
    torch.distributed.all_gather(gather_t_tensor, t)
    return torch.cat(gather_t_tensor)


def set_seed(seed=1234):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def get_model(cfg, ds):
    Net = importlib.import_module(cfg.model).Net
    net = Net(cfg)
    if cfg.pretrained_weights is not None:
        print(f'{cfg.local_rank}: loading weights from',cfg.pretrained_weights)
        state_dict = torch.load(cfg.pretrained_weights, map_location='cpu')
        if "model" in state_dict.keys():
            state_dict = state_dict['model']
        state_dict = {key.replace('module.',''):val for key,val in state_dict.items()}
        if cfg.pop_weights is not None:
            print(f'popping {cfg.pop_weights}')
            to_pop = []
            for key in state_dict:
                for item in cfg.pop_weights:
                    if item in key:
                        to_pop += [key]
            for key in to_pop:
                print(f'popping {key}')
                state_dict.pop(key)

        net.load_state_dict(state_dict, strict=cfg.pretrained_weights_strict)
        print(f'{cfg.local_rank}: weights loaded from',cfg.pretrained_weights)
    
    return net


def create_checkpoint(cfg, model, optimizer, epoch, scheduler=None, scaler=None):

    
    state_dict = model.state_dict()
    if cfg.save_weights_only:
        checkpoint = {"model": state_dict}
        return checkpoint
    
    checkpoint = {
        "model": state_dict,
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
    }

    if scheduler is not None:
        checkpoint["scheduler"] = scheduler.state_dict()

    if scaler is not None:
        checkpoint["scaler"] = scaler.state_dict()
    return checkpoint

def load_checkpoint(cfg, model, optimizer, scheduler=None, scaler=None):
    
    print(f'loading ckpt {cfg.resume_from}')
    checkpoint = torch.load(cfg.resume_from, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler_dict = checkpoint['scheduler']
    if scaler is not None:    
        scaler.load_state_dict(checkpoint['scaler'])
        
    epoch = checkpoint['epoch']
    return model, optimizer, scheduler_dict, scaler, epoch


def get_dataset(df, cfg, mode='train'):
    
    #modes train, val, index
    print(f"Loading {mode} dataset")
    
    if mode == 'train':
        dataset = get_train_dataset(df, cfg)
#     elif mode == 'train_val':
#         dataset = get_val_dataset(df, cfg)
    elif mode == 'val':
        dataset = get_val_dataset(df, cfg)
    elif mode == 'test':
        dataset = get_test_dataset(df, cfg)
    else:
        pass
    return dataset

def get_dataloader(ds, cfg, mode='train'):
    
    if mode == 'train':
        dl = get_train_dataloader(ds, cfg)
    elif mode =='val':
        dl = get_val_dataloader(ds, cfg)
    elif mode =='test':
        dl = get_test_dataloader(ds, cfg)
    return dl


def get_train_dataset(train_df, cfg):
    print("Loading train dataset")

    train_dataset = cfg.CustomDataset(train_df, cfg, aug=cfg.train_aug, mode="train")
    if cfg.data_sample > 0:
        train_dataset = torch.utils.data.Subset(train_dataset, np.arange(cfg.data_sample))
    return train_dataset


def get_train_dataloader(train_ds, cfg):

    if cfg.distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(
            train_ds, num_replicas=cfg.world_size, rank=cfg.local_rank, shuffle=True, seed=cfg.seed
        )
    else:
        try:
            if cfg.weighted_random_sampler:
                sample_weights = train_ds.sample_weights
                num_samples = train_ds.num_samples_per_epoch
                num_batches = num_samples // cfg.batch_size
                print(f'Num samples per epoch : {num_samples}; num batches : {num_batches}')
                sampler = WeightedRandomSampler(sample_weights, num_samples= num_samples )
            else:
                sampler = None
        except:
            sampler = None

    train_dataloader = DataLoader(
        train_ds,
        sampler=sampler,
        shuffle=(sampler is None),
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        collate_fn=cfg.tr_collate_fn,
        drop_last=cfg.drop_last,
        worker_init_fn=worker_init_fn,
    )
    print(f"train: dataset {len(train_ds)}, dataloader {len(train_dataloader)}")
    return train_dataloader


def get_val_dataset(val_df, cfg, allowed_targets=None):
    print("Loading val dataset")
    val_dataset = cfg.CustomDataset(val_df, cfg, aug=cfg.val_aug, mode="val")
    return val_dataset

# def get_val_index_dataset(train_df, train_dataset):
#     print("Loading val dataset")
#     val_dataset = cfg.CustomDataset(val_df, cfg, aug=cfg.val_aug, mode="val")
#     return val_dataset

def get_val_dataloader(val_ds, cfg):

    if cfg.distributed and cfg.eval_ddp:
        sampler = OrderedDistributedSampler(
            val_ds, num_replicas=cfg.world_size, rank=cfg.local_rank
        )
    else:
        sampler = SequentialSampler(val_ds)

    if cfg.batch_size_val is not None:
        batch_size = cfg.batch_size_val
    else:
        batch_size = cfg.batch_size
    val_dataloader = DataLoader(
        val_ds,
        sampler=sampler,
        batch_size=batch_size,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        collate_fn=cfg.val_collate_fn,
        worker_init_fn=worker_init_fn,
    )
    print(f"valid: dataset {len(val_ds)}, dataloader {len(val_dataloader)}")
    return val_dataloader


def get_test_dataset(test_df, cfg):
    print("Loading test dataset")
    test_dataset = cfg.CustomDataset(test_df, cfg, aug=cfg.val_aug, mode="test")
    return test_dataset


def get_test_dataloader(test_ds, cfg):

    if cfg.distributed and cfg.eval_ddp:
        sampler = OrderedDistributedSampler(
            test_ds, num_replicas=cfg.world_size, rank=cfg.local_rank
        )
    else:
        sampler = SequentialSampler(test_ds)

    if cfg.batch_size_val is not None:
        batch_size = cfg.batch_size_val
    else:
        batch_size = cfg.batch_size
    test_dataloader = DataLoader(
        test_ds,
        sampler=sampler,
        batch_size=batch_size,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        collate_fn=cfg.val_collate_fn,
        worker_init_fn=worker_init_fn,
    )
    print(f"test: dataset {len(test_ds)}, dataloader {len(test_dataloader)}")
    return test_dataloader


def AdamW_LLRD(model, cfg):
    
    opt_parameters = []    
    named_parameters = list(model.named_parameters()) 
        
    # for bias and LayerNorm.weight layers.
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    init_lr = cfg.lr[0]
    head_lr = cfg.lr[1]
    lr = init_lr
    
    # === head ======================================================  
    
    params_0 = [p for n,p in named_parameters if ( not "backbone" in n) 
                and any(nd in n for nd in no_decay)]
    params_1 = [p for n,p in named_parameters if ( not "backbone" in n)
                and not any(nd in n for nd in no_decay)]
    
    head_params = {"params": params_0, "lr": head_lr, "weight_decay": 0.0}    
    opt_parameters.append(head_params)
        
    head_params = {"params": params_1, "lr": head_lr, "weight_decay": cfg.weight_decay}    
    opt_parameters.append(head_params)
                
    # === 12 Hidden layers ==========================================================
    
    for layer in range(11,-1,-1):        
        params_0 = [p for n,p in named_parameters if f"encoder.layer.{layer}." in n 
                    and any(nd in n for nd in no_decay)]
        params_1 = [p for n,p in named_parameters if f"encoder.layer.{layer}." in n 
                    and not any(nd in n for nd in no_decay)]
        
        layer_params = {"params": params_0, "lr": lr, "weight_decay": 0.0}
        opt_parameters.append(layer_params)   
                            
        layer_params = {"params": params_1, "lr": lr, "weight_decay": cfg.weight_decay}
        opt_parameters.append(layer_params)       
        
        lr *= 0.9     
        
    # === Embeddings layer ==========================================================
    
    params_0 = [p for n,p in named_parameters if "embeddings" in n 
                and any(nd in n for nd in no_decay)]
    params_1 = [p for n,p in named_parameters if "embeddings" in n
                and not any(nd in n for nd in no_decay)]
    
    embed_params = {"params": params_0, "lr": lr, "weight_decay": 0.0} 
    opt_parameters.append(embed_params)
        
    embed_params = {"params": params_1, "lr": lr, "weight_decay": cfg.weight_decay} 
    opt_parameters.append(embed_params)        
    
    return AdamW(opt_parameters, lr=init_lr)

def AdamW_grouped_LLRD(model,cfg):
        
    opt_parameters = []       # To be passed to the optimizer (only parameters of the layers you want to update).
    named_parameters = list(model.named_parameters()) 
    
    # According to AAAMLP book by A. Thakur, we generally do not use any decay 
    # for bias and LayerNorm.weight layers.
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    set_2 = ["layer.4", "layer.5", "layer.6", "layer.7"]
    set_3 = ["layer.8", "layer.9", "layer.10", "layer.11"]
    init_lr = cfg.lr[0]
    
    for i, (name, params) in enumerate(named_parameters):  
        
        weight_decay = 0.0 if any(p in name for p in no_decay) else 0.01
 
        if ("embeddings" in name) or ("encoder" in name):            
            # For first set, set lr to 1e-6 (i.e. 0.000001)
            lr = init_lr       
            
            # For set_2, increase lr to 0.00000175
            lr = init_lr * 1.75 if any(p in name for p in set_2) else lr
            
            # For set_3, increase lr to 0.0000035 
            lr = init_lr * 3.5 if any(p in name for p in set_3) else lr
            
            opt_parameters.append({"params": params,
                                   "weight_decay": weight_decay,
                                   "lr": lr})  
            
        # For regressor and pooler, set lr to 0.0000036 (slightly higher than the top layer).                
        if name.startswith("head"):               
            lr = cfg.lr[1]
            
            opt_parameters.append({"params": params,
                                   "weight_decay": weight_decay,
                                   "lr": lr})    
    
    return AdamW(opt_parameters, lr=init_lr)

def get_optimizer(model, cfg):

    # params = [{"params": [param for name, param in model.named_parameters()], "lr": cfg.lr,"weight_decay":cfg.weight_decay}]
    params = model.parameters()

    if cfg.optimizer == "Adam":
        optimizer = optim.Adam(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    elif cfg.optimizer == "Adam_plus":
        paras = list(model.named_parameters())
        no_decay = ["bias", "LayerNorm.bias"]
        params = [{"params": [param for name, param in paras if (not any(nd in name for nd in no_decay))],
                   "lr": cfg.lr,
                   "weight_decay":cfg.weight_decay},
                  {"params": [param for name, param in paras if (any(nd in name for nd in no_decay))],
                   "lr": cfg.lr,
                   "weight_decay":0.},
                 ]        
        optimizer = optim.Adam(params, lr=cfg.lr)        
    elif cfg.optimizer == "AdamW_plus":
        paras = list(model.named_parameters())
        no_decay = ["bias", "LayerNorm.bias"]
        params = [{"params": [param for name, param in paras if (not any(nd in name for nd in no_decay))],
                   "lr": cfg.lr,
                   "weight_decay":cfg.weight_decay},
                  {"params": [param for name, param in paras if (any(nd in name for nd in no_decay))],
                   "lr": cfg.lr,
                   "weight_decay":0.},
                 ]        
        optimizer = AdamW(params, lr=cfg.lr)         
    elif cfg.optimizer == "AdamW_plus2":
        paras = list(model.named_parameters())
        no_decay = ["bias", "LayerNorm.bias"]
        params = [{"params": [param for name, param in paras if (not any(nd in name for nd in no_decay))],
                   "lr": cfg.lr,
                   "weight_decay":cfg.weight_decay},
                  {"params": [param for name, param in paras if (any(nd in name for nd in no_decay))],
                   "lr": cfg.lr,
                   "weight_decay":0.},
                 ]
        optimizer = AdamW_torch(params, lr=cfg.lr, eps=cfg.optim_eps, betas=cfg.optim_betas, weight_decay=cfg.weight_decay)
    elif cfg.optimizer == "AdamW_plus3":
        paras = list(model.named_parameters())
        no_decay = ["bias", "LayerNorm.bias"]
        params = [{"params": [param for name, param in paras if ("backbone" in name) & (not any(nd in name for nd in no_decay))],"lr": cfg.lr[0],"weight_decay":cfg.weight_decay},
                  {"params": [param for name, param in paras if ("backbone" in name) & (any(nd in name for nd in no_decay))],"lr": cfg.lr[0],"weight_decay":0.},
                  {"params": [param for name, param in paras if (not "backbone" in name) & (not any(nd in name for nd in no_decay))],"lr": cfg.lr[1],"weight_decay":cfg.weight_decay},
                  {"params": [param for name, param in paras if (not "backbone" in name) & (any(nd in name for nd in no_decay))],"lr": cfg.lr[1],"weight_decay":0.},]
        optimizer = AdamW_torch(params, lr=cfg.lr[0], eps=cfg.optim_eps, betas=cfg.optim_betas, weight_decay=cfg.weight_decay)
    elif cfg.optimizer == "AdamW_mixed":
        params = [
            {
                "params": [
                    param for name, param in model.named_parameters() if "backbone" in name
                ],
                "lr": cfg.lr[0],
            },
            {
                "params": [
                    param for name, param in model.named_parameters() if not "backbone" in name
                ],
                "lr": cfg.lr[1],
            },
        ]
        optimizer = AdamW(params, lr=cfg.lr[1], weight_decay=cfg.weight_decay)
    elif cfg.optimizer == 'AdamW_LLRD':
        optimizer = AdamW_LLRD(model,cfg)
    elif cfg.optimizer == 'AdamW_grouped_LLRD':
        optimizer = AdamW_grouped_LLRD(model,cfg)    
    
    
    elif cfg.optimizer == "Adam_mixed_plus":
        paras = list(model.named_parameters())
        no_decay = ["bias", "LayerNorm.bias"]
        params = [{"params": [param for name, param in paras if ("backbone" in name) & (not any(nd in name for nd in no_decay))],"lr": cfg.lr[0],"weight_decay":cfg.weight_decay},
                  {"params": [param for name, param in paras if ("backbone" in name) & (any(nd in name for nd in no_decay))],"lr": cfg.lr[0],"weight_decay":0.},
                  {"params": [param for name, param in paras if (not "backbone" in name) & (not any(nd in name for nd in no_decay))],"lr": cfg.lr[1],"weight_decay":cfg.weight_decay},
                  {"params": [param for name, param in paras if (not "backbone" in name) & (any(nd in name for nd in no_decay))],"lr": cfg.lr[1],"weight_decay":0.},
                  
                  
                  
                 ]
        optimizer = optim.Adam(params, lr=cfg.lr[1])        
        
      
        
    elif cfg.optimizer == "AdamW":
        optimizer = AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    elif cfg.optimizer == "Adafactor":
        optimizer = Adafactor(params, lr=cfg.lr, weight_decay=cfg.weight_decay, scale_parameter=False, relative_step=False)
    elif cfg.optimizer == "Adafactor_mixed":
        params = [
            {
                "params": [
                    param for name, param in model.named_parameters() if "backbone" in name
                ],
                "lr": cfg.lr[0],
            },
            {
                "params": [
                    param for name, param in model.named_parameters() if not "backbone" in name
                ],
                "lr": cfg.lr[1],
            },
        ]
        optimizer = Adafactor(params, lr=cfg.lr[1], weight_decay=cfg.weight_decay, scale_parameter=False, relative_step=False)
    elif cfg.optimizer == "Adabelief":
        from adabelief_pytorch import AdaBelief
        optimizer = AdaBelief(params, lr=cfg.lr, eps=1e-16, betas=(0.9, 0.999), weight_decouple=True, rectify=False)
    elif cfg.optimizer == "Adabelief_mixed":
        from adabelief_pytorch import AdaBelief
        params = [
            {
                "params": [
                    param for name, param in model.named_parameters() if "backbone" in name
                ],
                "lr": cfg.lr[0],
            },
            {
                "params": [
                    param for name, param in model.named_parameters() if not "backbone" in name
                ],
                "lr": cfg.lr[1],
            },
        ]
        optimizer = AdaBelief(params, lr=cfg.lr[1], eps=1e-16, betas=(0.9, 0.999), weight_decouple=True, rectify=False)
    elif cfg.optimizer == "RAdam":
        optimizer = optim.RAdam(params, lr=cfg.lr, betas=(0.95, 0.999), #changed to .95
                                            eps=1e-08, weight_decay=cfg.weight_decay)
    elif cfg.optimizer == "CellPoseRAdam":
        import torch_optimizer as optim2
        optimizer = optim2.RAdam(params, lr=cfg.lr, betas=(0.95, 0.999), #changed to .95
                                            eps=1e-08, weight_decay=cfg.weight_decay)
        optimizer.current_lr = cfg.lr
        
        for param_group in optimizer.param_groups:
                param_group['lr'] = cfg.lr
        
    elif cfg.optimizer == "SGD":
        optimizer = optim.SGD(
            params,
            lr=cfg.lr,
            momentum=cfg.sgd_momentum,
            nesterov=cfg.sgd_nesterov,
            weight_decay=cfg.weight_decay,
        )
    elif cfg.optimizer == "fused_SGD":
        import apex

        optimizer = apex.optimizers.FusedSGD(
            params, lr=cfg.lr, momentum=0.9, nesterov=True, weight_decay=cfg.weight_decay
        )
    elif cfg.optimizer == "fused_Adam":
        import apex

        optimizer = apex.optimizers.FusedAdam(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    elif cfg.optimizer == "SGD_AGC":
        from nfnets import SGD_AGC

        optimizer = SGD_AGC(
            named_params=model.named_parameters(),  # Pass named parameters
            lr=cfg.lr,
            momentum=0.9,
            clipping=0.1,  # New clipping parameter
            weight_decay=cfg.weight_decay,
            nesterov=True,
        )

    return optimizer


def get_scheduler(cfg, optimizer, total_steps):

    if cfg.schedule == "steplr":
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=cfg.epochs_step * (total_steps // cfg.batch_size) // cfg.world_size,
            gamma=0.5,
        )
    elif cfg.schedule == "cosine":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=cfg.warmup * (total_steps // cfg.batch_size) // cfg.world_size,
            num_training_steps=cfg.epochs * (total_steps // cfg.batch_size) // cfg.world_size,
        )
    elif cfg.schedule == "linear":
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=cfg.epochs * (total_steps // cfg.batch_size) // cfg.world_size,
        )

        print("num_steps", (total_steps // cfg.batch_size) // cfg.world_size)

    else:
        scheduler = None

    return scheduler


def setup_neptune(cfg):
    neptune_run = neptune.init(
        project=cfg.neptune_project,
        tags=cfg.tags,
        mode=cfg.neptune_connection_mode,
        capture_stdout=False,
        capture_stderr=False,
        source_files=[f'models/{cfg.model}.py',f'data/{cfg.dataset}.py']
    )


    neptune_run["cfg"] = cfg.__dict__

    return neptune_run


def get_data(cfg):

    # setup dataset

    print(f"reading {cfg.train_df}")
    df = pd.read_csv(cfg.train_df)

    if cfg.test:
        test_df = pd.read_csv(cfg.test_df)
    else:
        test_df = None
    
    if cfg.fold == -1:
        val_df = df[df["fold"] == 0]
    else:
        val_df = df[df["fold"] == cfg.fold]
        
    train_df = df[df["fold"] != cfg.fold]
        
    return train_df, val_df, test_df


def save_first_batch(feature_dict, cfg):
    print("Saving first batch of images")
    images = feature_dict["input"].detach().cpu().numpy()
    targets = feature_dict["target"].detach().cpu().numpy()
    boxes_batch = feature_dict["boxes"]

    for i, (image, target, boxes) in enumerate(zip(images, targets, boxes_batch)):
        fig, ax = plt.subplots(figsize=(13, 13))
        print(f"image_{i}: min {image[0].min()}, max {image[0].max()}")
        ax.imshow(image[0])  # just one channel / greyscale
        boxes = boxes.detach().cpu().numpy()
        for ii in range(len(boxes)):
            w = boxes[ii, 2] - boxes[ii, 0]
            h = boxes[ii, 3] - boxes[ii, 1]
            rect = patches.Rectangle((boxes[ii, 1], boxes[ii, 0]), h, w, linewidth=2, edgecolor='g', facecolor='none')
            ax.add_patch(rect)
        fig.suptitle(f"Target: {target}")
        fig.savefig(f"{cfg.output_dir}/fold{cfg.fold}/batch1_image{i}_seed{cfg.seed}.png")
        plt.close()


def save_first_batch_preds(feature_dict, output_dict, cfg):
    print("Saving preds of first batch of images")
    images = feature_dict["input"].detach().cpu().numpy()
    targets = feature_dict["target"].detach().cpu().numpy()
    class_preds = output_dict["class_logits"].softmax(1).detach().cpu().numpy()
    boxes_batch = feature_dict["boxes"]
    pred_boxes_batch = output_dict["detections"].detach().cpu().numpy()
    # pred_boxes_batch = pred_boxes_batch[:, :, 4]
    # print(pred_boxes_batch.shape)

    for i, (image, boxes, boxes_pred) in enumerate(zip(images, boxes_batch, pred_boxes_batch)):
        fig, ax = plt.subplots(figsize=(13, 13))
        ax.imshow(image[0])  # just one channel / greyscale
        boxes = boxes.detach().cpu().numpy()
        for ii in range(len(boxes)):
            w = boxes[ii, 2] - boxes[ii, 0]
            h = boxes[ii, 3] - boxes[ii, 1]
            rect = patches.Rectangle((boxes[ii, 1], boxes[ii, 0]), h, w, linewidth=2, edgecolor='g', facecolor='none')
            ax.add_patch(rect)
        for ii in range(len(boxes_pred)):
            if boxes_pred[ii, 4] > 0.3:
                w = boxes_pred[ii, 2] - boxes_pred[ii, 0]
                h = boxes_pred[ii, 3] - boxes_pred[ii, 1]
                rect = patches.Rectangle((boxes_pred[ii, 0], boxes_pred[ii, 1]), w, h, linewidth=2, edgecolor='r', facecolor='none')
                ax.add_patch(rect)
                ax.text(boxes_pred[ii, 0] + w, boxes_pred[ii, 1] + h, f"{boxes_pred[ii, 4]}")
        fig.suptitle(f"Target: {targets[i]}, Preds: {class_preds[i]}")
        fig.savefig(f"{cfg.output_dir}/fold{cfg.fold}/preds_batch1_image{i}_seed{cfg.seed}.png")
        plt.close()


def upload_s3(cfg):
    BUCKET_NAME = cfg.s3_bucket_name
    ACCESS_KEY = cfg.s3_access_key
    SECRET_KEY = cfg.s3_secret_key
    session = Session(aws_access_key_id=ACCESS_KEY,
                aws_secret_access_key=SECRET_KEY)
    s3 = session.resource('s3')

    s3.Bucket(BUCKET_NAME).upload_file(f"{cfg.output_dir}/fold{cfg.fold}/val_data_seed{cfg.seed}.pth", f"output/{cfg.name}/fold{cfg.fold}/val_data_seed{cfg.seed}.pth")
    s3.Bucket(BUCKET_NAME).upload_file(f"{cfg.output_dir}/fold{cfg.fold}/test_data_seed{cfg.seed}.pth", f"output/{cfg.name}/fold{cfg.fold}/test_data_seed{cfg.seed}.pth")
    s3.Bucket(BUCKET_NAME).upload_file(f"{cfg.output_dir}/fold{cfg.fold}/submission_seed{cfg.seed}.csv", f"output/{cfg.name}/fold{cfg.fold}/submission_seed{cfg.seed}.csv")


def flatten(t):
    return [item for sublist in t for item in sublist]

def set_pandas_display():
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows',10000)
    pd.set_option('display.width', 10000)
    pd.set_option('display.float_format', lambda x: '%.3f' % x)

def dumpobj(file, obj):
    with open(file, 'wb') as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
def loadobj(file):
    with open(file, 'rb') as handle:
        return pickle.load(handle)

def get_level(level_str):
    ''' get level'''
    l_names = {logging.getLevelName(lvl).lower(): lvl for lvl in [10, 20, 30, 40, 50]} # noqa
    return l_names.get(level_str.lower(), logging.INFO)

def get_logger(name, level_str):
    ''' get logger'''
    logger = logging.getLogger(name)
    logger.setLevel(get_level(level_str))
    handler = logging.StreamHandler()
    handler.setLevel(level_str)
    handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')) # pylint: disable=C0301 # noqa
    logger.addHandler(handler)

    return logger


class AWP:
    """
    https://www.kaggle.com/code/wht1996/feedback-nn-train/notebook
    adversarial learning
    """

    def __init__(
        self,
        model,
        optimizer,
        cfg,
        adv_param="weight",
        adv_lr=1,
        adv_eps=0.2,
        start_epoch=0,
        adv_step=1,
        scaler=None
    ):
        self.model = model
        self.optimizer = optimizer
        self.cfg = cfg
        self.adv_param = adv_param
        self.adv_lr = adv_lr
        self.adv_eps = adv_eps
        self.start_epoch = start_epoch
        self.adv_step = adv_step
        self.backup = {}
        self.backup_eps = {}
        self.scaler = scaler

    def attack_backward(self, batch, epoch):
        if (self.adv_lr == 0) or (epoch < self.start_epoch):
            return None

        self._save()
        for i in range(self.adv_step):
            self._attack_step()
            with torch.cuda.amp.autocast():
                output_dict = self.model(batch)
                adv_loss = output_dict["loss"]
                if self.cfg.grad_accumulation != 0:
                    adv_loss /= self.cfg.grad_accumulation
            self.optimizer.zero_grad()
            self.scaler.scale(adv_loss).backward()

        self._restore()

    def _attack_step(self):
        e = 1e-6
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and self.adv_param in name:
                norm1 = torch.norm(param.grad)
                norm2 = torch.norm(param.data.detach())
                if norm1 != 0 and not torch.isnan(norm1):
                    r_at = self.adv_lr * param.grad / (norm1 + e) * (norm2 + e)
                    param.data.add_(r_at)
                    param.data = torch.min(
                        torch.max(param.data, self.backup_eps[name][0]), self.backup_eps[name][1]
                    )
                # param.data.clamp_(*self.backup_eps[name])

    def _save(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and self.adv_param in name:
                if name not in self.backup:
                    self.backup[name] = param.data.clone()
                    grad_eps = self.adv_eps * param.abs().detach()
                    self.backup_eps[name] = (
                        self.backup[name] - grad_eps,
                        self.backup[name] + grad_eps,
                    )

    def _restore(self,):
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]
        self.backup = {}
        self.backup_eps = {}


def set_pandas_display():
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows',10000)
    pd.set_option('display.width', 10000)
    pd.set_option('display.float_format', lambda x: '%.3f' % x)

def dumpobj(file, obj):
    with open(file, 'wb') as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
def loadobj(file):
    with open(file, 'rb') as handle:
        return pickle.load(handle)

def get_level(level_str):
    ''' get level'''
    l_names = {logging.getLevelName(lvl).lower(): lvl for lvl in [10, 20, 30, 40, 50]} # noqa
    return l_names.get(level_str.lower(), logging.INFO)

def get_logger(name, level_str):
    ''' get logger'''
    logger = logging.getLogger(name)
    logger.setLevel(get_level(level_str))
    handler = logging.StreamHandler()
    handler.setLevel(level_str)
    handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')) # pylint: disable=C0301 # noqa
    logger.addHandler(handler)

    return logger

def prepare_mask_data(string):
    # fetching all the values from the string
    all_values = map(int, string.split(" "))
    # preparing the usable arrays
    starterIndex, pixelCount = [], []
    for index, value in enumerate(all_values):
        if index % 2:
            # storing even indexed values in pixelCount
            pixelCount.append(value)
        else:
            # storing odd indexed values in starterIndex
            starterIndex.append(value)
    return starterIndex, pixelCount
    
def fetch_pos_pixel_indexes(indexes, counts):
    final_arr = []
    for index, counts in zip(indexes, counts):
        # adding all the values from starterIndex to range of positive pixel counts
        final_arr += [index + i for i in range(counts)]
    return final_arr

def prepare_mask(string, height, width):
    # preparing the respective arrays
    indexes, counts = prepare_mask_data(string)
    # preparing all the pixel indexes those have mask values
    pos_pixel_indexes = fetch_pos_pixel_indexes(indexes, counts)
    # forming the flattened array
    mask_array = np.zeros(height * width)
    # updating values in the array
    mask_array[pos_pixel_indexes] = 1
    # reshaping the masks
    return mask_array.reshape(height, width)

def fetch_filename(ids, root_dir=''):
    imgls = glob.glob(f'{root_dir}/*/*/*/*.png')
    imgnmdf = pd.Series(imgls).str.split('/', expand = True).iloc[:,-4:]
    imgnmdf =  pd.concat((imgnmdf.iloc[:,1].str.split('_', expand = True), 
                          imgnmdf.iloc[:,-1].str.split('_', expand = True).iloc[:, [1]],
                          imgnmdf.iloc[:,[-1]]), axis = 1)
    imgnmdf.columns = 'case day idd filename'.split()
    imgnmdf['idd'] = imgnmdf['idd'].str.zfill(4)
    imgnmdf['id'] = imgnmdf.apply(lambda x: f'{x.case}_{x.day}_slice_{x.idd}', 1)
    imgnmdf['filedir'] = list( map(os.path.dirname, imgls))
    imgnmdf = imgnmdf.set_index('id').loc[ids]
    filenames = imgnmdf.filename.tolist()
    filedirs = imgnmdf.filedir.tolist()
    return filenames, filedirs

class PreprocessDataset(Dataset):

    #def __init__(self, df, mode, tokenizer, max_pos, smoothbreaks = False):
    def __init__(self, mode="train", data_dir = 'datamount', windows = [(400, 1800)]):
        self.dicnmls = glob.glob(f'{data_dir}/{mode}_images/**/*.dcm', recursive = True)
        self.dicnmls = sorted(self.dicnmls)
        self.windows = windows # [ (500, 1800), (400, 650), (80, 300)]
        self.datadir = f'{data_dir}/{mode}_images/'
        self.jsondir = f'{data_dir}/{mode}_images_json/'
        self.jpgdir = f'{data_dir}/{mode}_images_jpg/'
        idx = 0
        
    def __len__(self):
        return len(self.dicnmls)

    def __getitem__(self, idx):
        
        dcmfile = self.dicnmls[idx]
        X, metaj = self.read_dicom(dcmfile)
        img = self.windowsfn(X, self.windows)
        
        outjfile = dcmfile.replace(self.datadir, self.jsondir) + '.json'
        outifile = dcmfile.replace(self.datadir, self.jpgdir) + '.jpg'
        
        Path(str(Path(outjfile).parent)).mkdir(parents=True, exist_ok=True)
        Path(str(Path(outifile).parent)).mkdir(parents=True, exist_ok=True)
        
        json_string = json.dumps(metaj)
        with open(outjfile, 'w') as outfile:
            json.dump(json_string, outfile)
            
        cv2.imwrite(outifile, img)
        
        return np.array([1])
    
    def read_dicom(self, dcmfile):
        if 'placeholder' in dcmfile:
            return np.zeros((512,512))-1000
        D = pydicom.dcmread(dcmfile)
        metaj = self.dicom_dataset_to_dict(D)
        #zpos = D.ImagePositionPatient[-1]
        # D.PhotometricInterpretation = 'YBR_FULL'
        m = float(D.RescaleSlope)
        b = float(D.RescaleIntercept)
        D = D.pixel_array.astype('float')*m
        D = D.astype('float')+b
        return D, metaj

    def list_dcm_dir(self, dcmfile):
        dcmfiles = sorted(glob.glob(f'{cfg.data_folder}/{idd}/*'))
        dcmnums = [int(i.split('/')[-1].split('.')[0]) for i in dcmfiles]
        dcmfiles = [x for _, x in sorted(zip(dcmnums, dcmfiles))]
        return dcmfiles
    
    def windowfn(self, img, WL=400, WW=1800):
        upper, lower = WL+WW//2, WL-WW//2
        X = np.clip(img.copy(), lower, upper)
        X = X - np.min(X)
        X = X / np.max(X)
        X = (X*255.0).astype('uint8')
        return X
    
    def windowsfn(self, X, windows):
        return cv2.merge([self.windowfn(X, *w) for t, w in enumerate(windows)])
    
    def dicom_dataset_to_dict(self, dicom_header):
        dicom_dict = {}
        repr(dicom_header)
        for dicom_value in dicom_header.values():
            if dicom_value.tag == (0x7fe0, 0x0010):
                # discard pixel data
                continue
            if type(dicom_value.value) == pydicom.dataset.Dataset:
                dicom_dict[dicom_value.tag] = dicom_dataset_to_dict(dicom_value.value)
            else:
                v = self._convert_value(dicom_value.value)
                dicom_dict[dicom_value.tag] = v
        return dicom_dict
    
    def _sanitise_unicode(self, s):
        return s.replace(u"\u0000", "").strip()
    
    def _convert_value(self, v):
        t = type(v)
        if t in (list, int, float):
            cv = v
        elif t == str:
            cv = self._sanitise_unicode(v)
        elif t == bytes:
            s = v.decode('ascii', 'replace')
            cv = _sanitise_unicode(s)
        elif t == pydicom.valuerep.DSfloat:
            cv = float(v)
        elif t == pydicom.valuerep.IS:
            cv = int(v)
        elif t == pydicom.valuerep.PersonName:
            cv = str(v)
        else:
            cv = repr(v)
        return cv

def read_dicom(dcmfile):
    if 'placeholder' in dcmfile:
        return np.zeros((512,512))-1000
    D = pydicom.dcmread(dcmfile)
    # D.PhotometricInterpretation = 'YBR_FULL'
    m = float(D.RescaleSlope)
    b = float(D.RescaleIntercept)
    D = D.pixel_array.astype('float')*m
    D = D.astype('float')+b
    return D

def windowfn(img, WL=400, WW=1800):
    upper, lower = WL+WW//2, WL-WW//2
    X = np.clip(img.copy(), lower, upper)
    X = X - np.min(X)
    X = X / np.max(X)
    X = (X*255.0).astype('uint8')
    return X

def windowsfn(X, windows=[ (500, 1800), (400, 650), (80, 300)]):
    return cv2.merge([windowfn(X, *w) for t, w in enumerate(windows)])

