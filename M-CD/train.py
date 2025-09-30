import os.path as osp
import os
import sys
import time
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torch.nn.parallel import DistributedDataParallel

from dataloader.dataloader import floga_data_loader
from models.builder import EncoderDecoder as segmodel
from dataloader.flogaDataset import FLOGADataset
from utils.init_func import group_weight
from utils.lr_policy import WarmUpPolyLR
from utils.pyt_utils import load_model, save_checkpoint
from engine.engine import Engine
from eval import FlogaEvaluator
import shutil
import logging

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', type=str, default='floga')
parser.add_argument('--gpu_id', type=int, choices=[0,1,2,3], default=3, required=False)
args = parser.parse_args()

torch.cuda.set_device(args.gpu_id)
dataset_name = args.dataset_name
print("DATASET NAME::  ", dataset_name)
if dataset_name == 'floga':
    from configs.config_floga import config
else:
    raise NotImplementedError('Not a valid dataset name')

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
# Create a file handler
file_handler = logging.FileHandler(config.log_file)
file_handler.setLevel(logging.INFO)
# Create a logging format
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
# Add the file handler to the logger
logger.addHandler(file_handler)

cudnn.benchmark = True
seed = config.seed
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

# data loader
train_loader, train_sampler = floga_data_loader(config, 'train')
val_loader, _ = floga_data_loader(config, 'val')

# config network and criterion
criterion = nn.CrossEntropyLoss(reduction='mean', ignore_index=2)
BatchNorm2d = nn.BatchNorm2d

config.bands_idx = list(config['datasets']['selected_bands'][config['datasets']['data_source']].values())

model=segmodel(cfg=config, criterion=criterion, norm_layer=BatchNorm2d)

# group weight and config optimizer
base_lr = config.lr
params_list = []
params_list = group_weight(params_list, model, BatchNorm2d, base_lr)

if config.optimizer == 'AdamW':
    optimizer = torch.optim.AdamW(params_list, lr=base_lr, betas=(0.9, 0.999), weight_decay=config.weight_decay)
elif config.optimizer == 'SGDM':
    optimizer = torch.optim.SGD(params_list, lr=base_lr, momentum=config.momentum, weight_decay=config.weight_decay)
else:
    raise NotImplementedError

# config lr policy
total_iteration = config.nepochs * config.niters_per_epoch
lr_policy = WarmUpPolyLR(base_lr, config.lr_power, total_iteration, config.niters_per_epoch * config.warm_up_epoch)

device = f"cuda:{args.gpu_id}"
model.to(device)

optimizer.zero_grad()
model.train()

best_f1 = -1 # Track the best F1 for model saving
best_epoch = -1  # Track the epoch with the best F1 for model saving
best_checkpoint_path = None

logger.info("Training started")
for epoch in range(1, config.nepochs+1):
    bar_format = '{desc}[{elapsed}<{remaining},{rate_fmt}]'
    pbar = tqdm(range(config.niters_per_epoch), file=sys.stdout,
                bar_format=bar_format)
    dataloader = iter(train_loader)

    sum_loss = 0

    for idx in pbar:
        try:
            minibatch = next(dataloader)
        except StopIteration:
            break
        if (dataset_name == 'floga'):
            As = minibatch['S2_before_image'][:, config.bands_idx, :, :]
            Bs = minibatch['S2_after_image'][:, config.bands_idx, :, :]
            gts = minibatch['label']
        else:
            As = minibatch['A']
            Bs = minibatch['B']
            gts = minibatch['gt']

        As = As.cuda(non_blocking=True)
        Bs = Bs.cuda(non_blocking=True)
        gts = gts.cuda(non_blocking=True)

        loss = model(As, Bs, gts)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #* Change learning rate
        current_idx = (epoch-1) * config.niters_per_epoch + idx 
        lr = lr_policy.get_lr(current_idx)
        for i in range(len(optimizer.param_groups)):
            optimizer.param_groups[i]['lr'] = lr

        sum_loss += loss
        print_str = 'Epoch {}/{}'.format(epoch, config.nepochs) \
                + ' Iter {}/{}:'.format(idx + 1, config.niters_per_epoch) \
                + ' lr=%.4e' % lr \
                + ' loss=%.4f total_loss=%.4f' % (loss, (sum_loss / (idx + 1)))
        pbar.set_description(print_str, refresh=False)
        del loss
        
    logger.info(f'Epoch: {epoch} {"-"*10} train_loss: {sum_loss / len(pbar)}')

    if (epoch >= config.checkpoint_start_epoch) and (epoch % config.checkpoint_step == 0) or (epoch == config.nepochs):
        save_checkpoint(model, config.checkpoint_dir / f"epoch-{epoch}.pth", epoch, idx, optimizer)
    save_checkpoint(model, config.checkpoint_dir / "epoch-curr.pth", epoch, idx, optimizer)
    torch.cuda.empty_cache()
    model.eval() 
    with torch.no_grad():
        evaluator = FlogaEvaluator(device=device, loader=val_loader,  cfg=config, model=model)
        round_metrics = evaluator()
        logger.info(f"{'*'*10}Epoch_{epoch}{'*'*10}")
        logger.info(f'Validation acc = {round_metrics["accuracy"]}%')
        logger.info(f'Precision = {round_metrics["precision"]}')
        logger.info(f'Recall = {round_metrics["recall"]}')
        logger.info(f'F1 Score = {round_metrics["f1"]}')
        logger.info(f'MeanIoU = {round_metrics["meanIoU"]}')
        logger.info(f'iou = {round_metrics["iou"]}')
        
        round_f1 = round_metrics['f1'][1]
            # Determine if the model performance improved
        if round_f1 > best_f1:
            # If the model improves, remove the saved checkpoint for this epoch
            best_epoch = epoch
            best_f1 = round_f1
            save_checkpoint(model, config.checkpoint_dir / "epoch-best.pth", epoch, idx, optimizer)
            with open(config.checkpoint_dir / 'best_weights.txt', 'w') as f:
                f.write(f"Best epoch: {best_epoch}\nBest F1 Score: {best_f1}\n")
    model.train()

print('-'*10, "Start Testing", '-'*10)
torch.cuda.empty_cache()
model = load_model(model, config.checkpoint_dir / "epoch-best.pth")
model.eval()
with torch.no_grad():
        test_loader, _ = floga_data_loader(config, mode='test')
        evaluator = FlogaEvaluator(device=device, loader=test_loader,  cfg=config, model=model)
        test_metrics = evaluator()
        logger.info(f"{'*'*10}Test Metrics{'*'*10}")
        logger.info(f"Validation acc = {test_metrics['accuracy']}%")
        logger.info(f"Precision = {test_metrics['precision']}")
        logger.info(f"Recall = {test_metrics['recall']}")
        logger.info(f"F1 Score = {test_metrics['f1']}")
        logger.info(f"MeanIoU = {test_metrics['meanIoU']}")
        logger.info(f"iou = {test_metrics['iou']}")
    