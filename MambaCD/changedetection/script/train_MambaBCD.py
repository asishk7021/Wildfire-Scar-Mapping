import argparse
import os
from datetime import datetime
from pathlib import Path
import json
import re
import numpy as np

from MambaCD.changedetection.configs.config import get_config

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from MambaCD.changedetection.datasets.make_data_loader import ChangeDetectionDataset, FLOGADataset, make_data_loader
from MambaCD.changedetection.utils_func.metrics import Evaluator, initialize_metrics
from MambaCD.changedetection.utils_func.preprocessing import load_dataset_configs
from MambaCD.changedetection.models.MambaBCD import STMambaBCD
from torch.nn.parallel import DataParallel
import MambaCD.changedetection.utils_func.lovasz_loss as L

class Trainer(object):
    def __init__(self, args):
        torch.cuda.set_device(args.gpu_id)
        self.args = args
        config = get_config(args)
        self.device = args.device
        self.model_path = args.model_path
        configs = load_dataset_configs(args.configs_path)
        self.sen2_bands_idx = list(configs['datasets']['selected_bands'][configs['datasets']['data_source']].values()) #HACK Change here
        self.train_data_loader = make_data_loader(args) #* Returns an instance of `torch` DataLoader
        
        #* Load validation dataset
        if (args.dataset == 'sen2' or args.dataset == 'mod'):
            dataset = FLOGADataset('val', configs, clc=True)
            self.val_data_loader = DataLoader(dataset, num_workers=configs['datasets']['num_workers'], batch_size=4,
                                 shuffle=False, drop_last=False, pin_memory=True) #? Should the batch size be 1
        else:
            dataset = ChangeDetectionDataset(self.args.test_dataset_path, self.args.test_data_name_list, 256, None, 'test')
            self.val_data_loader = DataLoader(dataset, batch_size=1, num_workers=4, drop_last=False)
        
        #* Load test dataset
        if (args.dataset == 'sen2' or args.dataset == 'mod'):
            dataset = FLOGADataset('test', configs, clc=True)
            self.test_data_loader = DataLoader(dataset, num_workers=configs['datasets']['num_workers'], batch_size=4,
                                 shuffle=False, drop_last=False, pin_memory=True)
        else:
            dataset = ChangeDetectionDataset(self.args.test_dataset_path, self.args.test_data_name_list, 256, None, 'test')
            self.test_data_loader = DataLoader(dataset, batch_size=1, num_workers=4, drop_last=False)
        
        self.evaluator = Evaluator(num_class=2)
        self.cm, self.iou = initialize_metrics(self.device)

        self.deep_model = STMambaBCD(
            pretrained=args.pretrained_weight_path,
            patch_size=config.MODEL.VSSM.PATCH_SIZE, #* Used as kernel size for Conv2d
            in_chans=config.MODEL.VSSM.IN_CHANS, 
            num_classes=config.MODEL.NUM_CLASSES, 
            depths=config.MODEL.VSSM.DEPTHS, 
            dims=config.MODEL.VSSM.EMBED_DIM, 
            # ===================
            ssm_d_state=config.MODEL.VSSM.SSM_D_STATE,
            ssm_ratio=config.MODEL.VSSM.SSM_RATIO,
            ssm_rank_ratio=config.MODEL.VSSM.SSM_RANK_RATIO,
            ssm_dt_rank=("auto" if config.MODEL.VSSM.SSM_DT_RANK == "auto" else int(config.MODEL.VSSM.SSM_DT_RANK)),
            ssm_act_layer=config.MODEL.VSSM.SSM_ACT_LAYER, #* default = 'silu'
            ssm_conv=config.MODEL.VSSM.SSM_CONV, #* default = 3
            ssm_conv_bias=config.MODEL.VSSM.SSM_CONV_BIAS, #* default = True
            ssm_drop_rate=config.MODEL.VSSM.SSM_DROP_RATE, #* default = 0
            ssm_init=config.MODEL.VSSM.SSM_INIT, #* default = 'v0'
            forward_type=config.MODEL.VSSM.SSM_FORWARDTYPE, #* default = 'v2'
            # ===================
            mlp_ratio=config.MODEL.VSSM.MLP_RATIO, #* default = 4
            mlp_act_layer=config.MODEL.VSSM.MLP_ACT_LAYER, #* default = 'gelu'
            mlp_drop_rate=config.MODEL.VSSM.MLP_DROP_RATE, #* default = 0
            # ===================
            drop_path_rate=config.MODEL.DROP_PATH_RATE, #* default = 0
            patch_norm=config.MODEL.VSSM.PATCH_NORM, #* default = True
            norm_layer=config.MODEL.VSSM.NORM_LAYER, #* default = 'ln'
            downsample_version=config.MODEL.VSSM.DOWNSAMPLE, #* default = 'v2'
            patchembed_version=config.MODEL.VSSM.PATCHEMBED, #* default = 'v2'
            gmlp=config.MODEL.VSSM.GMLP, #* default = False
            use_checkpoint=config.TRAIN.USE_CHECKPOINT,
            ) 
        # self.deep_model = DataParallel(self.deep_model, device_ids=[0,1])
        # self.deep_model.module.to('cuda:0')
        self.deep_model.to(self.device)
        self.model_save_path = os.path.join(args.model_param_path, args.dataset,
                                            args.model_type + '_' + datetime.now().strftime("%Y%m%d%H%M%S")) if self.model_path is None else self.model_path.parent
        self.lr = args.learning_rate
        self.epochs = args.epochs

        os.makedirs(self.model_save_path, exist_ok=True)

        if self.model_path is not None:
            if not os.path.isfile(self.model_path):
                raise RuntimeError("=> no checkpoint found at '{}'".format(args.resume))
            checkpoint = torch.load(self.model_path, map_location=self.device)
            model_dict = {}
            state_dict = self.deep_model.state_dict()
            for k, v in checkpoint.items():
                if k in state_dict:
                    model_dict[k] = v
            state_dict.update(model_dict)
            self.deep_model.load_state_dict(state_dict)

        self.optim = optim.AdamW(self.deep_model.parameters(),
                                 lr=args.learning_rate,
                                 weight_decay=args.weight_decay)
        with open(f"{self.model_save_path}/training_logs.txt", 'a') as f:
            f.write(f'{self.optim.defaults}\n')
    def __call__(self):
        if (self.model_path is not None):
            self.start_epoch = int(re.search(r'(\d+)_model\.pth$', str(self.model_path)).group(1))
            best_f1 = float(re.search(r'Epoch is \d+, f1 is ([0-9.]+)', open(f"{self.model_save_path}/best_model.txt").read()).group(1))
        else:
            self.start_epoch = 0
            best_f1 = 0.0
        best_round = {}
        
        if (self.device.startswith('cuda')):
            torch.cuda.empty_cache()
        
        elem_num = len(self.train_data_loader)
        for e in tqdm(range(self.start_epoch, self.epochs)):
            epoch_train_loss = 0
            for data in tqdm(self.train_data_loader): #TODO re initialize after every epoch for shuffling purpose
                if (self.args.dataset == 'sen2'):
                    pre_change_imgs = data['S2_before_image'][:, self.sen2_bands_idx, :, :]
                    post_change_imgs = data['S2_after_image'][:, self.sen2_bands_idx, :, :]
                    labels = data['label']
                else:
                    pre_change_imgs, post_change_imgs, labels, _ = data

                pre_change_imgs = pre_change_imgs.to(torch.float32).to(self.device)
                post_change_imgs = post_change_imgs.to(torch.float32).to(self.device)
                labels = labels.to(torch.long).to(self.device)
                output_1 = self.deep_model(pre_change_imgs, post_change_imgs)

                self.optim.zero_grad()
                ce_loss_1 = F.cross_entropy(output_1, labels, ignore_index=2)
                lovasz_loss = L.lovasz_softmax(F.softmax(output_1, dim=1), labels, ignore=2)
                # print(f"Cross-entropy loss: {ce_loss_1}     lovasz_loss: {lovasz_loss}")
                final_loss = ce_loss_1 + 0.75 * lovasz_loss
                epoch_train_loss += final_loss.item()

                final_loss.backward()
                self.optim.step()
            
            epoch_train_loss = epoch_train_loss / elem_num
            with open(f"{self.model_save_path}/training_logs.txt", 'a') as f:
                current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                f.write(f'[{current_time}] Epoch is {e}, overall train loss is {epoch_train_loss}\n')
            print(f'Epoch is {e}, overall train loss is {epoch_train_loss}')
            
            print('---------starting validation-----------')
            self.deep_model.eval()
            round_metrics = self.metrics_validation()
            with open(f"{self.model_save_path}/metrics_{e}.json", 'w') as f:
                json.dump(round_metrics, f)
            round_f1 = round_metrics['f1'][1]
            if (round_f1 > best_f1):
                torch.save(self.deep_model.state_dict(),
                            os.path.join(self.model_save_path, f'best_model.pth'))
                best_f1 = round_f1
                with open(f"{self.model_save_path}/best_model.txt", 'w') as f:
                    f.write(f'Epoch is {e}, f1 is {best_f1}\n')
                best_round = round_metrics.copy()
            torch.save(self.deep_model.state_dict(),
                           os.path.join(self.model_save_path, f'curr_model.pth'))
            if ((e+1) % 50 == 0):
                torch.save(self.deep_model.state_dict(),
                           os.path.join(self.model_save_path, f'{e + 1}_model.pth'))
            self.deep_model.train()
        print('The accuracy of the best round is: ', best_round)
        
        print('---------starting testing-----------')
        #* Load checkpoint
        checkpoint = torch.load(f"{self.model_save_path}/best_model.pth")
        model_dict = {}
        state_dict = self.deep_model.state_dict()
        for k, v in checkpoint.items():
            if k in state_dict:
                model_dict[k] = v
            else:
                print(f"{k} not in model dict")
        state_dict.update(model_dict)
        self.deep_model.load_state_dict(state_dict)
        
        self.deep_model.eval()
        if (self.device.startswith('cuda')):
            torch.cuda.empty_cache()
        test_metrics = self.metrics_validation(mode='test')
        with open(f"{self.model_save_path}/test_metrics.json", 'w') as f:
            json.dump(test_metrics, f)
    def validation(self):
        print('---------starting evaluation-----------')
        if (self.device == 'cuda'):
            torch.cuda.empty_cache()
       
        metrics = {
            "f1_score": 0,
            "oa": 0,
            "rec": 0,
            "pre": 0,
            "iou": 0,
            "kc": 0
        }
        with torch.no_grad():
            for itera, data in enumerate(self.val_data_loader):
                self.evaluator.reset()
                # Load images
                if (self.args.dataset == 'sen2'):
                    pre_change_imgs = data['S2_before_image'][:, self.sen2_bands_idx, :, :]
                    post_change_imgs = data['S2_after_image'][:, self.sen2_bands_idx, :, :]
                    labels = data['label']
                else:
                    pre_change_imgs, post_change_imgs, labels, _ = data

                pre_change_imgs = pre_change_imgs.to(self.device)
                post_change_imgs = post_change_imgs.to(self.device)
                labels = labels.to(self.device).long()

                output_1 = self.deep_model(pre_change_imgs, post_change_imgs)
                
                output_1 = output_1.numpy(force=True)
                output_1 = np.argmax(output_1, axis=1)
                labels = labels.numpy(force=True)

                self.evaluator.add_batch(labels, output_1)
                metrics["f1_score"] += self.evaluator.Pixel_F1_score()
                metrics["oa"] += self.evaluator.Pixel_Accuracy()
                metrics["rec"] += self.evaluator.Pixel_Recall_Rate()
                metrics["pre"] += self.evaluator.Pixel_Precision_Rate()
                metrics["iou"] += self.evaluator.Intersection_over_Union()
                metrics["kc"] += self.evaluator.Kappa_coefficient()

        #Calculate average
        n_samples = len(self.val_data_loader)
        for k in metrics.keys():
            metrics[k] /= n_samples
        
        print(f'Recall rate is {metrics["rec"]}, Precision rate is {metrics["pre"]}, OA is {metrics["oa"]}, '
              f'F1 score is {metrics["f1_score"]}, IoU is {metrics["iou"]}, Kappa coefficient is {metrics["kc"]}')
                
        return metrics

    def metrics_validation(self, mode='val'):
        print('---------starting evaluation-----------')
        if (self.device.startswith('cuda')):
            torch.cuda.empty_cache()            

        with torch.no_grad():
            self.cm.reset()
            self.iou.reset()
            for data in tqdm(self.val_data_loader if mode=='val' else self.test_data_loader):
                # Load images
                if (self.args.dataset == 'sen2'):
                    pre_change_imgs = data['S2_before_image'][:, self.sen2_bands_idx, :, :]
                    post_change_imgs = data['S2_after_image'][:, self.sen2_bands_idx, :, :]
                    labels = data['label']
                else:
                    pre_change_imgs, post_change_imgs, labels, _ = data

                pre_change_imgs = pre_change_imgs.to(self.device)
                post_change_imgs = post_change_imgs.to(self.device)
                labels = labels.to(self.device).long()

                output_1 = self.deep_model(pre_change_imgs, post_change_imgs)
                
                preds = output_1.argmax(1).to(dtype=torch.int8)
                self.cm.compute(preds, labels)
                self.iou.update(preds, labels)
        
        acc = self.cm.accuracy()
        score = self.cm.f1_score()
        prec = self.cm.precision()
        rec = self.cm.recall()
        ious = self.iou.compute()
        mean_iou = ious[:2].mean()
        floga_metrics = {
            'precision': (100 * prec[0].item(), 100 * prec[1].item()),
            'recall': (100 * rec[0].item(), 100 * rec[1].item()),
            'accuracy': (100 * acc[0].item(), 100 * acc[1].item()),
            'f1': (100 * score[0].item(), 100 * score[1].item()),
            'iou': (100 * ious[0].item(), 100 * ious[1].item()),
            'meanIoU': 100 * mean_iou.item()
        }
        return floga_metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='../MambaCD/changedetection/configs/vssm1/vssm_base_224.yaml')
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )
    parser.add_argument('--configs_path', type=str, default='../MambaCD/changedetection/configs/FLOGA_configs.json', required=False)
    parser.add_argument('--pretrained_weight_path', type=str)
    parser.add_argument('--dataset', type=str, default='sen2')
    parser.add_argument('--type', type=str, default='train')
    parser.add_argument('--train_dataset_path', type=str, default='../MambaCD/datasets/SYSU/train')
    parser.add_argument('--train_data_list_path', type=str, default='../MambaCD/datasets/SYSU/train_list.txt')
    parser.add_argument('--test_dataset_path', type=str, default='../MambaCD/datasets/SYSU/test')
    parser.add_argument('--test_data_list_path', type=str, default='../MambaCD/datasets/SYSU/test_list.txt')
    parser.add_argument('--shuffle', type=bool, default=True)
    parser.add_argument('--batch_size', type=int, default=6)
    parser.add_argument('--crop_size', type=int, default=256)
    parser.add_argument('--train_data_name_list', type=list)
    parser.add_argument('--test_data_name_list', type=list)
    parser.add_argument('--start_iter', type=int, default=0)
    parser.add_argument('--gpu_id', type=int, choices=[0, 1, 2, 3], default=3)
    parser.add_argument('--max_iters', type=int, default=150000)
    parser.add_argument('--model_type', type=str, default='MambaBCD')
    parser.add_argument('--epochs', type=int, default=200, required=False)
    parser.add_argument('--model_param_path', type=str, default='../MambaCD/data/results/')
    parser.add_argument('--model_path', type=Path, required=False, default=None)
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'], required=False)

    parser.add_argument('--resume', type=str)
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=5e-4)

    args = parser.parse_args()
    args.device = f"cuda:{args.gpu_id}"

    if (args.mode == 'train'):
        train_model = Trainer(args)
        train_model()
    elif (args.mode == 'test'):
        test_model = Trainer(args)
        print(test_model.metrics_validation(mode='test'))


if __name__ == "__main__":
    main()
