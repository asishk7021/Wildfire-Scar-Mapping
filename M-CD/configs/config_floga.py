import os
import os.path as osp
from pathlib import Path
import sys
import time
from easydict import EasyDict as edict
import argparse

C = edict({
    "dataset_type": "sen2_60_mod_500",  # Format: sen2_<sen2_gsd>_mod_<mod_gsd>
    "distributed": False,
    "seed": 999,
    "paths": edict({
        "dataset": "/mnt/FLOGA/data",
    }),
    "datasets": edict({
        "train": "allEvents_60-20-20_r1_V6_train.pkl",
        "val": "allEvents_60-20-20_r1_V6_val.pkl",
        "test": "allEvents_60-20-20_r1_V6_test.pkl",
        "data_source": "sen2",  # Options: "sen2", "mod"
        "scale_input": "clamp_scale_10000",  # Options: "normalize", "min-max", "clamp_scale_<value>", "clamp_<value>", a list of custom [min, max] values, null. <value> is an integer depicting the clamping threshold
        "img_size": 256,
        "batch_size": 18,
        "num_workers": 16,
        "use_shuffle": True,
        "only_positives": False,  # Use only positive patches
        "nan_value": 0,  # the value to replace NaNs with
        "augmentation": False,
        "oversampling": False,  # Options: false or float (0 <= float <= 1)
        "selected_bands": edict({  # Include only the bands needed for the experiments. Indices will be filled in during execution
            "sen2": edict({
                "B02": -1,
                "B03": -1,
                "B04": -1,
                "B05": -1,
                "B06": -1,
                "B07": -1,
                "B11": -1,
                "B12": -1,
                "B8A": -1
            }),
            "mod": edict({
                "B01": -1,
                "B02": -1,
                "B03": -1,
                "B04": -1,
                "B05": -1,
                "B06": -1,
                "B07": -1
            })
        }),
        "mod_bands": edict({  # For reference
            "500": edict({
                "B01": 0,
                "B02": 1,
                "B03": 2,
                "B04": 3,
                "B05": 4,
                "B06": 5,
                "B07": 6
            })
        }),
        "sen2_bands": edict({  # For reference
            "10": edict({
                "B02": 0,
                "B03": 1,
                "B04": 2,
                "B08": 3
            }),
            "20": edict({
                "B02": 0,
                "B03": 1,
                "B04": 2,
                "B05": 3,
                "B06": 4,
                "B07": 5,
                "B11": 6,
                "B12": 7,
                "B8A": 8
            }),
            "60": edict({
                "B01": 0,
                "B02": 1,
                "B03": 2,
                "B04": 3,
                "B05": 4,
                "B06": 5,
                "B07": 6,
                "B09": 7,
                "B11": 8,
                "B12": 9,
                "B8A": 10
            })
        }),
        "sen2_mod_500_band_mapping": edict({  # For reference
            "B02": "B03",  # Blue
            "B03": "B04",  # Green
            "B04": "B01",  # Red
            "B08": "B02",  # NIR
            "B12": "B07",  # SWIR
            "B8A": "B02"  # NIR
        }),
        "sen2_mean": edict({
            "10": [],
            "20": [
                63.8612,
                73.0030,
                78.0166,
                100.7361,
                137.4804,
                151.7485,
                144.9945,
                105.9401,
                162.0981
            ],
            "60": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        }),
        "sen2_std": edict({
            "10": [],
            "20": [
                288.5572,
                318.7534,
                354.1387,
                430.6897,
                573.3617,
                634.2242,
                614.6827,
                454.1967,
                680.0145
            ],
            "60": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        }),
        "mod_mean": edict({
            "500": [1, 1, 1, 1, 1, 1, 1]
        }),
        "mod_std": edict({
            "500": [1, 1, 1, 1, 1, 1, 1]
        })
    })
})
config = C
cfg = C

tmp = C['dataset_type'].split('_')
gsd = {tmp[0]: tmp[1], tmp[2]: tmp[3]}  # format: "sen2_xx_mod_yy"
data_source = C['datasets']['data_source']
for band in C['datasets']['selected_bands'][data_source].keys():
    C['datasets']['selected_bands'][data_source][band] = C['datasets'][f'{data_source}_bands'][gsd[data_source]][band]

# remoteip = os.popen('pwd').read()
C.root_dir = Path(os.path.abspath(os.path.join(os.getcwd(), './')))
C.abs_dir = Path(osp.realpath("."))

# Dataset config
"""Dataset Path"""
C.dataset_name = 'FLOGA'
# C.root_folder = '/mnt/FLOGA/data'
C.num_classes = 2
C.ignore_index = 2
C.class_names =  ['background', 'change']

"""Image Config"""
# C.background = 255
C.image_height = C.datasets.img_size #256
C.image_width = C.datasets.img_size #256
C.norm_mean = C['datasets']['sen2_mean']['60']
C.norm_std = C['datasets']['sen2_std']['60']

""" Settings for network, this would be different for each kind of model"""
C.backbone = 'sigma_small' # sigma_tiny / sigma_small / sigma_base
C.pretrained_model = None # do not need to change
C.decoder = 'MambaDecoder' # 'MLPDecoder'
C.decoder_embed_dim = 512
C.optimizer = 'AdamW'

"""Train Config"""
C.lr = 6e-5
C.lr_power = 0.9
C.momentum = 0.9
C.weight_decay = 0.01
# C.batch_size = 8
C.nepochs = 200
# C.niters_per_epoch = C.num_train_imgs // C.batch_size  + 1
# C.num_workers = 16
# C.train_scale_array = [1]
# C.train_scale_array = None
C.warm_up_epoch = 10

C.fix_bias = True
C.bn_eps = 1e-3
C.bn_momentum = 0.1

"""Eval Config"""
# C.eval_iter = 1
C.eval_stride_rate = 2 / 3
C.eval_scale_array = [1] 
C.eval_flip = False
C.eval_crop_size = [C.datasets.img_size, C.datasets.img_size]

"""Store Config"""
C.checkpoint_start_epoch = 0
C.checkpoint_step = 50

"""Path Config"""
def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)
add_path(osp.join(C.root_dir))

exp_time = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
C.log_dir = Path(osp.abspath(f'data/results/' + 'log_' + C.dataset_name + '_' + C.backbone + '_' + C.decoder + '/' + exp_time))
C.checkpoint_dir = Path(osp.abspath(osp.join(C.log_dir, "checkpoints")))
C.checkpoint_dir.mkdir(exist_ok=True, parents=True)

C.log_file = C.log_dir  / 'training.log'

if __name__ == '__main__':
    print(config.nepochs)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-tb', '--tensorboard', default=False, action='store_true')
    args = parser.parse_args()