"""
This script is used to visualize the model predictions on a single image.
"""
import numpy as np
from pathlib import Path
import pyjson5
from flask import Flask, request
import matplotlib as mpl
import matplotlib.pyplot as plt

import torch
from uuid import uuid4
import json
from utils import (
    resume_or_start,
    init_model
)


def generate_mask():
    configs = pyjson5.load(open('/mnt/FLOGA/FLOGA-github/configs/inference_configs.json', 'r'))
    configs['method'] = 'snunet'
    model_configs = pyjson5.load(open(Path('/mnt/FLOGA/FLOGA-github/configs') / 'method' / f'{configs["method"]}.json', 'r'))
    configs['mode'] = 'eval'
    configs['paths']['load_state'] = '/mnt/FLOGA/FLOGA-github/data/results/snunet/20240703123347/checkpoints/0/best_segmentation.pt'

    run_path, resume_from_checkpoint, _ = resume_or_start(configs, model_configs)
    results_path = Path('/mnt/FLOGA/gradio_outputs/snunet/')
    results_path.mkdir(exist_ok=True, parents=True)

    gsd = {'sen2': '60', 'mod': '500'}
    data_source = configs['datasets']['data_source']
    for band in configs['datasets']['selected_bands'][data_source].keys():
        configs['datasets']['selected_bands'][data_source][band] = configs['datasets'][f'{data_source}_bands'][gsd[data_source]][band]

    inp_channels = len(configs['datasets']['selected_bands'][data_source])

    device = 'cpu'


    bands = configs['datasets']['selected_bands'][data_source]
    selected_bands_idx = {band: order_id for order_id, (band, _) in enumerate(bands.items())}

    if data_source == 'sen2':
        if set(['B08', 'B04', 'B03']) <= set(configs['datasets']['selected_bands']['sen2'].keys()):
            # NIR, Red, Green
            plot_bands = [selected_bands_idx[band] for band in ['B08', 'B04', 'B03']]
        else:
            # NIR, Red, Green
            plot_bands = [selected_bands_idx[band] for band in ['B8A', 'B04', 'B03']]
    else:
        if set(['B02', 'B01', 'B04']) <= configs['datasets']['selected_bands']['mod'].keys():
            # NIR, Red, Green
            plot_bands = [selected_bands_idx[band] for band in ['B02', 'B01', 'B04']]
        else:
            # NIR, Red, Green
            plot_bands = [selected_bands_idx[band] for band in ['B02', 'B01']]
    cmap = mpl.colors.LinearSegmentedColormap.from_list('Custom cmap', [(0, 0, 0, 10), (0.09019607843137255, 0.7450980392156863, 0.8117647058823529, 1.0), (0.8647058823529412, 0.30980392156862746, 0.45882352941176474, 1.0)], 3)
    
    with torch.no_grad():
        checkpoint = torch.load(resume_from_checkpoint, map_location=device)
        model = init_model(configs, model_configs, checkpoint, inp_channels, device, run_path=run_path)[0]
        model.eval()
    
    json_data = {
        "before_img_fp":"/mnt/FLOGA/data/dataset/2021/sample00000020_145_2021.sen2_60_pre.npy",
        "after_img_fp":"/mnt/FLOGA/data/dataset/2021/sample00000020_145_2021.sen2_60_post.npy",
        "label_fp":"/mnt/FLOGA/data/dataset/2021/sample00000020_145_2021.label.npy"
    }
    
    before_img_f = open(json_data['before_img_fp'], 'rb')
    after_img_f = open(json_data['after_img_fp'], 'rb')
    label_f = None if 'label_fp' not in json_data else open(json_data['label_fp'], 'rb')
    before_img = np.load(before_img_f)
    before_img_f.close()
    before_img = np.expand_dims(before_img, axis=0)
    before_img = before_img.astype(np.float32)
    
    after_img = np.load(after_img_f)
    after_img_f.close()
    after_img = np.expand_dims(after_img, axis=0)
    after_img = after_img.astype(np.float32)
    with torch.no_grad():
        before_img = torch.from_numpy(before_img)
        before_img = before_img.to(torch.float32)
        after_img = torch.from_numpy(after_img)
        after_img = after_img.to(torch.float32)

    label = None
    if label_f is not None:
        label = np.load(label_f)
        label = label.astype(np.float32)
        label_f.close()

    thresh = int(configs['datasets']['scale_input'].split('_')[-1])
    
    with torch.no_grad():
        before_img = torch.clamp(before_img, max=thresh) / thresh
        after_img = torch.clamp(after_img, max=thresh) / thresh

    bands = list(configs['datasets']['selected_bands'][data_source].values())
    with torch.no_grad():
        before_img = before_img[:, bands, :, :].to(device)
        after_img = after_img[:, bands, :, :].to(device)
    with torch.no_grad():
        output = model(before_img, after_img)
        predictions = output.argmax(1).to(dtype=torch.int8)

        before_img = before_img.squeeze()
        after_img = after_img.squeeze()

    id = str(uuid4())
    results_path.joinpath(id).mkdir(exist_ok=True)
    results_fp = {
        "before_img_fp": str(results_path / id /'prediction_visualization_ID_before_img.png'),
        "after_img_fp": str(results_path / id /'prediction_visualization_ID_after_img.png'),
        "label_img_fp": str(results_path / id /'prediction_visualization_ID_label_img.png'),
        "predictions_img_fp": str(results_path / id /'prediction_visualization_ID_predictions_img.png')
    }
    with torch.no_grad():
        before_img = before_img[plot_bands, :, :].detach().cpu().numpy()
        before_img = np.clip(before_img, a_min=0, a_max=1)
    mpl.image.imsave(results_fp["before_img_fp"], np.moveaxis(before_img, 0, -1))

    with torch.no_grad():
        after_img = after_img[plot_bands, :, :].detach().cpu().numpy()
        after_img = np.clip(after_img, a_min=0, a_max=1)
    mpl.image.imsave(results_fp["after_img_fp"], np.moveaxis(after_img, 0, -1))


    if label is not None:
        label = label.squeeze()
        mpl.image.imsave(results_fp["label_img_fp"], label, vmin=0, vmax=2, cmap=cmap)
    else:
        results_fp["label_img_fp"] = ''
    
    with torch.no_grad():
        predictions = predictions.squeeze().cpu().detach().numpy()
    mpl.image.imsave(results_fp["predictions_img_fp"], predictions, vmin=0, vmax=2, cmap=cmap)

    return results_fp

if __name__ == "__main__":
    print(generate_mask())