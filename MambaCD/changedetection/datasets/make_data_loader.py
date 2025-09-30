import sys
import argparse
import os
from pathlib import Path
import random
import pandas as pd
import pickle
import imageio
import torch
import numpy as np
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms.functional as TF
from MambaCD.changedetection.datasets import imutils
from MambaCD.changedetection.utils_func.preprocessing import load_dataset_configs

def img_loader(path): #TODO update this function to include bands
    # img = np.load(path).astype('float32') #TODO set bands to get 9 channels
    img = np.array(imageio.imread(path), np.float32)
    return img

def one_hot_encoding(image, num_classes=8):
    # Create a one hot encoded tensor
    one_hot = np.eye(num_classes)[image.astype(np.uint8)]

    # Move the channel axis to the front
    # one_hot = np.moveaxis(one_hot, -1, 0)

    return one_hot

class FLOGADataset(Dataset):
    def __init__(self, mode, configs, clc=False, clouds=False, sea=False):
        self.mode = mode # 'train' or 'test'
        self.configs = configs
        self.augmentation = configs['datasets']['augmentation'] #* default=False

        self.ds_path = Path(configs['paths']['dataset']) / configs['dataset_type']

        # Read the pickle files containing information on the splits
        patches = pickle.load(open(self.ds_path / configs['datasets'][mode], 'rb'))
        print(f"Number of patches: {len(list(patches.keys()))}")
        self.events_df = pd.DataFrame([{**{'sample_key': k}, **patches[k]} for k in sorted(list(patches.keys()))])
        self.events_df.dropna(axis=0, inplace=True)

        # Keep the positive indices in a separate list (useful for under/oversampling)
        self.positives_idx = list(self.events_df[self.events_df['positive_flag']]['sample_key'].values)

        # format: "sen2_xx_mod_yy"
        tmp = configs['dataset_type'].split('_')
        self.gsd = {tmp[0]: tmp[1], tmp[2]: tmp[3]}

        self.clc = clc
        self.sea = sea
        self.clouds = clouds

        self.selected_bands = {}
        self.means = {}
        self.stds = {}
        for k, v in self.gsd.items():
            self.selected_bands[k] = configs['datasets']['selected_bands'][k].values()
            self.means[k] = [m for i, m in enumerate(configs['datasets'][f'{k}_mean'][v]) if i in self.selected_bands[k]]
            self.stds[k] = [m for i, m in enumerate(configs['datasets'][f'{k}_std'][v]) if i in self.selected_bands[k]]

    def scale_img(self, sample):
        '''
        Scales the given images with the method defined in the config file.
        The input `sample` is a dictionary mapping image name -> image array.
        '''
        scaled_sample = sample.copy()

        for sample_name, sample_img in sample.items():
            if ('label' in sample_name) or ('cloud' in sample_name) or ('key' in sample_name) or ('positive' in sample_name) or ('sea' in sample_name) or ('clc' in sample_name):
                scaled_sample[sample_name] = sample_img
            elif self.configs['datasets']['scale_input'] == 'normalize':
                if 'S2' in sample_name:
                    scaled_sample[sample_name] = TF.normalize(sample_img, mean=self.means['sen2'], std=self.stds['sen2'])
                elif 'MOD' in sample_name:
                    scaled_sample[sample_name] = TF.normalize(sample_img, mean=self.means['mod'], std=self.stds['mod'])
            elif self.configs['datasets']['scale_input'] == 'min-max':
                mins = sample_img.min(dim=-1).values.min(dim=-1).values
                maxs = sample_img.max(dim=-1).values.max(dim=-1).values

                uniq_mins = mins.unique()
                uniq_maxs = maxs.unique()
                if not (((len(uniq_mins) == 1) and (uniq_mins.item() == 0.)) and ((len(uniq_maxs) == 1) and (uniq_maxs.item() == 0.))):
                    # Some images are all-zeros so scaling returns a NaN image
                    new_ch = []
                    for ch in range(sample_img.shape[0]):
                        if mins[ch] == maxs[ch]:
                            # Some channels contain only a single value, so scaling returns all-NaN
                            # We convert it to all-zeros
                            new_ch.append(torch.zeros(*sample_img[ch, :, :].shape)[None, :, :])
                        else:
                            new_ch.append(((sample_img[ch, :, :] - mins[:, None, None][ch]) / (maxs[:, None, None][ch] - mins[:, None, None][ch]))[None, :, :])

                    scaled_sample[sample_name] = torch.cat(new_ch, dim=0)
            elif isinstance(self.configs['datasets']['scale_input'], list):
                new_min, new_max = [torch.tensor(i) for i in self.configs['datasets']['scale_input']]

                mins = sample_img.min(dim=-1).values.min(dim=-1).values
                maxs = sample_img.max(dim=-1).values.max(dim=-1).values

                uniq_mins = mins.unique()
                uniq_maxs = maxs.unique()
                if not (((len(uniq_mins) == 1) and (uniq_mins.item() == 0.)) and ((len(uniq_maxs) == 1) and (uniq_maxs.item() == 0.))):
                    # Some images are all-zeros so scaling returns a NaN image
                    new_ch = []
                    for ch in range(sample_img.shape[0]):
                        if mins[ch] == maxs[ch]:
                            # Some channels contain only a single value, so scaling returns all-NaN
                            # We convert it to all-zeros
                            new_ch.append(torch.zeros(*sample_img[ch, :, :].shape)[None, :, :])
                        else:
                            new_ch.append(((sample_img[ch, :, :] - mins[:, None, None][ch]) / (maxs[:, None, None][ch] - mins[:, None, None][ch]))[None, :, :])

                    scaled_sample[sample_name] = torch.mul(torch.cat(new_ch, dim=0), (new_max - new_min)) + new_min
            elif self.configs['datasets']['scale_input'].startswith('clamp_scale'):
                thresh = int(self.configs['datasets']['scale_input'].split('_')[-1])
                scaled_sample[sample_name] = torch.clamp(sample_img, max=thresh)
                scaled_sample[sample_name] = scaled_sample[sample_name] / thresh
            elif self.configs['datasets']['scale_input'].startswith('clamp'):
                thresh = int(self.configs['datasets']['scale_input'].split('_')[-1])
                scaled_sample[sample_name] = torch.clamp(sample_img, max=thresh)

        return scaled_sample

    def load_img(self, sample):
        '''
        Loads the images associated with a single event. The input `sample` is a list of filenames for
        the event.

        Returns a dictionary mapping image name -> image array.
        '''
        loaded_sample = {}

        for sample_info in sample.index:
            if sample_info == 'sample_key':
                loaded_sample['key'] = sample[sample_info]
            elif sample_info == 'positive_flag':
                loaded_sample['positive'] = sample[sample_info]
            elif ('label' in sample_info):
                if sample[sample_info].suffix == '.npy':
                    loaded_sample[sample_info] = torch.from_numpy(np.load(sample[sample_info]))
                else:
                    loaded_sample[sample_info] = torch.load(sample[sample_info])
            elif self.clouds and ('cloud' in sample_info):
                if sample[sample_info].suffix == '.npy':
                    loaded_sample[sample_info] = torch.from_numpy(np.load(sample[sample_info]).astype(np.float32))
                else:
                    loaded_sample[sample_info] = torch.load(sample[sample_info])
            elif 'sea' in sample_info:
                if self.sea:
                    if sample[sample_info].suffix == '.npy':
                        loaded_sample[sample_info] = torch.from_numpy(np.load(sample[sample_info]))
                    else:
                        loaded_sample[sample_info] = torch.load(sample[sample_info])
            elif 'S2' in sample_info:
                if sample[sample_info].suffix == '.npy':
                    loaded_sample[sample_info] = torch.from_numpy(np.load(sample[sample_info]).astype(np.float32)).to(torch.float32)
                else:
                    loaded_sample[sample_info] = torch.load(sample[sample_info]).to(torch.float32)
            elif 'MOD' in sample_info:
                if sample[sample_info].suffix == '.npy':
                    loaded_sample[sample_info] = torch.from_numpy(np.load(sample[sample_info]).astype(np.float32)).to(torch.float32)
                else:
                    loaded_sample[sample_info] = torch.load(sample[sample_info]).to(torch.float32)
            elif self.clc and ('clc' in sample_info):
                if sample[sample_info].suffix == '.npy':
                    loaded_sample[sample_info] = torch.from_numpy(np.load(sample[sample_info]).astype(np.float32))
                else:
                    loaded_sample[sample_info] = torch.load(sample[sample_info])

        return loaded_sample

    def fillna(self, sample):
        '''
        Fills NaN values in the sample with the constant specified in the config.

        It also replaces the corresponding values in the label with the number '2' which will be ignored during training.
        '''
        filled_sample = sample.copy()

        nan_idx = []
        label = []
        for sample_name, s in sample.items():
            if 'label' in sample_name:
                label.append(sample_name)
            elif ('cloud' in sample_name) or ('clc' in sample_name):
                continue
            elif ('before' in sample_name) or ('after' in sample_name):
                nan_idx.append(torch.isnan(s))
                filled_sample[sample_name] = torch.nan_to_num(s, nan=self.configs['datasets']['nan_value'])

        for lbl in label:
            for nan_id in nan_idx:
                for band_id in nan_id:
                    filled_sample[lbl][band_id] = 2

        return filled_sample

    def augment(self, sample):
        '''
        Applies the following augmentations:
        - Random horizontal flipping (possibility = 0.5)
        - Random vertical flipping (possibility = 0.5)
        - Random rotation (-15 to +15 deg)
        '''
        aug_sample = sample.copy()

        # Horizontal flip
        if random.random() > 0.5:
            for sample_name, s in aug_sample.items():
                if sample_name not in ['key', 'positive']:
                    aug_sample[sample_name] = TF.hflip(s)

        # Vertical flip
        if random.random() > 0.5:
            for sample_name, s in aug_sample.items():
                if sample_name not in ['key', 'positive']:
                    aug_sample[sample_name] = TF.vflip(s)

        # Rotation
        if random.random() > 0.5:
            angle = random.uniform(-15, 15)
            for sample_name, s in aug_sample.items():
                if sample_name not in ['key', 'positive']:
                    if s.dim() == 2:
                        # For some reason `TF.rotate()` cannot handle 2D input
                        aug_sample[sample_name] = TF.rotate(torch.unsqueeze(s, 0), angle=angle).squeeze()
                    else:
                        aug_sample[sample_name] = TF.rotate(s, angle=angle)

        return aug_sample

    def __len__(self):
        return self.events_df.shape[0]

    def __getitem__(self, event_id):
        batch = self.events_df.iloc[event_id]

        # Load images
        batch = self.load_img(batch)

        # Replace NaN values with constant
        batch = self.fillna(batch)

        # Normalize images
        if self.configs['datasets']['scale_input'] is not None:
            batch = self.scale_img(batch)

        # Augment images
        if self.augmentation:
            batch = self.augment(batch)

        return batch

class ChangeDetectionDataset(Dataset):
    def __init__(self, dataset_path, data_list, crop_size, max_iters=None, type='train', data_loader=img_loader):
        self.dataset_path = dataset_path
        self.data_list = data_list
        self.loader = data_loader
        self.type = type
        self.data_pro_type = self.type

        if max_iters is not None:
            self.data_list = self.data_list * int(np.ceil(float(max_iters) / len(self.data_list)))
            self.data_list = self.data_list[0:max_iters]
        self.crop_size = crop_size

    def __transforms(self, aug, pre_img, post_img, label):
        if aug:
            pre_img, post_img, label = imutils.random_crop_new(pre_img, post_img, label, self.crop_size)
            pre_img, post_img, label = imutils.random_fliplr(pre_img, post_img, label) #TODO check how this affects 3D array
            pre_img, post_img, label = imutils.random_flipud(pre_img, post_img, label) #TODO check how this affects 3D array
            pre_img, post_img, label = imutils.random_rot(pre_img, post_img, label) #TODO check how this affects 3D array

        pre_img = imutils.normalize_img(pre_img)  # imagenet normalization
        pre_img = np.transpose(pre_img, (2, 0, 1))

        post_img = imutils.normalize_img(post_img)  # imagenet normalization
        post_img = np.transpose(post_img, (2, 0, 1))

        return pre_img, post_img, label

    def __getitem__(self, index):
        pre_path = os.path.join(self.dataset_path, 'T1', self.data_list[index])
        post_path = os.path.join(self.dataset_path, 'T2', self.data_list[index])
        label_path = os.path.join(self.dataset_path, 'GT', self.data_list[index])
        pre_img = self.loader(pre_path)
        post_img = self.loader(post_path)
        label = self.loader(label_path)
        label = label / 255

        if 'train' in self.data_pro_type:
            pre_img, post_img, label = self.__transforms(True, pre_img, post_img, label)
        else:
            pre_img, post_img, label = self.__transforms(False, pre_img, post_img, label)
            label = np.asarray(label)

        data_idx = self.data_list[index]
        return pre_img, post_img, label, data_idx

    def __len__(self):
        return len(self.data_list)

class SemanticChangeDetectionDatset(Dataset):
    def __init__(self, dataset_path, data_list, crop_size, max_iters=None, type='train', data_loader=img_loader):
        self.dataset_path = dataset_path
        self.data_list = data_list
        self.loader = data_loader
        self.type = type
        self.data_pro_type = self.type

        if max_iters is not None:
            self.data_list = self.data_list * int(np.ceil(float(max_iters) / len(self.data_list)))
            self.data_list = self.data_list[0:max_iters]
        self.crop_size = crop_size

    def __transforms(self, aug, pre_img, post_img, cd_label, t1_label, t2_label):
        if aug:
            pre_img, post_img, cd_label, t1_label, t2_label = imutils.random_crop_mcd(pre_img, post_img, cd_label, t1_label, t2_label, self.crop_size)
            pre_img, post_img, cd_label, t1_label, t2_label = imutils.random_fliplr_mcd(pre_img, post_img, cd_label, t1_label, t2_label)
            pre_img, post_img, cd_label, t1_label, t2_label = imutils.random_flipud_mcd(pre_img, post_img, cd_label, t1_label, t2_label)
            pre_img, post_img, cd_label, t1_label, t2_label = imutils.random_rot_mcd(pre_img, post_img, cd_label, t1_label, t2_label)

        pre_img = imutils.normalize_img(pre_img)  # imagenet normalization
        pre_img = np.transpose(pre_img, (2, 0, 1))

        post_img = imutils.normalize_img(post_img)  # imagenet normalization
        post_img = np.transpose(post_img, (2, 0, 1))

        return pre_img, post_img, cd_label, t1_label, t2_label

    def __getitem__(self, index):
        if 'train' in self.data_pro_type:
            pre_path = os.path.join(self.dataset_path, 'T1', self.data_list[index] + '.png')
            post_path = os.path.join(self.dataset_path, 'T2', self.data_list[index] + '.png')
            T1_label_path = os.path.join(self.dataset_path, 'GT_T1', self.data_list[index] + '.png')
            T2_label_path = os.path.join(self.dataset_path, 'GT_T2', self.data_list[index] + '.png')
            cd_label_path = os.path.join(self.dataset_path, 'GT_CD', self.data_list[index] + '.png')
        else:
            pre_path = os.path.join(self.dataset_path, 'T1', self.data_list[index])
            post_path = os.path.join(self.dataset_path, 'T2', self.data_list[index])
            T1_label_path = os.path.join(self.dataset_path, 'GT_T1', self.data_list[index])
            T2_label_path = os.path.join(self.dataset_path, 'GT_T2', self.data_list[index])
            cd_label_path = os.path.join(self.dataset_path, 'GT_CD', self.data_list[index])

        pre_img = self.loader(pre_path)
        post_img = self.loader(post_path)
        t1_label = self.loader(T1_label_path)
        t2_label = self.loader(T2_label_path)
        cd_label = self.loader(cd_label_path)
        cd_label = cd_label / 255

        if 'train' in self.data_pro_type:
            pre_img, post_img, cd_label, t1_label, t2_label = self.__transforms(True, pre_img, post_img, cd_label, t1_label, t2_label)
        else:
            pre_img, post_img, cd_label, t1_label, t2_label = self.__transforms(False, pre_img, post_img, cd_label, t1_label, t2_label)
            cd_label = np.asarray(cd_label)
            t1_label = np.asarray(t1_label)
            t2_label = np.asarray(t2_label)

        data_idx = self.data_list[index]
        return pre_img, post_img, cd_label, t1_label, t2_label, data_idx

    def __len__(self):
        return len(self.data_list)

class DamageAssessmentDatset(Dataset):
    def __init__(self, dataset_path, data_list, crop_size, max_iters=None, type='train', data_loader=img_loader):
        self.dataset_path = dataset_path
        self.data_list = data_list
        self.loader = data_loader
        self.type = type
        self.data_pro_type = self.type

        if max_iters is not None:
            self.data_list = self.data_list * int(np.ceil(float(max_iters) / len(self.data_list)))
            self.data_list = self.data_list[0:max_iters]
        self.crop_size = crop_size

    def __transforms(self, aug, pre_img, post_img, loc_label, clf_label):
        if aug:
            pre_img, post_img, loc_label, clf_label = imutils.random_crop_bda(pre_img, post_img, loc_label, clf_label, self.crop_size)
            pre_img, post_img, loc_label, clf_label = imutils.random_fliplr_bda(pre_img, post_img, loc_label, clf_label)
            pre_img, post_img, loc_label, clf_label = imutils.random_flipud_bda(pre_img, post_img, loc_label, clf_label)
            pre_img, post_img, loc_label, clf_label = imutils.random_rot_bda(pre_img, post_img, loc_label, clf_label)

        pre_img = imutils.normalize_img(pre_img)  # imagenet normalization
        pre_img = np.transpose(pre_img, (2, 0, 1))

        post_img = imutils.normalize_img(post_img)  # imagenet normalization
        post_img = np.transpose(post_img, (2, 0, 1))

        return pre_img, post_img, loc_label, clf_label

    def __getitem__(self, index):
        if 'train' in self.data_pro_type: 
            parts = self.data_list[index].rsplit('_', 2)

            pre_img_name = f"{parts[0]}_pre_disaster_{parts[1]}_{parts[2]}.png"
            post_img_name = f"{parts[0]}_post_disaster_{parts[1]}_{parts[2]}.png"

            pre_path = os.path.join(self.dataset_path, 'images', pre_img_name)
            post_path = os.path.join(self.dataset_path, 'images', post_img_name)
            
            loc_label_path = os.path.join(self.dataset_path, 'masks', pre_img_name)
            clf_label_path = os.path.join(self.dataset_path, 'masks', post_img_name)
        else:
            pre_path = os.path.join(self.dataset_path, 'images', self.data_list[index] + '_pre_disaster.png')
            post_path = os.path.join(self.dataset_path, 'images', self.data_list[index] + '_post_disaster.png')
            loc_label_path = os.path.join(self.dataset_path, 'masks', self.data_list[index]+ '_pre_disaster.png')
            clf_label_path = os.path.join(self.dataset_path, 'masks', self.data_list[index]+ '_post_disaster.png')

        pre_img = self.loader(pre_path)
        post_img = self.loader(post_path)
        loc_label = self.loader(loc_label_path)[:,:,0]
        clf_label = self.loader(clf_label_path)[:,:,0]

        if 'train' in self.data_pro_type:
            pre_img, post_img, loc_label, clf_label = self.__transforms(True, pre_img, post_img, loc_label, clf_label)
            clf_label[clf_label == 0] = 255
        else:
            pre_img, post_img, loc_label, clf_label = self.__transforms(False, pre_img, post_img, loc_label, clf_label)
            loc_label = np.asarray(loc_label)
            clf_label = np.asarray(clf_label)

        data_idx = self.data_list[index]
        return pre_img, post_img, loc_label, clf_label, data_idx

    def __len__(self):
        return len(self.data_list)

def make_data_loader(args, **kwargs):  # **kwargs could be omitted
    if 'SYSU' in args.dataset or 'LEVIR-CD+' in args.dataset or 'WHU' in args.dataset:
        dataset = ChangeDetectionDataset(args.train_dataset_path, args.train_data_name_list, args.crop_size, args.max_iters, args.type)
        # train_sampler = DistributedSampler(dataset, shuffle=True)
        data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=args.shuffle, **kwargs, num_workers=16,
                                 drop_last=False)
        return data_loader
    elif 'xBD' in args.dataset:
        dataset = DamageAssessmentDatset(args.train_dataset_path, args.train_data_name_list, args.crop_size, args.max_iters, args.type)
        data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=args.shuffle, **kwargs, num_workers=6,
                                 drop_last=False)
        return data_loader
    elif 'SECOND' in args.dataset:
        dataset = SemanticChangeDetectionDatset(args.train_dataset_path, args.train_data_name_list, args.crop_size, args.max_iters, args.type)
        data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=args.shuffle, **kwargs, num_workers=16,
                                 drop_last=False)
        return data_loader
    elif 'sen2' in args.dataset or 'mod' in args.dataset:
        configs = load_dataset_configs(args.configs_path)
        dataset = FLOGADataset('train', configs)
        data_loader = DataLoader(dataset, num_workers=configs['datasets']['num_workers'], batch_size=configs['datasets']['batch_size'],
                                 shuffle=True, drop_last=False, pin_memory=True)
        return data_loader
    else:
        raise NotImplementedError


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="SECOND DataLoader Test")
    parser.add_argument('--dataset', type=str, default='sen2')
    parser.add_argument('--max_iters', type=int, default=30000)
    parser.add_argument('--type', type=str, default='train')
    parser.add_argument('--dataset_path', type=str, default='D:/Workspace/Python/STCD/data/ST-WHU-BCD') #Can be ignored
    parser.add_argument('--data_list_path', type=str, default='./ST-WHU-BCD/train_list.txt') #Can be ignored
    parser.add_argument('--shuffle', type=bool, default=True)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--data_name_list', type=list)
    parser.add_argument('--configs_path', type=str, default='../MambaCD/changedetection/configs/FLOGA_configs.json', required=False)

    args = parser.parse_args()

    # with open(args.data_list_path, "r") as f:
    #     # data_name_list = f.read()
    #     data_name_list = [data_name.strip() for data_name in f]
    # args.data_name_list = data_name_list
    train_data_loader = make_data_loader(args)
    for i, data in enumerate(train_data_loader):
        if (args.dataset == 'sen2'):
            bands_idx = [1, 2, 3, 4, 5, 6, 8, 9, 10]
            pre_img = data['S2_before_image'][:, bands_idx, :, :]
            post_img = data['S2_after_image'][:, bands_idx, :, :]
            labels = data['label']
        else:
            pre_img, post_img, labels, _ = data
        pre_data, post_data = Variable(pre_img), Variable(post_img)
        labels = Variable(labels)
        print(i, "个inputs", pre_data.data.size(), "labels", labels.data.size())
