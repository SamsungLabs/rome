import lmdb
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils import data
from torchvision import transforms
import albumentations as A
from argparse import ArgumentParser
import io
from PIL import Image, ImageOps
import random
import cv2
import pickle

from src.utils import args as args_utils
from src.utils.point_transforms import parse_3dmm_param


class LMDBDataset(data.Dataset):
    def __init__(self,
                 data_root,
                 image_size,
                 keys,
                 phase,
                 align_source=False,
                 align_target=False,
                 align_scale=1.0,
                 augment_geometric_source=False,
                 augment_geometric_target=False,
                 augment_color=False,
                 output_aug_warp=False,
                 aug_warp_size=-1,
                 epoch_len=-1,
                 return_keys=False,
                 ):
        super(LMDBDataset, self).__init__()
        self.envs = []
        for i in range(128):
            self.envs.append(lmdb.open(f'{data_root}/chunks/{i}_lmdb', max_readers=1, readonly=True,
                                       lock=False, readahead=False, meminit=False))

        self.keys = keys
        self.phase = phase

        self.image_size = image_size

        self.return_keys = return_keys
        self.align_source = align_source
        self.align_target = align_target
        self.align_scale = align_scale
        self.augment_geometric_source = augment_geometric_source
        self.augment_geometric_target = augment_geometric_target
        self.augment_color = augment_color
        self.output_aug_warp = output_aug_warp

        self.epoch_len = epoch_len

        # Transforms
        if self.augment_color:
            self.aug = A.Compose(
                [A.ColorJitter(hue=0.02, p=0.8)],
                additional_targets={f'image{k}': 'image' for k in range(1, 2)})
        self.to_tensor = transforms.ToTensor()

        if self.align_source:
            grid = torch.linspace(-1, 1, self.image_size)
            v, u = torch.meshgrid(grid, grid)
            self.identity_grid = torch.stack([u, v, torch.ones_like(u)], dim=2).view(1, -1, 3)

        if self.output_aug_warp:
            self.aug_warp_size = aug_warp_size

            # Greate a uniform meshgrid, which is used for warping calculation from deltas
            tick = torch.linspace(0, 1, self.aug_warp_size)
            v, u = torch.meshgrid(tick, tick)
            grid = torch.stack([u, v, torch.zeros(self.aug_warp_size, self.aug_warp_size)], dim=2)

            self.grid = (grid * 255).numpy().astype('uint8')  # aug_warp_size x aug_warp_size x 3

    @staticmethod
    def to_tensor_keypoints(keypoints, size):
        keypoints = torch.from_numpy(keypoints).float()
        keypoints /= size
        keypoints[..., :2] -= 0.5
        keypoints *= 2

        return keypoints

    def __getitem__(self, index):
        n = 1
        t = 1

        chunk, keys_ = self.keys[index]
        env = self.envs[chunk]

        if self.phase == 'train':
            indices = torch.randperm(len(keys_))[:2]
            keys = [keys_[i] for i in indices]

        else:
            keys = keys_

        data_dict = {
            'image': [],
            'mask': [],
            'size': [],
            'face_scale': [],
            'keypoints': [],
            'params_3dmm': {'R': [], 'offset': [], 'roi_box': [], 'size': []},
            'params_ffhq': {'theta': []},
            'crop_box': []}

        with env.begin(write=False) as txn:
            for key in keys:
                item = pickle.loads(txn.get(key.encode()))

                image = Image.open(io.BytesIO(item['image'])).convert('RGB')
                mask = Image.open(io.BytesIO(item['mask']))

                data_dict['image'].append(image)
                data_dict['mask'].append(mask)

                data_dict['size'].append(item['size'])
                data_dict['face_scale'].append(item['face_scale'])
                data_dict['keypoints'].append(item['keypoints_2d'])

                R, offset, _, _ = parse_3dmm_param(item['3dmm']['param'])

                data_dict['params_3dmm']['R'].append(R)
                data_dict['params_3dmm']['offset'].append(offset)
                data_dict['params_3dmm']['roi_box'].append(item['3dmm']['bbox'])
                data_dict['params_3dmm']['size'].append(item['size'])

                data_dict['params_ffhq']['theta'].append(item['transform_ffhq']['theta'])

        # Geometric augmentations and resize
        data_dict = self.preprocess_data(data_dict)
        data_dict['image'] = [np.asarray(img).copy() for img in data_dict['image']]

        # Augment color
        if self.augment_color:
            imgs_dict = {(f'image{k}' if k > 0 else 'image'): img for k, img in enumerate(data_dict['image'])}
            data_dict['image'] = list(self.aug(**imgs_dict).values())

        # Augment with local warpings
        if self.output_aug_warp:
            warp_aug = self.augment_via_warp([self.grid] * (n + t), self.aug_warp_size)
            warp_aug = torch.stack([self.to_tensor(w) for w in warp_aug], dim=0)
            warp_aug = (warp_aug.permute(0, 2, 3, 1)[..., :2] - 0.5) * 2

        imgs = torch.stack([self.to_tensor(img) for img in data_dict['image']])
        masks = torch.stack([self.to_tensor(mask) for mask in data_dict['mask']])
        keypoints = torch.FloatTensor(data_dict['keypoints'])

        R = torch.FloatTensor(data_dict['params_3dmm']['R'])
        offset = torch.FloatTensor(data_dict['params_3dmm']['offset'])
        roi_box = torch.FloatTensor(data_dict['params_3dmm']['roi_box'])[:, None]
        size = torch.FloatTensor(data_dict['params_3dmm']['size'])[:, None, None]
        theta = torch.FloatTensor(data_dict['params_ffhq']['theta'])
        crop_box = torch.FloatTensor(data_dict['crop_box'])[:, None]
        face_scale = torch.FloatTensor(data_dict['face_scale'])

        if self.align_source or self.align_target:
            # Align input images using theta
            eye_vector = torch.zeros(theta.shape[0], 1, 3)
            eye_vector[:, :, 2] = 1

            theta_ = torch.cat([theta, eye_vector], dim=1).float()

            # Perform 2x zoom-in compared to default theta
            scale = torch.zeros_like(theta_)
            scale[:, [0, 1], [0, 1]] = self.align_scale
            scale[:, 2, 2] = 1

            theta_ = torch.bmm(theta_, scale)[:, :2]

            align_warp = self.identity_grid.repeat_interleave(theta_.shape[0], dim=0)
            align_warp = align_warp.bmm(theta_.transpose(1, 2)).view(theta_.shape[0], self.image_size, self.image_size,
                                                                     2)

            if self.align_source:
                source_imgs_aligned = F.grid_sample(imgs[:n], align_warp[:n])
                source_masks_aligned = F.grid_sample(masks[:n], align_warp[:n])

            if self.align_target:
                target_imgs_aligned = F.grid_sample(imgs[-t:], align_warp[-t:])
                target_masks_aligned = F.grid_sample(masks[-t:], align_warp[-t:])

        output_data_dict = {
            'source_img': source_imgs_aligned if self.align_source else F.interpolate(imgs[:n], size=self.image_size,
                                                                                      mode='bilinear'),
            'source_mask': source_masks_aligned if self.align_source else F.interpolate(masks[:n], size=self.image_size,
                                                                                        mode='bilinear'),
            'source_keypoints': keypoints[:n],

            'target_img': target_imgs_aligned if self.align_target else F.interpolate(imgs[-t:], size=self.image_size,
                                                                                      mode='bilinear'),
            'target_mask': target_masks_aligned if self.align_target else F.interpolate(masks[-t:],
                                                                                        size=self.image_size,
                                                                                        mode='bilinear'),
            'target_keypoints': keypoints[-t:]
        }
        if self.return_keys:
            output_data_dict['keys'] = keys

        if self.output_aug_warp:
            output_data_dict['source_warp_aug'] = warp_aug[:n]
            output_data_dict['target_warp_aug'] = warp_aug[-t:]

        return output_data_dict

    def preprocess_data(self, data_dict):
        MIN_SCALE = 0.67
        n = 1
        t = 1

        for i in range(len(data_dict['image'])):
            image = data_dict['image'][i]
            mask = data_dict['mask'][i]
            size = data_dict['size'][i]
            face_scale = data_dict['face_scale'][i]
            keypoints = data_dict['keypoints'][i]

            use_geometric_augs = (i < n) and self.augment_geometric_source or (i == n) and self.augment_geometric_target

            if use_geometric_augs and face_scale >= MIN_SCALE:
                # Random sized crop
                min_scale = MIN_SCALE / face_scale
                seed = random.random()
                scale = seed * (1 - min_scale) + min_scale
                translate_x = random.random() * (1 - scale)
                translate_y = random.random() * (1 - scale)

            elif i > n:
                pass  # use params of the previous frame

            else:
                translate_x = 0
                translate_y = 0
                scale = 1

            crop_box = (size * translate_x,
                        size * translate_y,
                        size * (translate_x + scale),
                        size * (translate_y + scale))

            size_box = (crop_box[2] - crop_box[0], crop_box[3] - crop_box[1])

            keypoints[..., 0] = (keypoints[..., 0] - crop_box[0]) / size_box[0] - 0.5
            keypoints[..., 1] = (keypoints[..., 1] - crop_box[1]) / size_box[1] - 0.5
            keypoints *= 2

            data_dict['keypoints'][i] = keypoints

            image = image.crop(crop_box)
            image = image.resize((self.image_size * 2, self.image_size * 2), Image.BICUBIC)

            mask = mask.crop(crop_box)
            mask = mask.resize((self.image_size * 2, self.image_size * 2), Image.BICUBIC)

            data_dict['image'][i] = image
            data_dict['mask'][i] = mask

            # Normalize crop_box to work with coords in [-1, 1]
            crop_box = ((translate_x - 0.5) * 2,
                        (translate_y - 0.5) * 2,
                        (translate_x + scale - 0.5) * 2,
                        (translate_y + scale - 0.5) * 2)

            data_dict['crop_box'].append(crop_box)

        return data_dict

    @staticmethod
    def augment_via_warp(images, image_size):
        # Implementation is based on DeepFaceLab repo
        # https://github.com/iperov/DeepFaceLab
        #
        # Performs an elastic-like transform for a uniform grid accross the image
        image_aug = []

        for image in images:
            cell_count = 8 + 1
            cell_size = image_size // (cell_count - 1)

            grid_points = np.linspace(0, image_size, cell_count)
            mapx = np.broadcast_to(grid_points, (cell_count, cell_count)).copy()
            mapy = mapx.T

            mapx[1:-1, 1:-1] = mapx[1:-1, 1:-1] + np.random.normal(
                size=(cell_count - 2, cell_count - 2)) * cell_size * 0.1
            mapy[1:-1, 1:-1] = mapy[1:-1, 1:-1] + np.random.normal(
                size=(cell_count - 2, cell_count - 2)) * cell_size * 0.1

            half_cell_size = cell_size // 2

            mapx = cv2.resize(mapx, (image_size + cell_size,) * 2)[half_cell_size:-half_cell_size,
                   half_cell_size:-half_cell_size].astype(np.float32)
            mapy = cv2.resize(mapy, (image_size + cell_size,) * 2)[half_cell_size:-half_cell_size,
                   half_cell_size:-half_cell_size].astype(np.float32)

            image_aug += [cv2.remap(image, mapx, mapy, cv2.INTER_CUBIC)]

        return image_aug

    def __len__(self):
        if self.epoch_len == -1:
            return len(self.keys)
        else:
            return self.epoch_len


class DataModule(object):
    @staticmethod
    def add_argparse_args(parent_parser: ArgumentParser, *, use_argument_group=True):
        if use_argument_group:
            parser = parent_parser.add_argument_group("dataset")
            parser_out = parent_parser
        else:
            parser = ArgumentParser(parents=[parent_parser], add_help=False)
            parser_out = parser

        parser.add_argument('--batch_size', default=4, type=int)
        parser.add_argument('--test_batch_size', default=1, type=int)
        parser.add_argument('--num_workers', default=16, type=int)
        parser.add_argument('--data_root', type=str)
        parser.add_argument('--image_size', default=256, type=int)
        parser.add_argument('--augment_geometric_source', default='True', type=args_utils.str2bool,
                            choices=[True, False])
        parser.add_argument('--augment_geometric_target', default='True', type=args_utils.str2bool,
                            choices=[True, False])
        parser.add_argument('--augment_color', default='True', type=args_utils.str2bool, choices=[True, False])
        parser.add_argument('--return_keys', default='False', type=args_utils.str2bool, choices=[True, False])
        parser.add_argument('--num_source_frames', default=1, type=int)
        parser.add_argument('--num_target_frames', default=1, type=int)
        parser.add_argument('--keys_name', default='keys_diverse_pose')

        parser.add_argument('--align_source', default='False', type=args_utils.str2bool, choices=[True, False])
        parser.add_argument('--align_target', default='False', type=args_utils.str2bool, choices=[True, False])
        parser.add_argument('--align_scale', default=1.0, type=float)

        parser.add_argument('--output_aug_warp', default='True', type=args_utils.str2bool, choices=[True, False])
        parser.add_argument('--aug_warp_size', default=128, type=int)

        # These parameters can be used for debug
        parser.add_argument('--train_epoch_len', default=-1, type=int)
        parser.add_argument('--test_epoch_len', default=-1, type=int)

        return parser_out

    def __init__(self, args):
        super(DataModule, self).__init__()
        self.ddp = args.num_gpus > 1
        self.batch_size = args.batch_size
        self.test_batch_size = args.test_batch_size
        self.num_workers = args.num_workers
        self.data_root = args.data_root
        self.image_size = args.image_size
        self.align_source = args.align_source
        self.align_target = args.align_target
        self.align_scale = args.align_scale
        self.augment_geometric_source = args.augment_geometric_source
        self.augment_geometric_target = args.augment_geometric_target
        self.augment_color = args.augment_color
        self.return_keys = args.return_keys
        self.output_aug_warp = args.output_aug_warp
        self.aug_warp_size = args.aug_warp_size
        self.train_epoch_len = args.train_epoch_len
        self.test_epoch_len = args.test_epoch_len

        self.keys = {
            'test': pickle.load(open(f'{self.data_root}/lists/test_keys.pkl', 'rb')),
            'train': pickle.load(open(f'{self.data_root}/lists/train_keys.pkl', 'rb'))}

    def train_dataloader(self):
        train_dataset = LMDBDataset(self.data_root,
                                    self.image_size,
                                    self.keys['train'],
                                    'train',
                                    self.align_source,
                                    self.align_target,
                                    self.align_scale,
                                    self.augment_geometric_source,
                                    self.augment_geometric_target,
                                    self.augment_color,
                                    self.output_aug_warp,
                                    self.aug_warp_size,
                                    self.train_epoch_len,
                                    self.return_keys)

        shuffle = True
        sampler = None
        if self.ddp:
            shuffle = False
            sampler = data.distributed.DistributedSampler(train_dataset)

        return (
            data.DataLoader(train_dataset,
                            batch_size=self.batch_size,
                            num_workers=self.num_workers,
                            pin_memory=True,
                            shuffle=shuffle,
                            sampler=sampler,
                            drop_last=True),
            sampler
        )

    def test_dataloader(self):
        test_dataset = LMDBDataset(self.data_root,
                                   self.image_size,
                                   self.keys['test'],
                                   'test',
                                   self.align_source,
                                   self.align_target,
                                   self.align_scale,
                                   return_keys=self.return_keys,
                                   epoch_len=self.test_epoch_len)

        sampler = None
        if self.ddp:
            sampler = data.distributed.DistributedSampler(test_dataset, shuffle=False)

        return data.DataLoader(test_dataset,
                               batch_size=self.test_batch_size,
                               num_workers=self.num_workers,
                               pin_memory=True,
                               sampler=sampler)