import warnings

import os
import importlib
import argparse
from glob import glob

import face_alignment
from torchvision import transforms
from PIL import Image
import numpy as np
import torch
from pytorch3d.io import save_obj
import torch.nn.functional as F
import torch.nn as nn
from MODNet.src.models.modnet import MODNet

from data_utils import calc_ffhq_alignment
from src.rome import ROME
from src.utils import args as args_utils
from src.utils.processing import process_black_shape, prepare_input_data, tensor2image
from src.utils.visuals import obtain_modnet_mask, mask_errosion

warnings.filterwarnings("ignore")


class Infer(object):
    def __init__(self, args):
        super(Infer, self).__init__()

        # Initialize and apply general options
        torch.manual_seed(args.random_seed)
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Initialize model
        if args.verbose:
            print('Initialize model.')

        self.model = ROME(args).eval().to(self.device)
        self.image_size = 256
        self.source_transform = transforms.Compose([
            # transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])

        # Load pre-trained weights
        if args.model_checkpoint:
            ckpt_loaded = torch.load(args.model_checkpoint, map_location='cpu')
            missing_keys, unexpected_keys = self.model.load_state_dict(ckpt_loaded, strict=False)
        self.setup_modnet()
        self.mask_hard_threshold = 0.5

        self.data_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])

    def setup_modnet(self):
        pretrained_ckpt = self.args.modnet_path

        modnet = nn.DataParallel(MODNet(backbone_pretrained=False))

        modnet.load_state_dict(torch.load(pretrained_ckpt, map_location='cpu'))
        self.modnet = modnet.eval().to(self.device)
        self.fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D,
                                               flip_input=False, device='cuda' if torch.cuda.is_available() else 'cpu')

    def process_source_for_input_dict(self, source_img: Image, data_transform, crop_center=False):
        data_dict = {}
        source_pose = self.fa.get_landmarks_from_image(np.asarray(source_img))[0]

        if crop_center or source_img.size[0] != source_img.size[1]:
            pose = source_pose
            center = ((pose.min(0) + pose.max(0)) / 2).round().astype(int)
            size = int(max(pose[:, 0].max() - pose[:, 0].min(), pose[:, 1].max() - pose[:, 1].min()))
            center[1] -= size // 6
            source_img = source_img.crop((center[0] - size, center[1] - size, center[0] + size, center[1] + size))

        source_img = source_img.resize((self.image_size, self.image_size), Image.ANTIALIAS)
        data_dict['source_img'] = data_transform(source_img)[None].to(self.device)

        pred_mask = obtain_modnet_mask(data_dict['source_img'][0], self.modnet, ref_size=512)[0]

        data_dict['source_mask'] = torch.from_numpy(pred_mask).float().to(self.device).unsqueeze(0)[None]
        data_dict['source_keypoints'] = torch.from_numpy(self.fa.get_landmarks_from_image(np.asarray(source_img))[0])[
            None]

        if (data_dict['source_mask'].shape) == 3:
            data_dict['source_mask'] = data_dict['source_mask'][..., -1]
        return self.preprocess_dict(data_dict)

    def preprocess_dict(self, data_dict):
        args = self.args

        imgs = data_dict['source_img'].cpu()
        masks = data_dict['source_mask'].cpu()

        image_size = self.image_size

        lm_2d = data_dict['source_keypoints'][0].detach().cpu().numpy()
        transform_ffhq = calc_ffhq_alignment(lm_2d, size=imgs.shape[2], device=self.device)

        theta = torch.FloatTensor(transform_ffhq['theta'])[None]

        if args.align_source:
            grid = torch.linspace(-1, 1, image_size)
            v, u = torch.meshgrid(grid, grid)
            identity_grid = torch.stack([u, v, torch.ones_like(u)], dim=2).view(1, -1, 3)

        if args.align_source:
            # Align input images using theta
            if imgs.shape[0] > 1:
                raise Exception('works only with single size')
            eye_vector = torch.zeros(theta.shape[0], 1, 3)
            eye_vector[:, :, 2] = 1
            theta_ = torch.cat([theta, eye_vector], dim=1).float()

            # Perform 2x zoom-in compared to default theta
            scale = torch.zeros_like(theta_)
            scale[:, [0, 1], [0, 1]] = args.align_scale
            scale[:, 2, 2] = 1

            theta_ = torch.bmm(theta_, scale)[:, :2]
            align_warp = identity_grid.repeat_interleave(theta_.shape[0], dim=0)
            align_warp = align_warp.bmm(theta_.transpose(1, 2)).view(theta_.shape[0], image_size, image_size, 2)

            source_imgs = F.grid_sample(imgs, align_warp)
            source_masks = F.grid_sample(masks, align_warp)
        else:
            source_imgs, source_masks = imgs, masks

        source_keypoints = torch.from_numpy(self.fa.get_landmarks_from_image(tensor2image(source_imgs[0]))[0])[None]
        output_data_dict = {
            'source_img': source_imgs,
            'source_mask': source_masks,
            'source_keypoints': (source_keypoints / (image_size / 2) - 1),
        }
        return output_data_dict

    def process_driver_img(self, data_dict: dict, driver_image: Image, crop_center=False):
        driver_pose = self.fa.get_landmarks_from_image(np.asarray(driver_image))[0]

        if crop_center or driver_image.size[0] != driver_image.size[1]:
            pose = driver_pose
            center = ((pose.min(0) + pose.max(0)) / 2).round().astype(int)
            size = int(max(pose[:, 0].max() - pose[:, 0].min(), pose[:, 1].max() - pose[:, 1].min()))
            center[1] -= size // 6
            driver_image = driver_image.crop((center[0] - size, center[1] - size, center[0] + size, center[1] + size))

        data_dict['target_img'] = self.data_transform(driver_image)[None]
        data_dict['target_mask'] = torch.zeros_like(data_dict['target_img'])
        landmark_input = np.asarray(driver_image)
        kp_scale = landmark_input.shape[0] // 2
        data_dict['target_keypoints'] = \
        torch.from_numpy(self.fa.get_landmarks_from_image(landmark_input)[0] / kp_scale - 1)[None]
        return data_dict

    def reuse_source_image(self, driver_image):
        pass

    @torch.no_grad()
    def evaluate(self, source_image, driver_image,
                 neutral_pose: bool = False, source_information_for_reuse: dict = None, crop_center=False):
        if source_information_for_reuse is not None:
            data_dict = source_information_for_reuse.get('data_dict')
            if data_dict is None:
                data_dict = self.process_source_for_input_dict(source_image, self.source_transform, crop_center)
        else:
            data_dict = self.process_source_for_input_dict(source_image, self.source_transform, crop_center)
        data_dict = self.process_driver_img(data_dict, driver_image, crop_center)
        for k, v in data_dict.items():
            data_dict[k] = data_dict[k].to(self.device)

        out = self.model(data_dict,
                         neutral_pose=neutral_pose,
                         source_information=source_information_for_reuse)
        out['source_information']['data_dict'] = data_dict
        return out

    def run_example(self):
        src_path = 'data/imgs/taras1.jpg'
        driver_path = 'data/imgs/taras1.jpg'

        driver_img = Image.open(src_path)
        source_image = Image.open(driver_path)
        out = self.evaluate(source_image, driver_img, crop_center=True)
        render_result = tensor2image(out['render_masked'].cpu())
        shape_result = tensor2image(out['pred_target_shape_img'][0].cpu())
        print('Successfully rendered')


def main(args):
    infer = Infer(args)
    infer.run_example()


if __name__ == "__main__":
    print('Start infer!')
    default_modnet_path = 'MODNet/pretrained/modnet_photographic_portrait_matting.ckpt'
    default_model_path = 'data/rome.pth'

    parser = argparse.ArgumentParser(conflict_handler='resolve')
    parser.add_argument('--save_dir', default='.', type=str)
    parser.add_argument('--save_render', default='True', type=args_utils.str2bool, choices=[True, False])
    parser.add_argument('--model_checkpoint', default=default_model_path, type=str)
    parser.add_argument('--modnet_path', default=default_modnet_path, type=str)
    parser.add_argument('--random_seed', default=0, type=int)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--verbose', default='False', type=args_utils.str2bool, choices=[True, False])
    args, _ = parser.parse_known_args()

    parser = importlib.import_module(f'src.rome').ROME.add_argparse_args(parser)

    args = parser.parse_args()
    args.deca_path = 'DECA'

    main(args)
