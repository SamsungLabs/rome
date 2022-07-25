import numpy as np
import torch
from torch import nn
from torch import optim
from torch.utils import data
import torch.nn.functional as F
import io
from torchvision import transforms
import cv2
import yaml

from torch.utils.data.dataloader import default_collate


def create_batch(indexes, target_dataset):
    res = []
    for idx in indexes:
        res.append(target_dataset[idx])
    return default_collate(res)


def swap_source_target(target_idx_in_dataset, batch, dataset):
    assert len(target_idx_in_dataset) == len(batch['target_img'])
    for i, target_idx in enumerate(target_idx_in_dataset):
        for key in ['target_img', 'target_mask', 'target_keypoints']:
            batch[key][i] = dataset[target_idx][key]
    return batch


def process_black_shape(shape_img):
    black_mask = shape_img == 0.0
    shape_img[black_mask] = 1.0
    shape_img_opa = torch.cat([shape_img, (black_mask.float()).mean(-1, keepdim=True)], dim=-1)
    return shape_img_opa[..., :256, :256]


def process_white_shape(shape_img):
    black_mask = shape_img == 1.0
    shape_img[black_mask] = 1.0
    shape_img_opa = torch.cat([shape_img, (black_mask.float()).mean(-1, keepdim=True)], dim=-1)
    return shape_img_opa[..., :256, :256]


def mask_update_by_vec(masks, thres_list):
    for i, th in enumerate(thres_list):
        masks[i] = masks[i] > th
        masks[i] = mask_errosion(masks[i].numpy() * 255)
    return masks


def obtain_modnet_mask(im: torch.tensor, modnet: nn.Module,
                       ref_size=512, ):
    transes = [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    im_transform = transforms.Compose(transes)
    im = im_transform(im)
    im = im[None, :, :, :]

    im_b, im_c, im_h, im_w = im.shape
    if max(im_h, im_w) < ref_size or min(im_h, im_w) > ref_size:
        if im_w >= im_h:
            im_rh = ref_size
            im_rw = int(im_w / im_h * ref_size)
        elif im_w < im_h:
            im_rw = ref_size
            im_rh = int(im_h / im_w * ref_size)
    else:
        im_rh = im_h
        im_rw = im_w
    im_rw = im_rw - im_rw % 32
    im_rh = im_rh - im_rh % 32
    im = F.interpolate(im, size=(im_rh, im_rw), mode='area')
    _, _, matte = modnet(im, True)
    # resize and save matte
    matte = F.interpolate(matte, size=(im_h, im_w), mode='area')
    matte = matte[0][0].data.cpu().numpy()
    return matte[None]


def mask_errosion(mask):
    kernel = np.ones((9, 9), np.uint8)
    resmask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return torch.from_numpy(resmask / 255)


def save_kp(img_path, kp_save_path, fa):
    kp_input = io.imread(img_path)
    l = fa.get_landmarks(kp_input, return_bboxes=True)
    if l is not None and l[-1] is not None:
        keypoints, _, bboxes = l
        areas = []
        for bbox in bboxes:
            areas.append((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
        i = np.argmax(areas)

        np.savetxt(kp_save_path, keypoints[i])
        return True
    return False


def tensor2image(tensor):
    image = tensor.detach().cpu().numpy()
    image = image * 255.
    image = np.maximum(np.minimum(image, 255), 0)
    image = image.transpose(1, 2, 0)[:, :, [2, 1, 0]]
    return image.astype(np.uint8).copy()
