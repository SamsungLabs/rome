import torch
import numpy as np
from torch import optim
import math
import torch.nn.functional as F



def get_similarity_transform_matrix(scale, rotation, translation):
    eye_matrix = torch.eye(3).type(scale.type()).to(scale.device)

    # Scale transform
    S = eye_matrix.clone()

    S[0, 0] = scale
    S[1, 1] = scale

    # Rotation transform
    R = eye_matrix.clone()

    rotation = rotation.clamp(-math.pi / 2, math.pi)

    rotation_cos = rotation.cos()
    rotation_sin = rotation.sin()

    R[0, 0] = rotation_cos
    R[0, 1] = -rotation_sin

    R[1, 0] = rotation_sin
    R[1, 1] = rotation_cos

    # Translation transform
    T = eye_matrix.clone()

    T[0, 2] = translation[0]
    T[1, 2] = translation[1]

    theta = (S @ R @ T)
    theta = theta[:2]

    return theta


def estimate_similarity_transform(source_points, target_points):
    # Params
    scale = torch.FloatTensor([1]).to(source_points.device).requires_grad_()
    rotation = torch.FloatTensor([0]).to(source_points.device).requires_grad_()
    translation = torch.FloatTensor([0, 0]).to(source_points.device).requires_grad_()

    params = [scale, rotation, translation]
    opt = optim.LBFGS(params)

    transform_args = params

    def closure():
        opt.zero_grad()

        transform_matrix = get_similarity_transform_matrix(*transform_args)
        pred_aligned_points = source_points @ transform_matrix.transpose(0, 1)

        loss = ((pred_aligned_points - target_points) ** 2).mean()
        loss.backward()

        return loss

    for i in range(5):
        opt.step(closure)

    inv_theta = get_similarity_transform_matrix(*transform_args)

    # Align input images using theta
    eye_vector = torch.zeros(1, 3)
    eye_vector[:, 2] = 1
    eye_vector = eye_vector.to(source_points.device)

    inv_theta_ = torch.cat([inv_theta, eye_vector], dim=0)
    theta = inv_theta_.inverse()

    theta = theta[:2]
    inv_theta = inv_theta[:2]

    transform = {
        'theta': theta.detach().cpu().numpy(),
        'inv_theta': inv_theta.detach().cpu().numpy(),
        'scale': scale.detach().cpu().numpy(),
        'rotation': rotation.detach().cpu().numpy(),
        'translation': translation.detach().cpu().numpy()}

    return transform


def calc_ffhq_alignment(lm, size=512, device='cpu'):
    lm_chin = lm[0: 17]  # left-right
    lm_eyebrow_left = lm[17: 22]  # left-right
    lm_eyebrow_right = lm[22: 27]  # left-right
    lm_nose = lm[27: 31]  # top-down
    lm_nostrils = lm[31: 36]  # top-down
    lm_eye_left = lm[36: 42]  # left-clockwise
    lm_eye_right = lm[42: 48]  # left-clockwise
    lm_mouth_outer = lm[48: 60]  # left-clockwise
    lm_mouth_inner = lm[60: 68]  # left-clockwise

    # Calculate auxiliary vectors.
    eye_left = np.mean(lm_eye_left, axis=0)
    eye_right = np.mean(lm_eye_right, axis=0)
    eye_avg = (eye_left + eye_right) * 0.5
    eye_to_eye = eye_right - eye_left
    mouth_left = lm_mouth_outer[0]
    mouth_right = lm_mouth_outer[6]
    mouth_avg = (mouth_left + mouth_right) * 0.5
    eye_to_mouth = mouth_avg - eye_avg

    # Choose oriented crop rectangle.
    x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
    x /= np.hypot(*x)
    x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
    y = np.flipud(x) * [-1, 1]
    c = eye_avg + eye_to_mouth * 0.1

    bbox = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
    bbox = (torch.from_numpy(bbox).float() / size - 0.5) * 2
    bbox = torch.cat([bbox, torch.ones(4, 1)], dim=1)

    gt_bbox = torch.FloatTensor([[-1, -1], [-1, 1], [1, 1], [1, -1]])

    bbox = bbox.to(device)
    gt_bbox = gt_bbox.to(device)

    return estimate_similarity_transform(bbox, gt_bbox)


def tensor2image(tensor):
    image = tensor.detach().cpu().numpy()
    image = image * 255.
    image = np.maximum(np.minimum(image, 255), 0)
    image = image.transpose(1, 2, 0)[:, :, [2, 1, 0]]
    return image.astype(np.uint8).copy()


def preprocess_dict(data_dict, fa, image_size, align_source, align_target, align_scale, device):
    image_size_ = data_dict['source_img'].shape[-1]
    image_size = image_size

    imgs = data_dict['source_img'].cpu()
    masks = data_dict['source_mask'].cpu()

    lm_2d = data_dict['source_keypoints'][0].detach().cpu().numpy()
    transform_ffhq = calc_ffhq_alignment(lm_2d, size=imgs.shape[2])

    theta = torch.FloatTensor(transform_ffhq['theta'])[None]

    if align_source:
        grid = torch.linspace(-1, 1, image_size)
        v, u = torch.meshgrid(grid, grid)
        identity_grid = torch.stack([u, v, torch.ones_like(u)], dim=2).view(1, -1, 3)

    if align_source or align_target:
        # Align input images using theta
        eye_vector = torch.zeros(theta.shape[0], 1, 3)
        eye_vector[:, :, 2] = 1
        theta_ = torch.cat([theta, eye_vector], dim=1).float()

        # Perform 2x zoom-in compared to default theta
        scale = torch.zeros_like(theta_)
        scale[:, [0, 1], [0, 1]] = align_scale
        scale[:, 2, 2] = 1

        theta_ = torch.bmm(theta_, scale)[:, :2]

        align_warp = identity_grid.repeat_interleave(theta_.shape[0], dim=0)
        align_warp = align_warp.bmm(theta_.transpose(1, 2)).view(theta_.shape[0], image_size, image_size, 2)

        if align_source:
            source_imgs_aligned = F.grid_sample(imgs, align_warp)
            source_masks_aligned = F.grid_sample(masks, align_warp)
            source_keypoints = torch.from_numpy(fa.get_landmarks_from_image(tensor2image(source_imgs_aligned[0]))[0])[
                None]
    output_data_dict = {
        'source_img': source_imgs_aligned[0].to(device) if align_source else
        F.interpolate(imgs, size=image_size, mode='bilinear')[0],
        'source_mask': source_masks_aligned[0].to(device) if align_source else
        F.interpolate(masks, size=image_size, mode='bilinear')[0],
        'source_keypoints': (source_keypoints.to(device) / (image_size / 2) - 1)[0],
    }
    return output_data_dict
