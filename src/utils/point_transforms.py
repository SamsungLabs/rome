import torch
from torch import optim
import math



def parse_3dmm_param(param):
    """matrix pose form
    param: shape=(trans_dim+shape_dim+exp_dim,), i.e., 62 = 12 + 40 + 10
    """

    # pre-defined templates for parameter
    n = param.shape[0]
    if n == 62:
        trans_dim, shape_dim, exp_dim = 12, 40, 10
    elif n == 72:
        trans_dim, shape_dim, exp_dim = 12, 40, 20
    elif n == 141:
        trans_dim, shape_dim, exp_dim = 12, 100, 29
    else:
        raise Exception(f'Undefined templated param parsing rule')

    R_ = param[:trans_dim].reshape(3, -1)
    R = R_[:, :3]
    offset = R_[:, -1].reshape(3, 1)
    alpha_shp = param[trans_dim:trans_dim + shape_dim].reshape(-1, 1)
    alpha_exp = param[trans_dim + shape_dim:].reshape(-1, 1)

    return R, offset, alpha_shp, alpha_exp

def world_to_camera(pts_world, params):
    R, offset, roi_box, size = params['R'], params['offset'], params['roi_box'], params['size']
    crop_box = params['crop_box'] if 'crop_box' in params.keys() and len(params['crop_box']) else None

    if pts_world.shape[0] < R.shape[0]:
        pts_camera = pts_world.repeat_interleave(R.shape[0] // pts_world.shape[0], dim=0)
    
    elif pts_world.shape[0] > R.shape[0]:
        num_repeats = pts_world.shape[0] // R.shape[0]

        R = R.repeat_interleave(num_repeats, dim=0)
        offset = offset.repeat_interleave(num_repeats, dim=0)
        roi_box = roi_box.repeat_interleave(num_repeats, dim=0)
        size = size.repeat_interleave(num_repeats, dim=0)
        if crop_box is not None:
            crop_box = crop_box.repeat_interleave(num_repeats, dim=0)

        pts_camera = pts_world.clone()

    else:
        pts_camera = pts_world.clone()

    pts_camera[..., 2] += 0.5
    pts_camera *= 2e5

    pts_camera = pts_camera @ R.transpose(1, 2) + offset.transpose(1, 2)

    pts_camera[..., 0] -= 1
    pts_camera[..., 2] -= 1
    pts_camera[..., 1] = 120 - pts_camera[..., 1]
    
    sx, sy, ex, ey = [chunk[..., 0] for chunk in roi_box.split(1, dim=2)]
    scale_x = (ex - sx) / 120
    scale_y = (ey - sy) / 120
    scale_z = (scale_x + scale_y) / 2
    
    pts_camera[..., 0] = pts_camera[..., 0] * scale_x + sx
    pts_camera[..., 1] = pts_camera[..., 1] * scale_y + sy
    pts_camera[..., 2] = pts_camera[..., 2] * scale_z

    pts_camera /= size
    pts_camera[..., 0] -= 0.5
    pts_camera[..., 1] -= 0.5
    pts_camera[..., :2] *= 2

    if crop_box is not None:
        crop_shift_x = (crop_box[..., 0] + crop_box[..., 2]) / 2
        crop_shift_y = (crop_box[..., 1] + crop_box[..., 3]) / 2
        
        pts_camera[..., 0] -= crop_shift_x
        pts_camera[..., 1] -= crop_shift_y
    
        crop_scale_x = (crop_box[..., 2] - crop_box[..., 0]) / 2
        crop_scale_y = (crop_box[..., 3] - crop_box[..., 1]) / 2
        crop_scale_z = (crop_scale_x + crop_scale_y) / 2

        pts_camera[..., 0] /= crop_scale_x
        pts_camera[..., 1] /= crop_scale_y
        pts_camera[..., 2] /= crop_scale_z
    
    return pts_camera

def camera_to_world(pts_camera, params):
    R, offset, roi_box, size = params['R'], params['offset'], params['roi_box'], params['size']
    crop_box = params['crop_box'] if 'crop_box' in params.keys() and len(params['crop_box']) else None

    if pts_camera.shape[0] < R.shape[0]:
        pts_world = pts_camera.repeat_interleave(R.shape[0] // pts_camera.shape[0], dim=0)
    
    elif pts_camera.shape[0] > R.shape[0]:
        num_repeats = pts_camera.shape[0] // R.shape[0]

        R = R.repeat_interleave(num_repeats, dim=0)
        offset = offset.repeat_interleave(num_repeats, dim=0)
        roi_box = roi_box.repeat_interleave(num_repeats, dim=0)
        size = size.repeat_interleave(num_repeats, dim=0)
        if crop_box is not None:
            crop_box = crop_box.repeat_interleave(num_repeats, dim=0)

        pts_world = pts_camera.clone()

    else:
        pts_world = pts_camera.clone()

    if crop_box is not None:
        crop_scale_x = (crop_box[..., 2] - crop_box[..., 0]) / 2
        crop_scale_y = (crop_box[..., 3] - crop_box[..., 1]) / 2
        crop_scale_z = (crop_scale_x + crop_scale_y) / 2

        pts_world[..., 0] *= crop_scale_x
        pts_world[..., 1] *= crop_scale_y
        pts_world[..., 2] *= crop_scale_z

        crop_shift_x = (crop_box[..., 0] + crop_box[..., 2]) / 2
        crop_shift_y = (crop_box[..., 1] + crop_box[..., 3]) / 2

        pts_world[..., 0] += crop_shift_x
        pts_world[..., 1] += crop_shift_y
        
    pts_world[..., :2] /= 2
    pts_world[..., 0] += 0.5
    pts_world[..., 1] += 0.5
    pts_world *= size

    sx, sy, ex, ey = [chunk[..., 0] for chunk in roi_box.split(1, dim=2)]
    scale_x = (ex - sx) / 120
    scale_y = (ey - sy) / 120
    scale_z = (scale_x + scale_y) / 2
    
    pts_world[..., 0] = (pts_world[..., 0] - sx) / scale_x
    pts_world[..., 1] = (pts_world[..., 1] - sy) / scale_y
    pts_world[..., 2] = pts_world[..., 2] / scale_z

    pts_world[..., 0] += 1
    pts_world[..., 2] += 1
    pts_world[..., 1] = -(pts_world[..., 1] - 120)
    
    pts_world = (pts_world - offset.transpose(1, 2)) @ torch.linalg.inv(R.transpose(1, 2))
    
    pts_world /= 2e5
    pts_world[..., 2] -= 0.5
    
    return pts_world

###############################################################################

def align_ffhq_with_zoom(pts_camera, params, zoom_factor=0.6):
    R, offset = params['theta'].split([2, 1], dim=2)
    crop_box = params['crop_box'] if 'crop_box' in params.keys() and len(params['crop_box']) else None

    if pts_camera.shape[0] != R.shape[0]:
        pts_camera = pts_camera.repeat_interleave(R.shape[0], dim=0)
    else:
        pts_camera = pts_camera.clone()
    
    pts_camera = pts_camera @ R.transpose(1, 2) + offset.transpose(1, 2)

    # Zoom into face
    pts_camera *= zoom_factor

    if crop_box is not None:
        crop_shift_x = (crop_box[..., 0] + crop_box[..., 2]) / 2
        crop_shift_y = (crop_box[..., 1] + crop_box[..., 3]) / 2

        pts_camera[..., 0] -= crop_shift_x
        pts_camera[..., 1] -= crop_shift_y
        
        crop_scale_x = (crop_box[..., 2] - crop_box[..., 0]) / 2
        crop_scale_y = (crop_box[..., 3] - crop_box[..., 1]) / 2
        
        pts_camera[..., 0] /= crop_scale_x
        pts_camera[..., 1] /= crop_scale_y
    
    return pts_camera

###############################################################################

def get_transform_matrix(scale, rotation, translation):
    b = scale.shape[0]
    dtype = scale.dtype
    device = scale.device

    eye_matrix = torch.eye(4, dtype=dtype, device=device)[None].repeat_interleave(b, dim=0)

    # Scale transform
    S = eye_matrix.clone()

    if scale.shape[1] == 3:
        S[:, 0, 0] = scale[:, 0]
        S[:, 1, 1] = scale[:, 1]
        S[:, 2, 2] = scale[:, 2]
    else:
        S[:, 0, 0] = scale[:, 0]
        S[:, 1, 1] = scale[:, 0]
        S[:, 2, 2] = scale[:, 0]

    # Rotation transform
    R = eye_matrix.clone()

    rotation = rotation.clamp(-math.pi/2, math.pi)

    yaw, pitch, roll = rotation.split(1, dim=1)
    yaw, pitch, roll = yaw[:, 0], pitch[:, 0], roll[:, 0] # squeeze angles
    yaw_cos = yaw.cos()
    yaw_sin = yaw.sin()
    pitch_cos = pitch.cos()
    pitch_sin = pitch.sin()
    roll_cos = roll.cos()
    roll_sin = roll.sin()

    R[:, 0, 0] = yaw_cos * pitch_cos
    R[:, 0, 1] = yaw_cos * pitch_sin * roll_sin - yaw_sin * roll_cos
    R[:, 0, 2] = yaw_cos * pitch_sin * roll_cos + yaw_sin * roll_sin

    R[:, 1, 0] = yaw_sin * pitch_cos
    R[:, 1, 1] = yaw_sin * pitch_sin * roll_sin + yaw_cos * roll_cos
    R[:, 1, 2] = yaw_sin * pitch_sin * roll_cos - yaw_cos * roll_sin

    R[:, 2, 0] = -pitch_sin
    R[:, 2, 1] = pitch_cos * roll_sin
    R[:, 2, 2] = pitch_cos * roll_cos

    # Translation transform
    T = eye_matrix.clone()

    T[:, 0, 3] = translation[:, 0]
    T[:, 1, 3] = translation[:, 1]
    T[:, 2, 3] = translation[:, 2]

    theta = S @ R @ T

    return theta

def estimate_transform_from_keypoints(keypoints, aligned_keypoints, dilation=True, shear=False):
    b, n = keypoints.shape[:2]
    device = keypoints.device
    dtype = keypoints.dtype

    keypoints = keypoints.to(device)
    aligned_keypoints = aligned_keypoints.to(device)

    keypoints = torch.cat([keypoints, torch.ones(b, n, 1, device=device, dtype=dtype)], dim=2)

    if not dilation and not shear:
        # scale, yaw, pitch, roll, dx, dy, dz
        param = torch.tensor([[1,   0, 0, 0,   0, 0, 0]], device=device, dtype=dtype)

        scale, rotation, translation = param.repeat_interleave(b, dim=0).split([1, 3, 3], dim=1)
        params = [scale, rotation, translation]

    elif dilation and not shear:
        # scale_x, scale_y, scale_z, yaw, pitch, roll, dx, dy, dz
        param = torch.tensor([[1, 1, 1,   0, 0, 0,   0, 0, 0]], device=device, dtype=dtype)

        scale, rotation, translation = param.repeat_interleave(b, dim=0).split([3, 3, 3], dim=1)
        params = [scale, rotation, translation]

    elif dilation and shear:
        # full affine matrix
        theta = torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]], device=device, dtype=dtype)
        theta = theta[None].repeat_interleave(b, dim=0)
        params = [theta]

    # Solve for a given transform
    params = [p.clone().requires_grad_() for p in params]

    opt = optim.LBFGS(params)

    def closure():
        opt.zero_grad()

        if not shear:
            theta = get_transform_matrix(*params)[:, :3]
        else:
            theta = params[0]
        
        pred_aligned_keypoints = keypoints @ theta.transpose(1, 2)

        loss = ((pred_aligned_keypoints - aligned_keypoints)**2).mean()
        loss.backward()

        return loss

    for i in range(5):
        opt.step(closure)

    if not shear:
        theta = get_transform_matrix(*params).detach()
    else:
        theta = params[0].detach()

        eye = torch.zeros(b, 4, device=device, dtype=dtype)
        eye[:, 2] = 1

        theta = torch.cat([theta, eye], dim=1)

    return theta, params