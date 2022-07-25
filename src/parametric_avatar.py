import torch
import torch.nn as nn
from torch.nn import functional as F
import sys
import os
from pytorch3d.structures import Meshes
import pickle as pkl

from DECA.decalib.utils import util
from DECA.decalib.models.lbs import blend_shapes, batch_rodrigues
from DECA.decalib.deca import DECA
import DECA.decalib.utils.config as config

from src.utils import harmonic_encoding
from src.utils.params import batch_cont2matrix
from src.utils.processing import create_regressor


class ParametricAvatar(DECA):
    def __init__(self,
                 image_size,
                 deca_path=None,
                 use_scalp_deforms=False,
                 use_neck_deforms=False,
                 subdivide_mesh=False,
                 use_details=False,
                 use_tex=False,
                 external_params=None,
                 device=torch.device('cpu'),
                 ):

        self.flame_config = cfg = config.cfg
        config.cfg.deca_dir = deca_path

        cfg.model.topology_path = os.path.join(cfg.deca_dir, 'data', 'head_template.obj')
        cfg.model.addfiles_path = external_params.rome_data_dir
        cfg.model.flame_model_path = os.path.join(cfg.deca_dir, 'data', 'generic_model.pkl')
        cfg.model.flame_lmk_embedding_path = os.path.join(cfg.deca_dir, 'data', 'landmark_embedding.npy')
        cfg.model.face_mask_path = os.path.join(cfg.deca_dir, 'data', 'uv_face_mask.png')
        cfg.model.face_eye_mask_path = os.path.join(cfg.deca_dir, 'data', 'uv_face_eye_mask.png')
        cfg.pretrained_modelpath = os.path.join(cfg.deca_dir, 'data', 'deca_model.tar')
        cfg.model.use_tex = use_tex
        cfg.dataset.image_size = image_size
        cfg.model.uv_size = image_size

        sys.path.append(os.path.join(deca_path, 'data'))
        super().__init__(cfg, device=device)

        self.device = device
        self.image_size = image_size
        self.use_scalp_deforms = use_scalp_deforms
        self.use_neck_deforms = use_neck_deforms
        self.subdivide_mesh = subdivide_mesh
        self.external_params = external_params.__dict__
        self.use_tex = use_tex
        self.use_details = use_details
        self.finetune_flame_encoder = False
        self.train_flame_encoder_from_scratch = False
        self.mask_for_face = None
        # Modify default FLAME config
        flame_config = cfg
        flame_config.model.uv_size = image_size

        self.cfg = flame_config

        grid_s = torch.linspace(0, 1, 224)
        v, u = torch.meshgrid(grid_s, grid_s)
        self.register_buffer('identity_grid_deca', torch.stack([u, v], dim=2)[None], persistent=False)

        grid_s = torch.linspace(-1, 1, image_size)
        v, u = torch.meshgrid(grid_s, grid_s)
        self.register_buffer('uv_grid', torch.stack([u, v], dim=2)[None])
        harmonized_uv_grid = harmonic_encoding.harmonic_encoding(self.uv_grid, num_encoding_functions=6)
        self.register_buffer('harmonized_uv_grid', harmonized_uv_grid[None])

        # Load scalp-related data
        external_params.data_dir = deca_path

        self.hair_list = pkl.load(open(f'{external_params.rome_data_dir}/hair_list.pkl', 'rb'))
        self.neck_list = pkl.load(open(f'{external_params.rome_data_dir}/neck_list.pkl', 'rb'))

        self.hair_edge_list = pkl.load(open(f'{external_params.rome_data_dir}/hair_edge_list.pkl', 'rb'))
        self.neck_edge_list = pkl.load(open(f'{external_params.rome_data_dir}/neck_edge_list.pkl', 'rb'))

        true_uvcoords = torch.load(f'{external_params.rome_data_dir}/vertex_uvcoords.pth')
        self.render = self.render.to(device)

        self.register_buffer('hair_faces', torch.load(f'{external_params.rome_data_dir}/hair_faces.pth'))
        self.register_buffer('neck_faces', torch.load(f'{external_params.rome_data_dir}/neck_faces.pth'))

        self.deforms_mask = torch.zeros(1, 5023, 1, device=device)

        self.hair_mask = torch.zeros(1, 5023, 1, device=device)
        self.neck_mask = torch.zeros(1, 5023, 1, device=device)
        self.face_mask = torch.zeros(1, 5023, 1, device=device)

        if self.use_scalp_deforms:
            self.deforms_mask[:, self.hair_list] = 1.0
        if self.use_neck_deforms:
            self.deforms_mask[:, self.neck_list] = 1.0

        self.hair_mask[:, self.hair_edge_list] = 1.0
        self.neck_mask[:, self.neck_edge_list] = 1.0
        self.true_uvcoords = true_uvcoords.to(device)

        def rm_from_list(a, b):
            return list(set(a) - set(b))

        # TODO save list to pickle
        hard_not_deform_list = [3587, 3594, 3595, 3598, 3600, 3630, 3634,
                                3635, 3636, 3637, 3643, 3644, 3646, 3649,
                                3650, 3652, 3673, 3676, 3677, 3678, 3679,
                                3680, 3681, 3685, 3691, 3693, 3695, 3697,
                                3698, 3701, 3703, 3707, 3709, 3713, 3371,
                                3372, 3373, 3374, 3375, 3376, 3377, 3378,
                                3379, 3382, 3383, 3385, 3387, 3389, 3392,
                                3393, 3395, 3397, 3399, 3413, 3414, 3415,
                                3416, 3417, 3418, 3419, 3420, 3421, 3422,
                                3423, 3424, 3441, 3442, 3443, 3444, 3445,
                                3446, 3447, 3448, 3449, 3450, 3451, 3452,
                                3453, 3454, 3455, 3456, 3457, 3458, 3459,
                                3460, 3461, 3462, 3463, 3494, 3496, 3510,
                                3544, 3562, 3578, 3579, 3581, 3583]
        exclude_list = [3382, 3377, 3378, 3379, 3375, 3374, 3544, 3494, 3496,
                        3462, 3463, 3713, 3510, 3562, 3372, 3373, 3376, 3371]

        hard_not_deform_list = list(rm_from_list(hard_not_deform_list, exclude_list))

        # if self.use_neck_deforms and self.external_params.get('updated_neck_mask', False):
        self.deforms_mask[:, hard_not_deform_list] = 0.0
        self.face_mask[:, self.hair_edge_list] = 0.0
        self.face_mask[:, self.neck_edge_list] = 0.0

        self.register_buffer('faces_hair_mask', util.face_vertices(self.hair_mask, self.render.faces))
        self.register_buffer('faces_neck_mask', util.face_vertices(self.neck_mask, self.render.faces))
        self.register_buffer('faces_face_mask', util.face_vertices(self.face_mask, self.render.faces))

        if self.external_params.get('deform_face_scale_coef', 0.0) > 0.0:
            self.face_deforms_mask = torch.ones_like(self.deforms_mask).cpu() / \
                                     self.external_params.get('deform_face_scale_coef')
            self.face_deforms_mask[:, self.neck_list] = 1.0
            self.face_deforms_mask[:, self.hair_list] = 1.0

            if self.external_params.get('deform_face'):
                # put original deformation ofr face zone, scaling applied only for ears & eyes
                verts_uvs = self.true_uvcoords
                face_vertices = F.grid_sample(self.uv_face_mask, verts_uvs[None]).squeeze() > 0.0
                self.face_deforms_mask[:, face_vertices] = 1.0
        else:
            self.face_deforms_mask = None

        # Create distill model
        if self.external_params.get('use_distill', False):
            self._setup_linear_model()

    def _setup_linear_model(self):
        n_online_params = self.flame_config.model.n_exp + self.flame_config.model.n_pose + self.flame_config.model.n_cam
        self.hair_basis_reg = create_regressor('resnet50', self.external_params.get('n_scalp', 60))
        state_dict = torch.load(self.external_params['path_to_linear_hair_model'], map_location='cpu')

        self.hair_basis_reg.load_state_dict(state_dict)
        self.hair_basis_reg.eval()

        if self.external_params.get('use_mobile_version', False):
            self.online_regressor = create_regressor('mobilenet_v2', n_online_params)
            state_dict = torch.load(self.external_params['path_to_mobile_model'], map_location='cpu')

            self.online_regressor.load_state_dict(state_dict)
            self.online_regressor.eval()

        # Load basis
        self.u_full = torch.load(os.path.join(f'{self.external_params["rome_data_dir"]}', 'u_full.pt')).to(self.device)
        # Create mean deforms
        self.mean_deforms = torch.load(
            os.path.join(f'{self.external_params["rome_data_dir"]}', 'mean_deform.pt'),
            map_location='cpu').to(self.device)

        self.mean_hair = torch.zeros(5023, 3, device=self.mean_deforms.device)
        self.mean_neck = torch.zeros(5023, 3, device=self.mean_deforms.device)
        self.mean_hair[self.hair_list] = self.mean_deforms[self.hair_list]
        self.mean_neck[self.neck_list] = self.mean_deforms[self.neck_list]

    @staticmethod
    def calc_crop(l_old, r_old, t_old, b_old, scale):
        size = (r_old - l_old + b_old - t_old) / 2 * 1.1
        size *= scale

        center = torch.stack([r_old - (r_old - l_old) / 2.0,
                              b_old - (b_old - t_old) / 2.0], dim=1)

        l_new = center[:, 0] - size / 2
        r_new = center[:, 0] + size / 2
        t_new = center[:, 1] - size / 2
        b_new = center[:, 1] + size / 2

        l_new = l_new[:, None, None]
        r_new = r_new[:, None, None]
        t_new = t_new[:, None, None]
        b_new = b_new[:, None, None]

        return l_new, r_new, t_new, b_new

    def preprocess_image(self, image, keypoints) -> torch.Tensor:
        old_size = image.shape[2]

        keypoints = (keypoints + 1) / 2

        l_old = torch.min(keypoints[..., 0], dim=1)[0]
        r_old = torch.max(keypoints[..., 0], dim=1)[0]
        t_old = torch.min(keypoints[..., 1], dim=1)[0]
        b_old = torch.max(keypoints[..., 1], dim=1)[0]

        l_new, r_new, t_new, b_new = self.calc_crop(l_old, r_old, t_old, b_old, scale=1.25)

        warp_to_crop = self.identity_grid_deca.clone().repeat_interleave(image.shape[0], dim=0)

        warp_to_crop[..., 0] = warp_to_crop[..., 0] * (r_new - l_new) + l_new
        warp_to_crop[..., 1] = warp_to_crop[..., 1] * (b_new - t_new) + t_new
        warp_to_crop = (warp_to_crop - 0.5) * 2

        if not hasattr(self, 'identity_grid_input'):
            grid_s = torch.linspace(0, 1, old_size)
            v, u = torch.meshgrid(grid_s, grid_s)
            device = warp_to_crop.device
            dtype = warp_to_crop.type()
            self.register_buffer('identity_grid_input', torch.stack([u, v], dim=2)[None].cpu().type(dtype),
                                 persistent=False)

        crop_bbox = [l_new[..., 0], t_new[..., 0], r_new[..., 0], b_new[..., 0]]

        return F.grid_sample(image, warp_to_crop.float()), warp_to_crop.float(), crop_bbox

    def encode(self, images, crop_bbox):
        batch_size = images.shape[0]

        if self.finetune_flame_encoder or self.train_flame_encoder_from_scratch:
            e_flame_code_parameters = self.E_flame(images)
        else:
            with torch.no_grad():
                e_flame_code_parameters = self.E_flame(images)

        if not self.train_flame_encoder_from_scratch:
            codedict = self.decompose_code(e_flame_code_parameters, self.param_dict)
        else:
            codedict = e_flame_code_parameters

        codedict['images'] = images

        codedict['pose_vec'] = codedict['pose']

        pose = codedict['pose'].view(batch_size, -1, 3)
        angle = torch.norm(pose + 1e-8, dim=2, keepdim=True)
        rot_dir = pose / angle
        codedict['pose_rot_mats'] = batch_rodrigues(
            torch.cat([angle, rot_dir], dim=2).view(-1, 4)
        ).view(batch_size, pose.shape[1], 3, 3)  # cam & jaw | jaw | jaw & eyes

        if 'cont_pose' in codedict.keys():
            pose_rot_mats = batch_cont2matrix(codedict['cont_pose'])
            codedict['pose_rot_mats'] = torch.cat([pose_rot_mats, codedict['pose_rot_mats']], dim=1)  # cam | cam & neck

        if self.use_details:
            detailcode = self.E_detail(images)
            codedict['detail'] = detailcode

        # Modify camera params to include uncrop using crop_bbox
        if crop_bbox is not None:
            crop_w = crop_bbox[2] - crop_bbox[0]
            crop_h = crop_bbox[3] - crop_bbox[1]
            crop_s = (crop_w + crop_h) / 2
            crop_s = crop_s[:, 0]

            cam = codedict['cam']
            scale_orig = cam[..., 0]
            scale_crop = scale_orig * crop_s
            crop_y = cam[..., 1] + 1 / scale_orig + (2 * crop_bbox[0][:, 0] - 1) / scale_crop
            crop_z = cam[..., 2] - 1 / scale_orig - (2 * crop_bbox[1][:, 0] - 1) / scale_crop
            cam_crop = torch.stack([scale_crop, crop_y, crop_z], dim=-1)
            codedict['cam'] = cam_crop

        return codedict

    def get_parametric_vertices(self, codedict, neutral_pose):
        cam_rot_mats = codedict['pose_rot_mats'][:, :1]
        batch_size = cam_rot_mats.shape[0]

        eye_rot_mats = neck_rot_mats = None

        if codedict['pose_rot_mats'].shape[1] >= 3:
            neck_rot_mats = codedict['pose_rot_mats'][:, 1:2]
            jaw_rot_mats = codedict['pose_rot_mats'][:, 2:3]
        else:
            jaw_rot_mats = codedict['pose_rot_mats'][:, 1:2]

        if codedict['pose_rot_mats'].shape[1] == 4:
            eye_rot_mats = codedict['pose_rot_mats'][:, 3:]

        # Use zero global camera pose inside FLAME fitting class
        cam_rot_mats_ = torch.eye(3).to(cam_rot_mats.device).expand(batch_size, 1, 3, 3)
        # Shaped vertices
        verts_neutral = self.flame.reconstruct_shape(codedict['shape'])

        # Visualize shape
        default_cam = torch.zeros_like(codedict['cam'])[:, :3]  # default cam has orthogonal projection
        default_cam[:, :1] = 5.0

        _, verts_neutral_frontal, _ = util.batch_orth_proj(verts_neutral, default_cam, flame=self.flame)
        shape_neutral_frontal = self.render.render_shape(verts_neutral, verts_neutral_frontal)

        # Apply expression and pose
        if neutral_pose:
            verts_parametric, rot_mats, root_joint, neck_joint = self.flame.reconstruct_exp_and_pose(
                verts_neutral, torch.zeros_like(codedict['exp']))
        else:
            verts_parametric, rot_mats, root_joint, neck_joint = self.flame.reconstruct_exp_and_pose(
                verts_neutral,
                codedict['exp'],
                cam_rot_mats_,
                neck_rot_mats,
                jaw_rot_mats,
                eye_rot_mats
            )

        # Add neck rotation
        if neck_rot_mats is not None:
            neck_rot_mats = neck_rot_mats.repeat_interleave(verts_parametric.shape[1], dim=1)
            verts_parametric = verts_parametric - neck_joint[:, None]
            verts_parametric = torch.matmul(neck_rot_mats.transpose(2, 3), verts_parametric[..., None])[..., 0]
            verts_parametric = verts_parametric + neck_joint[:, None]

        # Visualize exp verts
        _, verts_parametric_frontal, _ = util.batch_orth_proj(verts_parametric, default_cam, flame=self.flame)
        shape_parametric_frontal = self.render.render_shape(verts_parametric, verts_parametric_frontal)

        return cam_rot_mats, root_joint, verts_parametric, shape_neutral_frontal, shape_parametric_frontal

    def estimate_texture(self, source_image: torch.Tensor, source_mask: torch.Tensor,
                         texture_encoder: torch.nn.Module) -> torch.Tensor:
        autoenc_inputs = torch.cat([source_image, source_mask], dim=1)
        neural_texture = texture_encoder(autoenc_inputs)
        if neural_texture.shape[-1] != 256:
            neural_texture = F.interpolate(neural_texture, (256, 256))

        return neural_texture

    @torch.no_grad()
    def deform_source_mesh(self, verts_parametric, neural_texture, deformer_nets):
        unet_deformer = deformer_nets['unet_deformer']
        vertex_deformer = deformer_nets['mlp_deformer']
        batch_size = verts_parametric.shape[0]

        verts_uvs = self.true_uvcoords[:, :, None, :2]  # 1 x V x 1 x 2

        verts_uvs = verts_uvs.repeat_interleave(batch_size, dim=0)

        # bs x 3 x H x W
        verts_texture = self.render.world2uv(verts_parametric) * 5

        enc_verts_texture = harmonic_encoding.harmonic_encoding(verts_texture.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        deform_unet_inputs = torch.cat([neural_texture.detach(), enc_verts_texture.detach()], dim=1)

        uv_deformations_codes = unet_deformer(deform_unet_inputs)

        mlp_input_uv_z = F.grid_sample(uv_deformations_codes, verts_uvs, align_corners=False)[..., 0].permute(0, 2, 1)

        mlp_input_uv = F.grid_sample(self.uv_grid.repeat(batch_size, 1, 1, 1).permute(0, 3, 1, 2),
                                     verts_uvs, align_corners=False)[..., 0]
        mlp_input_uv = harmonic_encoding.harmonic_encoding(mlp_input_uv.permute(0, 2, 1), 6, )

        mlp_input_uv_deformations = torch.cat([mlp_input_uv_z, mlp_input_uv], dim=-1)

        if self.mask_for_face is None:
            self.mask_for_face = F.grid_sample((F.interpolate(self.uv_face_eye_mask.repeat(batch_size, 1, 1, 1)
                                                              , uv_deformations_codes.shape[-2:])),
                                               verts_uvs, align_corners=False)[..., 0].permute(0, 2, 1) > 0.5

        bs, v, ch = mlp_input_uv_deformations.shape
        deformation_project = vertex_deformer(mlp_input_uv_deformations.view(-1, ch))
        predefined_mask = None
        if predefined_mask is not None:
            deforms = torch.tanh(deformation_project.view(bs, -1, 3).contiguous())
            verts_empty_deforms = torch.zeros(batch_size, verts_uvs.shape[1], 3,
                                              dtype=verts_uvs.dtype, device=verts_uvs.device)
            verts_empty_deforms = verts_empty_deforms.scatter_(1, predefined_mask[None, :, None].expand(bs, -1, 3),
                                                               deforms)
            # self.deforms_mask.nonzero()[None].repeat(bs, 1, 1), deforms)
            verts_deforms = verts_empty_deforms
        else:
            verts_deforms = torch.tanh(deformation_project.view(bs, v, 3).contiguous())

        if self.mask_for_face is not None and self.external_params.get('deform_face_tightness', 0.0) > 0.0:
            #      We slightly deform areas along the face
            self.deforms_mask[self.mask_for_face[[0]]] = self.external_params.get('deform_face_tightness', 0.0)

        verts_deforms = verts_deforms * self.deforms_mask
        return verts_deforms

    def add_details(self, target_codedict, verts, verts_target, uvs, shape_target):
        bs = verts.shape[0]
        uv_z = self.D_detail(
            torch.cat(
                [
                    target_codedict['pose_vec'][:, 3:],
                    target_codedict['exp'],
                    target_codedict['detail']
                ], dim=1)
        )

        vertex_normals = util.vertex_normals(verts, self.render.faces.expand(bs, -1, -1))
        uv_detail_normals = self.displacement2normal(uv_z, verts, vertex_normals)
        detail_normal_images = F.grid_sample(uv_detail_normals, uvs, align_corners=False)

        face_mask_deca = F.grid_sample(self.uv_face_mask.repeat_interleave(bs, dim=0), uvs)
        detail_shape_target = self.render.render_shape(verts, verts_target,
                                                       detail_normal_images=detail_normal_images)
        detail_shape_target = detail_shape_target * face_mask_deca + shape_target * (1 - face_mask_deca)
        return detail_shape_target

    @torch.no_grad()
    def decode(self, target_codedict, neutral_pose,
               deformer_nets=None, verts_deforms=None, neural_texture=None):
        batch_size = target_codedict['batch_size']

        # Visualize shape
        default_cam = torch.zeros_like(target_codedict['cam'])[:, :3]  # default cam has orthogonal projection
        default_cam[:, :1] = 5.0

        cam_rot_mats, root_joint, verts_template, \
        shape_neutral_frontal, shape_parametric_frontal = self.get_parametric_vertices(target_codedict, neutral_pose)

        if deformer_nets['mlp_deformer']:
            verts_deforms = self.deform_source_mesh(verts_template, neural_texture, deformer_nets)

            # Obtain visualized frontal vertices
            faces = self.render.faces.expand(batch_size, -1, -1)

            vertex_normals = util.vertex_normals(verts_template, faces)
            verts_deforms = verts_deforms * vertex_normals

        verts_final = verts_template + verts_deforms

        _, verts_final_frontal, _ = util.batch_orth_proj(verts_final, default_cam, flame=self.flame)
        shape_final_frontal = self.render.render_shape(verts_final, verts_final_frontal)

        _, verts_target, landmarks_target = util.batch_orth_proj(
            verts_final.clone(), target_codedict['cam'],
            root_joint, cam_rot_mats, self.flame)

        shape_target = self.render.render_shape(verts_final, verts_target)
        _, verts_final_posed, _ = util.batch_orth_proj(verts_final.clone(), default_cam, flame=self.flame)
        shape_final_posed = self.render.render_shape(verts_final, verts_final_posed)
        hair_neck_face_mesh_faces = torch.cat([self.faces_hair_mask, self.faces_neck_mask, self.faces_face_mask],
                                              dim=-1)

        ops = self.render(verts_final, verts_target, face_masks=hair_neck_face_mesh_faces)

        alphas = ops['alpha_images']
        soft_alphas = ops['soft_alpha_images']
        uvs = ops['uvcoords_images'].permute(0, 2, 3, 1)[..., :2]
        normals = ops['normal_images']
        coords = ops['vertice_images']

        # Grid sample outputs
        rendered_texture = None
        if neural_texture is not None:
            rendered_texture = F.grid_sample(neural_texture, uvs, mode='bilinear')

        if self.use_details:
            detail_shape_target = self.add_details(target_codedict, verts_final, verts_target, uvs, shape_target)

        opdict = {
            'rendered_texture': rendered_texture,
            'rendered_texture_detach_geom': None,
            'vertices': verts_final,
            'vertices_target': verts_target,
            'vertices_deforms': verts_deforms,
            'landmarks': landmarks_target,
            'alphas': alphas,
            'coords': coords,
            'normals': normals,
            'neural_texture': neural_texture,
        }

        if self.use_tex:
            opdict['flametex_images'] = ops.get('images')

        visdict = {
            'shape_neutral_frontal_images': shape_neutral_frontal,
            'shape_parametric_frontal_images': shape_parametric_frontal,
            'shape_final_frontal_images': shape_final_frontal,
            'shape_final_posed_images': shape_final_posed,
            'shape_images': shape_target,
            'shape_target_displ_images': detail_shape_target if self.use_details else None
        }
        for k, v in visdict.items():
            if v is None: continue
            visdict[k] = F.interpolate(v, size=self.image_size, mode='bilinear')

        return opdict, visdict

    def encode_by_distill(self, target_image):
        delta_blendshapes = blend_shapes(self.hair_basis_reg(target_image),
                                         self.u_full.reshape(5023, 3, -1)) + self.mean_deforms
        return delta_blendshapes

    def forward(
            self,
            source_image,
            source_mask,
            source_keypoints,
            target_image,
            target_keypoints,
            neutral_pose=False,
            deformer_nets=None,
            neural_texture=None,
            source_information: dict = {},
    ) -> dict:
        source_image_crop, _, source_crop_bbox = self.preprocess_image(source_image, source_keypoints)
        target_image_crop, _, target_crop_bbox = self.preprocess_image(target_image, target_keypoints)

        if neural_texture is None:
            source_codedict = self.encode(source_image_crop, source_crop_bbox)
            source_information = {}
            source_information['shape'] = source_codedict['shape']
            source_information['codedict'] = source_codedict

        target_codedict = self.encode(target_image_crop, target_crop_bbox)

        target_codedict['shape'] = source_information.get('shape')
        target_codedict['batch_size'] = target_image.shape[0]
        delta_blendshapes = None

        if neural_texture is None:
            neural_texture = self.estimate_texture(source_image, source_mask, deformer_nets['neural_texture_encoder'])
            source_information['neural_texture'] = neural_texture

        if self.external_params['use_distill']:
            delta_blendshapes = self.encode_by_distill(source_image)

            if self.external_params.get('use_mobile_version', False):
                output2 = self.online_regressor(target_image)
                codedict_ = {}
                full_online = self.flame_config.model.n_exp + self.flame_config.model.n_pose + self.flame_config.model.n_cam
                codedict_['shape'] = target_codedict['shape']
                codedict_['batch_size'] = target_codedict['batch_size']
                codedict_['exp'] = output2[:, : self.flame_config.model.n_exp]
                codedict_['pose'] = output2[:,
                                    self.flame_config.model.n_exp: self.flame_config.model.n_exp + self.flame_config.model.n_pose]
                codedict_['cam'] = output2[:, full_online - self.flame_config.model.n_cam:full_online]
                pose = codedict_['pose'].view(codedict_['batch_size'], -1, 3)
                angle = torch.norm(pose + 1e-8, dim=2, keepdim=True)
                rot_dir = pose / angle
                codedict_['pose_rot_mats'] = batch_rodrigues(
                    torch.cat([angle, rot_dir], dim=2).view(-1, 4)
                ).view(codedict_['batch_size'], pose.shape[1], 3, 3)
                target_codedict = codedict_

            deformer_nets = {
                'unet_deformer': None,
                'mlp_deformer': None,
            }

        opdict, visdict = self.decode(
            target_codedict,
            neutral_pose,
            deformer_nets,
            neural_texture=neural_texture,
            verts_deforms=delta_blendshapes
        )

        outputs = {
            'rendered_texture': opdict['rendered_texture'],
            'source_neural_texture': opdict['neural_texture'],
            'pred_target_normal': opdict['normals'],
            'pred_target_shape_img': visdict['shape_images'],
            'pred_target_shape_displ_img': visdict.get('shape_target_displ_images'),
            'pred_target_keypoints': opdict['landmarks'],
            'vertices': opdict['vertices'],
            'pred_target_hard_mask': opdict['alphas'],
            'vertices_target': opdict['vertices_target'],
            'target_shape_final_posed_img': visdict['shape_final_posed_images'],
            'target_shape_final_frontal_img': visdict['shape_final_frontal_images'],
            'target_shape_parametric_frontal_img': visdict['shape_parametric_frontal_images'],
            'target_shape_neutral_frontal_img': visdict['shape_neutral_frontal_images'],
            'source_information': source_information,
        }

        return outputs
