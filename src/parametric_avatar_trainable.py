from collections import OrderedDict

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
from src.parametric_avatar import ParametricAvatar


class ParametricAvatarTrainable(ParametricAvatar):

    def estimate_texture(self, source_image: torch.Tensor, source_mask: torch.Tensor,
                         texture_encoder: torch.nn.Module) -> torch.Tensor:
        autoenc_inputs = torch.cat([source_image, source_mask], dim=1)
        neural_texture = texture_encoder(autoenc_inputs)
        if neural_texture.shape[-1] != 256:
            neural_texture = F.interpolate(neural_texture, (256, 256))

        return neural_texture

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

        vert_texture_codes = torch.tanh(uv_deformations_codes[:, :3])
        vert_texture_coord_inp = torch.tanh(enc_verts_texture[:, :3])

        verts_deforms = verts_deforms * self.deforms_mask
        return verts_deforms, vert_texture_codes, vert_texture_coord_inp, uv_deformations_codes

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

    def decode(self, target_codedict, neutral_pose,
               deformer_nets=None, verts_deforms=None, neural_texture=None):
        images = target_codedict['images']
        batch_size = images.shape[0]

        # Visualize shape
        default_cam = torch.zeros_like(target_codedict['cam'])[:, :3]  # default cam has orthogonal projection
        default_cam[:, :1] = 5.0

        cam_rot_mats, root_joint, verts_template, \
        shape_neutral_frontal, shape_parametric_frontal = self.get_parametric_vertices(target_codedict, neutral_pose)

        if deformer_nets['mlp_deformer']:
            verts_deforms, vert_texture_codes, \
            vert_texture_coord_inp, uv_deformations_codes = self.deform_source_mesh(verts_template, neural_texture, deformer_nets)

            # Obtain visualized frontal vertices
            faces = self.render.faces.expand(batch_size, -1, -1)

            vertex_normals = util.vertex_normals(verts_template, faces)
            verts_deforms = verts_deforms * vertex_normals

        verts_final = verts_template + verts_deforms
        verts_deforms_texture = self.render.world2uv(verts_deforms) * 5

        _, verts_final_frontal, _ = util.batch_orth_proj(verts_final, default_cam, flame=self.flame)
        shape_final_frontal = self.render.render_shape(verts_final, verts_final_frontal)

        verts_target_hair, verts_final_hair = None, None
        verts_final_neck, verts_target_neck = None, None
        soft_alphas_hair_only, soft_alphas_hair = None, None
        soft_alphas_neck_only, soft_alphas_neck = None, None
        self.detach_silhouettes = True

        use_deformations = True
        soft_alphas_detach_hair, soft_alphas_detach_neck = None, None
        # Obtain visualized frontal vertices

        if self.use_scalp_deforms or self.external_params.get('predict_hair_silh', False):
            verts_final_hair = verts_final.clone().detach()
            verts_final_hair_only = verts_final[:, self.hair_list].clone()
            # if not self.external_params.get('detach_neck', True):
            verts_final_hair[:, self.hair_list] = verts_final_hair_only

            _, verts_target_hair, _ = util.batch_orth_proj(
                verts_final_hair,
                target_codedict['cam'],
                root_joint,
                cam_rot_mats,
                self.flame
            )

        if self.use_neck_deforms or self.external_params.get('predict_hair_silh'):
            verts_final_neck = verts_final.clone()
            verts_final_neck[:, self.hair_list] = verts_final_neck[:, self.hair_list].detach()

            _, verts_target_neck, _ = util.batch_orth_proj(
                verts_final_neck,
                target_codedict['cam'],
                root_joint,
                cam_rot_mats,
                self.flame
            )

        if self.external_params.get('use_laplace_vector_coef', False):
            vertices_laplace_list = torch.ones_like(verts_final[..., 0]) * self.external_params.get(
                'laplacian_reg_weight', 0.0)
            vertices_laplace_list[:, self.hair_list] = self.external_params.get('laplacian_reg_hair_weight', 0.01)
            vertices_laplace_list[:, self.neck_list] = self.external_params.get('laplacian_reg_neck_weight', 10.0)

        # Project verts into target camera
        _, verts_target, landmarks_target = util.batch_orth_proj(
            verts_final.clone(),
            target_codedict['cam'],
            root_joint,
            cam_rot_mats,
            self.flame
        )
        shape_target = self.render.render_shape(verts_final, verts_target)

        with torch.no_grad():
            _, verts_final_posed, _ = util.batch_orth_proj(verts_final.clone(), default_cam, flame=self.flame)

            shape_final_posed = self.render.render_shape(verts_final, verts_final_posed)

        # Render and parse the outputs
        hair_neck_face_mesh_faces = torch.cat([self.faces_hair_mask, self.faces_neck_mask, self.faces_face_mask],
                                              dim=-1)

        ops = self.render(verts_final, verts_target, face_masks=hair_neck_face_mesh_faces)

        alphas = ops['alpha_images']
        soft_alphas = ops['soft_alpha_images']
        uvs = ops['uvcoords_images'].permute(0, 2, 3, 1)[..., :2]
        normals = ops['normal_images']
        coords = ops['vertice_images']

        if self.detach_silhouettes:
            verts_final_detach_hair = verts_final.clone()
            verts_final_detach_neck = verts_final.clone()

            verts_final_detach_hair[:, self.hair_list] = verts_final_detach_hair[:, self.hair_list].detach()
            verts_final_detach_neck[:, self.neck_list] = verts_final_detach_neck[:, self.neck_list].detach()

            verts_target_detach_hair = verts_target.clone()
            verts_target_detach_neck = verts_target.clone()

            verts_target_detach_hair[:, self.hair_list] = verts_target_detach_hair[:, self.hair_list].detach()
            verts_target_detach_neck[:, self.neck_list] = verts_target_detach_neck[:, self.neck_list].detach()

            ops_detach_hair = self.render(
                verts_final_detach_hair,
                verts_target_detach_hair,
                faces=self.faces_subdiv if self.subdivide_mesh else None,
                render_only_soft_silhouette=True,
            )

            ops_detach_neck = self.render(
                verts_final_detach_neck,
                verts_target_detach_neck,
                faces=self.faces_subdiv if self.subdivide_mesh else None,
                render_only_soft_silhouette=True,
            )

            soft_alphas_detach_hair = ops_detach_hair['soft_alpha_images']
            soft_alphas_detach_neck = ops_detach_neck['soft_alpha_images']

        verts_vis_mask = ops['vertices_visibility']
        verts_hair_vis_mask = ops['vertices_visibility'][:, self.hair_list]
        verts_neck_vis_mask = ops['vertices_visibility'][:, self.neck_list]
        hard_alphas_hair_only = (ops['area_alpha_images'][:, 0:1] == 1.0).float()
        hard_alphas_neck_only = (ops['area_alpha_images'][:, 1:2] == 1.0).float()
        hard_alphas_face_only = (ops['area_alpha_images'][:, 2:3] == 1.0).float()

        if self.use_scalp_deforms or self.external_params.get('predict_hair_silh'):
            ops_hair = self.render(
                    verts_final_hair[:, self.hair_edge_list],
                    verts_target_hair[:, self.hair_edge_list],
                    faces=self.hair_faces,
                    render_only_soft_silhouette=True,
            )

            soft_alphas_hair = ops_hair.get('soft_alpha_images')

        if self.use_neck_deforms or self.external_params.get('predict_hair_silh'):
            # Render whole
            ops_neck = self.render(
                    verts_final_neck[:, self.neck_edge_list],
                    verts_target_neck[:, self.neck_edge_list],
                    faces=self.neck_faces,
                    render_only_soft_silhouette=True,
            )

            soft_alphas_neck = ops_neck['soft_alpha_images']

            if self.external_params.get('use_whole_segmentation', False):
                soft_alphas_neck = soft_alphas

        # Grid sample outputs
        rendered_texture = None
        rendered_texture_detach_geom = None
        if neural_texture is not None:
            rendered_texture = F.grid_sample(neural_texture, uvs, mode='bilinear')
            rendered_texture_detach_geom = F.grid_sample(neural_texture, uvs.detach(), mode='bilinear')

        dense_vert_tensor = None
        dense_face_tensor = None
        dense_shape = None

        opdict = {
            'rendered_texture': rendered_texture,
            'rendered_texture_detach_geom': rendered_texture_detach_geom,
            'vertices': verts_final,
            'vertices_target': verts_target,
            'vertices_target_hair': verts_target_hair,
            'vertices_target_neck': verts_target_neck,
            'vertices_deforms': verts_deforms,
            'vertices_vis_mask': verts_vis_mask,
            'vertices_hair_vis_mask': verts_hair_vis_mask,
            'vertices_neck_vis_mask': verts_neck_vis_mask,
            'landmarks': landmarks_target,
            'alphas': alphas,
            'alpha_hair': None,
            'alpha_neck': None,
            'soft_alphas': soft_alphas,
            'soft_alphas_detach_hair': soft_alphas_detach_hair,
            'soft_alphas_detach_neck': soft_alphas_detach_neck,
            'soft_alphas_hair': soft_alphas_hair,
            'soft_alphas_neck': soft_alphas_neck,
            'hard_alphas_hair_only': hard_alphas_hair_only,
            'hard_alphas_neck_only': hard_alphas_neck_only,
            'hard_alphas_face_only': hard_alphas_face_only,
            'coords': coords,
            'normals': normals,
            'uvs': uvs,
            'source_uvs': None,
            'dense_verts': dense_vert_tensor,
            'dense_faces': dense_face_tensor,
            'dense_shape': dense_shape,
            'uv_deformations_codes': uv_deformations_codes,
            'source_warped_img': None
        }

        if self.use_tex:
            opdict['flametex_images'] = ops.get('images')

        if use_deformations:
            vert_texture_inp = torch.tanh(neural_texture[:, :3])

            opdict['deformations_out'] = verts_deforms_texture
            opdict['deformations_inp_texture'] = vert_texture_inp
            opdict['deformations_inp_coord'] = vert_texture_coord_inp
            opdict['deformations_inp_orig'] = vert_texture_codes
            opdict['vertices_laplace_list'] = None
            if self.external_params.get('fp_visualize_uv', True):
                opdict['face_parsing_uvs'] = None
            if self.neck_mask is not None:
                opdict['uv_neck_mask'] = self.render.world2uv(self.neck_mask).detach()[:, 0].repeat(batch_size, 1, 1, 1)

        visdict = {
            'shape_neutral_frontal_images': shape_neutral_frontal,
            'shape_parametric_frontal_images': shape_parametric_frontal,
            'shape_final_frontal_images': shape_final_frontal,
            'shape_final_posed_images': shape_final_posed,
            'shape_images': shape_target,
        }

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
        source_image_crop, source_warp_to_crop, source_crop_bbox = self.preprocess_image(source_image, source_keypoints)
        target_image_crop, target_warp_to_crop, target_crop_bbox = self.preprocess_image(target_image, target_keypoints)

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

        posed_final_shapes = F.interpolate(visdict['shape_final_posed_images'], size=self.image_size, mode='bilinear')
        frontal_final_shapes = F.interpolate(visdict['shape_final_frontal_images'], size=self.image_size, mode='bilinear')
        frontal_parametric_shapes = F.interpolate(visdict['shape_parametric_frontal_images'], size=self.image_size, mode='bilinear')
        frontal_neutral_shapes = F.interpolate(visdict['shape_neutral_frontal_images'], size=self.image_size, mode='bilinear')

        outputs = {
            'rendered_texture' : opdict['rendered_texture'],
            'rendered_texture_detach_geom': opdict['rendered_texture_detach_geom'],
            'pred_target_coord': opdict['coords'],
            'pred_target_normal': opdict['normals'],
            'pred_target_uv': opdict['uvs'],
            'pred_source_coarse_uv': opdict['source_uvs'],
            'pred_target_shape_img': visdict['shape_images'],
            'pred_target_hard_mask': opdict['alphas'],
            'pred_target_hard_hair_mask': opdict['alpha_hair'],
            'pred_target_hard_neck_mask': opdict['alpha_neck'],
            'pred_target_soft_mask': opdict['soft_alphas'],
            'pred_target_soft_detach_hair_mask': opdict['soft_alphas_detach_hair'],
            'pred_target_soft_detach_neck_mask': opdict['soft_alphas_detach_neck'],
            'pred_target_soft_hair_mask': opdict['soft_alphas_hair'],
            'pred_target_soft_neck_mask': opdict['soft_alphas_neck'],
            'pred_target_hard_hair_only_mask':opdict['hard_alphas_hair_only'],
            'pred_target_hard_neck_only_mask':opdict['hard_alphas_neck_only'],
            'pred_target_hard_face_only_mask':opdict['hard_alphas_face_only'],
            'pred_target_keypoints': opdict['landmarks'],
            'vertices': opdict['vertices'],
            'vertices_target': opdict['vertices_target'],
            'vertices_target_hair':opdict['vertices_target_hair'],
            'vertices_target_neck':opdict['vertices_target_neck'],
            'vertices_deforms':opdict['vertices_deforms'],
            'vertices_vis_mask':opdict['vertices_vis_mask'],
            'vertices_hair_vis_mask':opdict['vertices_hair_vis_mask'],
            'vertices_neck_vis_mask':opdict['vertices_neck_vis_mask'],
            'target_shape_final_posed_img': posed_final_shapes,
            'target_shape_final_frontal_img': frontal_final_shapes,
            'target_shape_parametric_frontal_img': frontal_parametric_shapes,
            'target_shape_neutral_frontal_img': frontal_neutral_shapes,
            'deformations_out': opdict.get('deformations_out'),
            'deformations_inp_texture': opdict.get('deformations_inp_texture'),
            'deformations_inp_coord': opdict.get('deformations_inp_coord'),
            'deformations_inp_orig': opdict.get('deformations_inp_orig'),
            'laplace_coefs': opdict.get('vertices_laplace_list'),
            'target_render_face_mask': opdict.get('predicted_segmentation'),
            'uv_neck_mask': opdict.get('uv_neck_mask'),
            'target_visual_faceparsing': opdict.get('face_parsing_uvs'),
            'target_warp_to_crop': target_warp_to_crop,
            'dense_verts': opdict.get('dense_verts'),
            'dense_faces': opdict.get('dense_faces'),
            'flametex_images': opdict.get('flametex_images'),
            'dense_shape': opdict.get('dense_shape'),
            'uv_deformations_codes': opdict['uv_deformations_codes'],
            'source_warped_img': opdict['source_warped_img'],
            'source_image_crop': source_image_crop,
        }

        return outputs
