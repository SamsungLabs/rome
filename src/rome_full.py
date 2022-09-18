import itertools

import torch
from torch import nn
import torch.nn.functional as F

from argparse import ArgumentParser
from pytorch3d.structures import Meshes

from src.networks.face_parsing import FaceParsing
from src.parametric_avatar_trainable import ParametricAvatarTrainable
from src.rome import ROME
from src.utils import args as args_utils
from src.utils import harmonic_encoding
from src.utils.visuals import mask_errosion
from src.losses import *
from src.networks import MultiScaleDiscriminator
from src.utils import misc, spectral_norm, weight_init


class TrainableROME(ROME):

    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.num_source_frames = args.num_source_frames
        self.num_target_frames = args.num_target_frames

        self.weights = {
            'adversarial': args.adversarial_weight,
            'feature_matching': args.feature_matching_weight,
            'l1': args.l1_weight,
            'vgg19': args.vgg19_weight,
            'vggface': args.vggface_weight,
            'vgggaze': args.vgggaze_weight,
            'unet_seg': args.unet_seg_weight,
            'seg': args.seg_weight,
            'seg_hair': args.seg_hair_weight,
            'seg_neck': args.seg_neck_weight,
            'seg_hard': args.seg_hard_weight,
            'seg_hard_neck': args.seg_hard_neck_weight,
            'chamfer': args.chamfer_weight,
            'chamfer_hair': args.chamfer_hair_weight,
            'chamfer_neck': args.chamfer_neck_weight,
            'keypoints_matching': args.keypoints_matching_weight,
            'eye_closure': args.eye_closure_weight,
            'lip_closure': args.lip_closure_weight,
            'shape_reg': args.shape_reg_weight,
            'exp_reg': args.exp_reg_weight,
            'tex_reg': args.tex_reg_weight,
            'light_reg': args.light_reg_weight,
            'laplacian_reg': args.laplacian_reg_weight,
            'edge_reg': args.edge_reg_weight,
            'l1_hair': args.l1_hair_weight,
            'repulsion_hair': args.repulsion_hair_weight,
            'repulsion': args.repulsion_weight,
            'normal_reg': args.normal_reg_weight}

        self.init_networks(args)
        self.discriminator = None
        self.init_losses(args)

        self.fp_masks_setup = ['cloth_neck', 'hair_face_ears', 'neck_cloth_face']
        self.face_parsing = FaceParsing(
            args.face_parsing_path,
            device='cuda' if args.num_gpus else 'cpu'
        )

        self.parametric_avatar = ParametricAvatarTrainable(
            args.model_image_size,
            args.deca_path,
            args.use_scalp_deforms,
            args.use_neck_deforms,
            args.subdivide_mesh,
            args.use_deca_details,
            args.use_flametex,
            args,
            device=args.device,
        )
        if args.adversarial_weight > 0:
            self.init_disc(args)

    def init_disc(self, args):
        if args.spn_layers:
            spn_layers = args_utils.parse_str_to_list(args.spn_layers, sep=',')

        self.discriminator = MultiScaleDiscriminator(
            min_channels=args.dis_num_channels,
            max_channels=args.dis_max_channels,
            num_blocks=args.dis_num_blocks,
            input_channels=3,
            input_size=args.model_image_size,
            num_scales=args.dis_num_scales)

        self.discriminator.apply(weight_init.weight_init(args.dis_init_type, args.dis_init_gain))
        if args.spn_apply_to_dis:
            self.discriminator.apply(lambda module: spectral_norm.apply_spectral_norm(module, apply_to=spn_layers))

        if args.deferred_neural_rendering_path and not args.reset_dis_weights:
            state_dict_full = torch.load(args.deferred_neural_rendering_path, map_location='cpu')
            state_dict = OrderedDict()
            for k, v in state_dict_full.items():
                if 'discriminator' in k:
                    state_dict[k.replace(f'discriminator.', '')] = v
            self.discriminator.load_state_dict(state_dict)
            print('Loaded discriminator state dict')

    def init_losses(self, args):
        if self.weights['adversarial']:
            self.adversarial_loss = AdversarialLoss()

        if self.weights['feature_matching']:
            self.feature_matching_loss = FeatureMatchingLoss()

        if self.weights['vgg19']:
            class PerceptualLossWrapper(object):
                def __init__(self, num_scales, use_gpu, use_fp16):
                    self.loss = PerceptualLoss(
                        num_scales=num_scales,
                        use_fp16=use_fp16)

                    if use_gpu:
                        self.loss = self.loss.cuda()

                    self.forward = self.loss.forward

            self.vgg19_loss = PerceptualLossWrapper(
                num_scales=args.vgg19_num_scales,
                use_gpu=args.num_gpus > 0,
                use_fp16=True if args.use_amp and args.amp_opt_level != 'O0' else False)

        if self.weights['vggface']:
            self.vggface_loss = VGGFace2Loss(pretrained_model=args.vggface_path,
                                             device='cuda' if args.num_gpus else 'cpu')

        if self.weights['vgggaze']:
            self.vgggaze_loss = GazeLoss('cuda' if args.num_gpus else 'cpu')

        if self.weights['keypoints_matching']:
            self.keypoints_matching_loss = KeypointsMatchingLoss()

        if self.weights['eye_closure']:
            self.eye_closure_loss = EyeClosureLoss()

        if self.weights['lip_closure']:
            self.lip_closure_loss = LipClosureLoss()

        if self.weights['unet_seg']:
            self.unet_seg_loss = SegmentationLoss(loss_type=args.unet_seg_type)

        if (self.weights['seg'] or self.weights['seg_hair'] or self.weights['seg_neck'] or
                self.weights['seg_hard']):
            self.seg_loss = MultiScaleSilhouetteLoss(num_scales=args.seg_num_scales, loss_type=args.seg_type)

        if self.weights['chamfer'] or self.weights['chamfer_hair'] or self.weights['chamfer_neck']:
            self.chamfer_loss = ChamferSilhouetteLoss(
                args.chamfer_num_neighbours,
                args.chamfer_same_num_points,
                args.chamfer_sample_outside_of_silhouette
            )

        if self.weights['laplacian_reg']:
            self.laplace_loss = LaplaceMeshLoss(args.laplace_reg_type)
        if self.args.predict_face_parsing_mask:
            self.face_parsing_loss = nn.CrossEntropyLoss(ignore_index=255,
                                                         reduction='mean')

        self.ssim = SSIM(data_range=1, size_average=True, channel=3)
        self.ms_ssim = MS_SSIM(data_range=1, size_average=True, channel=3)
        self.psnr = PSNR()
        self.lpips = LPIPS()

    def train(self, mode=False):
        if self.args.train_deferred_neural_rendering:
            self.autoencoder.train(mode)
            self.unet.train(mode)
            if self.args.unet_pred_mask and self.args.use_separate_seg_unet:
                self.unet_seg.train(mode)
            if self.discriminator is not None:
                self.discriminator.train(mode)

        elif self.args.train_texture_encoder:
            self.autoencoder.train(mode)

        if self.args.use_mesh_deformations:
            if self.args.use_unet_deformer:
                self.mesh_deformer.train(mode)
            if self.args.use_mlp_deformer:
                self.mlp_deformer.train(mode)
            if self.args.use_basis_deformer:
                self.basis_deformer.train(mode)

    def _forward(self, data_dict):
        deca_results = self.parametric_avatar.forward(
            data_dict['source_img'],
            data_dict['source_mask'],
            data_dict['source_keypoints'],
            data_dict['target_img'],
            data_dict['target_keypoints'],
            deformer_nets={
                'neural_texture_encoder': self.autoencoder,
                'unet_deformer': self.mesh_deformer,
                'mlp_deformer': self.mlp_deformer,
                'basis_deformer': self.basis_deformer,
            },
        )

        rendered_texture = deca_results['rendered_texture']
        rendered_texture_detach_geom = deca_results.pop('rendered_texture_detach_geom')

        for key, value in deca_results.items():
            data_dict[key] = value

        if self.args.predict_face_parsing_mask:
            target_fp_mask = self.face_parsing.forward(data_dict['target_img'])  # face, ears, neck, cloth, hair
            data_dict['target_hair_mask'] = (
                                                    target_fp_mask[:, 0] +
                                                    target_fp_mask[:, 1] +
                                                    target_fp_mask[:, 4]
                                            )[:, None]

            if self.args.include_neck_to_hair_mask:
                data_dict['target_hair_mask'] += target_fp_mask[:, 2][:, None]

            data_dict['target_face_mask'] = target_fp_mask[:, 0][:, None]
            data_dict['target_hair_only_mask'] = target_fp_mask[:, 4][:, None]

            neck_mask = target_fp_mask[:, 0] + target_fp_mask[:, 1] + \
                        target_fp_mask[:, 2] + target_fp_mask[:, 3] + \
                        target_fp_mask[:, 4]

            data_dict['target_neck_mask'] = neck_mask[:, None]

            data_dict['target_neck_only_mask'] = (
                                                         target_fp_mask[:, 2] +
                                                         target_fp_mask[:, 3]
                                                 )[:, None]

        if self.args.use_graphonomy_mask:
            graphonomy_mask = self.graphonomy.forward(data_dict['target_img'])
            data_dict['target_face_mask'] = graphonomy_mask[:, 1:2]

        if self.unet is not None:
            unet_inputs = rendered_texture * data_dict['pred_target_hard_mask']
            # hard mask the rendered texture to make consistent padding

            if self.args.unet_use_normals_cond:
                normals = data_dict['pred_target_normal'].permute(0, 2, 3, 1)
                normal_inputs = harmonic_encoding.harmonic_encoding(normals,
                                                                    self.args.num_harmonic_encoding_funcs).permute(0, 3,
                                                                                                                   1, 2)
                unet_inputs = torch.cat([unet_inputs, normal_inputs], dim=1)
                if self.args.reg_positive_z_normals:
                    data_dict['normals_z'] = normals[..., [-1]]
                if self.args.mask_according_to_normal:
                    normal_z_mask = normals[..., [-1]] > -0.3
                    unet_inputs = normal_z_mask.permute(0, 3, 1, 2) * unet_inputs
                    normal_z_mask = normals[..., [-1]] > -0.2
                    data_dict['pred_target_soft_detach_hair_mask'] = data_dict[
                                                                         'pred_target_soft_detach_hair_mask'] * normal_z_mask.permute(
                        0, 3, 1, 2)
                    data_dict['pred_target_soft_neck_mask'] = data_dict[
                                                                  'pred_target_soft_neck_mask'] * normal_z_mask.permute(
                        0, 3, 1, 2)

            if self.args.unet_use_uvs_cond:
                uvs = data_dict['pred_target_uv'][..., :2]
                uvs_inputs = harmonic_encoding.harmonic_encoding(uvs, self.args.num_harmonic_encoding_funcs).permute(0,
                                                                                                                     3,
                                                                                                                     1,
                                                                                                                     2)
                unet_inputs = torch.cat([unet_inputs, uvs_inputs], dim=1)

            if self.args.use_separate_seg_unet:
                data_dict['pred_target_img'] = torch.sigmoid(self.unet(unet_inputs))

                if self.args.unet_pred_mask:
                    data_dict['pred_target_unet_logits'] = self.unet_seg(unet_inputs)
                    data_dict['pred_target_unet_mask'] = torch.sigmoid(data_dict['pred_target_unet_logits']).detach()

            else:
                unet_outputs = self.unet(unet_inputs)

                data_dict['pred_target_img'] = torch.sigmoid(unet_outputs[:, :3])

                if self.args.unet_pred_mask:
                    data_dict['pred_target_unet_logits'] = unet_outputs[:, 3:]
                    data_dict['pred_target_unet_mask'] = torch.sigmoid(data_dict['pred_target_unet_logits']).detach()

            if self.args.adv_only_for_rendering:
                unet_inputs = rendered_texture_detach_geom * data_dict[
                    'pred_target_hard_mask']  # hard mask the rendered texture to make consistent padding
                if self.args.unet_use_normals_cond:
                    unet_inputs = torch.cat([unet_inputs, normal_inputs.detach()], dim=1)

                data_dict['pred_target_img_detach_geom'] = torch.sigmoid(self.unet(unet_inputs)[:, :3])

        else:
            data_dict['pred_target_img'] = data_dict['flametex_images']

        if self.args.train_only_face:
            data_dict['pred_target_img'] = data_dict['pred_target_img'] * data_dict['target_face_mask']
            data_dict['target_img'] = data_dict['target_img'] * data_dict['target_face_mask']

        if self.args.use_mesh_deformations:
            if self.args.laplacian_reg_only_deforms:
                verts = data_dict['vertices_deforms']
            else:
                verts = data_dict['vertices']

            faces = self.parametric_avatar.render.faces.expand(verts.shape[0], -1, -1).long()

            data_dict['mesh'] = Meshes(
                verts=verts,
                faces=faces
            )

        # Apply masks
        if not self.args.unet_pred_mask:
            target_mask = data_dict['target_mask']
        else:
            target_mask = data_dict['pred_target_unet_mask'].detach()

        if self.args.use_random_uniform_background:
            random_bg_color = torch.rand(target_mask.shape[0], 3, 1, 1, dtype=target_mask.dtype,
                                         device=target_mask.device)
            data_dict['pred_target_img'] = data_dict['pred_target_img'] * target_mask + random_bg_color * (
                    1 - target_mask)
            data_dict['target_img'] = data_dict['target_img'] * target_mask + random_bg_color * (1 - target_mask)
        # else:
        #     random_bg_color = torch.zeros(target_mask.shape[0], 3, 1, 1, dtype=target_mask.dtype, device=target_mask.device)
        return data_dict

    def calc_train_losses(self, data_dict: dict, mode: str = 'gen'):
        losses_dict = {}

        if mode == 'dis':
            losses_dict['dis_adversarial'] = (
                    self.weights['adversarial'] *
                    self.adversarial_loss(
                        real_scores=data_dict['real_score_dis'],
                        fake_scores=data_dict['fake_score_dis'],
                        mode='dis'))

        if mode == 'gen':
            if self.weights['adversarial']:
                losses_dict['gen_adversarial'] = (
                        self.weights['adversarial'] *
                        self.adversarial_loss(
                            fake_scores=data_dict['fake_score_gen'],
                            mode='gen'))

                losses_dict['feature_matching'] = (
                        self.weights['feature_matching'] *
                        self.feature_matching_loss(
                            real_features=data_dict['real_feats_gen'],
                            fake_features=data_dict['fake_feats_gen']))

            if self.weights['l1']:
                losses_dict['l1'] = self.weights['l1'] * F.l1_loss(data_dict['pred_target_img'],
                                                                   data_dict['target_img'])

            if self.weights['vgg19']:
                losses_dict['vgg19'] = self.weights['vgg19'] * self.vgg19_loss.forward(
                    data_dict['pred_target_img'],
                    data_dict['target_img']
                )

            if self.weights['vggface']:
                pred_target_warped_img = F.grid_sample(data_dict['pred_target_img'], data_dict['target_warp_to_crop'])
                target_warped_img = F.grid_sample(data_dict['target_img'], data_dict['target_warp_to_crop'])

                losses_dict['vggface'] = self.weights['vggface'] * self.vggface_loss.forward(
                    pred_target_warped_img,
                    target_warped_img
                )

                # For vis
                with torch.no_grad():
                    data_dict['pred_target_warped_img'] = F.interpolate(
                        pred_target_warped_img,
                        size=256,
                        mode='bilinear'
                    )

                    data_dict['target_warped_img'] = F.interpolate(
                        target_warped_img,
                        size=256,
                        mode='bilinear'
                    )

            if self.weights['vgggaze']:
                try:
                    losses_dict['vgggaze'] = self.weights['vgggaze'] * self.vgggaze_loss.forward(
                        data_dict['pred_target_img'],
                        data_dict['target_img'],
                        data_dict['target_keypoints']
                    )

                except:
                    losses_dict['vgggaze'] = torch.zeros(1).to(data_dict['target_img'].device).mean()

            if self.weights['keypoints_matching']:
                losses_dict['keypoints_matching'] = self.weights['keypoints_matching'] * self.keypoints_matching_loss(
                    data_dict['pred_target_keypoints'],
                    data_dict['target_keypoints'])

                if self.weights['eye_closure']:
                    losses_dict['eye_closure'] = self.weights['eye_closure'] * self.eye_closure_loss(
                        data_dict['pred_target_keypoints'],
                        data_dict['target_keypoints'])

                if self.weights['lip_closure']:
                    losses_dict['lip_closure'] = self.weights['lip_closure'] * self.lip_closure_loss(
                        data_dict['pred_target_keypoints'],
                        data_dict['target_keypoints'])

            if self.args.finetune_flame_encoder or self.args.train_flame_encoder_from_scratch:

                if self.args.flame_encoder_reg:
                    losses_dict['shape_reg'] = (torch.sum(data_dict['shape'] ** 2) / 2) * self.weights['shape_reg']
                    losses_dict['exp_reg'] = (torch.sum(data_dict['exp'] ** 2) / 2) * self.weights['exp_reg']
                    if 'flame_tex_params' in data_dict.keys():
                        losses_dict['tex_reg'] = (torch.sum(data_dict['flame_tex_params'] ** 2) / 2) * self.weights[
                            'tex_reg']
                    # losses_dict['light_reg'] = ((torch.mean(data_dict['flame_light_params'], dim=2)[:, :, None] - data_dict[
                    #     'flame_light_params']) ** 2).mean() * self.weights['light_reg']

            if self.args.train_deferred_neural_rendering and self.args.unet_pred_mask:
                losses_dict['seg_unet'] = (
                        self.weights['unet_seg'] *
                        self.unet_seg_loss(
                            data_dict['pred_target_unet_logits'],
                            data_dict['target_mask']
                        )
                )

            if self.args.use_mesh_deformations:
                if self.weights['seg']:
                    losses_dict['seg'] = self.seg_loss(
                        data_dict['pred_target_soft_mask'],
                        data_dict['target_mask']
                    ) * self.weights['seg']

                if self.weights['seg_hair']:
                    losses_dict['seg_hair'] = self.seg_loss(
                        data_dict['pred_target_soft_detach_neck_mask'],
                        data_dict['target_hair_mask']
                    ) * self.weights['seg_hair']

                if self.weights['seg_neck']:
                    losses_dict['seg_neck'] = self.seg_loss(
                        data_dict['pred_target_soft_detach_hair_mask'],
                        data_dict['target_neck_mask']
                    ) * self.weights['seg_neck']

                if self.weights['seg_hard']:
                    data_dict['pred_target_soft_hair_only_mask'] = (
                            data_dict['pred_target_soft_hair_mask'] *
                            data_dict['pred_target_hard_hair_only_mask']
                    )

                    batch_indices = torch.nonzero(
                        (
                                data_dict['target_hair_only_mask'].mean([1, 2, 3]) /
                                data_dict['target_face_mask'].mean([1, 2, 3])
                        ) > 0.3
                    )[:, 0]

                    if len(batch_indices) > 0:
                        losses_dict['seg_only_hair'] = self.seg_loss(
                            data_dict['pred_target_soft_hair_only_mask'][batch_indices],
                            data_dict['target_hair_only_mask'][batch_indices]
                        ) * self.weights['seg_hard']
                    else:
                        losses_dict['seg_only_hair'] = torch.zeros(1, device=batch_indices.device,
                                                                   dtype=data_dict['target_mask'].dtype).mean()

                    # Zero values for visualization
                    tmp = data_dict['target_hair_only_mask'].clone()
                    data_dict['target_hair_only_mask'] = torch.zeros_like(tmp)
                    if len(batch_indices) > 0:
                        data_dict['target_hair_only_mask'][batch_indices] = tmp[batch_indices]

                if self.weights['seg_hard_neck']:
                    pred_target_soft_neck_mask = (
                            data_dict['pred_target_soft_neck_mask'] *
                            data_dict['pred_target_hard_neck_only_mask']
                    )

                    losses_dict['seg_only_neck'] = self.seg_loss(
                        pred_target_soft_neck_mask,
                        data_dict['target_neck_only_mask']
                    ) * self.weights['seg_hard_neck']

                if self.weights['chamfer']:
                    (
                        losses_dict['chamfer_loss'],
                        data_dict['chamfer_pred_vertices'],
                        data_dict['chamfer_target_vertices']
                    ) = self.chamfer_loss(
                        data_dict['vertices_target'][..., :2],
                        data_dict['vertices_vis_mask'],
                        data_dict['pred_target_hard_mask'],
                        data_dict['target_mask']
                    )
                    losses_dict['chamfer_loss'] = losses_dict['chamfer_loss'] * self.weights['chamfer']

                if self.weights['repulsion_hair']:
                    points = data_dict['vertices_target_hair'][:, self.deca.hair_list, :]
                    valid_dists = knn_points(points, points, K=5)[0]
                    losses_dict['repulsion_hair_loss'] = self.weights['repulsion_hair'] * (
                        torch.exp((-valid_dists / 10))).mean()

                if self.weights['repulsion']:
                    points = data_dict['vertices_target']
                    valid_dists = knn_points(points, points, K=5)[0]
                    losses_dict['repulsion_loss'] = self.weights['repulsion'] * (
                        torch.exp((-valid_dists / 10))).mean()

                if self.weights['chamfer_hair']:
                    batch_indices = torch.nonzero(
                        (
                                data_dict['target_hair_only_mask'].mean([1, 2, 3]) /
                                data_dict['target_face_mask'].mean([1, 2, 3])
                        ) > 0.3
                    )[:, 0]

                    data_dict['chamfer_pred_hair_vertices'] = torch.ones_like(
                        data_dict['vertices_target_hair'][:, :len(self.deca.hair_list), :2]) * -100.0
                    data_dict['chamfer_target_hair_vertices'] = data_dict['chamfer_pred_hair_vertices'].clone()

                    if len(batch_indices) > 0:
                        (
                            losses_dict['chamfer_hair_loss'],
                            chamfer_pred_hair_vertices,
                            chamfer_target_hair_vertices
                        ) = self.chamfer_loss(
                            data_dict['vertices_target_hair'][batch_indices][:, self.deca.hair_list, :2],
                            data_dict['vertices_hair_vis_mask'][batch_indices],
                            data_dict['pred_target_hard_hair_only_mask'][batch_indices],
                            data_dict['target_hair_only_mask'][batch_indices]
                        )

                        data_dict['chamfer_pred_hair_vertices'][batch_indices] = chamfer_pred_hair_vertices
                        data_dict['chamfer_target_hair_vertices'][batch_indices] = chamfer_target_hair_vertices

                    else:
                        losses_dict['chamfer_hair_loss'] = torch.zeros(1, device=batch_indices.device,
                                                                       dtype=data_dict['target_mask'].dtype).mean()

                    chamfer_weight = self.chamfer_hair_scheduler.step() if self.chamfer_hair_scheduler is not None else \
                        self.weights['chamfer_hair']
                    losses_dict['chamfer_hair_loss'] = losses_dict['chamfer_hair_loss'] * chamfer_weight

                if self.weights['chamfer_neck']:
                    (
                        losses_dict['chamfer_neck_loss'],
                        data_dict['chamfer_pred_neck_vertices'],
                        data_dict['chamfer_target_neck_vertices']
                    ) = self.chamfer_loss(
                        data_dict['vertices_target_neck'][:, self.deca.neck_list, :2],
                        data_dict['vertices_neck_vis_mask'],
                        data_dict['pred_target_hard_neck_only_mask'],
                        data_dict['target_neck_only_mask']
                    )
                    chamfer_weight = self.weights['chamfer_neck']
                    losses_dict['chamfer_neck_loss'] = losses_dict['chamfer_neck_loss'] * chamfer_weight

                if self.weights['laplacian_reg']:
                    laplacian_weight = self.weights['laplacian_reg']
                    losses_dict['laplacian_reg'] = laplacian_weight * \
                                                   self.laplace_loss(data_dict['mesh'],
                                                                     data_dict.get('laplace_coefs'))

                if self.weights['edge_reg']:
                    losses_dict['edge_reg'] = self.weights['edge_reg'] * mesh_edge_loss(data_dict['mesh'])

                if self.weights['normal_reg']:
                    losses_dict['normal_reg'] = self.weights['normal_reg'] * mesh_normal_consistency(data_dict['mesh'])

                if self.args.reg_positive_z_normals:
                    losses_dict['normal_reg'] = torch.pow(torch.relu(-data_dict['normals_z']), 2).mean()
        loss = 0
        for k, v in losses_dict.items():
            loss += v

        return loss, losses_dict

    def calc_test_losses(self, data_dict: dict):
        losses_dict = {}

        if self.args.pretrain_global_encoder:
            losses_dict['cam'] = ((data_dict['pred_cam'] - data_dict['flame_cam_params']) ** 2).mean(0).sum()
            losses_dict['pose_rot'] = ((data_dict['pred_pose'][:, 0] - data_dict['flame_pose_params'][:, 0]) ** 2).mean(
                0)
            losses_dict['pose_dir'] = 1 - (
                    data_dict['pred_pose'][:, 1:4] *
                    data_dict['flame_pose_params'][:, 1:4]
            ).sum(-1).mean(0)

        if 'pred_target_img' in data_dict.keys() and data_dict['pred_target_img'] is not None:
            losses_dict['ssim'] = self.ssim(data_dict['pred_target_img'], data_dict['target_img']).mean()
            losses_dict['psnr'] = self.psnr(data_dict['pred_target_img'], data_dict['target_img'])
            losses_dict['lpips'] = self.lpips(data_dict['pred_target_img'], data_dict['target_img'])

            if self.args.model_image_size > 160:
                losses_dict['ms_ssim'] = self.ms_ssim(data_dict['pred_target_img'], data_dict['target_img']).mean()

        return losses_dict

    def prepare_input_data(self, data_dict):
        for k, v in data_dict.items():
            if isinstance(v, dict):
                for k_, v_ in v.items():
                    if self.args.num_gpus:
                        v_ = v_.cuda()
                    v[k_] = v_.view(-1, *v_.shape[2:])
                data_dict[k] = v
            else:
                if self.args.num_gpus:
                    v = v.cuda()
                data_dict[k] = v.view(-1, *v.shape[2:])

        return data_dict

    def forward(self,
                data_dict: dict,
                phase: str = 'test',
                optimizer_idx: int = 0,
                visualize: bool = False):
        assert phase in ['train', 'test']
        mode = self.optimizer_idx_to_mode[optimizer_idx]

        if mode == 'gen':
            data_dict = self.prepare_input_data(data_dict)
            data_dict = self._forward(data_dict)

            if phase == 'train':
                if self.args.adversarial_weight > 0:
                    self.discriminator.eval()

                    with torch.no_grad():
                        _, data_dict['real_feats_gen'] = self.discriminator(data_dict['target_img'])

                    if self.args.adv_only_for_rendering:
                        data_dict['fake_score_gen'], data_dict['fake_feats_gen'] = self.discriminator(
                            data_dict['pred_target_img_detach_geom'])
                    else:
                        data_dict['fake_score_gen'], data_dict['fake_feats_gen'] = self.discriminator(
                            data_dict['pred_target_img'])

                loss, losses_dict = self.calc_train_losses(data_dict, mode='gen')

            elif phase == 'test':
                loss = None
                losses_dict = self.calc_test_losses(data_dict)

            histograms_dict = {}

        elif mode == 'dis':
            # Backward through dis
            self.discriminator.train()

            data_dict['real_score_dis'], _ = self.discriminator(data_dict['target_img'])
            data_dict['fake_score_dis'], _ = self.discriminator(data_dict['pred_target_img'].detach().clone())

            loss, losses_dict = self.calc_train_losses(data_dict, mode='dis')

            histograms_dict = {}

        visuals = None
        if visualize:
            visuals = self.get_visuals(data_dict)

        return loss, losses_dict, histograms_dict, visuals, data_dict

    @torch.no_grad()
    def get_visuals(self, data_dict):
        data_dict['target_stickman'] = misc.draw_stickman(data_dict['target_keypoints'], self.args.model_image_size)
        data_dict['pred_target_stickman'] = misc.draw_stickman(data_dict['pred_target_keypoints'],
                                                               self.args.model_image_size)
        # This function creates an output grid of visuals
        visuals_data_dict = {}

        for k, v in data_dict.items():
            if isinstance(v, torch.Tensor):
                v = v.cpu()
            visuals_data_dict[k] = v

        if 'pred_source_shape_img' in data_dict.keys():
            visuals_data_dict['pred_source_shape_overlay_img'] = (
                                                                         data_dict['source_img'] +
                                                                         data_dict['pred_source_shape_img']
                                                                 ) * 0.5

        visuals_data_dict['pred_target_shape_overlay_img'] = (
                                                                     data_dict['target_img'] +
                                                                     data_dict['pred_target_shape_img']
                                                             ) * 0.5

        if 'chamfer_pred_vertices' in data_dict.keys():
            visuals_data_dict['chamfer_vis_pred_verties'] = misc.draw_keypoints(
                data_dict['target_img'],
                data_dict['chamfer_pred_vertices']
            )

            visuals_data_dict['chamfer_vis_target_verties'] = misc.draw_keypoints(
                data_dict['target_img'],
                data_dict['chamfer_target_vertices']
            )

        if 'chamfer_pred_hair_vertices' in data_dict.keys():
            visuals_data_dict['chamfer_vis_pred_hair_verties'] = misc.draw_keypoints(
                data_dict['target_img'],
                data_dict['chamfer_pred_hair_vertices']
            )

            visuals_data_dict['chamfer_vis_target_hair_verties'] = misc.draw_keypoints(
                data_dict['target_img'],
                data_dict['chamfer_target_hair_vertices']
            )

        if 'chamfer_pred_neck_vertices' in data_dict.keys():
            visuals_data_dict['chamfer_vis_pred_neck_verties'] = misc.draw_keypoints(
                data_dict['target_img'],
                data_dict['chamfer_pred_neck_vertices']
            )

            visuals_data_dict['chamfer_vis_target_neck_verties'] = misc.draw_keypoints(
                data_dict['target_img'],
                data_dict['chamfer_target_neck_vertices']
            )

        if 'pred_texture_warp' in data_dict.keys():
            visuals_data_dict['vis_pred_texture_warp'] = F.grid_sample(
                data_dict['source_img'],
                data_dict['pred_texture_warp'],
                mode='bilinear'
            )

        visuals = []

        uv_prep = lambda x: (x.permute(0, 3, 1, 2) + 1) / 2
        coords_prep = lambda x: (x + 1) / 2
        seg_prep = lambda x: torch.cat([x] * 3, dim=1)
        score_prep = lambda x: (x + 1) / 2

        visuals_list = [
            ['source_img', None],
            ['source_mask', seg_prep],
            ['source_warped_img', None],
            ['pred_source_coarse_uv', uv_prep],
            ['pred_texture_warp', uv_prep],
            ['vis_pred_texture_warp', None],
            ['target_img', None],
            ['pred_target_img', None],
            ['target_stickman', None],
            ['pred_target_stickman', None],
            ['pred_target_coord', coords_prep],
            ['pred_target_uv', uv_prep],
            ['pred_target_shape_img', None],
            ['pred_target_shape_displ_img', None],
            ['pred_target_shape_overlay_img', None],

            ['target_mask', seg_prep],
            ['pred_target_unet_mask', seg_prep],
            ['pred_target_hard_mask', seg_prep],
            ['pred_target_soft_mask', seg_prep],
            ['target_face_mask', seg_prep],
            ['pred_target_hard_face_only_mask', seg_prep],
            ['chamfer_vis_pred_verties', None],
            ['chamfer_vis_target_verties', None],

            ['target_hair_mask', seg_prep],
            ['target_hair_only_mask', seg_prep],
            ['pred_target_soft_hair_mask', seg_prep],
            ['pred_target_hard_hair_only_mask', seg_prep],
            ['pred_target_soft_hair_only_mask', seg_prep],
            ['chamfer_vis_pred_hair_verties', None],
            ['chamfer_vis_target_hair_verties', None],

            ['target_neck_mask', seg_prep],
            ['target_neck_only_mask', seg_prep],
            ['pred_target_soft_neck_mask', seg_prep],
            ['pred_target_hard_neck_only_mask', seg_prep],
            ['pred_target_soft_neck_only_mask', seg_prep],
            ['chamfer_vis_pred_neck_verties', None],
            ['chamfer_vis_target_neck_verties', None],

            ['pred_target_normal', coords_prep],
            ['pred_target_shading', None if self.args.shading_channels == 3 else seg_prep],
            ['pred_target_albedo', None],
            ['target_vertices_texture', coords_prep],
            ['target_shape_final_posed_img', None],
            ['target_shape_final_frontal_img', None],
            ['target_shape_parametric_frontal_img', None],
            ['target_shape_neutral_frontal_img', None],

            ['pred_target_warped_img', None],
            ['target_warped_img', None],

            ['dummy_vis', seg_prep]
        ]

        if (
                self.args.finetune_flame_encoder or self.args.train_flame_encoder_from_scratch) and not self.args.use_mesh_deformations:
            visuals_list = [
                ['source_img', None],
                ['source_mask', seg_prep],
                ['pred_texture_warp', uv_prep],
                ['vis_pred_texture_warp', None],
                ['target_img', None],
                ['pred_target_img', None],
                ['target_stickman', None],
                ['pred_target_stickman', None],
                ['pred_target_coord', coords_prep],
                ['pred_target_uv', uv_prep],
                ['pred_target_shape_img', None],
                ['pred_target_shape_displ_img', None],
                ['pred_target_shape_overlay_img', None],
                ['pred_target_warped_img', None],
                ['target_warped_img', None],
            ]

        if self.args.use_mlp_deformer:
            visuals_list.append(['deformations_out', coords_prep])
            visuals_list.append(['deformations_inp_orig', None])
            visuals_list.append(['deformations_inp_texture', None])
            visuals_list.append(['deformations_inp_coord', coords_prep])

        max_h = max_w = 0

        for tensor_name, preprocessing_op in visuals_list:
            if tensor_name in visuals_data_dict.keys() and visuals_data_dict[tensor_name] is not None:
                visuals += misc.prepare_visual(visuals_data_dict, tensor_name, preprocessing_op)

            if len(visuals):
                h, w = visuals[-1].shape[2:]
                max_h = max(h, max_h)
                max_w = max(w, max_w)

        visuals = torch.cat(visuals, 3)  # cat w.r.t. width
        visuals = visuals.clamp(0, 1)

        return visuals

    def gen_parameters(self):
        params = iter([])

        if self.args.train_deferred_neural_rendering or self.args.train_texture_encoder:
            print('Training autoencoder')
            params = itertools.chain(params, self.autoencoder.parameters())

        if self.args.train_deferred_neural_rendering:
            print('Training rendering unet')
            params = itertools.chain(params, self.unet.parameters())
            if self.args.unet_pred_mask and self.args.use_separate_seg_unet:
                print('Training seg unet')
                params = itertools.chain(params, self.unet_seg.parameters())

        if self.basis_deformer is not None:
            print('Training basis deformer')
            params = itertools.chain(params, self.basis_deformer.parameters())
            if self.args.train_basis:
                params = itertools.chain(params, self.vertex_deformer.parameters())

        if self.mesh_deformer is not None:
            print('Training mesh deformer')
            params = itertools.chain(params, self.mesh_deformer.parameters())

            if self.mlp_deformer is not None:
                print('Training MLP deformer')
                params = itertools.chain(params, self.mlp_deformer.parameters())

        if self.args.finetune_flame_encoder or self.args.train_flame_encoder_from_scratch:
            print('Training FLAME encoder')
            params = itertools.chain(params, self.deca.E_flame.parameters())

        for param in params:
            yield param

    def configure_optimizers(self):
        self.optimizer_idx_to_mode = {0: 'gen', 1: 'dis'}

        opts = {
            'adam': lambda param_groups, lr, beta1, beta2: torch.optim.Adam(
                params=param_groups,
                lr=lr,
                betas=(beta1, beta2)),
            'adamw': lambda param_groups, lr, beta1, beta2: torch.optim.AdamW(
                params=param_groups,
                lr=lr,
                betas=(beta1, beta2))}

        opt_gen = opts[self.args.gen_opt_type](
            self.gen_parameters(),
            self.args.gen_lr,
            self.args.gen_beta1,
            self.args.gen_beta2)

        if self.args.adversarial_weight > 0:
            opt_dis = opts[self.args.dis_opt_type](
                self.discriminator.parameters(),
                self.args.dis_lr,
                self.args.dis_beta1,
                self.args.dis_beta2)

            return [opt_gen, opt_dis]

        else:
            return [opt_gen]

    def configure_schedulers(self, opts):
        shds = {
            'step': lambda optimizer, lr_max, lr_min, max_iters: torch.optim.lr_scheduler.StepLR(
                optimizer=optimizer,
                step_size=max_iters,
                gamma=lr_max / lr_min),
            'cosine': lambda optimizer, lr_max, lr_min, max_iters: torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer=optimizer,
                T_max=max_iters,
                eta_min=lr_min)}

        if self.args.gen_shd_type != 'none':
            shd_gen = shds[self.args.gen_shd_type](
                opts[0],
                self.args.gen_lr,
                self.args.gen_shd_lr_min,
                self.args.gen_shd_max_iters)

        if self.args.adversarial_weight > 0 and self.args.dis_shd_type != 'none':
            shd_dis = shds[self.args.dis_shd_type](
                opts[1],
                self.args.dis_lr,
                self.args.dis_shd_lr_min,
                self.args.dis_shd_max_iters)

            return [shd_gen, shd_dis], [self.args.gen_shd_max_iters, self.args.dis_shd_max_iters], []

        elif self.args.gen_shd_type != 'none':
            return [shd_gen], [self.args.gen_shd_max_iters], []

        else:
            return [], [], []

    @torch.no_grad()
    def forward_infer(self, data_dict, neutral_pose: bool = False, source_information=None):
        if source_information is None:
            source_information = dict()

        parametric_output = self.parametric_avatar.forward(
            data_dict['source_img'],
            data_dict['source_mask'],
            data_dict['source_keypoints'],
            data_dict['target_img'],
            data_dict['target_keypoints'],
            deformer_nets={
                'neural_texture_encoder': self.autoencoder,
                'unet_deformer': self.mesh_deformer,
                'mlp_deformer': self.mlp_deformer,
                'basis_deformer': self.basis_deformer,
            },
            neutral_pose=neutral_pose,
            neural_texture=source_information.get('neural_texture'),
            source_information=source_information,
        )
        result_dict = {}
        rendered_texture = parametric_output.pop('rendered_texture')

        for key, value in parametric_output.items():
            result_dict[key] = value

        unet_inputs = rendered_texture * result_dict['pred_target_hard_mask']

        normals = result_dict['pred_target_normal'].permute(0, 2, 3, 1)
        normal_inputs = harmonic_encoding.harmonic_encoding(normals, 6).permute(0, 3, 1, 2)
        unet_inputs = torch.cat([unet_inputs, normal_inputs], dim=1)
        unet_outputs = self.unet(unet_inputs)

        pred_img = torch.sigmoid(unet_outputs[:, :3])
        pred_soft_mask = torch.sigmoid(unet_outputs[:, 3:])

        return_mesh = False
        if return_mesh:
            verts = result_dict['vertices_target'].cpu()
            faces = self.parametric_avatar.render.faces.expand(verts.shape[0], -1, -1).long()
            result_dict['mesh'] = Meshes(verts=verts, faces=faces)

        result_dict['pred_target_unet_mask'] = pred_soft_mask
        result_dict['pred_target_img'] = pred_img
        mask_pred = (result_dict['pred_target_unet_mask'][0].cpu() > self.mask_hard_threshold).float()
        mask_pred = mask_errosion(mask_pred.float().numpy() * 255)
        result_dict['render_masked'] = result_dict['pred_target_img'][0].cpu() * (mask_pred) + (1 - mask_pred)

        return result_dict

    @staticmethod
    def add_argparse_args(parent_parser: ArgumentParser, *, use_argument_group=True):
        if use_argument_group:
            parser = parent_parser.add_argument_group("model")
            parser_out = parent_parser
        else:
            parser = ArgumentParser(parents=[parent_parser], add_help=False)
            parser_out = parser

        parser.add_argument('--model_image_size', default=256, type=int)

        parser.add_argument('--predict_face_parsing_mask', action='store_true')
        parser.add_argument('--compute_face_parsing_mask', action='store_true')
        parser.add_argument('--face_parsing_path', default='')
        parser.add_argument('--face_parsing_mask_type', default='face')
        parser.add_argument('--include_neck_to_hair_mask', action='store_true')

        parser.add_argument('--use_graphonomy_mask', action='store_true')
        parser.add_argument('--graphonomy_path', default='')

        parser.add_argument('--segm_classes', default=0, type=int)
        parser.add_argument('--fp_visualize_uv', default='True', type=args_utils.str2bool, choices=[True, False])
        parser.add_argument('--use_fp_merge_classes', action='store_true')
        parser.add_argument('--update_silh_with_segm', action='store_true')
        parser.add_argument('--mask_silh_cloth', action='store_true')

        parser.add_argument('--adv_only_for_rendering', default='False', type=args_utils.str2bool,
                            choices=[True, False])

        parser.add_argument('--use_mesh_deformations', default='False', type=args_utils.str2bool, choices=[True, False])
        parser.add_argument('--subdivide_mesh', default='False', type=args_utils.str2bool, choices=[True, False])
        parser.add_argument('--detach_silhouettes', default='False', type=args_utils.str2bool, choices=[True, False])

        parser.add_argument('--train_deferred_neural_rendering', default='True', type=args_utils.str2bool,
                            choices=[True, False])
        parser.add_argument('--train_only_autoencoder', default='False', type=args_utils.str2bool,
                            choices=[True, False])
        parser.add_argument('--train_texture_encoder', default='False', type=args_utils.str2bool, choices=[True, False])
        parser.add_argument('--train_extra_flame_parameters', action='store_true')
        parser.add_argument('--train_flametex', default=False, type=args_utils.str2bool, choices=[True, False])

        parser.add_argument('--use_cam_encoder', default='False', type=args_utils.str2bool, choices=[True, False])
        parser.add_argument('--use_neck_pose_encoder', default='False', type=args_utils.str2bool, choices=[True, False])
        parser.add_argument('--use_light_encoder', default='False', type=args_utils.str2bool, choices=[True, False])
        parser.add_argument('--light_channels', default=8, type=int)

        parser.add_argument('--pretrain_global_encoder', default='False', type=args_utils.str2bool,
                            choices=[True, False],
                            help='fit the encoder from DECA')
        parser.add_argument('--train_global_encoder', default='False', type=args_utils.str2bool, choices=[True, False])

        parser.add_argument('--renderer_sigma', default=1e-8, type=float)
        parser.add_argument('--renderer_zfar', default=100.0, type=float)
        parser.add_argument('--renderer_type', default='soft_mesh')
        parser.add_argument('--renderer_texture_type', default='texture_uv')
        parser.add_argument('--renderer_normalized_alphas', default='False', type=args_utils.str2bool,
                            choices=[True, False])

        parser.add_argument('--deca_path', default='')
        parser.add_argument('--global_encoder_path', default='')
        parser.add_argument('--deferred_neural_rendering_path', default='')
        parser.add_argument('--deca_neutral_pose', default=False, type=args_utils.str2bool, choices=[True, False])

        parser.add_argument('--autoenc_cat_alphas', default='False', type=args_utils.str2bool, choices=[True, False])
        parser.add_argument('--autoenc_align_inputs', default='False', type=args_utils.str2bool, choices=[True, False])
        parser.add_argument('--autoenc_use_warp', default='False', type=args_utils.str2bool, choices=[True, False])
        parser.add_argument('--autoenc_num_channels', default=64, type=int)
        parser.add_argument('--autoenc_max_channels', default=512, type=int)
        parser.add_argument('--autoenc_num_warp_groups', default=4, type=int)
        parser.add_argument('--autoenc_num_warp_blocks', default=1, type=int)
        parser.add_argument('--autoenc_num_warp_layers', default=3, type=int)
        parser.add_argument('--autoenc_num_groups', default=4, type=int)
        parser.add_argument('--autoenc_num_bottleneck_groups', default=0, type=int)
        parser.add_argument('--autoenc_num_blocks', default=2, type=int)
        parser.add_argument('--autoenc_num_layers', default=4, type=int)
        parser.add_argument('--autoenc_block_type', default='bottleneck')
        parser.add_argument('--autoenc_use_psp', action='store_true')

        parser.add_argument('--neural_texture_channels', default=16, type=int)

        parser.add_argument('--finetune_flame_encoder', default='False', type=args_utils.str2bool,
                            choices=[True, False])
        parser.add_argument('--flame_encoder_reg', default='False', type=args_utils.str2bool, choices=[True, False])
        parser.add_argument('--train_flame_encoder_from_scratch', default='False', type=args_utils.str2bool,
                            choices=[True, False])

        parser.add_argument('--flame_num_shape_params', default=-1, type=int)
        parser.add_argument('--flame_num_exp_params', default=-1, type=int)
        parser.add_argument('--flame_num_tex_params', default=-1, type=int)

        parser.add_argument('--mesh_deformer_gain', default=0.001, type=float)

        parser.add_argument('--backprop_adv_only_into_unet', default='False', type=args_utils.str2bool,
                            choices=[True, False])
        parser.add_argument('--replace_bn_with_in', default='False', type=args_utils.str2bool, choices=[True, False])

        parser.add_argument('--num_harmonic_encoding_funcs', default=6, type=int)

        parser.add_argument('--unet_num_channels', default=64, type=int)
        parser.add_argument('--unet_max_channels', default=1024, type=int)
        parser.add_argument('--unet_num_groups', default=4, type=int)
        parser.add_argument('--unet_num_blocks', default=1, type=int)
        parser.add_argument('--unet_num_layers', default=2, type=int)
        parser.add_argument('--unet_block_type', default='conv')
        parser.add_argument('--unet_skip_connection_type', default='cat')
        parser.add_argument('--unet_use_normals_cond', action='store_true')
        parser.add_argument('--unet_use_vertex_cond', action='store_true')
        parser.add_argument('--unet_use_uvs_cond', action='store_true')
        parser.add_argument('--unet_pred_mask', action='store_true')
        parser.add_argument('--use_separate_seg_unet', default='True', type=args_utils.str2bool, choices=[True, False])

        parser.add_argument('--use_mlp_renderer', action='store_true')

        parser.add_argument('--use_shading_renderer', action='store_true')
        parser.add_argument('--shading_channels', default=1, type=int)

        parser.add_argument('--norm_layer_type', default='gn', type=str, choices=['bn', 'sync_bn', 'in', 'gn'])
        parser.add_argument('--activation_type', default='relu', type=str, choices=['relu', 'lrelu'])
        parser.add_argument('--conv_layer_type', default='ws_conv', type=str, choices=['conv', 'ws_conv'])

        parser.add_argument('--deform_norm_layer_type', default='gn', type=str, choices=['bn', 'sync_bn', 'in', 'gn'])
        parser.add_argument('--deform_activation_type', default='relu', type=str, choices=['relu', 'lrelu'])
        parser.add_argument('--deform_conv_layer_type', default='ws_conv', type=str, choices=['conv', 'ws_conv'])

        parser.add_argument('--dis_num_channels', default=64, type=int)
        parser.add_argument('--dis_max_channels', default=512, type=int)
        parser.add_argument('--dis_num_blocks', default=4, type=int)
        parser.add_argument('--dis_num_scales', default=1, type=int)

        parser.add_argument('--dis_init_type', default='xavier')
        parser.add_argument('--dis_init_gain', default=0.02, type=float)

        parser.add_argument('--adversarial_weight', default=0.0, type=float)
        parser.add_argument('--gen_adversarial_weight', default=-1.0, type=float)
        parser.add_argument('--feature_matching_weight', default=0.0, type=float)

        parser.add_argument('--vgg19_weight', default=0.0, type=float)
        parser.add_argument('--vgg19_num_scales', default=1, type=int)
        parser.add_argument('--vggface_weight', default=0.0, type=float)
        parser.add_argument('--vgggaze_weight', default=0.0, type=float)

        parser.add_argument('--unet_seg_weight', default=0.0, type=float)
        parser.add_argument('--unet_seg_type', default='bce_with_logits', type=str, choices=['bce_with_logits', 'dice'])

        parser.add_argument('--l1_weight', default=0.0, type=float)
        parser.add_argument('--l1_hair_weight', default=0.0, type=float)
        parser.add_argument('--repulsion_hair_weight', default=0.0, type=float)
        parser.add_argument('--repulsion_weight', default=0.0, type=float)

        parser.add_argument('--keypoints_matching_weight', default=1.0, type=float)
        parser.add_argument('--eye_closure_weight', default=1.0, type=float)
        parser.add_argument('--lip_closure_weight', default=0.5, type=float)
        parser.add_argument('--seg_weight', default=0.0, type=float)
        parser.add_argument('--seg_type', default='bce', type=str, choices=['bce', 'iou', 'mse'])
        parser.add_argument('--seg_num_scales', default=1, type=int)

        parser.add_argument('--seg_hard_weight', default=0.0, type=float)
        parser.add_argument('--seg_hair_weight', default=0.0, type=float)
        parser.add_argument('--seg_neck_weight', default=0.0, type=float)
        parser.add_argument('--seg_hard_neck_weight', default=0.0, type=float)
        parser.add_argument('--seg_ignore_face', action='store_true')

        parser.add_argument('--chamfer_weight', default=0.0, type=float)
        parser.add_argument('--chamfer_hair_weight', default=0.0, type=float)
        parser.add_argument('--chamfer_neck_weight', default=0.0, type=float)
        parser.add_argument('--chamfer_num_neighbours', default=1, type=int)
        parser.add_argument('--chamfer_same_num_points', action='store_true')
        parser.add_argument('--chamfer_remove_face', action='store_true')
        parser.add_argument('--chamfer_sample_outside_of_silhouette', action='store_true')

        parser.add_argument('--shape_reg_weight', default=1e-4, type=float)
        parser.add_argument('--exp_reg_weight', default=1e-4, type=float)
        parser.add_argument('--tex_reg_weight', default=1e-4, type=float)
        parser.add_argument('--light_reg_weight', default=1.0, type=float)

        parser.add_argument('--laplacian_reg_weight', default=0.0, type=float)
        parser.add_argument('--laplacian_reg_only_deforms', default='True', type=args_utils.str2bool,
                            choices=[True, False])
        parser.add_argument('--laplacian_reg_apply_to_hair_only', action='store_true')
        parser.add_argument('--laplacian_reg_hair_weight', default=0.0, type=float)
        parser.add_argument('--laplacian_reg_neck_weight', default=0.0, type=float)

        parser.add_argument('--laplacian_reg_weight_start', default=0.0, type=float)
        parser.add_argument('--laplacian_reg_weight_end', default=0.0, type=float)
        parser.add_argument('--chamfer_hair_weight_start', default=0.0, type=float)
        parser.add_argument('--chamfer_hair_weight_end', default=0.0, type=float)
        parser.add_argument('--chamfer_neck_weight_start', default=0.0, type=float)
        parser.add_argument('--chamfer_neck_weight_end', default=0.0, type=float)
        parser.add_argument('--scheduler_total_iter', default=50000, type=int)

        parser.add_argument('--deform_face_tightness', default=0.0, type=float)

        parser.add_argument('--use_whole_segmentation', action='store_true')
        parser.add_argument('--mask_hair_for_neck', action='store_true')
        parser.add_argument('--use_hair_from_avatar', action='store_true')

        # Basis deformations
        parser.add_argument('--use_basis_deformer', default='False', type=args_utils.str2bool,
                            choices=[True, False], help='')
        parser.add_argument('--use_unet_deformer', default='True', type=args_utils.str2bool,
                            choices=[True, False], help='')
        parser.add_argument('--pretrained_encoder_basis_path', default='')
        parser.add_argument('--pretrained_vertex_basis_path', default='')
        parser.add_argument('--num_basis', default=50, type=int)
        parser.add_argument('--basis_init', default='pca', type=str, choices=['random', 'pca'])
        parser.add_argument('--num_vertex', default=5023, type=int)
        parser.add_argument('--train_basis', default=True, type=args_utils.str2bool, choices=[True, False])
        parser.add_argument('--path_to_deca', default='/Vol0/user/v.sklyarova/cvpr/latent-texture-avatar/utils')

        parser.add_argument('--deformer_path', default=None)

        parser.add_argument('--edge_reg_weight', default=0.0, type=float)
        parser.add_argument('--normal_reg_weight', default=0.0, type=float)

        # Deformation Block arguments
        parser.add_argument('--use_scalp_deforms', default=False, action='store_true')
        parser.add_argument('--use_neck_deforms', default=False, action='store_true')

        parser.add_argument('--use_gaze_dir', default=True, type=args_utils.str2bool, choices=[True, False])
        parser.add_argument('--use_neck_dir', default=False, type=args_utils.str2bool, choices=[True, False])

        parser.add_argument('--harmonize_deform_input', default='True', type=args_utils.str2bool,
                            choices=[True, False], help='harmonize input in the deformation Unet module')
        parser.add_argument('--detach_deformation_vertices', default='True', type=args_utils.str2bool,
                            choices=[True, False], help='detach textured vertices')
        parser.add_argument('--predict_deformed_vertices', default=False, action='store_true',
                            help='predict new vertices')
        parser.add_argument('--output_unet_deformer_feats', default=3, type=int,
                            help='output features in the UNet')
        parser.add_argument('--use_mlp_deformer', default='True', type=args_utils.str2bool,
                            choices=[True, False], help='')
        parser.add_argument('--harmonize_uv_mlp_input', default=True, action='store_true',
                            help='harmonize uv positional encoding')
        parser.add_argument('--mask_render_inputs', default='True', type=args_utils.str2bool,
                            choices=[True, False], help='mask resampled texture for rendering')
        parser.add_argument('--per_vertex_deformation', default='True', type=args_utils.str2bool, choices=[True, False])
        parser.add_argument('--use_complex_mesh', default=False, action='store_true',
                            help='two mesh for faces with blending weights')
        parser.add_argument('--invert_opacity', default=False, action='store_true',
                            help='instead of use opacity direct use 1 - p')
        parser.add_argument('--predict_sep_opacity', default='False', type=args_utils.str2bool,
                            choices=[True, False], help='mask resampled texture for rendering')
        parser.add_argument('--multi_texture', default='False', type=args_utils.str2bool,
                            choices=[True, False], help='use the second autoencoder for separate texture predicting')
        parser.add_argument('--detach_deformations', default='False', type=args_utils.str2bool,
                            choices=[True, False], help='detach_deformations for training weight map')
        parser.add_argument('--use_extended_flame', default=False, action='store_true',
                            help='use extended flame template')
        parser.add_argument('--use_deca_details', default=False, type=args_utils.str2bool, choices=[True, False])
        parser.add_argument('--use_flametex', default=False, type=args_utils.str2bool, choices=[True, False])
        parser.add_argument('--mask_uv_opacity', default=False, action='store_true',
                            help='mask opacity for faces')
        parser.add_argument('--mask_uv_face', default=False, action='store_true',
                            help='mask uvs for faces')

        parser.add_argument('--train_only_face', default=False, action='store_true')
        parser.add_argument('--deform_face', default=False, action='store_true')
        parser.add_argument('--deform_along_normals', default=False, action='store_true')
        parser.add_argument('--deform_hair_along_normals', default=False, action='store_true')
        parser.add_argument('--deform_nothair_along_normals', default=False, action='store_true')

        parser.add_argument('--reg_scalp_only', default=False, action='store_true')
        parser.add_argument('--mask_according_to_normal', default=False, action='store_true')

        parser.add_argument('--mask_neck_deformation_uvs', default=False, action='store_true')
        parser.add_argument('--mask_ear_deformation_uvs', default=False, action='store_true',
                            help='mask uvs for faces')
        parser.add_argument('--mask_deformation_uvs', default=False, action='store_true',
                            help='mask uvs for input in deformation network')
        parser.add_argument('--mask_eye_deformation_uvs', default=False, action='store_true',
                            help='mask eye in uvs for input in deformation network')
        parser.add_argument('--mask_hair_deformation_uvs', default=False, action='store_true',
                            help='mask hair according ot segmentation in uvs for input in deformation network')
        parser.add_argument('--use_hard_mask', default=False, action='store_true',
                            help='use_hard masking procedure')
        parser.add_argument('--use_updated_vertices', default=False, action='store_true',
                            help='use updated vertices')

        parser.add_argument('--detach_deforms_neural_texture', default=False, action='store_true')
        parser.add_argument('--hard_masking_deformations', default=False, action='store_true')
        parser.add_argument('--double_subdivide', default=False, action='store_true')

        parser.add_argument('--use_post_rendering_augs', default='False', type=args_utils.str2bool,
                            choices=[True, False])
        parser.add_argument('--use_random_uniform_background', default='False', type=args_utils.str2bool,
                            choices=[True, False])
        parser.add_argument('--reset_dis_weights', default='False', type=args_utils.str2bool, choices=[True, False])

        parser.add_argument('--predict_hair_silh', default=False, action='store_true')
        parser.add_argument('--detach_neck', default='True', type=args_utils.str2bool,
                            choices=[True, False], help='')
        parser.add_argument('--updated_neck_mask', default='False', type=args_utils.str2bool,
                            choices=[True, False], help='')

        parser.add_argument('--reg_positive_z_normals', default=False, action='store_true',
                            help='use defomations for mesh regularization')
        parser.add_argument('--use_deformation_reg', default=False, action='store_true',
                            help='use defomations for mesh regularization')
        parser.add_argument('--use_laplace_vector_coef', default=False, action='store_true',
                            help='use defomations for mesh regularization')
        parser.add_argument('--mask_eye_things_all', default=False, action='store_true',
                            help='')
        parser.add_argument('--mask_hair_face_soft', default=False, action='store_true')

        parser.add_argument('--mlp_input_camera_conditioned', default=False, action='store_true')

        parser.add_argument('--lambda_diffuse_reg', default=0.0, type=float)
        parser.add_argument('--num_frequencies', default=6, type=int, help='frequency for harmonic encoding')

        parser.add_argument('--laplace_reg_type', default='uniform', type=str, choices=['uniform', 'cot', 'cotcurv'])
        parser.add_argument('--update_laplace_weight_every', default=0, type=int)
        parser.add_argument('--vggface_path', default='data/resnet50_scratch_dag.pth', type=str)
        parser.add_argument('--use_gcn', default=False, action='store_true', help='')
        parser.add_argument('--dump_mesh', default=False, action='store_true',
                            help='dump batch of meshes')
        parser.add_argument('--deform_face_scale_coef', default=0.0, type=float)

        parser.add_argument('--spn_apply_to_gen', default=False, action='store_true')
        parser.add_argument('--spn_apply_to_dis', default=False, action='store_true')
        parser.add_argument('--spn_layers', default='conv2d, linear')

        # Optimization options
        parser.add_argument('--gen_opt_type', default='adam')
        parser.add_argument('--gen_lr', default=1e-4, type=float)
        parser.add_argument('--gen_beta1', default=0.0, type=float)
        parser.add_argument('--gen_beta2', default=0.999, type=float)

        parser.add_argument('--gen_weight_decay', default=1e-4, type=float)
        parser.add_argument('--gen_weight_decay_layers', default='conv2d')
        parser.add_argument('--gen_weight_decay_params', default='weight')

        parser.add_argument('--gen_shd_type', default='none')
        parser.add_argument('--gen_shd_max_iters', default=2.5e5, type=int)
        parser.add_argument('--gen_shd_lr_min', default=1e-6, type=int)

        parser.add_argument('--dis_opt_type', default='adam')
        parser.add_argument('--dis_lr', default=4e-4, type=float)
        parser.add_argument('--dis_beta1', default=0.0, type=float)
        parser.add_argument('--dis_beta2', default=0.999, type=float)

        parser.add_argument('--dis_shd_type', default='none')
        parser.add_argument('--dis_shd_max_iters', default=2.5e5, type=int)
        parser.add_argument('--dis_shd_lr_min', default=4e-6, type=int)
        parser.add_argument('--device', default='cuda', type=str)
        parser.add_argument('--deca_path', default='')
        parser.add_argument('--rome_data_dir', default='')

        parser.add_argument('--use_distill', default=False, type=args_utils.str2bool, choices=[True, False])
        parser.add_argument('--use_mobile_version', default=False, type=args_utils.str2bool, choices=[True, False])

        return parser_out
