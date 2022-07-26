import torch
from torch import nn
from argparse import ArgumentParser
from pytorch3d.structures import Meshes

import src.networks as networks
from src.parametric_avatar import ParametricAvatar
from src.utils import args as args_utils
from src.utils import harmonic_encoding
from src.utils.visuals import mask_errosion


class ROME(nn.Module):
    @staticmethod
    def add_argparse_args(parent_parser: ArgumentParser, *, use_argument_group=True):
        if use_argument_group:
            parser = parent_parser.add_argument_group("model")
            parser_out = parent_parser
        else:
            parser = ArgumentParser(parents=[parent_parser], add_help=False)
            parser_out = parser

        parser.add_argument('--model_image_size', default=256, type=int)

        parser.add_argument('--align_source', default='True', type=args_utils.str2bool, choices=[True, False])
        parser.add_argument('--align_target', default='False', type=args_utils.str2bool, choices=[True, False])
        parser.add_argument('--align_scale', default=1.25, type=float)

        parser.add_argument('--use_mesh_deformations', default='False', type=args_utils.str2bool, choices=[True, False])
        parser.add_argument('--subdivide_mesh', default='False', type=args_utils.str2bool, choices=[True, False])
        parser.add_argument('--renderer_sigma', default=1e-8, type=float)
        parser.add_argument('--renderer_zfar', default=100.0, type=float)
        parser.add_argument('--renderer_type', default='soft_mesh')
        parser.add_argument('--renderer_texture_type', default='texture_uv')
        parser.add_argument('--renderer_normalized_alphas', default='False', type=args_utils.str2bool,
                            choices=[True, False])

        parser.add_argument('--deca_path', default='')
        parser.add_argument('--rome_data_dir', default='')


        parser.add_argument('--autoenc_cat_alphas', default='False', type=args_utils.str2bool, choices=[True, False])
        parser.add_argument('--autoenc_align_inputs', default='False', type=args_utils.str2bool, choices=[True, False])
        parser.add_argument('--autoenc_use_warp', default='False', type=args_utils.str2bool, choices=[True, False])
        parser.add_argument('--autoenc_num_channels', default=64, type=int)
        parser.add_argument('--autoenc_max_channels', default=512, type=int)
        parser.add_argument('--autoenc_num_groups', default=4, type=int)
        parser.add_argument('--autoenc_num_bottleneck_groups', default=0, type=int)
        parser.add_argument('--autoenc_num_blocks', default=2, type=int)
        parser.add_argument('--autoenc_num_layers', default=4, type=int)
        parser.add_argument('--autoenc_block_type', default='bottleneck')

        parser.add_argument('--neural_texture_channels', default=8, type=int)
        parser.add_argument('--num_harmonic_encoding_funcs', default=6, type=int)

        parser.add_argument('--unet_num_channels', default=64, type=int)
        parser.add_argument('--unet_max_channels', default=512, type=int)
        parser.add_argument('--unet_num_groups', default=4, type=int)
        parser.add_argument('--unet_num_blocks', default=1, type=int)
        parser.add_argument('--unet_num_layers', default=2, type=int)
        parser.add_argument('--unet_block_type', default='conv')
        parser.add_argument('--unet_skip_connection_type', default='cat')
        parser.add_argument('--unet_use_normals_cond', default=True, action='store_true')
        parser.add_argument('--unet_use_vertex_cond', action='store_true')
        parser.add_argument('--unet_use_uvs_cond', action='store_true')
        parser.add_argument('--unet_pred_mask', action='store_true')
        parser.add_argument('--use_separate_seg_unet', default='True', type=args_utils.str2bool, choices=[True, False])

        parser.add_argument('--norm_layer_type', default='gn', type=str, choices=['bn', 'sync_bn', 'in', 'gn'])
        parser.add_argument('--activation_type', default='relu', type=str, choices=['relu', 'lrelu'])
        parser.add_argument('--conv_layer_type', default='ws_conv', type=str, choices=['conv', 'ws_conv'])

        parser.add_argument('--deform_norm_layer_type', default='gn', type=str, choices=['bn', 'sync_bn', 'in', 'gn'])
        parser.add_argument('--deform_activation_type', default='relu', type=str, choices=['relu', 'lrelu'])
        parser.add_argument('--deform_conv_layer_type', default='ws_conv', type=str, choices=['conv', 'ws_conv'])
        parser.add_argument('--unet_seg_weight', default=0.0, type=float)
        parser.add_argument('--unet_seg_type', default='bce_with_logits', type=str, choices=['bce_with_logits', 'dice'])
        parser.add_argument('--deform_face_tightness', default=0.0, type=float)

        parser.add_argument('--use_whole_segmentation', action='store_true')
        parser.add_argument('--mask_hair_for_neck', action='store_true')
        parser.add_argument('--use_hair_from_avatar', action='store_true')

        # Basis deformations
        parser.add_argument('--use_scalp_deforms', default='True', type=args_utils.str2bool,
                            choices=[True, False], help='')
        parser.add_argument('--use_neck_deforms', default='True', type=args_utils.str2bool,
                            choices=[True, False], help='')
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
        parser.add_argument('--path_to_deca', default='DECA')

        parser.add_argument('--path_to_linear_hair_model',
                            default='data/linear_hair.pth')
        parser.add_argument('--path_to_mobile_model',
                            default='data/disp_model.pth')
        parser.add_argument('--n_scalp', default=60, type=int)

        parser.add_argument('--use_distill', default=False, type=args_utils.str2bool, choices=[True, False])
        parser.add_argument('--use_mobile_version', default=False, type=args_utils.str2bool, choices=[True, False])

        parser.add_argument('--deformer_path', default='data/rome.pth')

        parser.add_argument('--output_unet_deformer_feats', default=32, type=int,
                            help='output features in the UNet')

        parser.add_argument('--use_deca_details', default=False, type=args_utils.str2bool, choices=[True, False])
        parser.add_argument('--use_flametex', default=False, type=args_utils.str2bool, choices=[True, False])

        parser.add_argument('--upsample_type', default='nearest', type=str,
                            choices=['nearest', 'bilinear', 'bicubic'])

        parser.add_argument('--num_frequencies', default=6, type=int, help='frequency for harmonic encoding')
        parser.add_argument('--deform_face_scale_coef', default=0.0, type=float)
        parser.add_argument('--device', default='cpu', type=str)

        return parser_out

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.init_networks(args)

    def init_networks(self, args):
        self.autoencoder = networks.Autoencoder(
            args.autoenc_num_channels,
            args.autoenc_max_channels,
            args.autoenc_num_groups,
            args.autoenc_num_bottleneck_groups,
            args.autoenc_num_blocks,
            args.autoenc_num_layers,
            args.autoenc_block_type,
            input_channels=3 + 1,  # cat alphas
            input_size=args.model_image_size,
            output_channels=args.neural_texture_channels,
            norm_layer_type=args.norm_layer_type,
            activation_type=args.activation_type,
            conv_layer_type=args.conv_layer_type,
            use_psp=False,
        ).eval()

        self.basis_deformer = None
        self.vertex_deformer = None
        self.mask_hard_threshold = 0.6

        deformer_input_ch = args.neural_texture_channels

        deformer_input_ch += 3

        deformer_input_ch += 3 * args.num_frequencies * 2

        output_channels = self.args.output_unet_deformer_feats

        if self.args.use_unet_deformer:
            self.mesh_deformer = networks.UNet(
                args.unet_num_channels,
                args.unet_max_channels,
                args.unet_num_groups,
                args.unet_num_blocks,
                args.unet_num_layers,
                args.unet_block_type,
                input_channels=deformer_input_ch,
                output_channels=output_channels,
                skip_connection_type=args.unet_skip_connection_type,
                norm_layer_type=args.deform_norm_layer_type,
                activation_type=args.deform_activation_type,
                conv_layer_type=args.deform_conv_layer_type,
                downsampling_type='maxpool',
                upsampling_type='nearest',
            )

        input_mlp_feat = self.args.output_unet_deformer_feats + 2 * (1 + args.num_frequencies * 2)

        self.mlp_deformer = networks.MLP(
            num_channels=256,
            num_layers=8,
            skip_layer=4,
            input_channels=input_mlp_feat,
            output_channels=3,
            activation_type=args.activation_type,
            last_bias=False,
        )

        if self.args.use_basis_deformer:
            print('Create and load basis deformer.')
            self.basis_deformer = networks.EncoderResnet(
                pretrained_encoder_basis_path=args.pretrained_encoder_basis_path,
                norm_type='gn+ws',
                num_basis=args.num_basis)

            self.vertex_deformer = networks.EncoderVertex(
                path_to_deca_lib=args.path_to_deca,
                pretrained_vertex_basis_path=args.pretrained_vertex_basis_path,
                norm_type='gn+ws',
                num_basis=args.num_basis,
                basis_init=args.basis_init,
                num_vertex=args.num_vertex)

        self.parametric_avatar = ParametricAvatar(
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

        self.unet = networks.UNet(
            args.unet_num_channels,
            args.unet_max_channels,
            args.unet_num_groups,
            args.unet_num_blocks,
            args.unet_num_layers,
            args.unet_block_type,
            input_channels=args.neural_texture_channels + 3 * (
                    1 + args.unet_use_vertex_cond) * (1 + 6 * 2),  # unet_use_normals_cond
            output_channels=3 + 1,
            skip_connection_type=args.unet_skip_connection_type,
            norm_layer_type=args.norm_layer_type,
            activation_type=args.activation_type,
            conv_layer_type=args.conv_layer_type,
            downsampling_type='maxpool',
            upsampling_type='nearest',
        ).eval()

    @torch.no_grad()
    def forward(self, data_dict, neutral_pose: bool = False, source_information=None):
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
