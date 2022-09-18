import torch
from torch import nn
import torch
import torch.nn.functional as F
from pathlib import Path
from torch import nn
import cv2
import numpy as np

from typing import Union
from typing import Tuple, List
from rt_gene import RtGeneEstimator
from rt_gene import FaceBox


class GazeLoss(object):
    def __init__(self,
                 device: str,
                 gaze_model_types: Union[List[str], str] = ['vgg16',],
                 criterion: str = 'l1',
                 interpolate: bool = False,
                 layer_indices: tuple = (1, 6, 11, 18, 25),
#                  layer_indices: tuple = (4, 5, 6, 7), # for resnet 
#                  weights: tuple = (2.05625e-3, 2.78125e-4, 5.125e-5, 6.575e-8, 9.67e-10)
#                  weights: tuple = (1.0, 1e-1, 4e-3, 2e-6, 1e-8),
#                  weights: tuple = (0.0625, 0.125, 0.25, 1.0),
                 weights: tuple = (0.03125, 0.0625, 0.125, 0.25, 1.0),
                 ) -> None:
        super(GazeLoss, self).__init__()
        self.len_features = len(layer_indices)
        # checkpoints_paths_dict = {'vgg16':'/Vol0/user/n.drobyshev/latent-texture-avatar/losses/gaze_models/vgg_16_2_forward_sum.pt', 'resnet18':'/Vol0/user/n.drobyshev/latent-texture-avatar/losses/gaze_models/resnet_18_2_forward_sum.pt'}
        # if interpolate:
        checkpoints_paths_dict = {'vgg16': '/group-volume/orc_srr/multimodal/t.khakhulin/pretrained/gaze_net.pt',
                                'resnet18': '/group-volume/orc_srr/multimodal/t.khakhulin/pretrained/gaze_net.pt'}
            
        self.gaze_estimator = RtGeneEstimator(device=device,
                                              model_nets_path=[checkpoints_paths_dict[m] for m in gaze_model_types],
                                              gaze_model_types=gaze_model_types,
                                              interpolate = interpolate,
                                              align_face=True)

        if criterion == 'l1':
            self.criterion = nn.L1Loss()
        elif criterion == 'l2':
            self.criterion = nn.MSELoss()

        self.layer_indices = layer_indices
        self.weights = weights

    @torch.cuda.amp.autocast(False)
    def forward(self,
                inputs: Union[torch.Tensor, list],
                target: torch.Tensor,
                keypoints: torch.Tensor = None,
                interpolate=True) -> Union[torch.Tensor, list]:
        if isinstance(inputs, list):
            # Concat alongside the batch axis
            input_is_a_list = True
            num_chunks = len(inputs)
            chunk_size = inputs[0].shape[0]
            inputs = torch.cat(inputs)

        else:
            input_is_a_list = False
            
        if interpolate:   
            inputs = F.interpolate(inputs, (224, 224), mode='bicubic', align_corners=False)
            target = F.interpolate(target, (224, 224), mode='bicubic', align_corners=False)
        
        if keypoints is not None:
            keypoints_np = [(kp[:, :2].cpu().numpy() + 1) / 2 * tgt.shape[2] for kp, tgt in zip(keypoints, target)]
#             keypoints_np = [(kp[:, :2].cpu().numpy() + 1) / 2 * tgt.shape[2] for kp, tgt in zip(keypoints, target)]
#             keypoints_np = [(kp[:, :2].cpu().numpy()/tgt.shape[2]*224).astype(np.int32) for kp, tgt in zip(keypoints, target)]
            
            faceboxes = [FaceBox(left=kp[:, 0].min(),
                                               top=kp[:, 1].min(),
                                               right=kp[:, 0].max(),
                                               bottom=kp[:, 1].max()) for kp in keypoints_np]
        else:
            faceboxes = None
            keypoints_np = None

        target = target.float()
        inputs = inputs.float()

        with torch.no_grad():
            target_subjects = self.gaze_estimator.get_eye_embeddings(target,
                                                                     self.layer_indices,
                                                                     faceboxes,
                                                                     keypoints_np)

        # Filter subjects with visible eyes
        visible_eyes = [subject is not None and subject.eye_embeddings is not None for subject in target_subjects]

        if not any(visible_eyes):
            return torch.zeros(1).to(target.device)

        target_subjects = self.select_by_mask(target_subjects, visible_eyes)

        faceboxes = [subject.box for subject in target_subjects]
        keypoints_np = [subject.landmarks for subject in target_subjects]

        target_features = [[] for i in range(self.len_features)]
        for subject in target_subjects:
            for k in range(self.len_features):
                target_features[k].append(subject.eye_embeddings[k])
        target_features = [torch.cat(feats) for feats in target_features]

        eye_masks = self.draw_eye_masks(keypoints_np, target.shape[2], target.device)

        if input_is_a_list:
            visible_eyes *= num_chunks
            faceboxes *= num_chunks
            keypoints_np *= num_chunks
            eye_masks = torch.cat([eye_masks] * num_chunks)

        # Grads are masked
        inputs = inputs[visible_eyes]
#         inputs.retain_grad() # turn it on while debugging
        inputs_ = inputs * eye_masks + inputs.detach() * (1 - eye_masks)
#         inputs_.retain_grad() # turn it on while debugging
        
        # In order to apply eye masks for gradients, first calc the grads
#         inputs_ = inputs.detach().clone().requires_grad_()
#         inputs_ = inputs_ * eye_masks + inputs_.detach() * (1 - eye_masks)
        input_subjects = self.gaze_estimator.get_eye_embeddings(inputs_,
                                                                self.layer_indices,
                                                                faceboxes,
                                                                keypoints_np)

        input_features = [[] for i in range(self.len_features)]
        for subject in input_subjects:
            for k in range(self.len_features):
                input_features[k].append(subject.eye_embeddings[k])
        input_features = [torch.cat(feats) for feats in input_features]

        loss = 0

        for input_feature, target_feature, weight in zip(input_features, target_features, self.weights):
            if input_is_a_list:
                target_feature = torch.cat([target_feature.detach()] * num_chunks)

            loss += weight * self.criterion(input_feature, target_feature)
        
        return loss

    @staticmethod
    def select_by_mask(a, mask):
        return [v for (is_true, v) in zip(mask, a) if is_true]

    @staticmethod
    def draw_eye_masks(keypoints_np, image_size, device):
        ### Define drawing options ###
        edges_parts = [list(range(36, 42)), list(range(42, 48))]

        mask_kernel = np.ones((5, 5), np.uint8)

        ### Start drawing ###
        eye_masks = []

        for xy in keypoints_np:
            xy = xy[None, :, None].astype(np.int32)

            eye_mask = np.zeros((image_size, image_size, 3), np.uint8)

            for edges in edges_parts:
                eye_mask = cv2.fillConvexPoly(eye_mask, xy[0, edges], (255, 255, 255))

            eye_mask = cv2.dilate(eye_mask, mask_kernel, iterations=1)
            eye_mask = cv2.blur(eye_mask, mask_kernel.shape)
            eye_mask = torch.FloatTensor(eye_mask[:, :, [0]].transpose(2, 0, 1)) / 255.
            eye_masks.append(eye_mask)

        eye_masks = torch.stack(eye_masks).to(device)

        return eye_masks
