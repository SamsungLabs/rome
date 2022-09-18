import torch
import torch.nn.functional as F
from torch import nn

from typing import Union



class SegmentationLoss(nn.Module):
    def __init__(self, loss_type = 'bce_with_logits'):
        super(SegmentationLoss, self).__init__()
        self.loss_type = loss_type
        if loss_type == 'bce_with_logits':
            self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, 
                pred_seg_logits: Union[torch.Tensor, list], 
                target_segs: Union[torch.Tensor, list]) -> torch.Tensor:
        if isinstance(pred_seg_logits, list):
            # Concat alongside the batch axis
            pred_seg_logits = torch.cat(pred_seg_logits)
            target_segs = torch.cat(target_segs)

        if target_segs.shape[2] != pred_seg_logits.shape[2]:
            target_segs = F.interpolate(target_segs, size=pred_seg_logits.shape[2:], mode='bilinear')

        if self.loss_type == 'bce_with_logits':
            loss = self.criterion(pred_seg_logits, target_segs)
        
        elif self.loss_type == 'dice':
            pred_segs = torch.sigmoid(pred_seg_logits)

            intersection = (pred_segs * target_segs).view(pred_segs.shape[0], -1)
            cardinality = (pred_segs**2 + target_segs**2).view(pred_segs.shape[0], -1)
            loss = 1 - ((2. * intersection.mean(1)) / (cardinality.mean(1) + 1e-7)).mean(0)

        return loss


class MultiScaleSilhouetteLoss(nn.Module):
    def __init__(self, num_scales: int = 1, loss_type: str = 'bce'):
        super().__init__()
        self.num_scales = num_scales
        self.loss_type = loss_type
        if self.loss_type == 'bce':
            self.loss = nn.BCELoss()
        if self.loss_type == 'mse':
            self.loss = nn.MSELoss()

    def forward(self, inputs, targets):
        original_size = targets.size()[-1]
        loss = 0.0
        for i in range(self.num_scales):
            if i > 0:
                x = F.interpolate(inputs, size=original_size // (2 ** i))
                gt = F.interpolate(targets, size=original_size // (2 ** i))
            else:
                x = inputs
                gt = targets
            
            if self.loss_type == 'iou':
                intersection = (x * gt).view(x.shape[0], -1)
                union = (x + gt).view(x.shape[0], -1)
                loss += 1 - (intersection.mean(1) / (union - intersection).mean(1)).mean(0)
            
            elif self.loss_type == 'mse':
                loss += ((x - gt)**2).mean() * 0.5

            elif self.loss_type == 'bce':
                loss += self.loss(x, gt.float())
            elif self.loss_type == 'mse':
                loss += self.loss(x, gt.float())
        return loss / self.num_scales