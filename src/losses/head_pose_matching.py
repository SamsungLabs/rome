import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import numpy as np

from typing import Union



class HeadPoseMatchingLoss(nn.Module):
    def __init__(self, loss_type = 'l2'):
        super(HeadPoseMatchingLoss, self).__init__()
        self.loss_type = loss_type

    def forward(self, 
                pred_thetas: Union[torch.Tensor, list], 
                target_thetas: Union[torch.Tensor, list]) -> torch.Tensor:
        loss = 0

        if isinstance(pred_thetas, torch.Tensor):
            pred_thetas = [pred_thetas]
            target_thetas = [target_thetas]

        for pred_theta, target_theta in zip(pred_thetas, target_thetas):
            if self.loss_type == 'l1':
                loss += (pred_theta - target_theta).abs().mean()
            elif self.loss_type == 'l2':
                loss += ((pred_theta - target_theta)**2).mean()

        return loss