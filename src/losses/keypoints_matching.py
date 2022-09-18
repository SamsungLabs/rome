import torch
from torch import nn
import torch.nn.functional as F

from typing import List, Union



class KeypointsMatchingLoss(nn.Module):
    def __init__(self):
        super(KeypointsMatchingLoss, self).__init__()
        self.register_buffer('weights', torch.ones(68), persistent=False)
        self.weights[5:7] = 2.0
        self.weights[10:12] = 2.0
        self.weights[27:36] = 1.5
        self.weights[30] = 3.0
        self.weights[31] = 3.0
        self.weights[35] = 3.0
        self.weights[60:68] = 1.5
        self.weights[48:60] = 1.5
        self.weights[48] = 3
        self.weights[54] = 3

    def forward(self, 
                pred_keypoints: torch.Tensor,
                keypoints: torch.Tensor) -> torch.Tensor:
        diff = pred_keypoints - keypoints

        loss = (diff.abs().mean(-1) * self.weights[None] / self.weights.sum()).sum(-1).mean()

        return loss