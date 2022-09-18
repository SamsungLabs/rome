import torch
from torch import nn
import torch.nn.functional as F

from typing import List, Union



class LipClosureLoss(nn.Module):
    def __init__(self):
        super(LipClosureLoss, self).__init__()
        self.register_buffer('upper_lips', torch.LongTensor([61, 62, 63]), persistent=False)
        self.register_buffer('lower_lips', torch.LongTensor([67, 66, 65]), persistent=False)

    def forward(self, 
                pred_keypoints: torch.Tensor,
                keypoints: torch.Tensor) -> torch.Tensor:
        diff_pred = pred_keypoints[:, self.upper_lips] - pred_keypoints[:, self.lower_lips]
        diff = keypoints[:, self.upper_lips] - keypoints[:, self.lower_lips]

        loss = (diff_pred.abs().sum(-1) - diff.abs().sum(-1)).abs().mean()

        return loss