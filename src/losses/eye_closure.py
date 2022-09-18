import torch
from torch import nn
import torch.nn.functional as F

from typing import List, Union



class EyeClosureLoss(nn.Module):
    def __init__(self):
        super(EyeClosureLoss, self).__init__()
        self.register_buffer('upper_lids', torch.LongTensor([37, 38, 43, 44]), persistent=False)
        self.register_buffer('lower_lids', torch.LongTensor([41, 40, 47, 46]), persistent=False)

    def forward(self, 
                pred_keypoints: torch.Tensor,
                keypoints: torch.Tensor) -> torch.Tensor:
        diff_pred = pred_keypoints[:, self.upper_lids] - pred_keypoints[:, self.lower_lids]
        diff = keypoints[:, self.upper_lids] - keypoints[:, self.lower_lids]

        loss = (diff_pred.abs().sum(-1) - diff.abs().sum(-1)).abs().mean()

        return loss