import torch
from torch import nn
import torch.nn.functional as F

from typing import List



class FeatureMatchingLoss(nn.Module):
    def __init__(self, loss_type = 'l1', ):
        super(FeatureMatchingLoss, self).__init__()
        self.loss_type = loss_type

    def forward(self, 
                real_features: List[List[List[torch.Tensor]]], 
                fake_features: List[List[List[torch.Tensor]]]
        ) -> torch.Tensor:
        """
        features: a list of features of different inputs (the third layer corresponds to
                  features of a separate input to each of these discriminators)
        """
        loss = 0

        for real_feats_net, fake_feats_net in zip(real_features, fake_features):
            # *_feats_net corresponds to outputs of a separate discriminator
            loss_net = 0

            for real_feats_layer, fake_feats_layer in zip(real_feats_net, fake_feats_net):
                assert len(real_feats_layer) == 1 or len(real_feats_layer) == len(fake_feats_layer), 'Wrong number of real inputs'
                if len(real_feats_layer) == 1:
                    real_feats_layer = [real_feats_layer[0]] * len(fake_feats_layer)

                for real_feats_layer_i, fake_feats_layer_i in zip(real_feats_layer, fake_feats_layer):
                    if self.loss_type == 'l1':
                        loss_net += F.l1_loss(fake_feats_layer_i, real_feats_layer_i)
                    elif self.loss_type == 'l2':
                        loss_net += F.mse_loss(fake_feats_layer_i, real_feats_layer_i)

            loss_net /= len(fake_feats_layer) # normalize by the number of inputs
            loss_net /= len(fake_feats_net) # normalize by the number of layers
            loss += loss_net

        loss /= len(real_features) # normalize by the number of networks

        return loss