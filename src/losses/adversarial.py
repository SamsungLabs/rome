import torch
from torch import nn
from torch.nn import functional as F

from typing import List



class AdversarialLoss(nn.Module):
    def __init__(self, loss_type = 'hinge'):
        super(AdversarialLoss, self).__init__()
        # TODO: different adversarial loss types
        self.loss_type = loss_type

    def forward(self, 
                fake_scores: List[List[torch.Tensor]], 
                real_scores: List[List[torch.Tensor]] = None, 
                mode: str = 'gen') -> torch.Tensor:
        """
        scores: a list of lists of scores (the second layer corresponds to a
                separate input to each of these discriminators)
        """
        loss = 0

        if mode == 'dis':
            for real_scores_net, fake_scores_net in zip(real_scores, fake_scores):
                # *_scores_net corresponds to outputs of a separate discriminator
                loss_real = 0
                
                for real_scores_net_i in real_scores_net:
                    if self.loss_type == 'hinge':
                        loss_real += torch.relu(1.0 - real_scores_net_i).mean()
                    else:
                        raise # not implemented
                
                loss_real /= len(real_scores_net)

                loss_fake = 0
                
                for fake_scores_net_i in fake_scores_net:
                    if self.loss_type == 'hinge':
                        loss_fake += torch.relu(1.0 + fake_scores_net_i).mean()
                    else:
                        raise # not implemented
                
                loss_fake /= len(fake_scores_net)

                loss_net = loss_real + loss_fake
                loss += loss_net

        elif mode == 'gen':
            for fake_scores_net in fake_scores:
                assert isinstance(fake_scores_net, list), 'Expect a list of fake scores per discriminator'

                loss_net = 0

                for fake_scores_net_i in fake_scores_net:
                    if self.loss_type == 'hinge':
                        # *_scores_net_i corresponds to outputs for separate inputs
                        loss_net -= fake_scores_net_i.mean()

                    else:
                        raise # not implemented

                loss_net /= len(fake_scores_net) # normalize by the number of inputs
                loss += loss_net
        
        loss /= len(fake_scores) # normalize by the nubmer of discriminators

        return loss