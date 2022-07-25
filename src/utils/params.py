import torch
import torch.nn.functional as F


def batch_cont2matrix(module_input):
    ''' Decoder for transforming a latent representation to rotation matrices
        Implements the decoding method described in:
        "On the Continuity of Rotation Representations in Neural Networks"
        Code from https://github.com/vchoutas/expose
    '''
    batch_size = module_input.shape[0]
    reshaped_input = module_input.reshape(-1, 3, 2)

    # Normalize the first vector
    b1 = F.normalize(reshaped_input[:, :, 0].clone(), dim=1)

    dot_prod = torch.sum(
        b1 * reshaped_input[:, :, 1].clone(), dim=1, keepdim=True)
    # Compute the second vector by finding the orthogonal complement to it
    b2 = F.normalize(reshaped_input[:, :, 1] - dot_prod * b1, dim=1)
    # Finish building the basis by taking the cross product
    b3 = torch.cross(b1, b2, dim=1)
    rot_mats = torch.stack([b1, b2, b3], dim=-1)

    return rot_mats.view(batch_size, -1, 3, 3)
