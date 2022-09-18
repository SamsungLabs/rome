import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision import transforms
import sys
import os


class FaceParsing(object):
    def __init__(self,
                 path_to_face_parsing,
                 device='cuda'):
        super(FaceParsing, self).__init__()
        import sys
        sys.path.append(path_to_face_parsing)

        from model import BiSeNet

        n_classes = 19
        self.net = BiSeNet(n_classes=n_classes).to(device)
        save_pth = os.path.join(f'{path_to_face_parsing}/res/cp/79999_iter.pth')
        self.net.load_state_dict(torch.load(save_pth, map_location='cpu'))
        self.net.eval()

        self.mean = torch.FloatTensor([0.485, 0.456, 0.406]).to(device)
        self.std = torch.FloatTensor([0.229, 0.224, 0.225]).to(device)

        self.mask_types = {
            'face': [1, 2, 3, 4, 5, 6, 10, 11, 12, 13],
            'ears': [7, 8, 9],
            'neck': [14, 15],
            'cloth': [16],
            'hair': [17, 18],
        }

    @torch.no_grad()
    def forward(self, x):
        h, w = x.shape[2:]
        x = (x - self.mean[None, :, None, None]) / self.std[None, :, None, None]
        x = F.interpolate(x, size=(512, 512), mode='bilinear')
        y = self.net(x)[0]
        y = F.interpolate(y, size=(h, w), mode='bilinear')

        labels = y.argmax(1)

        mask = torch.zeros(x.shape[0], len(self.mask_types.keys()), h, w, dtype=x.dtype, device=x.device)

        for i, indices in enumerate(self.mask_types.values()):
            for j in indices:
                mask[:, i] += labels == j

        return mask