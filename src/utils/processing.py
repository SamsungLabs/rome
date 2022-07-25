import torch
import numpy as np
import torch.nn as nn
import torchvision.models as models


def create_regressor(model_name, num_params):
    models_mapping = {
                'resnet50': lambda x : models.resnet50(pretrained=True),
                'resnet18': lambda x : models.resnet18(pretrained=True),
                'efficientnet_b0': lambda x : models.efficientnet_b0(pretrained=True),
                'efficientnet_b3': lambda x : models.efficientnet_b3(pretrained=True),
                'mobilenet_v3_large': lambda x : models.mobilenet_v3_large(pretrained=True),
                'mobilenet_v3_small': lambda x : models.mobilenet_v3_small(pretrained=True),
                'mobilenet_v2': lambda x: models.mobilenet_v2(pretrained=True),
                 }
    model = models_mapping[model_name](True)
    if model_name == 'resnet50':
        model.fc = nn.Linear(in_features=2048, out_features=num_params, bias=True)
    elif model_name == 'resnet18':
        model.fc = nn.Linear(in_features=512, out_features=num_params, bias=True)
    elif model_name == 'mobilenet_v3_large':
        model.classifier[3] = nn.Linear(in_features=1280, out_features=num_params, bias=True)
    elif model_name == 'mobilenet_v3_small':
        model.classifier[3] = nn.Linear(in_features=1024, out_features=num_params, bias=True)    
    elif model_name == 'efficientnet_b3':
        model.classifier[1] = nn.Linear(in_features=1536, out_features=num_params, bias=True)
    elif model_name == 'efficientnet_b0':
        model.classifier[1] = nn.Linear(in_features=1280, out_features=num_params, bias=True)
    elif model_name == 'mobilenet_v2':
        model.classifier[1] = nn.Linear(in_features=1280, out_features=num_params, bias=True)
    else:
        model.fc = nn.Linear(in_features=10, out_features=num_params, bias=True)
    return model


def process_black_shape(shape_img):
    black_mask = shape_img == 0.0
    shape_img[black_mask] = 1.0
    return shape_img


def prepare_input_data(data_dict):
        for k, v in data_dict.items():
            if isinstance(v, dict):
                for k_, v_ in v.items():
                    v_ = v_.cuda()
                    v[k_] = v_.view(-1, *v_.shape[2:])
                data_dict[k] = v
            else:
                v = v.cuda()
                data_dict[k] = v.view(-1, *v.shape[2:])

        return data_dict


def tensor2image(tensor):
    image = tensor.detach().cpu().numpy()
    image = image * 255.
    image = np.maximum(np.minimum(image, 255), 0)
    image = image.transpose(1,2,0)[:,:,[2,1,0]]
    return image.astype(np.uint8).copy()

