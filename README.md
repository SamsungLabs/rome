# Realistic one-shot mesh-based avatars

![tease](media/tease.gif)

[**Paper**](https://arxiv.org/abs/2206.08343) | [**Project Page**](https://samsunglabs.github.io/rome)



This repository contains official inference code for ROME.

This code helps you to create personal avatar from just a single image. 
The resulted meshes can be animated and rendered with photorealistic quality.   


### Important disclaimer
To render ROME avatar with pretrained weights you need download [FLAME model](https://flame.is.tue.mpg.de/download.php) and [DECA](https://github.com/YadiraF/DECA) weights. 
DECA reconstructs a 3D head model with detailed facial geometry from a single input image for FLAME template.
Also, it can be replaced by another parametric model.


![tease](media/tease1.gif)

##  Getting started
Initialise submodules and download [DECA](https://github.com/YadiraF/DECA) & [MODNet](https://github.com/ZHKKKe/MODNet) weights

```sh
git submodule update --init --recursive
```

Install requirements and download ROME model: [gDrive](https://drive.google.com/file/d/1ao4-wYu8eRYbVklMB3Hrz2LAAfE4TsP6/view?usp=sharing), [y-disk](https://disk.yandex.ru/d/zfGijJPCbgNHUQ). 

Put model into ```data``` folder.

To verify the code with images run: 

```python
python3 infer.py --deca DECA --rome data
```

For linear basis download ROME model: [gDrive](https://drive.google.com/file/d/1CPZuQFQhjvWHzkonRxeLFLYA1mlTaXXv/view?usp=sharing) (or camera model for voxceleb [gDrive](https://drive.google.com/file/d/1TpOXsmBZxHNVX6cRFiZ3A1uNzFscA-Yd/view?usp=sharing)), [yDrive](https://disk.yandex.ru/d/u2hRXJGewJoCwQ)

```python
python3 infer.py --deca DECA --rome data  --use_distill True
```

### License

This code and model are available for scientific research purposes as defined in the LICENSE file. 
By downloading and using the project you agree to the terms in the LICENSE and DECA LICENSE.
Please note that the restriction on distributing this code for non-scientific purposes is limited.

## Links
This work is based on the great project [DECA](https://github.com/YadiraF/DECA). 
Also we acknowledge additional projects that were essential and speed up the developement.  
- [DECA](https://github.com/YadiraF/DECA) for FLAME regressor and useful functions 
- [Pytorch3D](https://pytorch3d.org/) for differentiable rendering,
- [face-alignment](https://github.com/1adrianb/face-alignment) for keypoints
- [VGGFace2-pytorch](https://github.com/cydonia999/VGGFace2-pytorch) for identity loss  
- [MODNet](https://github.com/ZHKKKe/MODNet), [FaceParsing](https://github.com/zllrunning/face-parsing.PyTorch), [Graphonomy](https://github.com/Gaoyiminggithub/Graphonomy) for fast segmentations   
- [H3DNet](https://github.com/) for evaluation geometry  


## Citation
If you found this code helpful, please consider citing: 

```
@inproceedings{Khakhulin2022ROME,
  author    = {Khakhulin, Taras and Sklyarova,  Vanessa and Lempitsky, Victor and Zakharov, Egor},
  title     = {Realistic One-shot Mesh-based Head Avatars},
  booktitle = {European Conference of Computer vision (ECCV)},
  year      = {2022}
}
```
