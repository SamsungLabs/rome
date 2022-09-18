from .adversarial import AdversarialLoss
from .feature_matching import FeatureMatchingLoss
from .keypoints_matching import KeypointsMatchingLoss
from .eye_closure import EyeClosureLoss
from .lip_closure import LipClosureLoss
from .head_pose_matching import HeadPoseMatchingLoss
from .perceptual import PerceptualLoss

from .segmentation import SegmentationLoss, MultiScaleSilhouetteLoss
from .chamfer_silhouette import ChamferSilhouetteLoss
from .equivariance import EquivarianceLoss, LaplaceMeshLoss
from .vgg2face import VGGFace2Loss
from .gaze import GazeLoss

from .psnr import PSNR
from .lpips import LPIPS
from pytorch_msssim import SSIM, MS_SSIM