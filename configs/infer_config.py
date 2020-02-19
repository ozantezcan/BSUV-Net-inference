"""
 Config file for inference with a trained model
"""

from utils import augmentations as aug


class SemanticSegmentation:
    root_path = "./utils/segmentation/"
    yaml_path = root_path + "config/ade20k-hrnetv2.yaml"
    encoder_path = root_path + "hrnet_v2/encoder_epoch_30.pth"
    decoder_path = root_path + "hrnet_v2/decoder_epoch_30.pth"


class BSUVNet:
    model_path = "./trained_models/BSUVNet_RGB.mdl"
    seg_ch = True
    mean_rgb = [0.485, 0.456, 0.406]
    std_rgb = [0.229, 0.224, 0.225]
    transforms_pre = []
    transforms_post = [aug.ToTensor(),
                       aug.NormalizeTensor(mean_rgb=mean_rgb, std_rgb=std_rgb,
                                           mean_seg=[0.5], std_seg=[0.5], segmentation_ch=seg_ch)
                       ]
    emtpy_bg = "automatic"   # Automatically create an empty BG frame as median of initial frames
    empty_win_len = 30  # Number of initial frames to be used for the empty BG model
    recent_bg = 10  # Number of last frames to be used for recent BG model
