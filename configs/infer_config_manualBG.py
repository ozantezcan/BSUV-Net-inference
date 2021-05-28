"""
 Config file for inference with a trained model
 For running the model with FPM, carefully read README file
"""

from utils import augmentations as aug
from utils import seg_utils as seg


class SemanticSegmentation:
    root_path = "./utils/segmentation/"
    yaml_path = root_path + "config/ade20k-hrnetv2.yaml"
    encoder_path = root_path + "hrnet_v2/encoder_epoch_30.pth"
    decoder_path = root_path + "hrnet_v2/decoder_epoch_30.pth"


class BSUVNet:

    
    # Model with FPM
    model_path = "./trained_models/BSUV-Net-2.0.mdl"
    seg_network = seg.segModel(SemanticSegmentation.yaml_path,
                               SemanticSegmentation.encoder_path,
                               SemanticSegmentation.decoder_path)
    

    # model without FPM
    # model_path = "./trained_models/Fast-BSUV-Net-2.0.mdl"
    # seg_network = None

    emtpy_bg = "manual"  # Automatically create an empty BG frame as median of initial frames
    empty_win_len = 30  # Number of initial frames to be used for the empty BG model when empty_bg="automatic"
    empty_bg_path = "examples/Candela_m1_10_empty_BG.jpg" # Path of the empty background. Only used when empty_bg="manual"
    recent_bg = 10  # Number of last frames to be used for recent BG model

    seg_ch = False if seg_network is None else True
    mean_rgb = [0.485, 0.456, 0.406]
    std_rgb = [0.229, 0.224, 0.225]
    transforms_pre = []
    transforms_post = [aug.ToTensor(),
                       aug.NormalizeTensor(mean_rgb=mean_rgb, std_rgb=std_rgb,
                                           mean_seg=[0.5], std_seg=[0.5], segmentation_ch=seg_ch)
                       ]
